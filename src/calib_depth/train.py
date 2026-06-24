"""Training entry point (run on the remote RTX 5090).

Implements the loop, param-grouped optimization, checkpointing and logging.
Building the foundation-model backbone is the only remaining GPU-side step
(see model.py / FRAMEWORK.md milestone 3).

    python -m calib_depth.train --data-root data/sensor-sample --epochs 20
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .benchmarks import BENCHMARKS, get_adapter
from .camera_model import Intrinsics
from .dataset import BenchmarkDepthDataset
from .losses import total_loss
from .model import CameraAwareDepthNet

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("train")

LOSS_WEIGHTS = {"silog": 1.0, "l3d": 0.5, "reproj": 0.05, "smooth": 0.1, "prior": 0.01}


def _first(batch):
    """Top-level collate (batch_size=1) — must be picklable for DataLoader workers."""
    return batch[0]


def save_ckpt(path: Path, model, opt, step: int) -> None:
    import torch
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step}, path)
    log.info("checkpoint -> %s (step %d)", path, step)


def main() -> None:
    import torch
    from torch.utils.data import DataLoader

    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default="av2", choices=BENCHMARKS,
                    help="dataset adapter: av2 | kitti | nuscenes | lyft")
    ap.add_argument("--data-root", type=Path, default=Path("data/sensor-sample"))
    ap.add_argument("--cams", nargs="+", default=None,
                    help="camera names (default: the benchmark's standard set)")
    ap.add_argument("--stride", type=int, default=1, help="subsample frames per scene")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr-backbone", type=float, default=1e-5)
    ap.add_argument("--lr-head", type=float, default=1e-4)
    ap.add_argument("--ckpt-dir", type=Path, default=Path("checkpoints"))
    ap.add_argument("--ckpt-every", type=int, default=500)
    ap.add_argument("--freeze-calib", action="store_true",
                    help="curriculum stage 1: keep theta at GT, fine-tune depth only")
    ap.add_argument("--unfreeze-backbone", action="store_true",
                    help="P-B: fine-tune the DAv2 backbone (low LR) instead of freezing it")
    args = ap.parse_args()

    logging.getLogger("lidar-depth").setLevel(logging.WARNING)  # silence per-frame projection logs
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cams = args.cams or get_adapter(args.benchmark).default_cams
    log.info("device=%s  benchmark=%s  data_root=%s  cams=%s", device, args.benchmark,
             args.data_root, cams)

    ds = BenchmarkDepthDataset.build(args.benchmark, args.data_root, cams=cams, stride=args.stride)
    log.info("dataset: %d frames over %d cameras", len(ds), len(cams))
    loader = DataLoader(ds, batch_size=1, shuffle=True, num_workers=6, collate_fn=_first)

    # Deliberately generic init at training resolution (518) so we can watch the
    # self-calibration recover each camera's true focal length from a wrong start.
    init = Intrinsics(fx=500.0, fy=500.0, cx=259.0, cy=259.0)
    model = CameraAwareDepthNet(num_cameras=len(cams), init_intrinsics=init,
                                freeze_backbone=not args.unfreeze_backbone).to(device)
    log.info("backbone %s | trainable params: %.2fM",
             "UNFROZEN" if args.unfreeze_backbone else "frozen",
             sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6)
    if args.freeze_calib:
        for p in model.calib.parameters():
            p.requires_grad_(False)

    groups = [
        {"params": [p for n, p in model.named_parameters() if "dav2" in n and p.requires_grad],
         "lr": args.lr_backbone},
        {"params": [p for n, p in model.named_parameters() if "dav2" not in n and p.requires_grad],
         "lr": args.lr_head},
    ]
    groups = [g for g in groups if g["params"]]  # drop empty groups (e.g. frozen backbone)
    opt = torch.optim.AdamW(groups)

    # fp32: the trainable model is tiny (<1 GB), and metric (metre-scale) 3D losses
    # overflow under fp16 -> NaN. No AMP; clip grads and skip non-finite steps.
    step = 0
    skipped = 0
    for epoch in range(args.epochs):
        for batch in loader:
            batch = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}
            cam_idx = torch.tensor([batch["cam_idx"]], device=device)
            img4 = batch["image"][None]
            loss_batch = {**batch, "image": img4}
            opt.zero_grad()
            depth, theta = model(img4, cam_idx)
            loss, parts = total_loss(depth, theta[0], loss_batch, LOSS_WEIGHTS)
            if not torch.isfinite(loss):
                skipped += 1
                step += 1
                continue
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 10.0)
            if not torch.isfinite(gnorm):  # don't let a bad grad poison theta
                opt.zero_grad(set_to_none=True)
                skipped += 1
                step += 1
                continue
            opt.step()
            if step % 50 == 0:
                fx_p, fy_p = float(theta[0, 0]), float(theta[0, 1])
                fx_g, fy_g = float(batch["theta_gt"][0]), float(batch["theta_gt"][1])
                log.info("ep %d step %d | loss %.4f | %s | fx %.0f/%.0f fy %.0f/%.0f (pred/GT)",
                         epoch, step, float(loss),
                         " ".join(f"{k}={v:.3f}" for k, v in parts.items()),
                         fx_p, fx_g, fy_p, fy_g)
            if step % args.ckpt_every == 0 and step > 0:
                save_ckpt(args.ckpt_dir / f"step_{step}.pt", model, opt, step)
            step += 1
    save_ckpt(args.ckpt_dir / "final.pt", model, opt, step)
    log.info("done.")


if __name__ == "__main__":
    main()
