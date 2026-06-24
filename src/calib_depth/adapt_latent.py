"""Few-shot per-camera latent adaptation (cross-dataset generalization test).

The cross-dataset eval showed the AV2-trained model does not zero-shot transfer:
``theta = latent[cam_idx] + 0.1·delta_head(feat)`` is dominated by the AV2-learned
per-camera latent, so on a new camera it predicts ≈the AV2 focal. This script
tests the obvious fix: freeze the ENTIRE network except the single per-camera
latent row and optimize just it (log fx, log fy, cx, cy, [k1, k2]) on a handful of
target-dataset frames, supervised only by the LiDAR 3D / reproj losses — exactly
the signal that recovered focal length "from a wrong init in ~100 steps" originally.

Reports depth + calibration metrics on HELD-OUT target frames, before vs after.

    python -m calib_depth.adapt_latent --benchmark kitti \
        --ckpt checkpoints_v3/final.pt --data-root <kitti_raw> --cam image_02 \
        --adapt-frames 20 --steps 300
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch

from .benchmarks import BENCHMARKS, get_adapter
from .camera_model import Intrinsics
from .dataset import BenchmarkDepthDataset
from .eval import depth_metrics, sample_map
from .losses import total_loss
from .model import CameraAwareDepthNet

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("adapt")
logging.getLogger("lidar-depth").setLevel(logging.WARNING)

TARGET_HW = (518, 518)
# latent-only adaptation: depth head is frozen, so the depth terms only steer theta
# through the 3D back-projection; weight the theta-driving terms.
WEIGHTS = {"silog": 0.5, "l3d": 1.0, "reproj": 0.1, "smooth": 0.0, "prior": 0.01}


def evaluate(model, ds, idxs, dev, n_train_cams) -> dict:
    model.eval()
    dep, fx_err, fy_err, cx_err, cy_err = [], [], [], [], []
    with torch.no_grad():
        for i in idxs:
            s = ds[int(i)]
            cam_idx = torch.tensor([min(s["cam_idx"], n_train_cams - 1)], device=dev)
            depth, theta = model(s["image"][None].to(dev), cam_idx)
            th, tg = theta[0].cpu().numpy(), s["theta_gt"].numpy()
            pred = sample_map(depth, s["uv"].numpy())
            dep.append(depth_metrics(pred, s["gt_depth"].numpy()))
            fx_err.append(abs(th[0] - tg[0]) / tg[0]); fy_err.append(abs(th[1] - tg[1]) / tg[1])
            cx_err.append(abs(th[2] - tg[2])); cy_err.append(abs(th[3] - tg[3]))
    agg = lambda k: float(np.mean([d[k] for d in dep]))
    return {"AbsRel": agg("AbsRel"), "RMSE": agg("RMSE"), "d1": agg("d1"),
            "fx%": np.mean(fx_err) * 100, "fy%": np.mean(fy_err) * 100,
            "cx": np.mean(cx_err), "cy": np.mean(cy_err)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--benchmark", default="kitti", choices=BENCHMARKS)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--split", default="")
    ap.add_argument("--cam", required=True, help="single target camera (-> latent[0])")
    ap.add_argument("--adapt-frames", type=int, default=20, help="few-shot adaptation budget")
    ap.add_argument("--eval-frames", type=int, default=100, help="held-out eval frames")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--adapt-head", action="store_true",
                    help="also unfreeze the depth ScaleMappingHead (recovers metric depth)")
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--no-aspect-prior", action="store_true",
                    help="drop the (log fx/fy)^2 prior (wrong under anisotropic 518^2 resize)")
    ap.add_argument("--distort", default=None,
                    help="inject Brown-Conrady radial distortion 'k1[,k2]' into target frames")
    ap.add_argument("--fisheye", default=None,
                    help="render target frames through a KB fisheye 'k1[,k2]'")
    ap.add_argument("--distort-bound", type=float, default=None,
                    help="widen the model radial bound (default 0.05 assumes near-pinhole; "
                         "must exceed the true |k| to be representable)")
    ap.add_argument("--out-ckpt", type=Path, default=None)
    args = ap.parse_args()

    distort = None
    if args.distort:
        ks = [float(x) for x in args.distort.split(",")]
        distort = (ks[0], ks[1] if len(ks) > 1 else 0.0)
    fisheye = None
    if args.fisheye:
        ks = [float(x) for x in args.fisheye.split(",")]
        fisheye = (ks[0], ks[1] if len(ks) > 1 else 0.0)
    if args.distort_bound is not None:
        # Justified: A4 (near-pinhole, |k|<=0.05) is violated by construction here;
        # the learnable radial bound must exceed the true distortion to represent it.
        import calib_depth.model as _M
        _M.DISTORTION_BOUND = args.distort_bound

    weights = dict(WEIGHTS)
    if args.no_aspect_prior:
        weights["aspect_w"] = 0.0
    if args.adapt_head:
        weights["smooth"] = 0.05  # light edge-aware regularization when depth is trainable

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.ckpt, map_location=dev)
    n_train_cams = int(ck["model"]["calib.latent"].shape[0])
    model = CameraAwareDepthNet(n_train_cams, Intrinsics(500, 500, 259, 259),
                                freeze_backbone=True).to(dev)
    model.load_state_dict(ck["model"])
    log.info("loaded %s (step %s); %d camera latents", args.ckpt, ck.get("step"), n_train_cams)

    # Freeze everything; train ONLY the per-camera latent (front cam -> row 0).
    for p in model.parameters():
        p.requires_grad_(False)
    model.calib.latent.requires_grad_(True)
    groups = [{"params": [model.calib.latent], "lr": args.lr}]
    if args.adapt_head:
        # also adapt the depth metric-scale head (small) so depth, not just theta, transfers
        for p in model.head.parameters():
            p.requires_grad_(True)
        groups.append({"params": list(model.head.parameters()), "lr": args.lr_head})
        log.info("adapting per-camera latent + depth head (%.1fk head params)",
                 sum(p.numel() for p in model.head.parameters()) / 1e3)
    else:
        log.info("adapting per-camera latent only")
    opt = torch.optim.Adam(groups)  # no weight decay

    root = args.data_root / args.split if args.split and (args.data_root / args.split).exists() \
        else args.data_root
    ds = BenchmarkDepthDataset.build(args.benchmark, root, cams=[args.cam], target_hw=TARGET_HW,
                                     distort=distort, fisheye=fisheye)
    n = len(ds)
    adapt_idx = list(range(min(args.adapt_frames, n)))
    pool = list(range(args.adapt_frames, n)) or list(range(n))  # held-out (fallback: reuse)
    eval_idx = np.linspace(0, len(pool) - 1, min(args.eval_frames, len(pool))).astype(int)
    eval_idx = [pool[j] for j in eval_idx]
    log.info("%s: %d frames | adapt on %d, eval on %d (held-out)",
             args.benchmark, n, len(adapt_idx), len(eval_idx))

    before = evaluate(model, ds, eval_idx, dev, n_train_cams)
    log.info("BEFORE  AbsRel %.3f RMSE %.2f d1 %.3f | fx %.1f%% fy %.1f%% cx %.1f cy %.1f",
             before["AbsRel"], before["RMSE"], before["d1"], before["fx%"], before["fy%"],
             before["cx"], before["cy"])

    # ---- few-shot adaptation loop (fp32; skip non-finite, like train.py) ----
    model.eval()  # keep frozen modules (norms) in eval; only latent is a leaf param
    skipped = 0
    for step in range(args.steps):
        i = adapt_idx[step % len(adapt_idx)]
        s = ds[int(i)]
        batch = {k: (v.to(dev) if hasattr(v, "to") else v) for k, v in s.items()}
        cam_idx = torch.tensor([min(s["cam_idx"], n_train_cams - 1)], device=dev)
        img4 = batch["image"][None]
        depth, theta = model(img4, cam_idx)
        loss, parts = total_loss(depth, theta[0], {**batch, "image": img4}, weights)
        if not torch.isfinite(loss):
            skipped += 1; continue
        opt.zero_grad(); loss.backward()
        trainable = [p for g in groups for p in g["params"]]
        gnorm = torch.nn.utils.clip_grad_norm_(trainable, 10.0)
        if not torch.isfinite(gnorm):
            opt.zero_grad(set_to_none=True); skipped += 1; continue
        opt.step()
        if step % 50 == 0 or step == args.steps - 1:
            fx_p, fy_p = float(theta[0, 0]), float(theta[0, 1])
            fx_g, fy_g = float(s["theta_gt"][0]), float(s["theta_gt"][1])
            log.info("step %3d | loss %.4f | %s | fx %.0f/%.0f fy %.0f/%.0f (pred/GT)",
                     step, float(loss), " ".join(f"{k}={v:.3f}" for k, v in parts.items()),
                     fx_p, fx_g, fy_p, fy_g)

    after = evaluate(model, ds, eval_idx, dev, n_train_cams)
    if skipped:
        log.info("skipped %d non-finite steps", skipped)

    print("\n========  FEW-SHOT LATENT ADAPTATION  (%s, %s)  ========" % (args.benchmark, args.cam))
    print("                AbsRel    RMSE     d1      fx%%     fy%%    cx(px)  cy(px)")
    for tag, m in [("before", before), ("after ", after)]:
        print("  %s     %6.3f  %6.2f  %6.3f  %6.1f  %6.1f  %6.1f  %6.1f" %
              (tag, m["AbsRel"], m["RMSE"], m["d1"], m["fx%"], m["fy%"], m["cx"], m["cy"]))
    print("========================================================")

    if args.out_ckpt:
        args.out_ckpt.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "step": ck.get("step"),
                    "adapted": args.benchmark}, args.out_ckpt)
        log.info("saved adapted ckpt -> %s", args.out_ckpt)


if __name__ == "__main__":
    main()
