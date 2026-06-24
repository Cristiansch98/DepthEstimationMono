"""Evaluation: depth accuracy + self-calibration accuracy + vehicle-distance error.

AV2 ships per-camera GT intrinsics and 3D vehicle cuboids, so all three are
measurable. Distances use the *learned* theta, so the vehicle-distance number
reflects the whole pipeline (depth + self-calibration) end to end.

    python -m calib_depth.eval --ckpt checkpoints/final.pt --data-root data/sensor --max-frames 300
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch

from .benchmarks import BENCHMARKS, get_adapter
from .camera_model import Intrinsics
from .dataset import BenchmarkDepthDataset
from .model import CameraAwareDepthNet

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("eval")
logging.getLogger("lidar-depth").setLevel(logging.WARNING)
logging.getLogger("vehicle-dist").setLevel(logging.WARNING)

TARGET_HW = (518, 518)


def depth_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    m = (gt > 0) & np.isfinite(pred)
    pred, gt = pred[m].clip(1e-3, None), gt[m]
    thresh = np.maximum(gt / pred, pred / gt)
    return {
        "AbsRel": float(np.mean(np.abs(pred - gt) / gt)),
        "RMSE": float(np.sqrt(np.mean((pred - gt) ** 2))),
        "d1": float((thresh < 1.25).mean()),
        "n": int(m.sum()),
    }


def sample_map(depth_map: torch.Tensor, uv: np.ndarray) -> np.ndarray:
    """Bilinear-sample a (1,1,H,W) depth map at pixel coords uv (N,2)."""
    h, w = depth_map.shape[-2:]
    g = torch.from_numpy(uv).float().clone()
    g[:, 0] = g[:, 0] / (w - 1) * 2 - 1
    g[:, 1] = g[:, 1] / (h - 1) * 2 - 1
    g = g.view(1, -1, 1, 2).to(depth_map.device)
    return torch.nn.functional.grid_sample(depth_map, g, align_corners=True).view(-1).cpu().numpy()


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--benchmark", default="av2", choices=BENCHMARKS)
    ap.add_argument("--data-root", type=Path, default=Path("data/sensor"))
    ap.add_argument("--split", default="val", help="subdir under data-root (av2: train/val)")
    ap.add_argument("--cams", nargs="+", default=None)
    ap.add_argument("--max-frames", type=int, default=300)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    adapter = get_adapter(args.benchmark)
    cams = args.cams or adapter.default_cams
    ck = torch.load(args.ckpt, map_location=dev)
    # Build the model with the *checkpoint's* camera count so cross-dataset eval
    # (a different #cameras than training) still loads. cam_idx is clamped below.
    n_train_cams = int(ck["model"]["calib.latent"].shape[0])
    model = CameraAwareDepthNet(n_train_cams, Intrinsics(500, 500, 259, 259),
                                freeze_backbone=True).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()
    log.info("loaded %s (step %s); model has %d camera latents",
             args.ckpt, ck.get("step"), n_train_cams)
    if args.benchmark != "av2":
        log.warning("CROSS-DATASET: per-camera latent is AV2-trained; front cam -> "
                    "latent[0]. Calibration transfer is carried only by the small "
                    "image-conditioned residual (see model.py).")

    root = args.data_root / args.split if (args.data_root / args.split).exists() else args.data_root
    ds = BenchmarkDepthDataset.build(args.benchmark, root, cams=cams, target_hw=TARGET_HW)
    idxs = np.linspace(0, len(ds) - 1, min(args.max_frames, len(ds))).astype(int)
    log.info("evaluating %d/%d %s val frames", len(idxs), len(ds), args.benchmark)

    dep, fx_err, fy_err, cx_err, cy_err = [], [], [], [], []
    veh_abs, veh_rel, veh_bucket = [], [], {"0-30": [], "30-60": [], "60+": []}

    for n, i in enumerate(idxs):
        s = ds[int(i)]
        img = s["image"][None].to(dev)
        cam_idx = torch.tensor([min(s["cam_idx"], n_train_cams - 1)], device=dev)
        depth, theta = model(img, cam_idx)
        th = theta[0].cpu().numpy()
        tg = s["theta_gt"].numpy()

        # depth metrics at LiDAR pixels (z-depth)
        pred = sample_map(depth, s["uv"].numpy())
        dep.append(depth_metrics(pred, s["gt_depth"].numpy()))
        fx_err.append(abs(th[0] - tg[0]) / tg[0]); fy_err.append(abs(th[1] - tg[1]) / tg[1])
        cx_err.append(abs(th[2] - tg[2])); cy_err.append(abs(th[3] - tg[3]))

        # vehicle distance: predicted euclidean range (uses learned theta) vs GT.
        # Adapter returns 3D-box targets where the benchmark has them ([] otherwise).
        try:
            targets = adapter.vehicle_targets(ds.refs[int(i)])
        except Exception:
            targets = []
        ow, oh = s["orig_wh"]
        sx, sy = TARGET_HW[1] / ow, TARGET_HW[0] / oh
        for t in targets:
            u, v = t.uv_center[0] * sx, t.uv_center[1] * sy
            if not (0 <= u < TARGET_HW[1] and 0 <= v < TARGET_HW[0]):
                continue
            z = float(sample_map(depth, np.array([[u, v]], dtype=np.float32))[0])
            xn, yn = (u - th[2]) / th[0], (v - th[3]) / th[1]
            range_pred = z * np.sqrt(xn * xn + yn * yn + 1)  # z -> euclidean range
            err = abs(range_pred - t.distance_m)
            veh_abs.append(err); veh_rel.append(err / t.distance_m)
            b = "0-30" if t.distance_m < 30 else ("30-60" if t.distance_m < 60 else "60+")
            veh_bucket[b].append(err)
        if (n + 1) % 50 == 0:
            log.info("  %d/%d frames", n + 1, len(idxs))

    def agg(key):
        return float(np.mean([d[key] for d in dep]))

    print("\n================  EVAL  ================")
    print(f"DEPTH (LiDAR):  AbsRel {agg('AbsRel'):.3f} | RMSE {agg('RMSE'):.2f} m | d<1.25 {agg('d1'):.3f}")
    print(f"CALIBRATION:    fx {np.mean(fx_err)*100:.2f}% | fy {np.mean(fy_err)*100:.2f}% | "
          f"cx {np.mean(cx_err):.1f}px | cy {np.mean(cy_err):.1f}px  (vs {args.benchmark} GT)")
    if veh_abs:
        va = np.array(veh_abs)
        near = va[: len(veh_bucket["0-30"]) + len(veh_bucket["30-60"])] if False else None
        close = np.concatenate([np.array(veh_bucket["0-30"]), np.array(veh_bucket["30-60"])]) \
            if (veh_bucket["0-30"] or veh_bucket["30-60"]) else np.array([])
        print(f"VEHICLE DIST:   MAE {va.mean():.2f} m | rel {np.mean(veh_rel)*100:.1f}%  (n={len(veh_abs)})")
        # P-D: headline number caps at 60 m (monocular is unreliable past that; LiDAR sparse)
        if close.size:
            print(f"    <=60 m (headline):  MAE {close.mean():.2f} m  (n={close.size})")
        for b, v in veh_bucket.items():
            if v:
                print(f"    {b:>6} m:  MAE {np.mean(v):.2f} m  (n={len(v)})")
    print("=======================================")


if __name__ == "__main__":
    main()
