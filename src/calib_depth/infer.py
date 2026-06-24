"""Inference + demo: a single camera frame -> predicted metric depth, estimated
intrinsics theta, and per-vehicle predicted-vs-GT distance.

On an AV2 val frame it overlays vehicle boxes labelled "pred / GT m" so the
end-goal is directly visible. Distances use the *estimated* theta, so the demo
exercises depth + self-calibration together.

ASSUMPTIONS: same as model.py (A1-A5). Predicted euclidean range = z * |ray|,
with z the predicted depth-along-axis and the ray built from estimated theta.

    python -m calib_depth.infer --ckpt checkpoints_v2/final.pt \
        --log-dir data/sensor/val/<id> --cam ring_front_center --out viz
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .camera_model import Intrinsics
from .model import CameraAwareDepthNet

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from vehicle_distance import vehicle_targets  # noqa: E402
import cv2  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("infer")
logging.getLogger("lidar-depth").setLevel(logging.WARNING)
logging.getLogger("vehicle-dist").setLevel(logging.WARNING)

CAMS = ["ring_front_center", "ring_front_left", "ring_front_right", "ring_side_left",
        "ring_side_right", "ring_rear_left", "ring_rear_right"]
HW = (518, 518)


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--log-dir", type=Path, required=True)
    ap.add_argument("--cam", default="ring_front_center")
    ap.add_argument("--lidar-ts", type=int, default=None)
    ap.add_argument("--out", type=Path, default=Path("viz"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = CameraAwareDepthNet(len(CAMS), Intrinsics(500, 500, 259, 259), freeze_backbone=True).to(dev)
    model.load_state_dict(torch.load(args.ckpt, map_location=dev)["model"])
    model.eval()

    img_full, targets, cam, cam_ts = vehicle_targets(args.log_dir, args.cam, args.lidar_ts)
    img = cv2.resize(img_full, (HW[1], HW[0]), interpolation=cv2.INTER_AREA)
    x = torch.from_numpy(img).permute(2, 0, 1).float().div(255)[None].to(dev)
    cam_idx = torch.tensor([CAMS.index(args.cam)], device=dev)
    depth, theta = model(x, cam_idx)
    th = theta[0].cpu().numpy()
    log.info("estimated theta: fx=%.1f fy=%.1f cx=%.1f cy=%.1f k1=%.4f (GT fx=%.1f@%dx%d)",
             th[0], th[1], th[2], th[3], th[4],
             cam.intrinsics.fx_px * HW[1] / cam.width_px, cam.width_px, cam.height_px)
    dmap = depth[0, 0].cpu().numpy()

    sx, sy = HW[1] / cam.width_px, HW[0] / cam.height_px
    rows = []
    for t in targets:
        u, v = t.uv_center[0] * sx, t.uv_center[1] * sy
        if not (0 <= u < HW[1] and 0 <= v < HW[0]):
            continue
        z = float(dmap[int(round(v)), int(round(u))])
        xn, yn = (u - th[2]) / th[0], (v - th[3]) / th[1]
        rng = z * float(np.sqrt(xn * xn + yn * yn + 1))
        rows.append((u, v, rng, t.distance_m, t.box_uv * np.array([sx, sy])))

    # figure: predicted depth + vehicle pred/GT labels
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(20, 8), dpi=120)
    a1.imshow(img); a1.set_title("input + vehicle distance  (pred / GT)", fontweight="bold")
    cmap = plt.get_cmap("turbo")
    for u, v, rng, gt, box in rows:
        col = cmap(min(gt / 60, 1.0))
        for e in [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
                  (0, 4), (1, 5), (2, 6), (3, 7)]:
            a1.plot([box[e[0], 0], box[e[1], 0]], [box[e[0], 1], box[e[1], 1]], color=col, lw=1.2)
        a1.text(u, v, f"{rng:.0f}/{gt:.0f}", color="white", fontsize=8, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.1", fc=col, ec="white", lw=0.4, alpha=0.85))
    a1.axis("off")
    im = a2.imshow(dmap, cmap="turbo", vmin=2, vmax=60)
    a2.set_title("predicted metric depth [m]  (theta self-estimated)", fontweight="bold")
    a2.axis("off"); fig.colorbar(im, ax=a2, fraction=0.025)
    out = args.out / f"infer_{args.cam}_{cam_ts}.png"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.1, facecolor="white")
    plt.close(fig)
    if rows:
        err = np.mean([abs(r[2] - r[3]) for r in rows])
        log.info("%d vehicles | mean |pred-GT| = %.2f m", len(rows), err)
    log.info("saved -> %s", out)


if __name__ == "__main__":
    main()
