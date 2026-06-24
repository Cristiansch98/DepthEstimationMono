"""Paper-quality cross-dataset qualitative figure.

One row per benchmark (Argoverse 2 / KITTI / nuScenes), four panels each:

    [ input RGB ] [ GT LiDAR depth ] [ predicted metric depth ] [ abs. error ]

GT and error are the sparse LiDAR returns (the only place we have ground truth),
coloured on a dimmed image; predicted depth is the dense map. Depth shares one
colourbar (m), error another (m). Each row is labelled with the dataset, its true
intrinsics, and the held-out metrics. Runs on the remote (needs torch); produces a
single high-DPI PNG.

    python -m calib_depth.figures --out viz_paper/cross_dataset_qualitative.png
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import cm
from matplotlib.colors import Normalize

from .camera_model import Intrinsics
from .dataset import BenchmarkDepthDataset
from .eval import depth_metrics, sample_map
from .model import CameraAwareDepthNet

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("figures")
logging.getLogger("lidar-depth").setLevel(logging.WARNING)

HW = (518, 518)
DMIN, DMAX = 2.0, 60.0          # depth colourbar (m)
EMAX = 12.0                      # error colourbar (m)
DEPTH_CMAP, ERR_CMAP = "turbo", "inferno"

# (title, benchmark, data-root, cam, ckpt, held-out metrics for the row label)
HOME = str(Path.home())
CONFIG = [
    ("Argoverse 2", "av2", f"{HOME}/Cubos_code/SelfCalibDepth/data/sensor/val",
     "ring_front_center", f"{HOME}/Cubos_code/SelfCalibDepth/checkpoints_v3/final.pt",
     dict(AbsRel=0.112, d1=0.884, fx=0.26, mode="in-domain")),
    ("KITTI", "kitti", f"{HOME}/Cubos_code/data_bench/kitti",
     "image_02", f"{HOME}/Cubos_code/SelfCalibDepth/checkpoints_adapt/kitti.pt",
     dict(AbsRel=0.086, d1=0.955, fx=1.1, mode="20-frame adapt")),
    ("nuScenes", "nuscenes", f"{HOME}/Cubos_code/data_bench/nuscenes",
     "CAM_FRONT", f"{HOME}/Cubos_code/SelfCalibDepth/checkpoints_adapt/nuscenes.pt",
     dict(AbsRel=0.119, d1=0.865, fx=0.7, mode="20-frame adapt")),
]


def _load_model(ckpt_path: str, dev: str):
    ck = torch.load(ckpt_path, map_location=dev)
    n = int(ck["model"]["calib.latent"].shape[0])
    model = CameraAwareDepthNet(n, Intrinsics(500, 500, 259, 259), freeze_backbone=True).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, n


@torch.no_grad()
def _pick_busy_frame(ds, scan: int = 25):
    """Return the index (within the first ``scan``) with the most LiDAR returns."""
    best, best_n = 0, -1
    for i in range(min(scan, len(ds))):
        n = len(ds[i]["gt_depth"])
        if n > best_n:
            best, best_n = i, n
    return best


def _dim(img: np.ndarray) -> np.ndarray:
    return (img.astype(np.float32) * 0.45 + 60).clip(0, 255).astype(np.uint8)


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("viz_paper/cross_dataset_qualitative.png"))
    ap.add_argument("--scan", type=int, default=25)
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    rows = []
    for title, bench, root, cam, ckpt, meta in CONFIG:
        log.info("rendering %s ...", title)
        model, ncam = _load_model(ckpt, dev)
        ds = BenchmarkDepthDataset.build(bench, Path(root), cams=[cam], target_hw=HW)
        idx = _pick_busy_frame(ds, args.scan)
        s = ds[idx]
        cam_idx = torch.tensor([min(s["cam_idx"], ncam - 1)], device=dev)
        depth, theta = model(s["image"][None].to(dev), cam_idx)
        dmap = depth[0, 0].cpu().numpy()
        img = (s["image"].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        uv, gt = s["uv"].numpy(), s["gt_depth"].numpy()
        pred_at = sample_map(depth, uv)
        err = np.abs(pred_at - gt)
        m = depth_metrics(pred_at, gt)
        th, tg = theta[0].cpu().numpy(), s["theta_gt"].numpy()
        ow, oh = s["orig_wh"]
        native_fx = tg[0] * ow / HW[1]   # undo the 518-resize -> sensor-native focal
        log.info("  %s frame %d | %d returns | AbsRel %.3f d1 %.3f", title, idx,
                 len(gt), m["AbsRel"], m["d1"])
        rows.append((title, meta, img, uv, gt, dmap, err, native_fx, (ow, oh)))

    # ---- compose ----
    nr = len(rows)
    fig, axes = plt.subplots(nr, 4, figsize=(17, 3.7 * nr), dpi=200)
    if nr == 1:
        axes = axes[None]
    dnorm, enorm = Normalize(DMIN, DMAX), Normalize(0, EMAX)
    col_titles = ["Input image", "Ground-truth LiDAR depth",
                  "Predicted metric depth", "Absolute error  |pred − GT|"]
    order_small_first = lambda d: np.argsort(-d)  # draw far/large first

    for r, (title, meta, img, uv, gt, dmap, err, native_fx, owh) in enumerate(rows):
        ax = axes[r]
        o = order_small_first(gt)
        ax[0].imshow(img)
        ax[1].imshow(_dim(img))
        ax[1].scatter(uv[o, 0], uv[o, 1], c=gt[o], s=6, cmap=DEPTH_CMAP, norm=dnorm)
        ax[2].imshow(dmap, cmap=DEPTH_CMAP, norm=dnorm)
        ax[3].imshow(_dim(img))
        oe = np.argsort(err)  # large errors on top
        ax[3].scatter(uv[oe, 0], uv[oe, 1], c=err[oe], s=6, cmap=ERR_CMAP, norm=enorm)
        for a in ax:
            a.set_xticks([]); a.set_yticks([])
        if r == 0:
            for c in range(4):
                ax[c].set_title(col_titles[c], fontsize=12, fontweight="bold", pad=8)
        # row label (left), with intrinsics + held-out metrics
        label = (f"{title}\n"
                 f"{owh[0]}×{owh[1]}\n"
                 f"fx≈{native_fx:.0f}px\n"
                 f"——————\n"
                 f"AbsRel {meta['AbsRel']:.3f}\n"
                 f"δ<1.25 {meta['d1']:.3f}\n"
                 f"fx err {meta['fx']:.1f}%\n"
                 f"[{meta['mode']}]")
        ax[0].set_ylabel(label, fontsize=10.5, rotation=0, ha="right", va="center",
                         labelpad=46, fontweight="bold")

    fig.suptitle("SelfCalibDepth — cross-dataset single-image metric depth & self-calibration",
                 fontsize=15, fontweight="bold", y=0.995)
    fig.subplots_adjust(left=0.085, right=0.91, top=0.93, bottom=0.02, wspace=0.04, hspace=0.08)
    # colourbars
    cax_d = fig.add_axes([0.925, 0.55, 0.012, 0.33])
    cax_e = fig.add_axes([0.925, 0.12, 0.012, 0.33])
    fig.colorbar(cm.ScalarMappable(dnorm, DEPTH_CMAP), cax=cax_d).set_label("depth [m]", fontsize=10)
    fig.colorbar(cm.ScalarMappable(enorm, ERR_CMAP), cax=cax_e).set_label("abs. error [m]", fontsize=10)
    fig.savefig(args.out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info("saved -> %s", args.out)


if __name__ == "__main__":
    main()
