"""LiDAR -> camera projection = the ground-truth depth signal for the framework.

This is the foundation of the self-calibrating monocular-depth framework: a
synchronized LiDAR sweep is projected into a ring-camera image (with ego-motion
compensation) to produce a *sparse metric depth map*. That sparse map is the
supervision target the depth network and the self-calibration are trained on.

It also renders a "cool" overlay (image + depth-coloured LiDAR returns), which
is a direct look at the supervision the model receives.

    python src/lidar_depth.py \
        --log-dir data/sensor-sample/val/<log_id> --cam ring_front_center --out viz
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from av2.geometry.camera.pinhole_camera import PinholeCamera
from av2.geometry.se3 import SE3
from av2.structures.sweep import Sweep
from av2.utils.io import read_city_SE3_ego, read_img

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("lidar-depth")


def nearest_timestamp(target: int, candidates: np.ndarray) -> int:
    """Return the candidate timestamp (ns) closest to ``target``."""
    return int(candidates[np.argmin(np.abs(candidates.astype(np.int64) - target))])


def list_timestamps(directory: Path, suffix: str) -> np.ndarray:
    return np.array(sorted(int(p.stem) for p in directory.glob(f"*{suffix}")), dtype=np.int64)


def sparse_depth_map(
    log_dir: Path,
    cam_name: str,
    lidar_ts: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, PinholeCamera, int, int]:
    """Build a sparse metric depth map by projecting one LiDAR sweep into a frame.

    Returns
    -------
    img        : (H, W, 3) uint8 camera image (nearest in time to the sweep)
    uv         : (N, 2) pixel coords of LiDAR returns that land in the image
    depth      : (N,)   metric depth (m) along the camera z-axis for those returns
    cam        : the PinholeCamera (carries intrinsics + ego_SE3_cam extrinsics)
    cam_ts     : chosen camera timestamp (ns)
    lidar_ts   : chosen LiDAR timestamp (ns)
    """
    cam = PinholeCamera.from_feather(log_dir, cam_name)
    cam_dir = log_dir / "sensors" / "cameras" / cam_name
    lidar_dir = log_dir / "sensors" / "lidar"

    lidar_times = list_timestamps(lidar_dir, ".feather")
    cam_times = list_timestamps(cam_dir, ".jpg")
    if lidar_ts is None:
        lidar_ts = int(lidar_times[len(lidar_times) // 2])  # a sweep in the middle
    cam_ts = nearest_timestamp(lidar_ts, cam_times)
    log.info("Pairing LiDAR %d <-> camera %d  (dt = %.1f ms)", lidar_ts, cam_ts,
             abs(lidar_ts - cam_ts) / 1e6)

    sweep = Sweep.from_feather(lidar_dir / f"{lidar_ts}.feather")
    points_ego = sweep.xyz  # (N, 3) in the egovehicle frame at LiDAR time

    poses = read_city_SE3_ego(log_dir)  # dict: timestamp_ns -> SE3 (city <- ego)
    city_SE3_ego_lidar_t: SE3 = poses[lidar_ts]
    city_SE3_ego_cam_t: SE3 = poses[cam_ts]

    uv, points_cam, is_valid = cam.project_ego_to_img_motion_compensated(
        points_ego, city_SE3_ego_cam_t=city_SE3_ego_cam_t, city_SE3_ego_lidar_t=city_SE3_ego_lidar_t
    )
    depth = points_cam[:, 2]  # camera-frame z = metric range to image plane
    uv, depth = uv[is_valid], depth[is_valid]

    img = read_img(cam_dir / f"{cam_ts}.jpg")
    log.info("Projected %d/%d LiDAR returns into the %dx%d image  (depth %.1f-%.1f m)",
             len(depth), len(points_ego), cam.width_px, cam.height_px, depth.min(), depth.max())
    return img, uv, depth, cam, cam_ts, lidar_ts


def densify_to_grid(uv: np.ndarray, depth: np.ndarray, h: int, w: int) -> np.ndarray:
    """Rasterize sparse returns onto an (H, W) grid (0 = no measurement)."""
    grid = np.zeros((h, w), dtype=np.float32)
    cols = np.clip(np.round(uv[:, 0]).astype(int), 0, w - 1)
    rows = np.clip(np.round(uv[:, 1]).astype(int), 0, h - 1)
    # nearest return wins where multiple project to the same pixel
    order = np.argsort(-depth)
    grid[rows[order], cols[order]] = depth[order]
    return grid


def render_overlay(img, uv, depth, out_path: Path, title: str) -> None:
    """Image with depth-coloured LiDAR returns overlaid (the supervision signal)."""
    fig, ax = plt.subplots(figsize=(16, 9), dpi=120)
    ax.imshow(img)
    order = np.argsort(-depth)  # draw far points first so near points sit on top
    sc = ax.scatter(uv[order, 0], uv[order, 1], c=depth[order], s=6, cmap="turbo",
                    vmin=2, vmax=min(80, float(depth.max())), alpha=0.9)
    cb = fig.colorbar(sc, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label("LiDAR ground-truth depth  [m]", color="black")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.axis("off")
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.1, facecolor="white")
    plt.close(fig)
    log.info("Saved overlay -> %s", out_path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Project LiDAR into a ring camera -> sparse depth GT.")
    ap.add_argument("--log-dir", type=Path, required=True)
    ap.add_argument("--cam", default="ring_front_center")
    ap.add_argument("--out", type=Path, default=Path("viz"))
    ap.add_argument("--lidar-ts", type=int, default=None)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    img, uv, depth, cam, cam_ts, lidar_ts = sparse_depth_map(args.log_dir, args.cam, args.lidar_ts)
    fx, fy = cam.intrinsics.fx_px, cam.intrinsics.fy_px
    title = (f"Argoverse 2 Sensor · {args.cam} · LiDAR-projected ground-truth depth\n"
             f"GT intrinsics fx={fx:.1f} fy={fy:.1f}  ({len(depth)} returns)  "
             f"— this is what the self-calibration must recover")
    render_overlay(img, uv, depth, args.out / f"{args.cam}_{cam_ts}_depth.png", title)

    grid = densify_to_grid(uv, depth, cam.height_px, cam.width_px)
    coverage = (grid > 0).mean() * 100
    log.info("Sparse depth map: %.2f%% of pixels have a LiDAR measurement", coverage)


if __name__ == "__main__":
    main()
