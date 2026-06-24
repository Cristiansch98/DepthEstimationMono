"""Per-vehicle ground-truth distance from AV2 3D cuboid annotations.

This supplies the supervision/evaluation signal for the framework's end goal:
"tell the real distance of vehicles" from a single image. AV2 ships 3D cuboids
(center, size, orientation, category) in the egovehicle frame; we project them
into a ring camera and read off the metric range to each vehicle.

Outputs:
  * ``vehicle_targets()``  -> list of (uv_center, distance_m, category, box_uv)
    usable as per-object GT for training and for distance-error evaluation.
  * a labelled overlay: camera image + projected 3D vehicle boxes annotated with
    their ground-truth distance, colour-coded by range.

    python src/vehicle_distance.py \
        --log-dir data/sensor/val/<log_id> --cam ring_front_center --out viz
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, NamedTuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from av2.geometry.camera.pinhole_camera import PinholeCamera
from av2.structures.cuboid import CuboidList
from av2.utils.io import read_city_SE3_ego, read_img

import sys
_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from lidar_depth import list_timestamps, nearest_timestamp  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("vehicle-dist")

# AV2 categories that are "vehicles" for this task.
VEHICLE_CATEGORIES = {
    "REGULAR_VEHICLE", "LARGE_VEHICLE", "BUS", "TRUCK", "BOX_TRUCK", "TRUCK_CAB",
    "VEHICULAR_TRAILER", "SCHOOL_BUS", "ARTICULATED_BUS", "MOTORCYCLE",
    "MESSAGE_BOARD_TRAILER", "RAILED_VEHICLE",
}
# 12 edges of the av2 cuboid vertex ordering (see Cuboid.vertices_m docstring).
BOX_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]


class VehicleTarget(NamedTuple):
    uv_center: np.ndarray   # (2,) image location of the cuboid centre
    distance_m: float       # metric range from the camera centre
    category: str
    box_uv: np.ndarray      # (8, 2) projected cuboid vertices


def vehicle_targets(log_dir: Path, cam_name: str, lidar_ts: int | None = None):
    """Project annotated vehicle cuboids into a camera frame -> per-vehicle GT distance."""
    cam = PinholeCamera.from_feather(log_dir, cam_name)
    cam_dir = log_dir / "sensors" / "cameras" / cam_name
    lidar_dir = log_dir / "sensors" / "lidar"

    lidar_times = list_timestamps(lidar_dir, ".feather")
    cam_times = list_timestamps(cam_dir, ".jpg")
    if lidar_ts is None:
        lidar_ts = int(lidar_times[len(lidar_times) // 2])
    cam_ts = nearest_timestamp(lidar_ts, cam_times)

    cuboids = CuboidList.from_feather(log_dir / "annotations.feather")
    # annotations are stamped at lidar sweep times; pick the sweep matching ours.
    anno_ts = nearest_timestamp(cam_ts, np.array([c.timestamp_ns for c in cuboids.cuboids]))
    cuboids = [c for c in cuboids.cuboids
               if c.timestamp_ns == anno_ts and c.category in VEHICLE_CATEGORIES]

    poses = read_city_SE3_ego(log_dir)
    city_SE3_ego_cam_t = poses[cam_ts]
    city_SE3_ego_lidar_t = poses[anno_ts]

    img = read_img(cam_dir / f"{cam_ts}.jpg")
    targets: List[VehicleTarget] = []
    for c in cuboids:
        verts_ego = c.vertices_m  # (8, 3) in the ego frame at annotation time
        uv, pts_cam, valid = cam.project_ego_to_img_motion_compensated(
            verts_ego, city_SE3_ego_cam_t=city_SE3_ego_cam_t, city_SE3_ego_lidar_t=city_SE3_ego_lidar_t)
        if valid.sum() < 4:  # require most of the box in front of the camera
            continue
        center_cam = pts_cam[valid].mean(0)
        dist = float(np.linalg.norm(center_cam))
        targets.append(VehicleTarget(uv.mean(0), dist, c.category, uv))
    targets.sort(key=lambda t: t.distance_m)
    log.info("%s @ %d: %d vehicles in view (%.1f-%.1f m)", cam_name, cam_ts, len(targets),
             targets[0].distance_m if targets else 0, targets[-1].distance_m if targets else 0)
    return img, targets, cam, cam_ts


def render(img, targets: List[VehicleTarget], out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(16, 10), dpi=120)
    ax.imshow(img)
    cmap = plt.get_cmap("turbo")
    dmax = max((t.distance_m for t in targets), default=80)
    h, w = img.shape[:2]
    for t in targets:
        col = cmap(min(t.distance_m / min(dmax, 80), 1.0))
        for a, b in BOX_EDGES:
            ax.plot([t.box_uv[a, 0], t.box_uv[b, 0]], [t.box_uv[a, 1], t.box_uv[b, 1]],
                    color=col, lw=1.6, alpha=0.9)
        u, v = t.uv_center
        if 0 <= u < w and 0 <= v < h:
            ax.text(u, v, f"{t.distance_m:.0f} m", color="white", fontsize=9, ha="center",
                    va="center", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.15", fc=col, ec="white", lw=0.5, alpha=0.85))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, min(dmax, 80)))
    cb = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label("ground-truth distance to vehicle  [m]")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.axis("off")
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.1, facecolor="white")
    plt.close(fig)
    log.info("Saved -> %s", out_path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", type=Path, required=True)
    ap.add_argument("--cam", default="ring_front_center")
    ap.add_argument("--out", type=Path, default=Path("viz"))
    ap.add_argument("--lidar-ts", type=int, default=None)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    img, targets, cam, cam_ts = vehicle_targets(args.log_dir, args.cam, args.lidar_ts)
    title = (f"Argoverse 2 · {args.cam} · ground-truth distance to vehicles\n"
             f"{len(targets)} vehicles — this is the metric target the model must predict from the image alone")
    render(img, targets, args.out / f"{args.cam}_{cam_ts}_vehicles.png", title)


if __name__ == "__main__":
    main()
