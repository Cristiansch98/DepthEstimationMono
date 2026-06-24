"""Argoverse 2 Sensor adapter — the reference implementation.

Thin wrapper over the proven ``src/lidar_depth.sparse_depth_map`` (motion-
compensated LiDAR -> ring-camera projection) so the unified path reproduces the
original AV2 pipeline exactly. This is the benchmark all others are matched to.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

from .base import BenchmarkAdapter, CameraIntrinsics, Frame, FrameRef

_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from lidar_depth import list_timestamps, sparse_depth_map  # noqa: E402

RING_CAMS = ["ring_front_center", "ring_front_left", "ring_front_right",
             "ring_side_left", "ring_side_right", "ring_rear_left", "ring_rear_right"]


class AV2Adapter(BenchmarkAdapter):
    name = "av2"
    default_cams = RING_CAMS

    def discover(self, root: Path, cams: List[str] | None = None,
                 stride: int = 1) -> List[FrameRef]:
        cams = cams or self.default_cams
        logs = sorted({p.parents[1] for p in root.rglob("calibration/intrinsics.feather")})
        refs: List[FrameRef] = []
        for log_dir in logs:
            lidar_dir = log_dir / "sensors" / "lidar"
            if not lidar_dir.exists():
                continue
            timestamps = list_timestamps(lidar_dir, ".feather")[::stride]
            for cam in cams:
                refs += [FrameRef(self.name, str(log_dir), cam, str(int(t))) for t in timestamps]
        return refs

    def load(self, ref: FrameRef) -> Frame:
        log_dir = Path(ref.scene)
        img, uv, depth, cam, _, _ = sparse_depth_map(log_dir, ref.cam, int(ref.frame))
        k = cam.intrinsics
        K = CameraIntrinsics(
            fx=float(k.fx_px), fy=float(k.fy_px), cx=float(k.cx_px), cy=float(k.cy_px),
            width=int(cam.width_px), height=int(cam.height_px),
            k1=float(getattr(k, "k1", 0.0)), k2=float(getattr(k, "k2", 0.0)),
            k3=float(getattr(k, "k3", 0.0)), p1=float(getattr(k, "p1", 0.0)),
            p2=float(getattr(k, "p2", 0.0)),
        )
        return Frame(image=img, uv=uv, depth=depth, intrinsics=K,
                     key=f"{ref.cam}_{ref.frame}")

    def vehicle_targets(self, ref: FrameRef):
        from vehicle_distance import vehicle_targets as _vt  # noqa: E402
        _, targets, _cam, _ = _vt(Path(ref.scene), ref.cam, int(ref.frame))
        return targets
