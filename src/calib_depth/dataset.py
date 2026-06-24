"""Benchmark -> training samples for SelfCalibDepth.

A single, benchmark-agnostic dataset. The only dataset-specific code lives in the
:mod:`calib_depth.benchmarks` adapters; here we turn any adapter's
:class:`~calib_depth.benchmarks.Frame` (image + sparse LiDAR depth + intrinsics)
into the tensors the model trains on:

    image      : (3, H, W) float in [0, 1]
    uv         : (N, 2)     LiDAR pixel coords (ground-truth depth locations)
    gt_depth   : (N,)       metric depth at those pixels
    lidar_cam  : (N, 3)     the LiDAR points in the camera frame (intrinsics-free)
    theta_gt   : (9,)       GT intrinsics (eval only / curriculum init)

``AV2SensorDepthDataset`` / ``discover_logs`` are kept as thin AV2 wrappers so the
existing train/eval/infer entry points keep working unchanged.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

import numpy as np

from .benchmarks import BenchmarkAdapter, FrameRef, get_adapter

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


class BenchmarkDepthDataset:
    """Torch-style dataset over any :class:`BenchmarkAdapter`.

    Torch/cv2 are imported lazily so non-training code stays light.
    ``target_hw`` (H, W, both multiples of 14) is the training resolution: the
    image is resized and the intrinsics + LiDAR pixel coords are scaled to match
    (independently in x and y, so a pinhole with fx != fy stays consistent).
    Metric depth and the 3D LiDAR points are scale-free and pass through.
    """

    def __init__(self, adapter: BenchmarkAdapter, refs: List[FrameRef],
                 target_hw: tuple[int, int] = (518, 518)):
        self.adapter = adapter
        self.refs = refs
        self.target_hw = target_hw

    @classmethod
    def build(cls, benchmark: str, root: Path, cams: Optional[List[str]] = None,
              stride: int = 1, target_hw: tuple[int, int] = (518, 518),
              distort: Optional[tuple[float, float]] = None,
              fisheye: Optional[tuple[float, float]] = None) -> "BenchmarkDepthDataset":
        adapter = get_adapter(benchmark)
        if distort is not None:
            from .synth_distort import DistortingAdapter
            adapter = DistortingAdapter(adapter, distort[0], distort[1])
        if fisheye is not None:
            from .synth_fisheye import FisheyeAdapter
            adapter = FisheyeAdapter(adapter, fisheye[0], fisheye[1])
        refs = adapter.discover(Path(root), cams=cams, stride=stride)
        return cls(adapter, refs, target_hw=target_hw)

    def __len__(self) -> int:
        return len(self.refs)

    def __getitem__(self, i: int) -> dict:
        import cv2
        import torch

        ref = self.refs[i]
        frame = self.adapter.load(ref)
        img, uv, depth, K = frame.image, frame.uv, frame.depth, frame.intrinsics

        # LiDAR points in the camera frame: intrinsics-independent geometric anchor.
        x = (uv[:, 0] - K.cx) / K.fx * depth
        y = (uv[:, 1] - K.cy) / K.fy * depth
        lidar_cam = np.stack([x, y, depth], axis=-1)

        Ht, Wt = self.target_hw
        sx, sy = Wt / K.width, Ht / K.height
        img = cv2.resize(img, (Wt, Ht), interpolation=cv2.INTER_AREA)
        uv = uv * np.array([sx, sy], dtype=np.float32)
        theta_gt = K.scaled(sx, sy).theta9()

        return {
            "image": torch.from_numpy(img).permute(2, 0, 1).float() / 255.0,
            "uv": torch.from_numpy(uv).float(),
            "gt_depth": torch.from_numpy(depth).float(),
            "lidar_cam": torch.from_numpy(lidar_cam).float(),
            "theta_gt": torch.from_numpy(theta_gt),
            "cam_idx": self._cam_index(ref.cam),
            "hw": (Ht, Wt),
            "orig_wh": (K.width, K.height),
        }

    # cameras are indexed in first-seen order across refs (stable per dataset)
    def _cam_index(self, cam: str) -> int:
        if not hasattr(self, "_cam_order"):
            seen: List[str] = []
            for r in self.refs:
                if r.cam not in seen:
                    seen.append(r.cam)
            self._cam_order = seen
        return self._cam_order.index(cam)


# --------------------------------------------------------------------------- #
# Backward-compatible AV2 wrappers (used by train.py / eval.py / infer.py).
# --------------------------------------------------------------------------- #
class AV2SensorDepthDataset(BenchmarkDepthDataset):
    """AV2 dataset with the original ``cam_names`` / ``.index`` interface.

    ``.index[i] == (log_dir: Path, cam_name: str, lidar_ts: int)`` is preserved so
    eval/infer can still fetch AV2 vehicle cuboids per frame.
    """

    def __init__(self, log_dirs: List[Path], cam_names: List[str], stride: int = 1,
                 target_hw: tuple[int, int] = (518, 518)):
        adapter = get_adapter("av2")
        self.cam_names = cam_names
        refs: List[FrameRef] = []
        from lidar_depth import list_timestamps  # reuse the proven lister
        for log_dir in log_dirs:
            lidar_dir = Path(log_dir) / "sensors" / "lidar"
            if not lidar_dir.exists():
                continue
            timestamps = list_timestamps(lidar_dir, ".feather")[::stride]
            for cam in cam_names:
                refs += [FrameRef("av2", str(log_dir), cam, str(int(t))) for t in timestamps]
        super().__init__(adapter, refs, target_hw=target_hw)

    def _cam_index(self, cam: str) -> int:
        return self.cam_names.index(cam)

    @property
    def index(self):
        return [(Path(r.scene), r.cam, int(r.frame)) for r in self.refs]


def discover_logs(root: Path) -> List[Path]:
    """All AV2 sensor-log folders (those containing a calibration/ dir) under ``root``."""
    return sorted({p.parents[1] for p in Path(root).rglob("calibration/intrinsics.feather")})
