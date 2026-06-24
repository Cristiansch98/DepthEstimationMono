"""Unified benchmark contract for SelfCalibDepth.

Every driving benchmark (Argoverse 2, KITTI, nuScenes, Lyft L5, …) is reduced to
the *same* per-frame tuple the framework was built on:

    image  : (H, W, 3) uint8   camera frame
    uv     : (N, 2)    float    pixel coords of LiDAR returns that land in the image
    depth  : (N,)      float    metric depth (camera-frame z) at those pixels
    K      : CameraIntrinsics   GT pinhole(+distortion) for that camera

That is exactly what ``src/lidar_depth.sparse_depth_map`` already produces for
AV2. A :class:`BenchmarkAdapter` is the *only* dataset-specific code; everything
downstream (resize/scale, the LiDAR-in-camera anchor, ``theta_gt``, the model,
losses, training, eval) is shared and benchmark-agnostic.

An adapter does two things:
  * ``discover(root)`` — cheaply enumerate every (scene, camera, frame) as a
    list of :class:`FrameRef` (no I/O of pixels/points);
  * ``load(ref)``      — do the heavy projection for one ref -> :class:`Frame`.

Heavy third-party SDKs (nuscenes-devkit, lyft_dataset_sdk, …) are imported
*inside* the concrete adapters so this package stays importable everywhere.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np

# Order of the 9-vector theta used everywhere in the framework.
THETA_NAMES = ("fx", "fy", "cx", "cy", "k1", "k2", "k3", "p1", "p2")


@dataclass
class CameraIntrinsics:
    """Pinhole + Brown-Conrady intrinsics in pixels, benchmark-independent.

    Field names mirror the framework's ``theta = (fx, fy, cx, cy, k1..p2)`` plus
    the image size, so an adapter just fills in whatever its calibration ships
    (distortion defaults to 0 for the rectified/near-pinhole datasets).
    """

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    k1: float = 0.0
    k2: float = 0.0
    k3: float = 0.0
    p1: float = 0.0
    p2: float = 0.0

    def theta9(self) -> np.ndarray:
        return np.array([getattr(self, n) for n in THETA_NAMES], dtype=np.float32)

    def scaled(self, sx: float, sy: float) -> "CameraIntrinsics":
        """Resize-equivalent intrinsics for an image scaled by (sx, sy).

        A pinhole with fx != fy stays consistent under anisotropic resize;
        distortion coefficients are dimensionless in normalized coords and so
        are unchanged.
        """
        return CameraIntrinsics(
            fx=self.fx * sx, fy=self.fy * sy, cx=self.cx * sx, cy=self.cy * sy,
            width=int(round(self.width * sx)), height=int(round(self.height * sy)),
            k1=self.k1, k2=self.k2, k3=self.k3, p1=self.p1, p2=self.p2,
        )


@dataclass(frozen=True)
class FrameRef:
    """A lightweight, picklable pointer to one trainable frame.

    ``scene`` / ``frame`` are stringified so the same ref type works for a
    file-tree dataset (AV2/KITTI: scene = log dir, frame = timestamp) and a
    token-database dataset (nuScenes/Lyft: scene = scene token, frame = sample
    token). The owning adapter is the only thing that interprets them.
    """

    benchmark: str
    scene: str
    cam: str
    frame: str
    extra: tuple = field(default=(), compare=True)


@dataclass
class Frame:
    """One loaded (image, sparse-LiDAR-depth, intrinsics) sample."""

    image: np.ndarray       # (H, W, 3) uint8
    uv: np.ndarray          # (N, 2) float
    depth: np.ndarray       # (N,) float, metric z along the optical axis
    intrinsics: CameraIntrinsics
    key: str                # stable id for logging / figure file names


class BenchmarkAdapter(ABC):
    """Turn a benchmark on disk into the unified frame contract."""

    #: short registry name, e.g. "av2"
    name: str = "base"
    #: cameras used when ``--cams`` is not given on the CLI
    default_cams: List[str] = []

    @abstractmethod
    def discover(self, root: Path, cams: List[str] | None = None,
                 stride: int = 1) -> List[FrameRef]:
        """Enumerate frame refs under ``root`` (no pixel/point I/O)."""

    @abstractmethod
    def load(self, ref: FrameRef) -> Frame:
        """Project one ref into a :class:`Frame` (the heavy step)."""

    # Optional: per-vehicle 3D boxes for the distance-to-vehicle metric.
    # Datasets that ship 3D object labels override this; default = none.
    def vehicle_targets(self, ref: FrameRef):  # pragma: no cover - optional
        """Return a list of objects with ``.uv_center``, ``.box_uv`` (8,2),
        ``.distance_m`` for the headline vehicle-distance metric, or [] if the
        benchmark has no 3D boxes for this frame."""
        return []
