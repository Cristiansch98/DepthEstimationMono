"""nuScenes adapter (uses nuscenes-devkit; imported lazily).

``root`` is the nuScenes ``dataroot`` (the dir holding ``samples/``, ``sweeps/``,
``v1.0-*`` metadata). Select the metadata version with the ``NUSCENES_VERSION``
env var (default ``v1.0-mini``). LIDAR_TOP is projected into each requested
camera, motion-compensated to the camera timestamp.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

from . import _nuscenes_core as core
from .base import BenchmarkAdapter, Frame, FrameRef

CAMS = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
        "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]


class NuScenesAdapter(BenchmarkAdapter):
    name = "nuscenes"
    default_cams = CAMS
    lidar_name = "LIDAR_TOP"

    def __init__(self):
        self._nusc = None
        self._dataroot = None

    def _ensure(self, root: Path):
        if self._nusc is not None:
            return
        from nuscenes.nuscenes import NuScenes  # lazy: heavy SDK
        version = os.environ.get("NUSCENES_VERSION", "v1.0-mini")
        self._dataroot = str(root)
        self._nusc = NuScenes(version=version, dataroot=str(root), verbose=False)

    def discover(self, root: Path, cams: List[str] | None = None,
                 stride: int = 1) -> List[FrameRef]:
        self._ensure(root)
        return core.discover(self._nusc, self.name, cams or self.default_cams, stride)

    def load(self, ref: FrameRef) -> Frame:
        self._ensure(Path(self._dataroot) if self._dataroot else Path("."))
        from nuscenes.utils.data_classes import LidarPointCloud
        from nuscenes.utils.geometry_utils import view_points
        from pyquaternion import Quaternion
        return core.load(self._nusc, self._dataroot, ref, self.lidar_name,
                         LidarPointCloud, view_points, Quaternion)
