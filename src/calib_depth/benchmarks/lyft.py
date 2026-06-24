"""Lyft Level 5 adapter (uses lyft_dataset_sdk; imported lazily).

Lyft L5 ships in the nuScenes format, and ``lyft_dataset_sdk`` mirrors the
nuscenes-devkit API, so this reuses the shared projection core. ``root`` is the
Lyft data dir; the JSON metadata is expected under ``<root>/<train|JSON_DIR>``
(``LYFT_JSON_DIR`` env var, default ``train_data``). The top LiDAR sensor is
named ``LIDAR_TOP`` in the SDK's combined cloud.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

from . import _nuscenes_core as core
from .base import BenchmarkAdapter, Frame, FrameRef

CAMS = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
        "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]


class LyftAdapter(BenchmarkAdapter):
    name = "lyft"
    default_cams = CAMS
    lidar_name = "LIDAR_TOP"

    def __init__(self):
        self._nusc = None
        self._dataroot = None

    def _ensure(self, root: Path):
        if self._nusc is not None:
            return
        from lyft_dataset_sdk.lyftdataset import LyftDataset  # lazy: heavy SDK
        json_dir = os.environ.get("LYFT_JSON_DIR", "train_data")
        self._dataroot = str(root)
        self._nusc = LyftDataset(data_path=str(root),
                                 json_path=str(root / json_dir), verbose=False)

    def discover(self, root: Path, cams: List[str] | None = None,
                 stride: int = 1) -> List[FrameRef]:
        self._ensure(root)
        return core.discover(self._nusc, self.name, cams or self.default_cams, stride)

    def load(self, ref: FrameRef) -> Frame:
        self._ensure(Path(self._dataroot) if self._dataroot else Path("."))
        from lyft_dataset_sdk.utils.data_classes import LidarPointCloud
        from lyft_dataset_sdk.utils.geometry_utils import view_points
        from pyquaternion import Quaternion
        return core.load(self._nusc, self._dataroot, ref, self.lidar_name,
                         LidarPointCloud, view_points, Quaternion)
