"""Unified, pluggable benchmark layer for SelfCalibDepth.

One contract (:class:`Frame` = image + sparse LiDAR depth + intrinsics) and one
registry, so the framework treats Argoverse 2, KITTI, nuScenes and Lyft L5
identically. Add a new benchmark = add one :class:`BenchmarkAdapter`.

    from calib_depth.benchmarks import get_adapter
    adapter = get_adapter("kitti")
    refs = adapter.discover(root, cams=["image_02"], stride=5)
    frame = adapter.load(refs[0])          # -> Frame(image, uv, depth, intrinsics)

Adapter classes are imported lazily so missing third-party SDKs (nuscenes-devkit,
lyft_dataset_sdk) never break ``import calib_depth.benchmarks``.
"""

from __future__ import annotations

from .base import (BenchmarkAdapter, CameraIntrinsics, Frame, FrameRef,
                   THETA_NAMES)

# name -> (module, class). Modules import their SDK only when instantiated.
_REGISTRY = {
    "av2": (".av2", "AV2Adapter"),
    "kitti": (".kitti", "KITTIAdapter"),
    "nuscenes": (".nuscenes", "NuScenesAdapter"),
    "lyft": (".lyft", "LyftAdapter"),
}

BENCHMARKS = tuple(_REGISTRY)


def get_adapter(name: str) -> BenchmarkAdapter:
    """Instantiate the adapter registered under ``name`` (e.g. ``"av2"``)."""
    import importlib

    if name not in _REGISTRY:
        raise KeyError(f"unknown benchmark {name!r}; choose from {BENCHMARKS}")
    module_name, class_name = _REGISTRY[name]
    module = importlib.import_module(module_name, __package__)
    return getattr(module, class_name)()


__all__ = ["BenchmarkAdapter", "CameraIntrinsics", "Frame", "FrameRef",
           "THETA_NAMES", "BENCHMARKS", "get_adapter"]
