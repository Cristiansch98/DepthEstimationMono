"""Shared LiDAR->camera projection for nuScenes-devkit-compatible datasets.

Both nuscenes-devkit and lyft_dataset_sdk expose the *same* token database API
(``sample`` / ``sample_data`` / ``calibrated_sensor`` / ``ego_pose`` records) and
the same ``LidarPointCloud`` / ``view_points`` helpers. This module implements
the canonical four-step transform (lidar sensor -> ego@lidar_t -> global ->
ego@cam_t -> camera), motion-compensating the sweep to the camera timestamp, so
the nuScenes and Lyft adapters share one tested code path.
"""

from __future__ import annotations

import os
from typing import List, Tuple

import numpy as np

from .base import CameraIntrinsics, Frame, FrameRef


def discover(nusc, benchmark: str, cams: List[str], stride: int = 1) -> List[FrameRef]:
    """One ref per (sample, camera). ``frame`` = sample token, ``scene`` = scene token."""
    refs: List[FrameRef] = []
    for i, sample in enumerate(nusc.sample):
        if i % stride:
            continue
        for cam in cams:
            if cam in sample["data"]:
                refs.append(FrameRef(benchmark, sample["scene_token"], cam, sample["token"]))
    return refs


def project(nusc, dataroot: str, ref: FrameRef, lidar_name: str,
            LidarPointCloud, view_points, Quaternion,
            min_dist: float = 1.0) -> Tuple[str, np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Project the motion-compensated sweep into ``ref.cam``.

    Returns (image_path, uv (N,2), depth (N,), K (3,3), width, height).
    """
    sample = nusc.get("sample", ref.frame)
    cam_sd = nusc.get("sample_data", sample["data"][ref.cam])
    lidar_sd = nusc.get("sample_data", sample["data"][lidar_name])

    pc = LidarPointCloud.from_file(os.path.join(dataroot, lidar_sd["filename"]))

    # lidar sensor -> ego @ lidar timestamp
    cs = nusc.get("calibrated_sensor", lidar_sd["calibrated_sensor_token"])
    pc.rotate(Quaternion(cs["rotation"]).rotation_matrix)
    pc.translate(np.array(cs["translation"]))
    # ego @ lidar t -> global
    ep = nusc.get("ego_pose", lidar_sd["ego_pose_token"])
    pc.rotate(Quaternion(ep["rotation"]).rotation_matrix)
    pc.translate(np.array(ep["translation"]))
    # global -> ego @ cam t
    ep = nusc.get("ego_pose", cam_sd["ego_pose_token"])
    pc.translate(-np.array(ep["translation"]))
    pc.rotate(Quaternion(ep["rotation"]).rotation_matrix.T)
    # ego @ cam t -> camera sensor
    cs = nusc.get("calibrated_sensor", cam_sd["calibrated_sensor_token"])
    pc.translate(-np.array(cs["translation"]))
    pc.rotate(Quaternion(cs["rotation"]).rotation_matrix.T)

    depth = pc.points[2, :]
    K = np.array(cs["camera_intrinsic"])
    uv = view_points(pc.points[:3, :], K, normalize=True)[:2, :].T  # (N, 2)

    w, h = int(cam_sd["width"]), int(cam_sd["height"])
    m = (depth > min_dist) & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    img_path = os.path.join(dataroot, cam_sd["filename"])
    return img_path, uv[m].astype(np.float32), depth[m].astype(np.float32), K, w, h


def load(nusc, dataroot: str, ref: FrameRef, lidar_name: str,
         LidarPointCloud, view_points, Quaternion) -> Frame:
    import cv2

    img_path, uv, depth, K, w, h = project(
        nusc, dataroot, ref, lidar_name, LidarPointCloud, view_points, Quaternion)
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    intr = CameraIntrinsics(fx=float(K[0, 0]), fy=float(K[1, 1]),
                            cx=float(K[0, 2]), cy=float(K[1, 2]), width=w, height=h)
    return Frame(image=img, uv=uv, depth=depth, intrinsics=intr,
                 key=f"{ref.cam}_{ref.frame[:12]}")
