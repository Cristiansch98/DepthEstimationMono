"""KITTI raw adapter — Velodyne -> camera projection.

Mirrors the AV2 LiDAR-as-GT-depth signal on the standard *KITTI raw* layout::

    <root>/<date>/                      e.g. 2011_09_26/
        calib_cam_to_cam.txt            (R_rect_00, P_rect_0x)
        calib_velo_to_cam.txt           (R, T : Velodyne -> cam0)
        <date>_drive_NNNN_sync/         a "scene" (= drive)
            image_02/data/0000000000.png
            velodyne_points/data/0000000000.bin

Projection (the KITTI-canonical chain):

    Y = P_rect_0c · R_rect_00 · [R|T]_velo->cam · [X 1]ᵀ
    uv = Y[:2]/Y[2] ,  depth = Y[2]   (z in the rectified camera frame)

KITTI images are already rectified, so distortion is 0 (pinhole). Intrinsics
come straight from P_rect: fx=P[0,0], fy=P[1,1], cx=P[0,2], cy=P[1,2].
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np

from .base import BenchmarkAdapter, CameraIntrinsics, Frame, FrameRef

#: KITTI camera name -> P_rect key in calib_cam_to_cam.txt
_CAM_TO_PRECT = {"image_02": "P_rect_02", "image_03": "P_rect_03"}


def _parse_calib(path: Path) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for line in path.read_text().splitlines():
        if ":" not in line:
            continue
        key, vals = line.split(":", 1)
        try:
            out[key.strip()] = np.array([float(x) for x in vals.split()], dtype=np.float64)
        except ValueError:
            pass
    return out


def _velo_to_cam_4x4(date_dir: Path) -> np.ndarray:
    c = _parse_calib(date_dir / "calib_velo_to_cam.txt")
    Tr = np.eye(4)
    Tr[:3, :3] = c["R"].reshape(3, 3)
    Tr[:3, 3] = c["T"]
    return Tr


class KITTIAdapter(BenchmarkAdapter):
    name = "kitti"
    default_cams = ["image_02"]  # left RGB

    def discover(self, root: Path, cams: List[str] | None = None,
                 stride: int = 1) -> List[FrameRef]:
        cams = cams or self.default_cams
        refs: List[FrameRef] = []
        # a scene = any dir containing velodyne_points/data
        for velo_dir in sorted(root.rglob("velodyne_points/data")):
            drive_dir = velo_dir.parents[1]
            for cam in cams:
                img_dir = drive_dir / cam / "data"
                if not img_dir.exists():
                    continue
                frames = sorted(p.stem for p in img_dir.glob("*.png"))[::stride]
                refs += [FrameRef(self.name, str(drive_dir), cam, f) for f in frames]
        return refs

    def load(self, ref: FrameRef) -> Frame:
        import cv2

        drive_dir = Path(ref.scene)
        date_dir = drive_dir.parent
        cam2cam = _parse_calib(date_dir / "calib_cam_to_cam.txt")

        R_rect = np.eye(4)
        R_rect[:3, :3] = cam2cam["R_rect_00"].reshape(3, 3)
        P_rect = cam2cam[_CAM_TO_PRECT[ref.cam]].reshape(3, 4)
        Tr = _velo_to_cam_4x4(date_dir)
        proj = P_rect @ R_rect @ Tr  # (3, 4) velodyne -> image

        velo = np.fromfile(drive_dir / "velodyne_points" / "data" / f"{ref.frame}.bin",
                           dtype=np.float32).reshape(-1, 4)
        pts = velo[velo[:, 0] > 0]  # drop points behind the LiDAR
        homog = np.concatenate([pts[:, :3], np.ones((len(pts), 1))], axis=1)
        Y = homog @ proj.T  # (N, 3)
        depth = Y[:, 2]
        uv = Y[:, :2] / depth[:, None]

        img = cv2.cvtColor(cv2.imread(str(drive_dir / ref.cam / "data" / f"{ref.frame}.png")),
                           cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        m = (depth > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
        uv, depth = uv[m].astype(np.float32), depth[m].astype(np.float32)

        K = CameraIntrinsics(fx=float(P_rect[0, 0]), fy=float(P_rect[1, 1]),
                             cx=float(P_rect[0, 2]), cy=float(P_rect[1, 2]),
                             width=w, height=h)
        return Frame(image=img, uv=uv, depth=depth, intrinsics=K,
                     key=f"{drive_dir.name}_{ref.cam}_{ref.frame}")
