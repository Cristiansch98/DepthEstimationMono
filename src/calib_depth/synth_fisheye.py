"""Controlled Kannala–Brandt (KB) fisheye synthesis — a *different* distortion-model
family than the Brown–Conrady test in ``synth_distort.py``.

The KB / OpenCV-fisheye model is *angular* (maps the incidence angle θ, not the
perspective radius tanθ), the standard model for real fisheye lenses (incl. KITTI-360):

    a = X/Z,  b = Y/Z,  r = √(a²+b²),  θ = atan(r)
    θ_d = θ (1 + k1 θ² + k2 θ⁴)
    (x', y') = (θ_d / r) · (a, b);   u = fx·x' + cx,  v = fy·y' + cy

We warp a pinhole frame into KB-fisheye optics (image + LiDAR projection) with known
GT, so testing under KB checks whether the distortion-robustness finding generalizes
beyond Brown–Conrady. NOTE: a pinhole source only spans its own (~±40° for KITTI) FOV,
so we cannot synthesise true >90° content — this stresses the *angular model* on the
available field, not a full 180° fisheye (that needs gated real data, e.g. KITTI-360).
"""

from __future__ import annotations

import numpy as np

from .benchmarks.base import CameraIntrinsics, Frame


def _theta_d(theta, k1, k2):
    return theta * (1 + k1 * theta**2 + k2 * theta**4)


def _invert_theta(theta_d, k1, k2, iters=15):
    """Solve θ from θ_d = θ(1+k1θ²+k2θ⁴) by Newton (θ_d as initial guess)."""
    th = theta_d.copy()
    for _ in range(iters):
        f = th * (1 + k1 * th**2 + k2 * th**4) - theta_d
        fp = 1 + 3 * k1 * th**2 + 5 * k2 * th**4
        th = th - f / np.maximum(fp, 1e-6)
    return np.clip(th, 0, np.pi / 2 - 1e-3)


def fisheye_frame(frame: Frame, k1: float, k2: float = 0.0) -> Frame:
    """Render ``frame`` through a KB fisheye with radial (k1, k2). fx,fy,cx,cy kept."""
    import cv2

    K = frame.intrinsics
    fx, fy, cx, cy = K.fx, K.fy, K.cx, K.cy
    H, W = frame.image.shape[:2]

    # --- inverse warp: for each fisheye output pixel, find the source pinhole pixel ---
    us, vs = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
    xprime, yprime = (us - cx) / fx, (vs - cy) / fy
    theta_d = np.sqrt(xprime**2 + yprime**2)
    theta = _invert_theta(theta_d, k1, k2)
    # perspective radius for that incidence angle; direction from (x',y')
    rp = np.tan(theta)
    scale = np.where(theta_d > 1e-9, rp / np.maximum(theta_d, 1e-9), 0.0)
    a, b = xprime * scale, yprime * scale            # = X/Z, Y/Z of the source ray
    map_x = (fx * a + cx).astype(np.float32)
    map_y = (fy * b + cy).astype(np.float32)
    # mask rays outside the source pinhole FOV (no content there) -> black
    valid = (map_x >= 0) & (map_x < W) & (map_y >= 0) & (map_y < H)
    map_x[~valid] = -1
    map_y[~valid] = -1
    img_f = cv2.remap(frame.image, map_x, map_y, cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    # --- forward KB-project the LiDAR pixels (true pinhole uv -> fisheye uv) ---
    a = (frame.uv[:, 0] - cx) / fx
    b = (frame.uv[:, 1] - cy) / fy
    r = np.sqrt(a**2 + b**2)
    theta = np.arctan(r)
    td = _theta_d(theta, k1, k2)
    s = np.where(r > 1e-9, td / np.maximum(r, 1e-9), 0.0)
    u2, v2 = fx * s * a + cx, fy * s * b + cy
    m = (u2 >= 0) & (u2 < W) & (v2 >= 0) & (v2 < H)
    uv_f = np.stack([u2, v2], axis=-1)[m].astype(np.float32)

    Kf = CameraIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy, width=W, height=H, k1=k1, k2=k2)
    return Frame(image=img_f, uv=uv_f, depth=frame.depth[m].astype(np.float32),
                 intrinsics=Kf, key=f"{frame.key}_kb{k1:+.2f}")


class FisheyeAdapter:
    """Wrap any benchmark adapter so frames are rendered through a KB fisheye."""

    def __init__(self, base, k1: float, k2: float = 0.0):
        self.base = base
        self.k1, self.k2 = k1, k2
        self.name = base.name
        self.default_cams = base.default_cams
        self.lidar_name = getattr(base, "lidar_name", None)

    def discover(self, *a, **kw):
        return self.base.discover(*a, **kw)

    def load(self, ref) -> Frame:
        return fisheye_frame(self.base.load(ref), self.k1, self.k2)

    def vehicle_targets(self, ref):
        return []
