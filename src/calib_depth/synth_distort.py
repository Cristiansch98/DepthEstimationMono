"""Controlled synthetic radial distortion — to probe where pinhole-assuming
foundation models (e.g. UniDepth) break and LiDAR few-shot self-calibration helps.

We take a (near-)pinhole frame and simulate a lens with *known* Brown-Conrady
radial coefficients (k1, k2). Both the **image** and the **LiDAR projection** are
warped consistently, so the ground-truth intrinsics become (fx, fy, cx, cy, k1, k2)
exactly. This makes distortion strength a controlled variable with known GT — the
clean way to test the hypothesis rather than relying on a gated fisheye dataset.

Geometry (K = pinhole matrix; a 3-D ray has normalized coords (xn, yn) = (X/Z, Y/Z)):
  * pinhole pixel   p_p = K · (xn, yn)
  * distorted pixel p_d = K · distort(xn, yn)
  ⇒ distorted image  I_d(p_d) = I_p( K · undistort(K⁻¹ p_d) )   (inverse warp / remap)
  ⇒ LiDAR uv         p_d = K · distort( K⁻¹ p_p )                (forward, from true uv)
Depth (camera-frame z) is unchanged by the lens model.
"""

from __future__ import annotations

import numpy as np

from .benchmarks.base import CameraIntrinsics, Frame


def _distort(x, y, k1, k2):
    r2 = x * x + y * y
    f = 1 + k1 * r2 + k2 * r2 * r2
    return x * f, y * f


def _undistort(xd, yd, k1, k2, iters=20):
    """Invert radial distortion by fixed-point iteration (more iters for large |k|)."""
    x, y = xd.copy(), yd.copy()
    for _ in range(iters):
        r2 = x * x + y * y
        f = 1 + k1 * r2 + k2 * r2 * r2
        x = xd / f
        y = yd / f
    return x, y


def distort_frame(frame: Frame, k1: float, k2: float = 0.0) -> Frame:
    """Return a copy of ``frame`` as seen through a lens with radial (k1, k2)."""
    import cv2

    K = frame.intrinsics
    fx, fy, cx, cy = K.fx, K.fy, K.cx, K.cy
    H, W = frame.image.shape[:2]

    # --- inverse-warp the image: for each output (distorted) pixel, find source ---
    us, vs = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    xd, yd = (us - cx) / fx, (vs - cy) / fy
    xu, yu = _undistort(xd, yd, k1, k2)
    map_x = (fx * xu + cx).astype(np.float32)
    map_y = (fy * yu + cy).astype(np.float32)
    img_d = cv2.remap(frame.image, map_x, map_y, cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    # --- forward-distort the LiDAR pixels (true uv -> distorted uv) ---
    xn, yn = (frame.uv[:, 0] - cx) / fx, (frame.uv[:, 1] - cy) / fy
    xdp, ydp = _distort(xn, yn, k1, k2)
    u2, v2 = fx * xdp + cx, fy * ydp + cy
    m = (u2 >= 0) & (u2 < W) & (v2 >= 0) & (v2 < H)
    uv_d = np.stack([u2, v2], axis=-1)[m].astype(np.float32)

    Kd = CameraIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy, width=W, height=H, k1=k1, k2=k2)
    return Frame(image=img_d, uv=uv_d, depth=frame.depth[m].astype(np.float32),
                 intrinsics=Kd, key=f"{frame.key}_k1{k1:+.2f}")


class DistortingAdapter:
    """Wrap any benchmark adapter so every loaded frame gets known radial distortion."""

    def __init__(self, base, k1: float, k2: float = 0.0):
        self.base = base
        self.k1, self.k2 = k1, k2
        self.name = base.name
        self.default_cams = base.default_cams
        self.lidar_name = getattr(base, "lidar_name", None)

    def discover(self, *a, **kw):
        return self.base.discover(*a, **kw)

    def load(self, ref) -> Frame:
        return distort_frame(self.base.load(ref), self.k1, self.k2)

    def vehicle_targets(self, ref):
        return []  # distortion remaps pixels; vehicle-box targets not used here
