"""Differentiable, *learnable* camera model — the self-calibration core.

Pinhole + Brown-Conrady radial/tangential distortion, parameterised by

    theta = (fx, fy, cx, cy, k1, k2, k3, p1, p2)

Everything here is pure geometry implemented in torch so gradients flow into
``theta``. The two consumers are:

  * ``back_project``  : (depth, pixel) -> 3D point in the camera frame, used by
                        the 3D-point loss against LiDAR.
  * ``ray_directions``: pixel grid -> unit bearing vectors, used to build the
                        camera-aware conditioning map (see ``ray_map.py``).

The actual calibration *values* are held by ``LearnableIntrinsics`` (a hybrid of
a per-camera latent and an image-conditioned residual head, see ``model.py``);
this module just consumes a ``theta`` tensor.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class Intrinsics:
    """A plain container mirroring ``theta`` for readability / logging."""

    fx: float
    fy: float
    cx: float
    cy: float
    k1: float = 0.0
    k2: float = 0.0
    k3: float = 0.0
    p1: float = 0.0
    p2: float = 0.0

    def to_theta(self) -> Tensor:
        return torch.tensor([self.fx, self.fy, self.cx, self.cy,
                             self.k1, self.k2, self.k3, self.p1, self.p2])


def _unpack(theta: Tensor):
    """theta: (..., 9) -> nine (...,) tensors."""
    fx, fy, cx, cy, k1, k2, k3, p1, p2 = theta.unbind(dim=-1)
    return fx, fy, cx, cy, k1, k2, k3, p1, p2


def normalized_grid(h: int, w: int, device=None, dtype=torch.float32):
    """Pixel-centre grid, returned as (u, v) each of shape (H, W)."""
    vs = torch.arange(h, device=device, dtype=dtype) + 0.5
    us = torch.arange(w, device=device, dtype=dtype) + 0.5
    v, u = torch.meshgrid(vs, us, indexing="ij")
    return u, v


def distort(x: Tensor, y: Tensor, theta: Tensor):
    """Apply Brown-Conrady distortion to normalized coords (forward model)."""
    _, _, _, _, k1, k2, k3, p1, p2 = _unpack(theta)
    r2 = x * x + y * y
    radial = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
    x_t = 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    y_t = p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    return x * radial + x_t, y * radial + y_t


def undistort(xd: Tensor, yd: Tensor, theta: Tensor, iters: int = 5):
    """Invert distortion by fixed-point iteration (differentiable)."""
    x, y = xd.clone(), yd.clone()
    _, _, _, _, k1, k2, k3, p1, p2 = _unpack(theta)
    for _ in range(iters):
        r2 = x * x + y * y
        radial = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
        x_t = 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
        y_t = p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
        x = (xd - x_t) / radial
        y = (yd - y_t) / radial
    return x, y


def project(points_cam: Tensor, theta: Tensor) -> Tensor:
    """3D camera-frame points (..., 3) -> pixel coords (..., 2), with distortion."""
    fx, fy, cx, cy, *_ = _unpack(theta)
    x = points_cam[..., 0] / points_cam[..., 2]
    y = points_cam[..., 1] / points_cam[..., 2]
    xd, yd = distort(x, y, theta)
    u = fx * xd + cx
    v = fy * yd + cy
    return torch.stack([u, v], dim=-1)


def ray_directions(theta: Tensor, h: int, w: int, device=None) -> Tensor:
    """Per-pixel unit bearing vectors in the camera frame -> (3, H, W).

    This is the camera-aware signal: change ``f`` and the rays fan out
    differently, so the conditioned network must produce different metric depth.
    """
    fx, fy, cx, cy, *_ = _unpack(theta)
    u, v = normalized_grid(h, w, device=device or theta.device, dtype=theta.dtype)
    xd = (u - cx) / fx
    yd = (v - cy) / fy
    x, y = undistort(xd, yd, theta)
    dirs = torch.stack([x, y, torch.ones_like(x)], dim=0)
    return dirs / dirs.norm(dim=0, keepdim=True)


def back_project(depth: Tensor, theta: Tensor) -> Tensor:
    """Depth map (H, W) [z along optical axis] + theta -> points (H, W, 3) in cam frame."""
    h, w = depth.shape[-2:]
    fx, fy, cx, cy, *_ = _unpack(theta)
    u, v = normalized_grid(h, w, device=depth.device, dtype=depth.dtype)
    xd = (u - cx) / fx
    yd = (v - cy) / fy
    x, y = undistort(xd, yd, theta)
    return torch.stack([x * depth, y * depth, depth], dim=-1)
