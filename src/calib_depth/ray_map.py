"""Camera-aware conditioning: turn calibration ``theta`` into a tensor the depth
decoder can consume, so identical pixels under different focal lengths yield
different metric depth.

We use the **ray map** representation (per-pixel unit bearing vectors) plus the
classic CamConvs normalized-coordinate channels. Concatenated, they give the
network an explicit, differentiable description of the (learned) camera geometry.
"""

from __future__ import annotations

import torch
from torch import Tensor

from . import camera_model as cm


def conditioning_map(theta: Tensor, h: int, w: int) -> Tensor:
    """Build the (C, H, W) camera-aware conditioning tensor from ``theta``.

    Channels:
        0..2  unit ray direction (x, y, z) in the camera frame
        3..4  CamConvs normalized pixel coords ((u-cx)/fx, (v-cy)/fy)
        5     per-pixel angle from the optical axis (FOV cue)
    """
    rays = cm.ray_directions(theta, h, w)                  # (3, H, W)
    fx, fy, cx, cy, *_ = theta.unbind(dim=-1)
    u, v = cm.normalized_grid(h, w, device=theta.device, dtype=theta.dtype)
    xn = (u - cx) / fx
    yn = (v - cy) / fy
    # acos has infinite gradient at +-1 (a ray exactly on the optical axis is
    # (0,0,1) -> z=1), which would send inf grads into theta. Clamp away from +-1.
    angle = torch.acos(rays[2].clamp(-1 + 1e-4, 1 - 1e-4))  # (H, W)
    return torch.cat([rays, xn[None], yn[None], angle[None]], dim=0)


def conditioning_channels() -> int:
    return 6
