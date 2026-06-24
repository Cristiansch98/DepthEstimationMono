"""Training objectives. The interesting ones (``loss_3d``, ``loss_reproj``) are
what make the *calibration* observable: they depend on ``theta`` and are anchored
by metric LiDAR, so gradients correct a wrong focal length.

All functions operate on a sparse LiDAR target given as (pixel uv, metric depth)
plus the corresponding 3D LiDAR points in the camera frame.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from . import camera_model as cm


def _sample(depth_map: Tensor, uv: Tensor) -> Tensor:
    """Bilinearly sample a (1,1,H,W) depth map at pixel coords uv (N,2) -> (N,)."""
    h, w = depth_map.shape[-2:]
    grid = uv.clone()
    grid[:, 0] = grid[:, 0] / (w - 1) * 2 - 1
    grid[:, 1] = grid[:, 1] / (h - 1) * 2 - 1
    grid = grid.view(1, -1, 1, 2)
    return F.grid_sample(depth_map, grid, align_corners=True).view(-1)


def loss_silog(pred_depth_map: Tensor, uv: Tensor, gt_depth: Tensor, lam: float = 0.85) -> Tensor:
    """Scale-invariant log depth error at LiDAR pixels (primary depth signal)."""
    pred = _sample(pred_depth_map, uv).clamp_min(1e-3)
    g = torch.log(pred) - torch.log(gt_depth.clamp_min(1e-3))
    return torch.sqrt((g**2).mean() - lam * g.mean() ** 2 + 1e-6)


def loss_3d(pred_depth_map: Tensor, theta: Tensor, uv: Tensor, lidar_cam: Tensor) -> Tensor:
    """Back-project predicted depth with ``theta`` and match the 3D LiDAR points.

    Depends on theta -> this is the term that calibrates the camera.
    """
    pred = _sample(pred_depth_map, uv).clamp_min(1e-3)
    fx, fy, cx, cy, *_ = theta.unbind(dim=-1)
    x, y = cm.undistort((uv[:, 0] - cx) / fx, (uv[:, 1] - cy) / fy, theta)
    pts = torch.stack([x * pred, y * pred, pred], dim=-1)
    return (pts - lidar_cam).norm(dim=-1).mean()


def loss_reproj(theta: Tensor, lidar_cam: Tensor, uv_gt: Tensor) -> Tensor:
    """Project LiDAR points with ``theta``; they must land at their true pixel."""
    uv_pred = cm.project(lidar_cam, theta)
    return (uv_pred - uv_gt).norm(dim=-1).mean()


def loss_smooth(depth_map: Tensor, image: Tensor) -> Tensor:
    """Edge-aware first-order smoothness."""
    dx = (depth_map[..., :, :-1] - depth_map[..., :, 1:]).abs()
    dy = (depth_map[..., :-1, :] - depth_map[..., 1:, :]).abs()
    ix = image.mean(1, keepdim=True)[..., :, :-1] - image.mean(1, keepdim=True)[..., :, 1:]
    iy = image.mean(1, keepdim=True)[..., :-1, :] - image.mean(1, keepdim=True)[..., 1:, :]
    return (dx * torch.exp(-ix.abs())).mean() + (dy * torch.exp(-iy.abs())).mean()


def loss_prior(theta: Tensor, h: int, w: int, aspect_w: float = 1.0) -> Tensor:
    """Weak stabilizing priors (full self-calibration -> only gentle nudges).

    ``aspect_w`` scales the fx~fy penalty. It is valid when the image keeps its
    native aspect ratio, but WRONG under an anisotropic resize (e.g. KITTI
    1242x375 -> 518x518 makes fx != fy), where it should be reduced/disabled.
    """
    fx, fy, cx, cy, k1, k2, k3, p1, p2 = theta.unbind(dim=-1)
    center = ((cx - w / 2) / w) ** 2 + ((cy - h / 2) / h) ** 2
    distort = k1**2 + k2**2 + k3**2 + p1**2 + p2**2
    aspect = (torch.log(fx / fy)) ** 2
    return center + 1e-2 * distort + aspect_w * aspect


def total_loss(pred_depth_map, theta, batch, weights: dict) -> tuple[Tensor, dict]:
    """Combine terms. ``batch`` carries uv, gt_depth, lidar_cam, image, (H,W)."""
    h, w = pred_depth_map.shape[-2:]
    terms = {
        "silog": loss_silog(pred_depth_map, batch["uv"], batch["gt_depth"]),
        "l3d": loss_3d(pred_depth_map, theta, batch["uv"], batch["lidar_cam"]),
        "reproj": loss_reproj(theta, batch["lidar_cam"], batch["uv"]),
        "smooth": loss_smooth(pred_depth_map, batch["image"]),
        "prior": loss_prior(theta, h, w, aspect_w=weights.get("aspect_w", 1.0)),
    }
    total = sum(weights.get(k, 0.0) * v for k, v in terms.items())
    return total, {k: float(v.detach()) for k, v in terms.items()}
