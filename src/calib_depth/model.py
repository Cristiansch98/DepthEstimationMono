"""The model: Depth Anything V2 + camera-aware metric mapping + self-calibration.

v2 (pinpointed improvements over v1 — see FRAMEWORK.md / info logs):

  P-A  Camera-conditioned scale mapping. DAv2's *relative* structure is excellent;
       what is missing is the metric scale, which is focal-dependent. So instead of
       regressing metric depth with a conv (v1), we map DAv2's relative disparity
       to metric via a global log-depth affine whose endpoints are predicted from
       the image feature AND the learned focal length (camera-aware), then add a
       small bounded local residual for detail.
  P-C  Bounded radial distortion re-enabled (true full self-calibration). k1,k2 are
       squashed through tanh to a small range so ``undistort`` cannot diverge.
  P-B  Optional backbone unfreeze (flag in train.py) for a stronger depth head.

ASSUMPTIONS (annotated):
  A1  DAv2 ``predicted_depth`` is a *relative, affine-invariant disparity* map
      (monotonic in 1/Z). We do not assume its sign/orientation — the two log-depth
      endpoints are free, so the net learns whichever way the disparity runs.
  A2  Relative -> metric is well approximated by a single global affine in
      log-depth per image (the standard affine-invariant-depth assumption), with a
      bounded (+-0.2 in log-space, ~+-22%) local residual for fine structure.
  A3  The metric scale correlates with focal length -> we condition the endpoints
      on (log fx, log fy). This is what makes depth camera-aware.
  A4  AV2 ring cameras are near-pinhole: radial distortion is small, so |k1|,|k2|
      bounded to 0.05 is sufficient and keeps undistort convergent. Tangential
      (p1,p2) and k3 are left at 0.
  A5  Metric depth is clamped to [0.1, 300] m (AV2 LiDAR range).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from . import ray_map
from .camera_model import Intrinsics

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DEPTH_MIN, DEPTH_MAX = 0.1, 300.0
DISTORTION_BOUND = 0.05          # A4
# v3: relax the local residual so the conv can express metric structure freely
# (v2/v2b's tight 0.2 bound was too rigid and lost to v1's free conv head on
# AbsRel). The camera-conditioned affine now acts as a well-posed init/prior;
# the conv refines up to exp(+-1.5) ~= +-4.5x. Pairs with backbone unfreeze (P-B).
RESIDUAL_BOUND = 1.5             # A2 (log-space)


class LearnableIntrinsics(nn.Module):
    """Per-camera latent + image-conditioned residual; learns (log_fx, log_fy, cx, cy)
    and, if enabled, bounded radial (k1, k2). theta is returned full 9-dim."""

    def __init__(self, num_cameras: int, init: Intrinsics, feat_dim: int = 256,
                 learn_distortion: bool = True):
        super().__init__()
        self.learn_distortion = learn_distortion
        n_core = 6 if learn_distortion else 4   # +2 for raw k1,k2
        core0 = torch.zeros(n_core)
        core0[0] = torch.log(torch.tensor(init.fx)); core0[1] = torch.log(torch.tensor(init.fy))
        core0[2] = init.cx; core0[3] = init.cy      # raw k start at 0 -> k=0
        self.latent = nn.Parameter(core0.repeat(max(num_cameras, 1), 1))
        self.delta_head = nn.Sequential(nn.Linear(feat_dim, 128), nn.GELU(), nn.Linear(128, n_core))
        self.delta_head[-1].weight.data.zero_()
        self.delta_head[-1].bias.data.zero_()

    def forward(self, cam_idx: Tensor, image_feat: Tensor) -> Tensor:
        raw = self.latent[cam_idx] + 0.1 * self.delta_head(image_feat)
        fx, fy = torch.exp(raw[..., 0]), torch.exp(raw[..., 1])
        cx, cy = raw[..., 2], raw[..., 3]
        z = torch.zeros_like(fx)
        if self.learn_distortion:
            k1 = DISTORTION_BOUND * torch.tanh(raw[..., 4])   # A4: bounded radial
            k2 = DISTORTION_BOUND * torch.tanh(raw[..., 5])
        else:
            k1 = k2 = z
        return torch.stack([fx, fy, cx, cy, k1, k2, z, z, z], dim=-1)


class CalibEncoder(nn.Module):
    """Tiny CNN -> global feature vector driving calibration + scale endpoints."""

    def __init__(self, out_dim: int = 256):
        super().__init__()
        c = [3, 32, 64, 128, 256]
        layers = []
        for i in range(len(c) - 1):
            layers += [nn.Conv2d(c[i], c[i + 1], 3, stride=2, padding=1),
                       nn.GroupNorm(8, c[i + 1]), nn.GELU()]
        self.net = nn.Sequential(*layers)
        self.proj = nn.Linear(c[-1], out_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(self.net(x).mean(dim=(-1, -2)))


class ScaleMappingHead(nn.Module):
    """P-A (v2b): map DAv2 disparity -> metric depth via a camera-conditioned
    affine in INVERSE-depth, then a bounded local log-residual.

    A2 (corrected): DAv2 output is affine-invariant *disparity* (disp = a/Z + b),
    so metric inverse-depth is affine in the disparity:  1/Z = s*sn + t  (s,t>0).
    v2 wrongly made depth linear in log(disp), which is the wrong curve and hurt
    AbsRel. We predict (log s, log t) per image, conditioned on (feat, log focal)."""

    def __init__(self, feat_dim: int, cond_ch: int):
        super().__init__()
        self.st_mlp = nn.Sequential(nn.Linear(feat_dim + 2, 128), nn.GELU(), nn.Linear(128, 2))
        # init: 1/Z = s*sn + t with s~0.32, t~0.0125  -> Z(sn=0)~80 m, Z(sn=1)~3 m
        self.st_mlp[-1].weight.data.mul_(0.01)
        self.st_mlp[-1].bias.data = torch.tensor([torch.log(torch.tensor(0.32)),
                                                  torch.log(torch.tensor(0.0125))])
        self.refine = nn.Sequential(
            nn.Conv2d(1 + cond_ch + 3, 48, 3, padding=1), nn.GroupNorm(8, 48), nn.GELU(),
            nn.Conv2d(48, 48, 3, padding=1), nn.GroupNorm(8, 48), nn.GELU(),
            nn.Conv2d(48, 1, 1))
        self.refine[-1].weight.data.zero_(); self.refine[-1].bias.data.zero_()

    def forward(self, sn: Tensor, feat: Tensor, theta: Tensor, cond: Tensor, image: Tensor) -> Tensor:
        log_focal = torch.log(theta[:, :2].clamp_min(1e-3))                  # A3
        log_s, log_t = self.st_mlp(torch.cat([feat, log_focal], -1)).unbind(-1)
        s, t = torch.exp(log_s)[:, None, None, None], torch.exp(log_t)[:, None, None, None]
        inv_z = s * sn + t                                                   # A2: affine in 1/Z
        res = RESIDUAL_BOUND * torch.tanh(self.refine(torch.cat([sn, cond, image], dim=1)))
        return (torch.exp(res) / inv_z).clamp(DEPTH_MIN, DEPTH_MAX)


class CameraAwareDepthNet(nn.Module):
    def __init__(self, num_cameras: int, init_intrinsics: Intrinsics,
                 backbone: str = "depth-anything/Depth-Anything-V2-Small-hf",
                 freeze_backbone: bool = True, learn_distortion: bool = True):
        super().__init__()
        from transformers import AutoModelForDepthEstimation
        self.dav2 = AutoModelForDepthEstimation.from_pretrained(backbone)
        for p in self.dav2.parameters():
            p.requires_grad_(not freeze_backbone)
        self.calib_encoder = CalibEncoder(256)
        self.calib = LearnableIntrinsics(num_cameras, init_intrinsics, 256, learn_distortion)
        self.cond_ch = ray_map.conditioning_channels()
        self.head = ScaleMappingHead(256, self.cond_ch)
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, image: Tensor, cam_idx: Tensor) -> tuple[Tensor, Tensor]:
        b, _, h, w = image.shape
        norm = (image - self.mean) / self.std
        rel = self.dav2(pixel_values=norm).predicted_depth.unsqueeze(1)      # (B,1,H,W)
        sn = (rel - rel.amin(dim=(-1, -2), keepdim=True)) / (
            rel.amax(dim=(-1, -2), keepdim=True) - rel.amin(dim=(-1, -2), keepdim=True) + 1e-6)
        feat = self.calib_encoder(norm)
        theta = self.calib(cam_idx, feat)
        cond = torch.stack([ray_map.conditioning_map(theta[i], h, w) for i in range(b)])
        depth = self.head(sn, feat, theta, cond, image)
        return depth, theta
