"""EUCM distortion-prediction head (prototype of S1+S4 for UniDepth).

Tests the core of suggestion S1: can a head *predict* lens distortion from a single
image (so a foundation model could handle distorted cameras zero-shot, no per-image
few-shot)? We keep the base focal/centre fixed (a single camera) and predict the
EUCM distortion (alpha, beta) from the image, trained with distortion augmentation
(S4) and a ray-matching loss against the known ground-truth ray field.

  train : random Brown-Conrady k1 per frame -> distorted image + GT rays;
          head(image) -> EUCM(alpha,beta) -> EUCM rays; loss = 1 - cos(pred, GT).
  eval  : predict EUCM on held-out distorted frames, feed to UniDepth.infer(camera=),
          measure depth vs the pinhole-head zero-shot baseline.

Train with the train venv (.venv); eval with the baselines venv (has unidepth).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .benchmarks import get_adapter
from .model import CalibEncoder
from .synth_distort import _undistort, distort_frame

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("eucm-head")
logging.getLogger("lidar-depth").setLevel(logging.WARNING)
HW = (518, 518)


class DistortionHead(nn.Module):
    """image -> EUCM distortion (alpha in (0,1), beta>0). Base fx,fy,cx,cy fixed."""

    def __init__(self, feat_dim: int = 256):
        super().__init__()
        self.enc = CalibEncoder(feat_dim)
        self.mlp = nn.Sequential(nn.Linear(feat_dim, 128), nn.GELU(), nn.Linear(128, 2))
        self.mlp[-1].weight.data.zero_()
        self.mlp[-1].bias.data.zero_()          # init alpha~0.5, beta~1 (near pinhole)

    def forward(self, image):                    # image (B,3,H,W) in [0,1]
        raw = self.mlp(self.enc(image))
        alpha = torch.sigmoid(raw[:, 0])
        beta = torch.nn.functional.softplus(raw[:, 1]) + 1e-3
        return alpha, beta


def _grid(h, w, dev):
    vs = torch.arange(h, device=dev, dtype=torch.float32) + 0.5
    us = torch.arange(w, device=dev, dtype=torch.float32) + 0.5
    v, u = torch.meshgrid(vs, us, indexing="ij")
    return u.reshape(-1), v.reshape(-1)


def eucm_rays(u, v, K, alpha, beta):
    """EUCM unprojection -> unit rays + validity mask (closed form, differentiable).

    EUCM is only defined where 1-(2a-1)*b*r2 > 0 (its FOV). We mask pixels outside
    that region out of the loss rather than letting the sqrt/division produce NaNs."""
    mx = (u - K["cx"]) / K["fx"]
    my = (v - K["cy"]) / K["fy"]
    r2 = mx**2 + my**2
    arg = 1 - (2 * alpha - 1) * beta * r2
    valid = arg > 1e-3
    s = torch.sqrt(arg.clamp_min(1e-3))
    denom = (alpha * s + (1 - alpha)).clamp_min(1e-3)
    mz = (1 - beta * alpha**2 * r2) / denom
    rays = torch.stack([mx, my, mz], -1)
    return rays / rays.norm(dim=-1, keepdim=True).clamp_min(1e-6), valid


def bc_rays(u, v, K, k1):
    """Brown-Conrady GT rays: undistort pixel -> ray (numpy then tensor)."""
    xd = ((u - K["cx"]) / K["fx"]).cpu().numpy()
    yd = ((v - K["cy"]) / K["fy"]).cpu().numpy()
    x, y = _undistort(xd, yd, k1, 0.0)
    rays = np.stack([x, y, np.ones_like(x)], -1)
    rays = rays / np.linalg.norm(rays, axis=-1, keepdims=True)
    return torch.from_numpy(rays.astype(np.float32)).to(u.device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default="kitti")
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--cam", default="image_02")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--k1-range", type=float, nargs=2, default=[-0.5, 0.0])
    ap.add_argument("--out", type=Path, default=Path("checkpoints_adapt/eucm_head.pt"))
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    import cv2

    base = get_adapter(args.benchmark)
    refs = base.discover(args.root, cams=[args.cam], stride=1)
    train_refs = refs[:max(1, len(refs) - 20)]
    log.info("EUCM head training | %s | %d frames | k1 in [%.2f,%.2f]",
             args.benchmark, len(train_refs), *args.k1_range)

    head = DistortionHead().to(dev).train()
    opt = torch.optim.Adam(head.parameters(), lr=5e-4)
    u, v = _grid(*HW, dev)
    rng = np.random.default_rng(0)
    skipped = 0

    for step in range(args.steps):
        ref = train_refs[step % len(train_refs)]
        k1 = float(rng.uniform(*args.k1_range))
        f = distort_frame(base.load(ref), k1, 0.0)
        K = f.intrinsics
        sx, sy = HW[1] / K.width, HW[0] / K.height
        Kd = {"fx": K.fx * sx, "fy": K.fy * sy, "cx": K.cx * sx, "cy": K.cy * sy}
        img = cv2.resize(f.image, (HW[1], HW[0]), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(img).permute(2, 0, 1).float().div(255)[None].to(dev)

        alpha, beta = head(x)
        pred, valid = eucm_rays(u, v, Kd, alpha[0], beta[0])
        gt = bc_rays(u, v, Kd, k1)
        cos = (pred * gt).sum(-1)
        loss = (1 - cos)[valid].mean()
        if not torch.isfinite(loss):
            skipped += 1; continue
        opt.zero_grad(); loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        if not torch.isfinite(gn):
            opt.zero_grad(set_to_none=True); skipped += 1; continue
        opt.step()
        if step % 50 == 0 or step == args.steps - 1:
            ang = float(torch.rad2deg(torch.acos(cos[valid].clamp(-1, 1))).mean())
            log.info("step %3d | k1=%.2f | ray-loss %.5f | mean ray err %.2f deg | a=%.3f b=%.3f",
                     step, k1, float(loss), ang, float(alpha[0]), float(beta[0]))
    if skipped:
        log.info("skipped %d non-finite steps", skipped)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"head": head.state_dict()}, args.out)
    log.info("saved EUCM head -> %s", args.out)


if __name__ == "__main__":
    main()
