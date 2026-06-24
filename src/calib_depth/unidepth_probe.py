"""Prototype S1+S2 on UniDepth, on synthetic KB-fisheye frames.

Validates the diagnosis (depth decoder is distortion-capable; only the pinhole
self-calibration head is the bottleneck) and prototypes the fix:

  zeroshot : UniDepth predicts a pinhole (its default)            -> fails on fisheye
  gtcam    : UniDepth GIVEN the GT Fisheye624 camera (=our KB)    -> S1 upside
  fewshot  : UniDepth GIVEN an EUCM camera fit from LiDAR (ours)  -> S1+S2

Depth sampled at LiDAR pixels; AbsRel/d1 vs GT (median-scaled too).
Run in the baselines venv (has unidepth) with PYTHONPATH=src.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch

from .benchmarks import get_adapter
from .synth_fisheye import FisheyeAdapter, _theta_d

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("ud-probe")
logging.getLogger("lidar-depth").setLevel(logging.WARNING)


def _metrics(pred, gt):
    m = (gt > 1e-3) & np.isfinite(pred) & (pred > 1e-3)
    pred, gt = pred[m], gt[m]
    thr = np.maximum(gt / pred, pred / gt)
    return dict(AbsRel=float(np.mean(np.abs(pred - gt) / gt)),
               d1=float((thr < 1.25).mean()), n=int(m.sum()))


def _sample(dmap, uv):
    h, w = dmap.shape
    c = np.clip(np.round(uv[:, 0]).astype(int), 0, w - 1)
    r = np.clip(np.round(uv[:, 1]).astype(int), 0, h - 1)
    return dmap[r, c]


def _fisheye624(K, k1, k2, dev):
    from unidepth.utils.camera import Fisheye624
    p = torch.zeros(1, 16, dtype=torch.float32, device=dev)
    p[0, 0], p[0, 1], p[0, 2], p[0, 3] = K.fx, K.fy, K.cx, K.cy
    p[0, 4], p[0, 5] = k1, k2          # radial th^3, th^5 (exactly our KB model)
    cam = Fisheye624(p)
    if getattr(cam, "K", None) is None:
        Km = torch.eye(3, device=dev)[None]
        Km[0, 0, 0], Km[0, 1, 1], Km[0, 0, 2], Km[0, 1, 2] = K.fx, K.fy, K.cx, K.cy
        cam.K = Km
    return cam


def _kb_forward_uv(a, b, K, k1, k2):
    """Our KB fisheye forward (numpy): normalized (a,b)=X/Z,Y/Z -> distorted pixel uv."""
    r = np.sqrt(a**2 + b**2)
    th = np.arctan(r)
    td = _theta_d(th, k1, k2)
    s = np.where(r > 1e-9, td / np.maximum(r, 1e-9), 0.0)
    return K.fx * s * a + K.cx, K.fy * s * b + K.cy


def _collect_corr(base, refs, k1, k2):
    """LiDAR correspondences for calibration: true camera-frame 3-D pts <-> distorted uv."""
    P3, UV = [], []
    for ref in refs:
        f = base.load(ref)
        K = f.intrinsics
        a = (f.uv[:, 0] - K.cx) / K.fx
        b = (f.uv[:, 1] - K.cy) / K.fy
        Z = f.depth
        u, v = _kb_forward_uv(a, b, K, k1, k2)
        m = (u >= 0) & (u < K.width) & (v >= 0) & (v < K.height) & (Z > 0.5)
        P3.append(np.stack([a * Z, b * Z, Z], -1)[m])
        UV.append(np.stack([u, v], -1)[m])
    return np.concatenate(P3).astype(np.float32), np.concatenate(UV).astype(np.float32), K


def _fit_kb(P3, UV, K, k1_gt, dev, steps=800):
    """Fit a KB/Fisheye624 camera (fx,fy,cx,cy,k1,k2) to LiDAR correspondences.

    We do NOT assume k1 is known — it is recovered from the LiDAR reprojection
    (our few-shot self-calibration). The model family is KB (matches a real fisheye)."""
    p3 = torch.from_numpy(P3).to(dev)
    uv = torch.from_numpy(UV).to(dev)
    x, y, z = p3[:, 0], p3[:, 1], p3[:, 2]
    a, b = x / z.clamp_min(1e-3), y / z.clamp_min(1e-3)
    r = torch.sqrt(a**2 + b**2).clamp_min(1e-9)
    th = torch.atan(r)
    # Fit only the unknown distortion k1, with base intrinsics fixed at the camera's
    # known focal/centre (distortion is what was *added*). Central LiDAR under-
    # constrains both higher-order terms (k2) and the principal point, so a low-order
    # fit with fixed base K is the well-posed choice — otherwise drift in cx/cy/k2
    # corrupts the PERIPHERAL rays (no LiDAR there) and breaks UniDepth's depth.
    fx, fy, cx, cy = (torch.tensor(float(v), device=dev) for v in (K.fx, K.fy, K.cx, K.cy))
    k1 = torch.zeros((), device=dev, requires_grad=True)
    opt = torch.optim.Adam([k1], lr=0.05)
    for _ in range(steps):
        th_d = th * (1 + k1 * th**2)
        s = th_d / r
        loss = ((fx * s * a + cx - uv[:, 0])**2 + (fy * s * b + cy - uv[:, 1])**2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    params = torch.tensor([float(fx), float(fy), float(cx), float(cy), float(k1.detach()), 0.0],
                          device=dev)
    log.info("fit KB (k1-only, fixed base K): k1=%.3f (GT %.3f)  reproj=%.2f px",
             float(k1.detach()), k1_gt, float(loss.detach())**0.5)
    return params


def _fisheye624_from(params, dev):
    from unidepth.utils.camera import Fisheye624
    p = torch.zeros(1, 16, dtype=torch.float32, device=dev)
    p[0, :4] = params[:4]
    p[0, 4], p[0, 5] = params[4], params[5]
    cam = Fisheye624(p)
    if getattr(cam, "K", None) is None:
        Km = torch.eye(3, device=dev)[None]
        Km[0, 0, 0], Km[0, 1, 1], Km[0, 0, 2], Km[0, 1, 2] = params[0], params[1], params[2], params[3]
        cam.K = Km
    return cam


@torch.no_grad()
def _depth(model, img_uint8, dev, camera=None):
    rgb = torch.from_numpy(img_uint8).permute(2, 0, 1).to(dev)
    out = model.infer(rgb, camera=camera) if camera is not None else model.infer(rgb)
    return out["depth"].squeeze().float().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default="kitti")
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--cam", default="image_02")
    ap.add_argument("--k1", type=float, default=4.0)
    ap.add_argument("--frames", type=int, default=15)
    ap.add_argument("--modes", nargs="+", default=["zeroshot", "gtcam"])
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    from unidepth.models import UniDepthV2
    model = UniDepthV2.from_pretrained("lpiccinelli/unidepth-v2-vitl14").to(dev).eval()
    base = get_adapter(args.benchmark)
    fish = FisheyeAdapter(base, args.k1, 0.0)
    refs = fish.discover(args.root, cams=[args.cam], stride=1)
    # disjoint calibration (first 20) vs held-out eval split
    calib_refs = refs[:20]
    eval_pool = refs[20:] if len(refs) > 20 else refs
    idxs = np.linspace(0, len(eval_pool) - 1, min(args.frames, len(eval_pool))).astype(int)
    eval_refs = [eval_pool[j] for j in idxs]
    log.info("UniDepth probe | %s | KB k1=%.2f | calib=%d eval=%d | modes=%s",
             args.benchmark, args.k1, len(calib_refs), len(eval_refs), args.modes)

    fs_params = None
    if "fewshot" in args.modes:
        P3, UV, Kc = _collect_corr(base, calib_refs, args.k1, 0.0)
        log.info("calibration correspondences: %d LiDAR points over %d frames", len(P3), len(calib_refs))
        fs_params = _fit_kb(P3, UV, Kc, args.k1, dev)  # fresh camera built per frame (infer mutates)

    res = {m: [] for m in args.modes}
    for ref in eval_refs:
        f = fish.load(ref)
        if len(f.depth) < 50:
            continue
        for m in args.modes:
            if m == "gtcam":
                cam = _fisheye624(f.intrinsics, args.k1, 0.0, dev)
            elif m == "fewshot":
                cam = _fisheye624_from(fs_params, dev)   # fresh each frame: infer() mutates camera
            else:
                cam = None
            try:
                d = _depth(model, f.image, dev, camera=cam)
                res[m].append(_metrics(_sample(d, f.uv), f.depth))
            except Exception as e:
                log.warning("%s failed: %s", m, str(e)[:120])

    print(f"\n====  UniDepth on {args.benchmark} KB-fisheye k1={args.k1}  ====")
    for m in args.modes:
        if res[m]:
            ar = np.mean([d["AbsRel"] for d in res[m]])
            d1 = np.mean([d["d1"] for d in res[m]])
            print(f"  {m:9s}: AbsRel {ar:.3f} | d1 {d1:.3f}  (n_frames={len(res[m])})")
    print("=" * 48)


if __name__ == "__main__":
    main()
