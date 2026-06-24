"""Zero-shot baselines on any benchmark, through the unified adapter layer.

Loads an external pretrained metric-depth model and scores it on the SAME frames
and metrics our model is evaluated on, by pulling (native-res image, LiDAR uv,
metric depth) from a :mod:`calib_depth.benchmarks` adapter. This is the
apples-to-apples comparison peer review requires.

We report metrics two ways:
  * metric   — predicted depth used as-is (the model's own metric scale);
  * scaled   — predicted depth median-aligned to GT per image (removes scale /
               calibration error, isolating relative-depth quality).

Models (``--model``):
  dav2-metric  Depth Anything V2 metric (HF, no extra install; same backbone family)
  zoedepth     ZoeDepth ZoeD_NK (torch hub)

    python -m calib_depth.baseline_eval --model dav2-metric --benchmark kitti \
        --root <kitti_raw> --cam image_02 --max-frames 100
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch

from .benchmarks import BENCHMARKS, get_adapter

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("baseline")
logging.getLogger("lidar-depth").setLevel(logging.WARNING)


def _metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    m = (gt > 1e-3) & np.isfinite(pred) & (pred > 1e-3)
    pred, gt = pred[m], gt[m]
    thr = np.maximum(gt / pred, pred / gt)
    return {"AbsRel": float(np.mean(np.abs(pred - gt) / gt)),
            "RMSE": float(np.sqrt(np.mean((pred - gt) ** 2))),
            "d1": float((thr < 1.25).mean()), "n": int(m.sum())}


def _sample(depth_hw: np.ndarray, uv: np.ndarray) -> np.ndarray:
    h, w = depth_hw.shape
    c = np.clip(np.round(uv[:, 0]).astype(int), 0, w - 1)
    r = np.clip(np.round(uv[:, 1]).astype(int), 0, h - 1)
    return depth_hw[r, c]


# --------------------------- model loaders --------------------------- #
def load_dav2_metric(dev, model_id="depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf"):
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    proc = AutoImageProcessor.from_pretrained(model_id)
    net = AutoModelForDepthEstimation.from_pretrained(model_id).to(dev).eval()

    @torch.no_grad()
    def predict(img_uint8: np.ndarray) -> np.ndarray:
        from PIL import Image
        h, w = img_uint8.shape[:2]
        inp = proc(images=Image.fromarray(img_uint8), return_tensors="pt").to(dev)
        d = net(**inp).predicted_depth                      # (1, h', w') metric meters
        d = torch.nn.functional.interpolate(d[:, None], size=(h, w), mode="bilinear",
                                            align_corners=False)[0, 0]
        return d.cpu().numpy()
    return predict


def load_zoedepth(dev):
    net = torch.hub.load("isl-org/ZoeDepth", "ZoeD_NK", pretrained=True,
                         trust_repo=True).to(dev).eval()

    @torch.no_grad()
    def predict(img_uint8: np.ndarray) -> np.ndarray:
        from PIL import Image
        return net.infer_pil(Image.fromarray(img_uint8))    # (h, w) metric meters
    return predict


def load_unidepth(dev, model_id="lpiccinelli/unidepth-v2-vitl14"):
    """UniDepth V2 — metric depth + self-predicted intrinsics (the key competitor)."""
    from unidepth.models import UniDepthV2
    net = UniDepthV2.from_pretrained(model_id).to(dev).eval()

    @torch.no_grad()
    def predict(img_uint8: np.ndarray) -> np.ndarray:
        h, w = img_uint8.shape[:2]
        rgb = torch.from_numpy(img_uint8).permute(2, 0, 1).to(dev)   # (3,H,W) uint8
        out = net.infer(rgb)                                          # predicts its own K
        d = out["depth"].squeeze()                                   # (h', w') metric meters
        if d.shape != (h, w):
            d = torch.nn.functional.interpolate(d[None, None], size=(h, w),
                                                mode="bilinear", align_corners=False)[0, 0]
        return d.cpu().numpy()
    return predict


LOADERS = {"dav2-metric": load_dav2_metric, "zoedepth": load_zoedepth,
           "unidepth": load_unidepth}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(LOADERS))
    ap.add_argument("--benchmark", required=True, choices=BENCHMARKS)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--cam", default=None)
    ap.add_argument("--max-frames", type=int, default=100)
    ap.add_argument("--max-depth", type=float, default=80.0)
    ap.add_argument("--distort", default=None,
                    help="inject Brown-Conrady radial distortion 'k1[,k2]'")
    ap.add_argument("--fisheye", default=None,
                    help="render through a KB fisheye 'k1[,k2]' (angular distortion model)")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    predict = LOADERS[args.model](dev)
    adapter = get_adapter(args.benchmark)
    if args.distort:
        from .synth_distort import DistortingAdapter
        ks = [float(x) for x in args.distort.split(",")]
        adapter = DistortingAdapter(adapter, ks[0], ks[1] if len(ks) > 1 else 0.0)
        log.info("synthetic Brown-Conrady: k1=%.3f k2=%.3f", ks[0], ks[1] if len(ks) > 1 else 0.0)
    if args.fisheye:
        from .synth_fisheye import FisheyeAdapter
        ks = [float(x) for x in args.fisheye.split(",")]
        adapter = FisheyeAdapter(adapter, ks[0], ks[1] if len(ks) > 1 else 0.0)
        log.info("synthetic KB fisheye: k1=%.3f k2=%.3f", ks[0], ks[1] if len(ks) > 1 else 0.0)
    refs = adapter.discover(args.root, cams=[args.cam] if args.cam else None, stride=1)
    if not refs:
        log.error("no frames for %s under %s", args.benchmark, args.root); return
    idxs = np.linspace(0, len(refs) - 1, min(args.max_frames, len(refs))).astype(int)
    log.info("%s | %s | %d/%d frames", args.model, args.benchmark, len(idxs), len(refs))

    met, scl = [], []
    for n, i in enumerate(idxs):
        frame = adapter.load(refs[int(i)])
        gt, uv = frame.depth, frame.uv
        keep = gt < args.max_depth
        gt, uv = gt[keep], uv[keep]
        if len(gt) < 50:
            continue
        pmap = predict(frame.image)
        pred = _sample(pmap, uv)
        met.append(_metrics(pred, gt))
        s = np.median(gt) / max(np.median(pred[pred > 1e-3]), 1e-6)
        scl.append(_metrics(pred * s, gt))
        if (n + 1) % 25 == 0:
            log.info("  %d/%d", n + 1, len(idxs))

    agg = lambda L, k: float(np.mean([d[k] for d in L]))
    print(f"\n====  BASELINE  {args.model}  on  {args.benchmark}  ({len(met)} frames, <{args.max_depth:.0f} m)  ====")
    print(f"  metric (as-is):     AbsRel {agg(met,'AbsRel'):.3f} | RMSE {agg(met,'RMSE'):.2f} | d1 {agg(met,'d1'):.3f}")
    print(f"  median-scaled:      AbsRel {agg(scl,'AbsRel'):.3f} | RMSE {agg(scl,'RMSE'):.2f} | d1 {agg(scl,'d1'):.3f}")
    print("=" * 64)


if __name__ == "__main__":
    main()
