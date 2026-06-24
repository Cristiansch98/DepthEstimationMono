"""Per-frame depth d1 over a video sequence -> a probabilistic representation.

The SAME metric as the static table (d1 = fraction of pixels with max(pred/gt,gt/pred)
< 1.25), but computed per frame across a rolling video and summarized as a distribution
(mean, std, percentiles) rather than a single average. The spread of the per-frame d1
captures temporal (in)consistency with the very same metric: a jumpy model has a wider,
lower-tailed per-frame d1 distribution than a smooth one at the same mean.

    NUSCENES_VERSION=v1.0-mini XFORMERS_DISABLED=1 PYTHONPATH=src <venv>/bin/python \
      -m calib_depth.seq_d1 --root <nuscenes> --scenes 3 --out viz_bev
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch

from .baseline_eval import _metrics, _sample
from .benchmarks import FrameRef, get_adapter
from .bev_tracker import load_unidepth

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("seq-d1")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--cam", default="CAM_FRONT")
    ap.add_argument("--scenes", type=int, default=3)
    ap.add_argument("--cap", type=float, default=80.0)
    ap.add_argument("--out", type=Path, default=Path("viz_bev/seq_d1.png"))
    ap.add_argument("--dump", type=Path, default=None, help="save per-scene per-frame d1 as JSON")
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    adapter = get_adapter("nuscenes")
    adapter.discover(args.root, cams=[args.cam])      # ensures the devkit is loaded
    nusc = adapter._nusc
    predict = load_unidepth(dev)

    series = []                                        # (scene_name, [per-frame d1])
    for si in range(args.scenes):
        scene = nusc.scene[si]
        toks, t = [], scene["first_sample_token"]
        while t:
            toks.append(t); t = nusc.get("sample", t)["next"]
        d1s = []
        for tk in toks:
            f = adapter.load(FrameRef("nuscenes", scene["token"], args.cam, tk))
            keep = f.depth <= args.cap
            if keep.sum() < 50:
                continue
            pred = _sample(predict(f.image), f.uv[keep])
            d1s.append(_metrics(pred, f.depth[keep])["d1"])
        series.append((scene["name"], d1s))
        log.info("scene %s: %d frames, d1 mean %.3f std %.3f", scene["name"], len(d1s),
                 np.mean(d1s), np.std(d1s))

    if args.dump:
        import json
        args.dump.parent.mkdir(parents=True, exist_ok=True)
        json.dump({name: s for name, s in series}, open(args.dump, "w"))
        log.info("dumped per-frame d1 -> %s", args.dump)

    pooled = np.array([v for _, s in series for v in s])
    print("\n========  PER-FRAME DEPTH d1 OVER VIDEO (UniDepthV2-L, nuScenes, <=%.0f m)  ========"
          % args.cap)
    print(f"  frames: {len(pooled)} over {len(series)} videos")
    print(f"  d1  mean {pooled.mean():.3f}  std {pooled.std():.3f}  "
          f"min {pooled.min():.3f}  p10 {np.percentile(pooled,10):.3f}  "
          f"p50 {np.percentile(pooled,50):.3f}  p90 {np.percentile(pooled,90):.3f}")
    print("  (std / low percentiles = temporal consistency, via the SAME d1 metric)")
    print("=" * 76)

    # figure: per-frame d1 time series (mean +/- std band) + distribution
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 5), dpi=150,
                                 gridspec_kw={"width_ratios": [1.6, 1]})
    cmap = plt.get_cmap("tab10")
    for i, (name, s) in enumerate(series):
        a1.plot(range(len(s)), s, "-o", ms=3, lw=1.3, color=cmap(i), label=name)
    a1.axhline(pooled.mean(), ls="--", color="#444", lw=1.2)
    a1.fill_between([0, max(len(s) for _, s in series) - 1],
                    pooled.mean() - pooled.std(), pooled.mean() + pooled.std(),
                    color="#888", alpha=0.15, label="pooled mean ± σ")
    a1.set_xlabel("frame index"); a1.set_ylabel("depth d1 (δ<1.25)")
    a1.set_ylim(0, 1.02); a1.grid(alpha=0.25); a1.legend(fontsize=8, loc="lower right")
    a1.set_title("Per-frame d1 along the video (same metric, kept per-frame)", fontweight="bold")

    a2.hist(pooled, bins=20, range=(0, 1), color="#1B9E9E", edgecolor="white", orientation="horizontal")
    a2.axhline(pooled.mean(), ls="--", color="#444", lw=1.2,
               label=f"mean {pooled.mean():.3f}")
    a2.axhline(pooled.mean() - pooled.std(), ls=":", color="#888", lw=1.0,
               label=f"±σ ({pooled.std():.3f})")
    a2.axhline(pooled.mean() + pooled.std(), ls=":", color="#888", lw=1.0)
    a2.set_ylim(0, 1.02); a2.set_xlabel("frame count"); a2.set_ylabel("depth d1")
    a2.legend(fontsize=8, loc="lower right")
    a2.set_title("Distribution of per-frame d1", fontweight="bold")
    fig.suptitle("Probabilistic d1 over rolling videos — spread = temporal consistency  "
                 "(UniDepth-V2-L · nuScenes)", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(args.out, bbox_inches="tight", facecolor="white"); plt.close(fig)
    log.info("saved -> %s", args.out)


if __name__ == "__main__":
    main()
