"""Figure: diagnosing + repairing UniDepth's distortion failure.

UniDepth's depth decoder is distortion-capable (it consumes any camera via rays), but
its self-calibration head predicts only a pinhole. Giving it a distortion camera — GT,
or one recovered by our 20-frame LiDAR self-calibration — restores depth to the oracle.
Numbers from info_logs/06_unidepth_analysis.md (synthetic KB-fisheye KITTI).

    python src/make_unidepth_fix.py --out viz_paper/unidepth_fix.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

K = ["k1 = 2.5", "k1 = 4.0"]
# (AbsRel, d1) per regime
ABSREL = {"UniDepth zero-shot (pinhole head)": [0.112, 0.158],
          "UniDepth + ours (LiDAR few-shot calib)": [0.087, 0.106],
          "UniDepth + GT camera (oracle)": [0.087, 0.106]}
D1 = {"UniDepth zero-shot (pinhole head)": [0.845, 0.695],
      "UniDepth + ours (LiDAR few-shot calib)": [0.927, 0.866],
      "UniDepth + GT camera (oracle)": [0.927, 0.866]}
COLORS = ["#B0B0B0", "#1B9E9E", "#222222"]
HATCH = [None, None, "//"]


def _panel(ax, data, title, ylabel, ylim=None):
    x = np.arange(len(K))
    w = 0.26
    for j, (label, vals) in enumerate(data.items()):
        bars = ax.bar(x + (j - 1) * w, vals, w, label=label, color=COLORS[j],
                      edgecolor="white", linewidth=0.6, hatch=HATCH[j], zorder=3)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + (ylim[1] * 0.012 if ylim else 0.003),
                    f"{v:g}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(K)
    ax.set_title(title, fontweight="bold", fontsize=11.5)
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(axis="y", alpha=0.3); ax.set_axisbelow(True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("viz_paper/unidepth_fix.png"))
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.8), dpi=200)
    _panel(a1, ABSREL, "Depth error under fisheye distortion (lower better)", "AbsRel", (0, 0.2))
    _panel(a2, D1, "Depth accuracy under fisheye distortion (higher better)", "δ < 1.25", (0, 1.05))
    a1.legend(fontsize=8.5, loc="upper left")
    fig.suptitle("Repairing UniDepth's distortion failure: a LiDAR few-shot camera "
                 "self-calibration matches the GT-camera oracle",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(args.out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
