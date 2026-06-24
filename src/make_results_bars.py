"""Paper-quality quantitative results figure for the cross-dataset study.

Three panels — depth AbsRel (↓), depth δ<1.25 (↑), focal-length error fx (↓, log) —
each grouping the three adaptation regimes (zero-shot · +latent · +latent+head) for
KITTI and nuScenes, with the in-domain Argoverse 2 result drawn as a reference line.
Pure-matplotlib; no GPU/data needed (numbers are the reported held-out metrics).

    python src/make_results_bars.py --out viz_paper/results_bars.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# held-out metrics (match PAPER.md §5.1)
REGIMES = ["zero-shot", "+latent\n(20 frames)", "+latent\n+head"]
DATA = {
    "KITTI":    {"AbsRel": [0.390, 0.402, 0.100], "d1": [0.026, 0.021, 0.953], "fx": [97.1, 0.2, 1.1]},
    "nuScenes": {"AbsRel": [0.274, 0.284, 0.122], "d1": [0.184, 0.172, 0.853], "fx": [45.3, 0.5, 0.7]},
}
AV2 = {"AbsRel": 0.112, "d1": 0.884, "fx": 0.26}   # in-domain reference
COLORS = {"KITTI": "#E8743B", "nuScenes": "#1B9E9E"}
PANELS = [("AbsRel", "Depth error  AbsRel  (lower better)", False),
          ("d1", "Depth accuracy  δ<1.25  (higher better)", False),
          ("fx", "Focal-length error  |Δfx|  [%]  (log, lower better)", True)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("viz_paper/results_bars.png"))
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.25,
                         "axes.axisbelow": True})
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), dpi=200)
    x = np.arange(len(REGIMES))
    w = 0.36

    for ax, (key, title, logy) in zip(axes, PANELS):
        for j, ds in enumerate(["KITTI", "nuScenes"]):
            vals = DATA[ds][key]
            bars = ax.bar(x + (j - 0.5) * w, vals, w, label=ds, color=COLORS[ds],
                          edgecolor="white", linewidth=0.6, zorder=3)
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v * (1.15 if logy else 1.0) + (0 if logy else 0.012),
                        f"{v:g}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
        # AV2 in-domain reference line
        ax.axhline(AV2[key], ls="--", lw=1.4, color="#444", zorder=2)
        ax.text(len(REGIMES) - 0.5, AV2[key], "  AV2 in-domain", color="#444",
                fontsize=8.5, va="bottom" if not logy else "top", ha="right", style="italic")
        if logy:
            ax.set_yscale("log")
            ax.set_ylim(0.1, 200)
        elif key == "d1":
            ax.set_ylim(0, 1.05)
        ax.set_xticks(x); ax.set_xticklabels(REGIMES)
        ax.set_title(title, fontsize=11.5, fontweight="bold")
        ax.set_axisbelow(True)
    axes[0].legend(loc="upper right", framealpha=0.9)

    fig.suptitle("Cross-dataset generalisation — AV2-trained model adapted to KITTI & nuScenes "
                 "(held-out frames)", fontsize=13.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(args.out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
