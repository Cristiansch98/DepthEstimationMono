"""Rolling-video BEV metric table — the temporal analogue of depth's d1.

Spatial accuracy: BEV-delta@tau = fraction of car observations within tau metres of
GT in BEV (position analogue of d1's ratio threshold; tau = nuScenes centre-distance
thresholds 1/2/4 m). Temporal consistency: track jitter (RMS step-acceleration) and
implausible-jump count, raw per-frame vs Kalman-smoothed. Numbers from
bev_tracker.py --consistency 6 (UniDepthV2-L, nuScenes, 6 videos, <=40 m, 867 car-obs).

    python src/make_bev_metric_table.py --out viz_bev/bev_metric_table.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# row -> [d@1, d@2, d@4, jitter(RMS accel m), jumps]
ROWS = [
    ("raw  (per-frame)",      "0.411", "0.637", "0.768", "4.19", "25"),
    ("+ Kalman (temporal)",   "0.332", "0.579", "0.762", "1.80", "9"),
]
HEADERS = ["", "BEV-δ@1m ↑", "BEV-δ@2m ↑", "BEV-δ@4m ↑", "jitter [m] ↓", "jumps ↓"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("viz_bev/bev_metric_table.png"))
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 2.5), dpi=200)
    ax.axis("off")
    t = ax.table(cellText=[list(r) for r in ROWS], colLabels=HEADERS, loc="center",
                 cellLoc="center", colWidths=[0.26, 0.155, 0.155, 0.155, 0.155, 0.12])
    t.auto_set_font_size(False); t.set_fontsize(11); t.scale(1, 1.9)
    for (r, c), cell in t.get_celld().items():
        cell.set_edgecolor("#ccc")
        if r == 0:
            cell.set_facecolor("#1B9E9E"); cell.set_text_props(color="white", fontweight="bold")
        elif 1 <= c <= 3:                          # the d1-analogue (spatial) columns
            cell.set_facecolor("#e8f6f5")
        elif c >= 4:                               # the temporal-consistency columns
            cell.set_facecolor("#fdeee7")
    fig.suptitle("Rolling-video BEV metric — spatial accuracy (d1 analogue) + temporal consistency\n"
                 "UniDepth-V2-L · nuScenes · 6 videos · 867 car-obs · ≤ 40 m",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.text(0.5, -0.02,
             "BEV-δ@τ = fraction of cars within τ m of GT (spatial, ↑).  "
             "jitter = RMS per-frame step-acceleration of tracks (temporal, ↓).  "
             "Kalman trades a little accuracy for 57% less jitter (3× fewer jumps).",
             ha="center", fontsize=8, style="italic", color="#555", wrap=True)
    fig.savefig(args.out, bbox_inches="tight", facecolor="white"); plt.close(fig)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
