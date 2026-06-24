"""Metric Depth Estimation table: zero-shot d1 (higher better) for UniDepthV2-L
on KITTI and nuScenes (vs LiDAR, depth <= 80 m). Numbers from baseline_eval.py.

    python src/make_metric_depth_table.py --out viz_paper/metric_depth_d1.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# dataset -> (frames, d1, AbsRel, RMSE)  [UniDepthV2-L, zero-shot, metric, <80 m]
ROWS = [
    ("KITTI  (image_02)", 108, 0.973, 0.090, 4.18),
    ("nuScenes  (CAM_FRONT)", 200, 0.922, 0.102, 6.44),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("viz_paper/metric_depth_d1.png"))
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    headers = ["Dataset", "Frames", "d1 (δ<1.25) ↑", "AbsRel ↓", "RMSE [m] ↓"]
    cells = [[r[0], str(r[1]), f"{r[2]:.3f}", f"{r[3]:.3f}", f"{r[4]:.2f}"] for r in ROWS]

    fig, ax = plt.subplots(figsize=(10.5, 2.4), dpi=200)
    ax.axis("off")
    t = ax.table(cellText=cells, colLabels=headers, loc="center", cellLoc="center",
                 colWidths=[0.30, 0.15, 0.21, 0.17, 0.17])
    t.auto_set_font_size(False); t.set_fontsize(11); t.scale(1, 1.8)
    for (r, c), cell in t.get_celld().items():
        cell.set_edgecolor("#ccc")
        if r == 0:
            cell.set_facecolor("#1B9E9E"); cell.set_text_props(color="white", fontweight="bold")
        elif c == 2:                                  # highlight the d1 column
            cell.set_facecolor("#e8f6f5"); cell.set_text_props(fontweight="bold")
    fig.suptitle("Metric Depth Estimation — zero-shot  d1 (↑)   ·   UniDepth-V2-L (ViT-L/14)",
                 fontsize=12.5, fontweight="bold", y=0.96)
    fig.text(0.5, 0.06, "Zero-shot, no fine-tuning · metric depth vs LiDAR · depth ≤ 80 m",
             ha="center", fontsize=8.5, style="italic", color="#555")
    fig.savefig(args.out, bbox_inches="tight", facecolor="white"); plt.close(fig)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
