"""Draw the SelfCalibDepth architecture (ray-map coupling) as a clean paper figure."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

INK = "#111111"
BLUE = "#2f6db0"
GREEN = "#2e8b57"
ORANGE = "#c8791a"


def box(ax, x, y, w, h, text, fc="white", ec=INK, fs=9, tc=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                                linewidth=1.3, edgecolor=ec, facecolor=fc))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc, wrap=True)


def arrow(ax, p, q, color=INK, style="-", lw=1.4, rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=13, lw=lw,
                                 color=color, linestyle=style,
                                 connectionstyle=f"arc3,rad={rad}"))


def main():
    fig, ax = plt.subplots(figsize=(9.2, 3.7), dpi=200)
    ax.set_xlim(0, 10); ax.set_ylim(0, 4); ax.axis("off")

    box(ax, 0.15, 1.6, 1.15, 0.8, "Single\nimage", fc="#eef3f9")
    box(ax, 1.7, 2.55, 1.9, 0.8, "Depth Anything V2\n(backbone)", fc="#eef3f9", ec=BLUE)
    box(ax, 1.7, 0.7, 1.9, 0.8, "Calib. encoder +\nlearnable intrinsics  θ", fc="#fdf1e3", ec=ORANGE)
    box(ax, 4.0, 0.7, 1.35, 0.8, "Ray map\nR(θ)", fc="#fdf1e3", ec=ORANGE)
    box(ax, 4.0, 2.55, 1.55, 0.8, "relative\ndisparity", fc="white")
    box(ax, 5.95, 1.55, 1.5, 0.95, "Metric head\n(camera-aware)", fc="#eef3f9", ec=BLUE)
    box(ax, 7.8, 1.55, 1.05, 0.95, "metric\ndepth  Z", fc="white")
    box(ax, 7.55, 0.15, 2.3, 0.7, "back-project with θ\n→ 3-D points", fc="#eafaf0", ec=GREEN)
    box(ax, 4.35, 0.15, 2.6, 0.7, "LiDAR points (cam. frame)\nL_3d  +  L_reproj  +  L_silog", fc="#eafaf0", ec=GREEN)

    arrow(ax, (1.3, 2.0), (1.7, 2.7))          # image -> dav2
    arrow(ax, (1.3, 1.85), (1.7, 1.2))         # image -> calib
    arrow(ax, (3.6, 2.95), (4.0, 2.95))        # dav2 -> rel
    arrow(ax, (3.6, 1.1), (4.0, 1.1))          # calib -> raymap
    arrow(ax, (5.35, 2.7), (6.0, 2.45), color=BLUE)   # rel -> head
    arrow(ax, (5.35, 1.15), (6.2, 1.6), color=ORANGE)  # raymap -> head
    arrow(ax, (1.3, 1.75), (6.0, 1.9), color="#888888", lw=1.0, rad=-0.25)  # image -> head (skip)
    arrow(ax, (7.45, 2.0), (7.8, 2.0))         # head -> Z
    arrow(ax, (8.3, 1.55), (8.7, 0.85), color=GREEN)   # Z -> backproject
    arrow(ax, (7.55, 0.5), (6.95, 0.5), color=GREEN)   # backproj -> lidar loss
    # gradient feedback to theta (dashed red)
    arrow(ax, (4.35, 0.38), (2.75, 0.7), color="#cc2b2b", style="--", lw=1.3, rad=-0.3)
    ax.text(3.4, 0.0, "gradient → corrects θ (self-calibration)", fontsize=7.5,
            color="#cc2b2b", ha="center")

    fig.savefig("viz_v3/arch_diagram.png", bbox_inches="tight", pad_inches=0.08, facecolor="white")
    print("wrote viz_v3/arch_diagram.png")


if __name__ == "__main__":
    Path("viz_v3").mkdir(exist_ok=True)
    main()
