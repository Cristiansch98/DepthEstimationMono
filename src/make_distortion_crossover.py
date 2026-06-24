"""Headline figure: robustness to lens distortion — UniDepth (zero-shot SOTA) vs
ours (LiDAR few-shot self-calibration with an explicit distortion model).

Controlled experiment: known radial distortion k1 injected into image + LiDAR
projection (src/calib_depth/synth_distort.py). UniDepth degrades and collapses as
|k1| grows; ours stays flat and overtakes it. Pure matplotlib (numbers are the
measured held-out metrics from info/2026-06-24_distortion-niche.txt).

    python src/make_distortion_crossover.py --out viz_paper/distortion_crossover.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# |k1| axis and metrics (AbsRel, d1); None = not measured
K = [0.0, 0.20, 0.30, 0.40, 0.50, 0.60]
DATA = {
    "KITTI": {
        "uni_absrel": [0.091, 0.074, 0.123, 0.151, 0.193, 0.284],
        "our_absrel": [0.100, 0.087, 0.106, 0.105, 0.107, 0.115],
        "uni_d1":     [0.973, 0.972, 0.883, 0.792, 0.706, 0.583],
        "our_d1":     [0.953, 0.951, 0.889, 0.840, 0.836, 0.828]},
    "nuScenes": {
        "uni_absrel": [0.105, None, None, 0.143, None, 0.776],
        "our_absrel": [0.122, None, None, 0.177, None, 0.252],
        "uni_d1":     [0.921, None, None, 0.781, None, 0.083],
        "our_d1":     [0.853, None, None, 0.790, None, 0.631]},
}
C = {"KITTI": "#E8743B", "nuScenes": "#1B9E9E"}


def _xy(vals):
    xs = [k for k, v in zip(K, vals) if v is not None]
    ys = [v for v in vals if v is not None]
    return xs, ys


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("viz_paper/distortion_crossover.png"))
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3,
                         "axes.axisbelow": True})
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.5, 5.0), dpi=200)

    for ds, d in DATA.items():
        c = C[ds]
        x, y = _xy(d["uni_absrel"]); a1.plot(x, y, "--o", color=c, mfc="white",
                                             label=f"UniDepth (zero-shot) · {ds}")
        x, y = _xy(d["our_absrel"]); a1.plot(x, y, "-s", color=c,
                                             label=f"Ours (few-shot) · {ds}")
        x, y = _xy(d["uni_d1"]); a2.plot(x, y, "--o", color=c, mfc="white")
        x, y = _xy(d["our_d1"]); a2.plot(x, y, "-s", color=c)

    a1.set_title("Depth error vs lens distortion  (lower better)", fontweight="bold")
    a1.set_xlabel("radial distortion  |k₁|"); a1.set_ylabel("AbsRel")
    a1.axvspan(0.30, 0.62, color="grey", alpha=0.08)
    a1.text(0.46, a1.get_ylim()[1] * 0.92, "strong / fisheye\nregime", ha="center",
            fontsize=8.5, color="#555", style="italic")
    a1.legend(fontsize=8.5, loc="upper left")

    a2.set_title("Depth accuracy vs lens distortion  (higher better)", fontweight="bold")
    a2.set_xlabel("radial distortion  |k₁|"); a2.set_ylabel("δ < 1.25")
    a2.set_ylim(0, 1.02); a2.axvspan(0.30, 0.62, color="grey", alpha=0.08)
    a2.plot([], [], "--o", color="#666", mfc="white", label="UniDepth (zero-shot)")
    a2.plot([], [], "-s", color="#666", label="Ours (few-shot)")
    a2.legend(fontsize=8.5, loc="lower left")

    fig.suptitle("Robustness to lens distortion: explicit-distortion LiDAR few-shot "
                 "overtakes a pinhole-assuming foundation model",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(args.out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
