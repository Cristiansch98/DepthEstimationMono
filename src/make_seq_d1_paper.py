"""Paper figure: probabilistic per-frame depth d1 over rolling videos.

The SAME metric as the static table (d1 = δ<1.25), kept per-frame and summarized as a
distribution. (a) per-frame d1 trajectories per video with the pooled mean±σ band —
temporal behaviour; (b) per-video d1 distributions (violins) ordered by mean — the
spread IS the temporal consistency. Reads the dumped values (viz_bev/seq_d1.json).

    python src/make_seq_d1_paper.py --data viz_bev/seq_d1.json --out viz_paper/seq_d1_paper.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("viz_bev/seq_d1.json"))
    ap.add_argument("--out", type=Path, default=Path("viz_paper/seq_d1_paper.png"))
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    data = json.load(open(args.data))
    names = list(data)
    arrs = [np.array(data[n]) for n in names]
    pooled = np.concatenate(arrs)
    mu, sd = pooled.mean(), pooled.std()
    order = np.argsort([a.mean() for a in arrs])      # videos sorted by mean d1

    plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.25,
                         "axes.axisbelow": True})
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15.5, 5.4), dpi=200,
                                 gridspec_kw={"width_ratios": [1.45, 1]})
    cmap = plt.get_cmap("viridis")

    # (a) per-frame trajectories + pooled mean ± sigma band
    maxlen = max(len(a) for a in arrs)
    a1.axhspan(mu - sd, mu + sd, color="#1B9E9E", alpha=0.12, zorder=0,
               label=f"pooled mean ± σ  ({mu:.3f} ± {sd:.3f})")
    a1.axhline(mu, ls="--", color="#0c6b6b", lw=1.3, zorder=1)
    for i, a in enumerate(arrs):
        a1.plot(range(len(a)), a, "-", lw=1.1, alpha=0.7, color=cmap(i / len(arrs)))
    worst = pooled.min()
    a1.scatter([np.argmin(arrs[int(np.argmin([a.min() for a in arrs]))])],
               [worst], facecolor="none", edgecolor="#D11", s=120, lw=1.6, zorder=5)
    a1.annotate(f"worst frame  d1={worst:.2f}", (3, worst + 0.02), color="#D11", fontsize=8.5)
    a1.set_xlabel("frame index along video"); a1.set_ylabel("depth d1  (δ < 1.25)")
    a1.set_xlim(0, maxlen - 1); a1.set_ylim(0.35, 1.02)
    a1.legend(loc="lower left", fontsize=8.5)
    a1.set_title("(a) Per-frame d1 along each video — same metric, kept per-frame",
                 fontsize=11.5, fontweight="bold")

    # (b) per-video violin distributions, ordered by mean d1
    parts = a2.violinplot([arrs[i] for i in order], positions=range(len(order)),
                          vert=False, showmeans=False, showextrema=False, widths=0.85)
    for i, b in enumerate(parts["bodies"]):
        b.set_facecolor(cmap(order[i] / len(arrs))); b.set_alpha(0.65); b.set_edgecolor("white")
    means = [arrs[i].mean() for i in order]
    a2.scatter(means, range(len(order)), c="k", s=22, zorder=4, label="per-video mean")
    a2.axvline(mu, ls="--", color="#0c6b6b", lw=1.3, label=f"pooled mean {mu:.3f}")
    a2.set_yticks(range(len(order)))
    a2.set_yticklabels([names[i].replace("scene-", "s") for i in order], fontsize=8)
    a2.set_xlabel("depth d1  (δ < 1.25)"); a2.set_xlim(0.35, 1.02)
    a2.legend(loc="lower left", fontsize=8.5)
    a2.set_title("(b) Per-video d1 distribution — spread = temporal consistency",
                 fontsize=11.5, fontweight="bold")

    fig.suptitle("Probabilistic depth d1 over rolling videos  ·  UniDepth-V2-L · nuScenes "
                 f"(10 videos, {len(pooled)} frames, ≤80 m)", fontsize=13.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(args.out, bbox_inches="tight", facecolor="white"); plt.close(fig)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
