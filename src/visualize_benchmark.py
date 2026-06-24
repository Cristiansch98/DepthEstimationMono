"""Unified LiDAR-depth overlay for ANY benchmark — the cross-dataset companion
to ``lidar_depth.py``.

Renders the supervision signal (camera image + depth-coloured LiDAR returns) the
exact same way for Argoverse 2, KITTI, nuScenes and Lyft L5, by going through the
shared :mod:`calib_depth.benchmarks` adapter layer. This is how we "include
images from other benchmarks in the same way as Argoverse 2": one command, one
look, four datasets.

    # Argoverse 2 (sample)
    python src/visualize_benchmark.py --benchmark av2 \
        --root data/sensor-sample --cam ring_front_center --num 2 --out viz_bench

    # KITTI raw  (on the remote, once a drive is present)
    python src/visualize_benchmark.py --benchmark kitti \
        --root /data/kitti_raw --cam image_02 --num 2 --out viz_bench

    # nuScenes / Lyft (need their devkit + data; version via env var)
    NUSCENES_VERSION=v1.0-mini python src/visualize_benchmark.py --benchmark nuscenes \
        --root /data/nuscenes --cam CAM_FRONT --num 2 --out viz_bench
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from calib_depth.benchmarks import BENCHMARKS, get_adapter
from lidar_depth import render_overlay

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("viz-bench")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark", required=True, choices=BENCHMARKS)
    ap.add_argument("--root", type=Path, required=True, help="dataset root / dataroot")
    ap.add_argument("--cam", default=None, help="camera name (default: adapter's first)")
    ap.add_argument("--num", type=int, default=2, help="how many frames to render")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--out", type=Path, default=Path("viz_bench"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    adapter = get_adapter(args.benchmark)
    cams = [args.cam] if args.cam else None
    refs = adapter.discover(args.root, cams=cams, stride=args.stride)
    if not refs:
        log.error("no frames found for %s under %s (check layout / cam name)",
                  args.benchmark, args.root)
        return
    log.info("%s: %d frame(s) discovered; rendering %d", args.benchmark, len(refs),
             min(args.num, len(refs)))

    # spread the chosen frames across the discovered set
    step = max(1, len(refs) // args.num)
    for ref in refs[::step][: args.num]:
        frame = adapter.load(ref)
        K = frame.intrinsics
        title = (f"{args.benchmark.upper()} · {ref.cam} · LiDAR-projected ground-truth depth\n"
                 f"GT intrinsics fx={K.fx:.1f} fy={K.fy:.1f}  ({len(frame.depth)} returns, "
                 f"{K.width}x{K.height})  — this is what the self-calibration must recover")
        out = args.out / f"{args.benchmark}_{frame.key}_depth.png"
        render_overlay(frame.image, frame.uv, frame.depth, out, title)


if __name__ == "__main__":
    main()
