"""Build a SelfCalibDepth training set from the public AV2 Sensor dataset.

Downloads only what the framework needs, *per log*:
  - calibration/intrinsics.feather + calibration/egovehicle_SE3_sensor.feather
  - city_SE3_egovehicle.feather               (ego poses for motion compensation)
  - every Kth LiDAR sweep                      (sparse depth ground truth)
  - the nearest camera frame to each sweep, for each requested ring camera

Selective (not whole multi-GB logs), parallel, and resumable (existing non-empty
files are skipped). Public HTTPS bucket -> no AWS credentials needed.

    python src/download_av2_sensor.py --split train --num-logs 20 \
        --cameras ring_front_center ring_front_left ring_front_right \
                  ring_side_left ring_side_right ring_rear_left ring_rear_right \
        --lidar-stride 2 --out data/sensor --workers 16
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional
from urllib.request import urlopen

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("dl-av2")

BUCKET = "https://argoverse.s3.amazonaws.com"
ROOT = "datasets/av2/sensor"
ALL_RING = ["ring_front_center", "ring_front_left", "ring_front_right",
            "ring_side_left", "ring_side_right", "ring_rear_left", "ring_rear_right"]


def _get(url: str, timeout: int = 60) -> bytes:
    with urlopen(url, timeout=timeout) as r:
        return r.read()


def list_prefix(prefix: str, delimiter: bool = False) -> tuple[List[str], List[str]]:
    """List keys (and common prefixes) under an S3 prefix, following pagination."""
    keys: List[str] = []
    prefixes: List[str] = []
    token: Optional[str] = None
    while True:
        url = f"{BUCKET}/?list-type=2&prefix={prefix}&max-keys=1000"
        if delimiter:
            url += "&delimiter=/"
        if token:
            from urllib.parse import quote
            url += f"&continuation-token={quote(token)}"
        xml = _get(url).decode()
        keys += re.findall(r"<Key>(.*?)</Key>", xml)
        prefixes += re.findall(r"<Prefix>(.*?)</Prefix>", xml)
        m = re.search(r"<NextContinuationToken>(.*?)</NextContinuationToken>", xml)
        if not m:
            break
        token = m.group(1)
    return keys, prefixes


def download_one(key: str, out_root: Path) -> int:
    """Download a single S3 key to out_root/<key-without-ROOT-prefix>. Returns bytes."""
    rel = key[len(f"{ROOT}/"):]
    dst = out_root / rel
    if dst.exists() and dst.stat().st_size > 0:
        return 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    data = _get(f"{BUCKET}/{key}")
    tmp = dst.with_suffix(dst.suffix + ".part")
    tmp.write_bytes(data)
    tmp.rename(dst)
    return len(data)


def stem_ts(key: str) -> int:
    return int(Path(key).stem)


def plan_log(log_prefix: str, cameras: List[str], lidar_stride: int) -> List[str]:
    """Return the list of S3 keys to fetch for one log."""
    wanted: List[str] = [
        f"{log_prefix}calibration/intrinsics.feather",
        f"{log_prefix}calibration/egovehicle_SE3_sensor.feather",
        f"{log_prefix}city_SE3_egovehicle.feather",
    ]
    lidar_keys, _ = list_prefix(f"{log_prefix}sensors/lidar/")
    lidar_keys = sorted(k for k in lidar_keys if k.endswith(".feather"))
    chosen_lidar = lidar_keys[::lidar_stride]
    wanted += chosen_lidar
    chosen_ts = [stem_ts(k) for k in chosen_lidar]

    for cam in cameras:
        cam_keys, _ = list_prefix(f"{log_prefix}sensors/cameras/{cam}/")
        cam_keys = sorted(k for k in cam_keys if k.endswith(".jpg"))
        if not cam_keys:
            continue
        cam_ts = [stem_ts(k) for k in cam_keys]
        picked = set()
        for lt in chosen_ts:
            j = min(range(len(cam_ts)), key=lambda i: abs(cam_ts[i] - lt))
            picked.add(cam_keys[j])
        wanted += sorted(picked)
    return wanted


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["train", "val", "test"], default="train")
    ap.add_argument("--num-logs", type=int, default=20)
    ap.add_argument("--cameras", nargs="+", default=ALL_RING)
    ap.add_argument("--lidar-stride", type=int, default=2)
    ap.add_argument("--out", type=Path, default=Path("data/sensor"))
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--skip", type=int, default=0, help="skip the first N logs (for sharded runs)")
    args = ap.parse_args()

    t0 = time.time()
    log.info("Listing %s logs ...", args.split)
    _, log_prefixes = list_prefix(f"{ROOT}/{args.split}/", delimiter=True)
    log_prefixes = sorted(p for p in log_prefixes if re.search(r"/[0-9a-f-]{36}/$", p))
    selected = log_prefixes[args.skip: args.skip + args.num_logs]
    log.info("Found %d logs total; selected %d (skip=%d). Cameras=%d, lidar_stride=%d",
             len(log_prefixes), len(selected), args.skip, len(args.cameras), args.lidar_stride)

    # Plan all keys (this lists each log's sensor dirs; parallelize the listing).
    all_keys: List[str] = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(plan_log, lp, args.cameras, args.lidar_stride): lp for lp in selected}
        for i, fut in enumerate(as_completed(futs), 1):
            keys = fut.result()
            all_keys += keys
            log.info("  planned log %d/%d  (+%d files, total %d)", i, len(selected), len(keys), len(all_keys))

    log.info("Downloading %d files with %d workers ...", len(all_keys), args.workers)
    total_bytes = 0
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(download_one, k, args.out) for k in all_keys]
        for fut in as_completed(futs):
            total_bytes += fut.result()
            done += 1
            if done % 200 == 0:
                log.info("  %d/%d files  (%.2f GB so far)", done, len(all_keys), total_bytes / 1e9)

    manifest = {
        "split": args.split, "num_logs": len(selected), "cameras": args.cameras,
        "lidar_stride": args.lidar_stride, "num_files": len(all_keys),
        "logs": [p.rstrip("/").split("/")[-1] for p in selected],
    }
    (args.out / f"manifest_{args.split}.json").write_text(json.dumps(manifest, indent=2))
    log.info("Done in %.1f min. New data: %.2f GB. Manifest -> %s",
             (time.time() - t0) / 60, total_bytes / 1e9, args.out / f"manifest_{args.split}.json")


if __name__ == "__main__":
    sys.exit(main())
