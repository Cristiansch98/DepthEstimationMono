# 04 · Environment, data & key constants

## Local Python env (this box)
- **Python 3.14 only**, no conda, no system pip/ensurepip. Venv at `.venv/`:
  ```bash
  python3.14 -m venv --without-pip .venv
  .venv/bin/python <(curl -s https://bootstrap.pypa.io/get-pip.py)
  .venv/bin/pip install -r requirements.txt
  ```
- Use `opencv-python-headless` (server lacks `libGL.so.1`). `av2 0.2.1` ships cp314 wheels.
- numpy 2.x: use `np.ptp(arr)` (no `ndarray.ptp()`).
- `torch 2.12.1` has cp314 + CUDA13 wheels, so the same venv *could* run it — but
  **do all GPU training on the remote 5090**, not locally.

## Remote GPU server (all training/inference)
- `thunderlane@10.162.44.13`, pass `ftmsm_thunderlane`, dir `~/Cubos_code/SelfCalibDepth`.
- NVIDIA RTX 5090, 32 GB, sm_120 (Blackwell). Python 3.10 venv, `torch 2.12.1+cu130`, `av2 0.3.6`.
- SSH: `SSH_ASKPASS=/tmp/askpass.sh DISPLAY=:0 setsid ssh -o StrictHostKeyChecking=no thunderlane@10.162.44.13 '<cmd>'`
  (askpass: `echo -e '#!/bin/bash\necho ftmsm_thunderlane' > /tmp/askpass.sh && chmod +x /tmp/askpass.sh`).
- Backbone: `depth-anything/Depth-Anything-V2-Small-hf` via transformers (24.8 M params).
  Train res 518×518; intrinsics + LiDAR uv scaled to match.
- **Gotcha:** `pkill -f calib_depth.train` matches its own SSH shell → use the
  bracket trick `pkill -f "[c]alib_depth.train"`, or launch without pkill.

## Data on disk
- **Motion-forecasting:** `data/motion-forecasting/<id>/` = 2 files (scenario parquet +
  log_map_archive json). 5 val scenarios (~1.5 MB). Public HTTPS, no AWS creds:
  `https://argoverse.s3.amazonaws.com/datasets/av2/motion-forecasting/`.
- **Sensor (depth training):** `data/sensor/` train = 40 logs (7.0 GB), val = 8 logs
  (1.4 GB). 7 ring cameras, lidar_stride=3. 2,519 sweeps / 17,630 frames / 48 annotation
  files. Built by `src/download_av2_sensor.py` (selective, 24-worker, resumable; ~180 MB
  & ~24 s per log). `discover_logs()` auto-finds logs — no manifest wiring needed.
- **Sensor-sample:** `data/sensor-sample/val/<log>/` — one 19 MB log for quick local checks.

## Key constants
- `ring_front_center` GT intrinsics: **fx=fy=1781.49, cx=775.51, cy=1022.94**, image **1550×2048** (portrait).
- Other ring cams: fx≈1687, **2048×1550** (landscape). front_center is the focal-diversity outlier
  → for real cross-camera generalization add stereo cams or a 2nd dataset.
- `annotations.feather` schema: timestamp_ns, track_uuid, category, length/width/height_m,
  quaternion (qw..qz), tx/ty/tz_m (ego frame), num_interior_pts. Vehicle categories:
  REGULAR_VEHICLE, LARGE_VEHICLE, BUS, BOX_TRUCK, TRUCK*, MOTORCYCLE, …
- Verified vehicle-distance range from cuboids: ~7–219 m.

## Repo / git
- `git` toplevel is `/home/cubos` (origin = unrelated `colors_osm`). **No dedicated remote
  for `~/Argoverse2` yet** — ask the user where to host before pushing this project.
- Repo convention (CLAUDE.md): mandatory `info/` folder; write a dated `.txt` after each
  significant run; read `info/` (and now `info_logs/`) before code to save tokens.
