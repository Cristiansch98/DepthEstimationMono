# 01 · Project overview

**Repo:** `~/Argoverse2` — work on the [Argoverse 2](https://www.argoverse.org/av2.html) AV dataset.
Two related deliverables share this repo:

## A. Argoverse 2 Motion-Forecasting visualizer (done)
Cool renders of the MF dataset built on the official `av2` API, plus a from-scratch
format explainer (`FORMAT.md`).
- `src/visualize_av2.py` — dark-themed "hero" PNG (HD map + all agent trajectories,
  focal agent as a plasma time-coloured trail with oriented boxes) + self-contained
  animated GIF (the bundled `av2 0.2.1` animation is broken on matplotlib ≥ 3.11).
- `src/inspect_format.py` — prints parquet schema + map JSON structure.
- Outputs in `viz/`. MF scenario = 11 s @ 10 Hz = 110 steps (50 observed / 60 predicted).

## B. SelfCalibDepth research framework (active focus)
**Goal:** learn **metric distance from a single camera image**, using synchronized
**LiDAR as ground-truth depth**, while the model **self-calibrates the camera**
(`f, c, distortion`) and stays **camera-aware** (generalizes across focal lengths).

User-selected scoping: full self-calibration · fine-tune a foundation model
(Depth Anything V2) · camera-aware generalization.

**Headline objective:** "real distance to vehicles from a single image." AV2 ships
3D cuboid annotations → exact per-vehicle metric distance for free, used both as an
optional loss and as the headline *distance-to-vehicle error* metric.

Full design → `FRAMEWORK.md`; package → `src/calib_depth/`; details → `02_architecture.md`.
