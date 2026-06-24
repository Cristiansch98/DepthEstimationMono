# Argoverse 2 — Motion Forecasting visualizer

Cool visualizations of the [Argoverse 2](https://www.argoverse.org/av2.html)
Motion Forecasting dataset, built on the official
[`av2`](https://pypi.org/project/av2/) API — plus a from-scratch explanation of
the data format in [`FORMAT.md`](FORMAT.md).

![hero](viz/0015197d_hero.png)

Each scenario is 11 s of city driving (110 frames @ 10 Hz): dozens of agents
moving over a local HD vector map, with one **focal agent** to be predicted.

## Setup

Python 3.14, no conda — a plain venv (`av2` ships cp314 wheels):

```bash
python3.14 -m venv --without-pip .venv      # this box lacks ensurepip
.venv/bin/python <(curl -s https://bootstrap.pypa.io/get-pip.py)
.venv/bin/pip install -r requirements.txt
# headless server: use opencv-python-headless (no libGL needed)
```

## Get a scenario (public, no credentials)

```bash
ID=0015197d-b916-43b6-bcaa-8a7d90d7b87d
DIR=data/motion-forecasting/$ID && mkdir -p $DIR
BASE=https://argoverse.s3.amazonaws.com/datasets/av2/motion-forecasting/val/$ID
curl -s $BASE/scenario_$ID.parquet      -o $DIR/scenario_$ID.parquet
curl -s $BASE/log_map_archive_$ID.json  -o $DIR/log_map_archive_$ID.json
```

## Run

```bash
# Static "hero" figures for every downloaded scenario
.venv/bin/python src/visualize_av2.py --data-root data/motion-forecasting --out viz

# Add an animated GIF (agents moving through time)
.venv/bin/python src/visualize_av2.py --data-root data/motion-forecasting --out viz --gif

# Understand the on-disk format of any scenario
.venv/bin/python src/inspect_format.py data/motion-forecasting/$ID
```

## Outputs

| File                       | What                                                    |
|----------------------------|---------------------------------------------------------|
| `viz/<id>_hero.png`        | HD map + all agent trajectories; focal agent as a time-coloured trail with oriented vehicle boxes |
| `viz/<id>_anim.gif`        | Animation: every agent as an oriented box, focal agent with a fading tail, observed → prediction phases |

### Legend
- **Plasma trail / boxes** — focal agent (the prediction target), coloured by time.
- **Green** — the autonomous vehicle (ego, `track_id == "AV"`).
- **Blue** — vehicles · **orange** — pedestrians · **pink/purple** — cyclists/bikes.
- **Gold patches** — pedestrian crossings · **grey lines** — HD-map lanes.

## Layout

```
src/visualize_av2.py   # hero figure + animation renderer
src/inspect_format.py  # prints the parquet schema + map JSON structure
FORMAT.md              # full data-format reference
data/motion-forecasting/<id>/   # downloaded scenarios (2 files each)
viz/                   # rendered PNGs / GIFs
info/                  # run logs
```

## Research framework — `SelfCalibDepth`

Learn **metric distance from a single image** by using synchronized **LiDAR as
ground-truth depth**, while **self-calibrating** the camera (`f, c, distortion`)
and fine-tuning **Depth Anything V2** with the calibration fed in (camera-aware).
Full design in [`FRAMEWORK.md`](FRAMEWORK.md); package in `src/calib_depth/`.

```bash
# The ground-truth signal: project a LiDAR sweep into a ring camera (proven)
.venv/bin/python src/lidar_depth.py \
    --log-dir data/sensor-sample/val/<log_id> --cam ring_front_center --out viz

# Training (run on the remote RTX 5090; needs torch + Depth Anything V2)
python -m calib_depth.train --data-root data/sensor-sample --epochs 20
```

The geometry (`camera_model.py`, `ray_map.py`) and losses are real and
cross-checked against av2 (0.000 px); the foundation-model backbone in
`model.py` is the GPU-side TODO (see FRAMEWORK.md roadmap).

## Notes
- The bundled `av2 0.2.1` animation (`viz.scenario_visualization`) is broken
  against matplotlib ≥ 3.11 (`Rectangle` angle became keyword-only), so the
  animation here is a self-contained reimplementation.
- MF provides pose + velocity but **not** box dimensions, so boxes use a fixed
  vehicle footprint (~4.6 × 2.0 m).
