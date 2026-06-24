# Argoverse 2 — Motion Forecasting format

This documents the **Motion Forecasting (MF)** subset of Argoverse 2, the
lightest and most visualization-friendly part of the dataset. Reproduce any of
the numbers below with:

```bash
python src/inspect_format.py data/motion-forecasting/<scenario_id>
```

## The dataset at a glance

| Property        | Value                                                      |
|-----------------|------------------------------------------------------------|
| # scenarios     | 250,000 (train 199,908 / val 24,988 / test 25,104)         |
| Duration        | 11 s per scenario, sampled at **10 Hz** → **110 timesteps** |
| Split of time   | first **50** steps (5 s) *observed*, last **60** *to predict* |
| Cities          | Austin, Miami, Pittsburgh, Palo Alto, Washington-DC, Dearborn |
| Coordinate frame| city frame, **metres** (x, y, z)                           |

A single scenario lives in one folder named by its UUID and holds **exactly two
files**:

```
<scenario_id>/
├── scenario_<scenario_id>.parquet       # the agents and how they move
└── log_map_archive_<scenario_id>.json   # the local HD vector map
```

---

## 1. `scenario_<id>.parquet` — trajectories

A **long-format** table: one row per `(track, timestep)`. Columns:

| Column            | Type     | Meaning                                                  |
|-------------------|----------|----------------------------------------------------------|
| `track_id`        | string   | Stable per-agent ID (the ego vehicle is literally `"AV"`)|
| `object_type`     | string   | `vehicle, pedestrian, motorcyclist, cyclist, bus, static, background, construction, riderless_bicycle, unknown` |
| `object_category` | int64    | Track quality 0–3 (see categories below)                 |
| `timestep`        | int64    | `0 … 109`                                                |
| `position_x/y`    | double   | Bounding-box centre in the city frame (m)                |
| `heading`         | double   | Yaw in radians, in the city frame                        |
| `velocity_x/y`    | double   | Instantaneous velocity (m/s)                             |
| `observed`        | bool     | `True` for the first 50 steps (the input window)         |
| `scenario_id`     | string   | UUID, repeated on every row                              |
| `focal_track_id`  | string   | Which track is *the* prediction target                   |
| `city`            | string   | City name                                                |
| `start/end_timestamp`, `num_timestamps` | — | scenario timing metadata             |

> Note: MF gives pose (position + heading) and velocity, **not** box length/width.
> The visualizer draws fixed-size oriented boxes (~4.6 × 2.0 m for vehicles).

### Track categories (`object_category`)

| value | name             | meaning                                                  |
|-------|------------------|----------------------------------------------------------|
| 0     | `TRACK_FRAGMENT` | low quality, only a few timestamps                       |
| 1     | `UNSCORED_TRACK` | decent quality, contextual input only                    |
| 2     | `SCORED_TRACK`   | high quality, scored in the *multi-agent* challenge      |
| 3     | `FOCAL_TRACK`    | the single agent the scenario was built around (scored)  |

---

## 2. Typed object model (`av2`)

`scenario_serialization.load_argoverse_scenario_parquet()` turns the parquet
into a clean nested object — this is what the visualizer consumes:

```
ArgoverseScenario
├── scenario_id      : str
├── city_name        : str
├── timestamps_ns    : int64[110]
├── focal_track_id   : str
└── tracks           : List[Track]
        ├── track_id      : str
        ├── object_type   : ObjectType        # enum
        ├── category      : TrackCategory      # enum
        └── object_states : List[ObjectState]
                ├── observed : bool
                ├── timestep : int             # 0..109
                ├── position : (x, y)          # metres
                ├── heading  : float           # radians
                └── velocity : (vx, vy)        # m/s
```

---

## 3. `log_map_archive_<id>.json` — local HD vector map

A small *local* crop of the city's HD map around the scenario. Three top-level
keys; load it with `ArgoverseStaticMap.from_json(path)`.

```
{
  "drivable_areas":       { id: { area_boundary: [ {x,y,z}, ... ] } },
  "lane_segments":        { id: LaneSegment, ... },
  "pedestrian_crossings": { id: { edge1: [...], edge2: [...] } }
}
```

### `LaneSegment` — the map is a *lane graph*, not just polylines

Each lane carries both **geometry** and **connectivity**, which is what makes
the map useful for prediction (you can walk the graph):

| Field                                | Meaning                                         |
|--------------------------------------|-------------------------------------------------|
| `id`                                 | unique lane id                                  |
| `lane_type`                          | `VEHICLE`, `BIKE`, `BUS`                         |
| `is_intersection`                    | inside an intersection?                         |
| `left/right_lane_boundary`           | polylines of `(x,y,z)` points                   |
| `left/right_mark_type`               | paint, e.g. `DOUBLE_SOLID_YELLOW`, `DASHED_WHITE`, `NONE` |
| `predecessors` / `successors`        | upstream / downstream lane ids (graph edges)    |
| `left/right_neighbor_id`             | adjacent lanes (lane changes)                   |

The **centerline** is not stored — it is interpolated on demand from the two
boundaries via `static_map.get_lane_segment_centerline(lane_id)`.

---

## Where the data comes from

The dataset is public over HTTPS (no AWS credentials needed). One scenario:

```bash
ID=0015197d-b916-43b6-bcaa-8a7d90d7b87d
BASE=https://argoverse.s3.amazonaws.com/datasets/av2/motion-forecasting/val/$ID
curl -O $BASE/scenario_$ID.parquet
curl -O $BASE/log_map_archive_$ID.json
```

For bulk downloads the official route is `s5cmd` against
`s3://argoverse/datasets/av2/motion-forecasting/`.

## The other AV2 subsets (same `av2` API, heavier data)

- **Sensor** — 1,000 logs with ring cameras + 2 LiDARs + 3D cuboids.
- **LiDAR** — 20,000 unlabeled LiDAR sequences for self-supervision.
- **Map Change (TbV)** — 1,000 logs for detecting real-world map changes.
