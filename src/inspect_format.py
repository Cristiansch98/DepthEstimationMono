"""Print the on-disk structure of an Argoverse 2 motion-forecasting scenario.

This is the companion to FORMAT.md: run it on any scenario folder to see the
exact parquet schema, the per-track object model, and the HD-map JSON layout.

    python src/inspect_format.py data/motion-forecasting/<scenario_id>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

from av2.datasets.motion_forecasting import scenario_serialization as ss
from av2.map.map_api import ArgoverseStaticMap


def rule(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def main(scenario_dir: Path) -> None:
    sid = scenario_dir.name
    parquet_path = scenario_dir / f"scenario_{sid}.parquet"
    map_path = scenario_dir / f"log_map_archive_{sid}.json"

    rule(f"SCENARIO  {sid}")
    print("Folder contains exactly two files:")
    for p in (parquet_path, map_path):
        print(f"  - {p.name:<55} {p.stat().st_size / 1024:6.1f} KiB")

    # ---- 1. The trajectory parquet ----------------------------------------
    rule("1. scenario_<id>.parquet  ->  long-format trajectory table")
    table = pq.read_table(parquet_path)
    print("Columns (one row per (track, timestep)):\n")
    for field in table.schema:
        print(f"  {field.name:<18} {str(field.type)}")
    df = table.to_pandas()
    print(f"\nrows = {len(df)}  =  {df['track_id'].nunique()} tracks "
          f"x up to {df['timestep'].nunique()} timesteps")
    print(f"sampling: {df['num_timestamps'].iloc[0]} steps @ 10 Hz "
          f"= {(df['num_timestamps'].iloc[0] - 1) / 10:.0f} s "
          f"(first 50 observed, remaining are the prediction horizon)")
    print(f"city = {df['city'].iloc[0]}   focal_track_id = {df['focal_track_id'].iloc[0]}")

    # ---- 2. The object model (via the typed API) --------------------------
    rule("2. Typed object model  (av2 ArgoverseScenario)")
    scenario = ss.load_argoverse_scenario_parquet(parquet_path)
    print("ArgoverseScenario")
    print(f"  .scenario_id      {scenario.scenario_id}")
    print(f"  .city_name        {scenario.city_name}")
    print(f"  .timestamps_ns    int64[{len(scenario.timestamps_ns)}]")
    print(f"  .focal_track_id   {scenario.focal_track_id}")
    print(f"  .tracks           List[Track]  (n={len(scenario.tracks)})")
    print("\n  Track")
    print("    .track_id         str")
    print("    .object_type      ObjectType   {vehicle, pedestrian, cyclist, bus, ...}")
    print("    .category         TrackCategory{TRACK_FRAGMENT, UNSCORED, SCORED, FOCAL}")
    print("    .object_states    List[ObjectState]")
    print("\n  ObjectState (one pose at one timestep)")
    print("    .observed  bool | .timestep int | .position (x,y) | .heading rad | .velocity (vx,vy)")

    focal = next(t for t in scenario.tracks if t.track_id == scenario.focal_track_id)
    s0 = focal.object_states[0]
    print(f"\n  Example - focal track {focal.track_id} ({focal.object_type.value}, "
          f"{focal.category.name}), first state:")
    print(f"    timestep={s0.timestep} observed={s0.observed} "
          f"pos=({s0.position[0]:.1f},{s0.position[1]:.1f}) "
          f"heading={s0.heading:.2f} vel=({s0.velocity[0]:.1f},{s0.velocity[1]:.1f})")

    from collections import Counter
    types = Counter(t.object_type.value for t in scenario.tracks)
    cats = Counter(t.category.name for t in scenario.tracks)
    print("\n  object_type histogram:", dict(types))
    print("  category   histogram:", dict(cats))

    # ---- 3. The HD vector map ---------------------------------------------
    rule("3. log_map_archive_<id>.json  ->  local HD vector map")
    raw = json.load(open(map_path))
    print("Top-level JSON keys:", list(raw.keys()))
    for k, v in raw.items():
        print(f"  {k:<22} {len(v)} entries")

    amap = ArgoverseStaticMap.from_json(map_path)
    print("\nLoaded as ArgoverseStaticMap:")
    print(f"  .vector_lane_segments        {len(amap.vector_lane_segments)} lanes")
    print(f"  .vector_drivable_areas       {len(amap.vector_drivable_areas)} polygons")
    print(f"  .vector_pedestrian_crossings {len(amap.vector_pedestrian_crossings)} crossings")

    lane = next(iter(amap.vector_lane_segments.values()))
    print("\n  LaneSegment fields (graph + geometry):")
    print(f"    id={lane.id}  lane_type={lane.lane_type.value}  is_intersection={lane.is_intersection}")
    print(f"    predecessors={lane.predecessors}  successors={lane.successors}")
    print(f"    left/right_neighbor_id={lane.left_neighbor_id}/{lane.right_neighbor_id}")
    print(f"    left_mark_type={lane.left_mark_type.value}  right_mark_type={lane.right_mark_type.value}")
    print(f"    left_lane_boundary  = polyline[{len(lane.left_lane_boundary.xyz)} x (x,y,z)]")
    print(f"    right_lane_boundary = polyline[{len(lane.right_lane_boundary.xyz)} x (x,y,z)]")
    cl = amap.get_lane_segment_centerline(lane.id)
    print(f"    centerline (derived) = polyline[{len(cl)} x (x,y,z)]")
    print("\nAll coordinates are in the city coordinate frame (metres).")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        target = Path(sys.argv[1])
    else:
        root = Path("data/motion-forecasting")
        target = sorted(p for p in root.iterdir() if p.is_dir())[0]
    main(target)
