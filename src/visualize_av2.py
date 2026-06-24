"""Cool visualizations for the Argoverse 2 *Motion Forecasting* dataset.

Each AV2 motion-forecasting scenario is just two files:

    <scenario_id>/
        scenario_<scenario_id>.parquet      # agent trajectories (the "what moves")
        log_map_archive_<scenario_id>.json  # local HD vector map (the "where")

This module loads those with the official ``av2`` API and renders:

    1. A dark-themed "hero" figure: HD map + every agent trajectory, with the
       focal agent drawn as a time-coloured trail with oriented vehicle boxes.
    2. The official animated GIF (agents moving over the map through time).

Run:
    python src/visualize_av2.py --data-root data/motion-forecasting --out viz
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")  # headless server -> render to files only
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon as MplPolygon

from av2.datasets.motion_forecasting import scenario_serialization as ss
from av2.datasets.motion_forecasting.data_schema import (
    ArgoverseScenario,
    ObjectType,
    TrackCategory,
)
from av2.map.map_api import ArgoverseStaticMap

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("av2-viz")

# ----------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------
BG = "#0d1117"          # GitHub-dark background
PANEL = "#161b22"
INK = "#e6edf3"
MUTED = "#7d8590"
DRIVABLE = "#171c24"    # subtle road fill
LANE_LINE = "#4c576a"   # lane boundaries / centerlines
CROSSWALK = "#8b7500"   # pedestrian crossings

# Colour per object type (everything that is *not* the focal / AV agent).
TYPE_COLORS: Dict[ObjectType, str] = {
    ObjectType.VEHICLE: "#4b9fea",
    ObjectType.BUS: "#2f6db0",
    ObjectType.PEDESTRIAN: "#f2a35e",
    ObjectType.MOTORCYCLIST: "#e0556b",
    ObjectType.CYCLIST: "#d44fb0",
    ObjectType.RIDERLESS_BICYCLE: "#9b59b6",
}
FOCAL_CMAP = plt.get_cmap("plasma")   # focal trail coloured by time
AV_COLOR = "#37d67a"                  # the ego vehicle (autonomous vehicle)
DEFAULT_OBJ = "#5a6473"


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------
def load_scenario(scenario_dir: Path) -> Tuple[ArgoverseScenario, ArgoverseStaticMap]:
    """Load a scenario (trajectories) and its local HD map from a folder."""
    sid = scenario_dir.name
    scenario_path = scenario_dir / f"scenario_{sid}.parquet"
    map_path = scenario_dir / f"log_map_archive_{sid}.json"
    log.info("Loading scenario %s", sid)
    scenario = ss.load_argoverse_scenario_parquet(scenario_path)
    static_map = ArgoverseStaticMap.from_json(map_path)
    log.info(
        "  tracks=%d  timesteps=%d  city=%s  lanes=%d  crossings=%d",
        len(scenario.tracks),
        len(scenario.timestamps_ns),
        scenario.city_name,
        len(static_map.vector_lane_segments),
        len(static_map.vector_pedestrian_crossings),
    )
    return scenario, static_map


# ----------------------------------------------------------------------------
# Geometry helpers
# ----------------------------------------------------------------------------
def track_xy(track) -> np.ndarray:
    """(N, 2) array of (x, y) positions for a track, ordered by timestep."""
    states = sorted(track.object_states, key=lambda s: s.timestep)
    return np.array([s.position for s in states], dtype=float)


def oriented_box(cx: float, cy: float, heading: float, length: float, width: float) -> np.ndarray:
    """Return the 4 corners of an oriented bounding box centred at (cx, cy)."""
    dx, dy = length / 2.0, width / 2.0
    corners = np.array([[dx, dy], [dx, -dy], [-dx, -dy], [-dx, dy]])
    c, s = np.cos(heading), np.sin(heading)
    rot = np.array([[c, -s], [s, c]])
    return corners @ rot.T + np.array([cx, cy])


# ----------------------------------------------------------------------------
# Map rendering
# ----------------------------------------------------------------------------
def draw_map(ax: plt.Axes, static_map: ArgoverseStaticMap) -> None:
    """Draw drivable areas, lane boundaries/centerlines and crosswalks."""
    # Drivable areas: filled background polygons.
    for da in static_map.vector_drivable_areas.values():
        xyz = da.xyz
        ax.add_patch(
            MplPolygon(xyz[:, :2], closed=True, facecolor=DRIVABLE, edgecolor="none", zorder=0)
        )

    # Lane segments: boundaries (solid) + centerlines (dashed).
    for lane in static_map.vector_lane_segments.values():
        for boundary in (lane.left_lane_boundary, lane.right_lane_boundary):
            xyz = boundary.xyz
            ax.plot(xyz[:, 0], xyz[:, 1], color=LANE_LINE, lw=0.8, zorder=1)
        cl = static_map.get_lane_segment_centerline(lane.id)
        ax.plot(cl[:, 0], cl[:, 1], color=LANE_LINE, lw=0.5, ls=(0, (4, 4)), alpha=0.6, zorder=1)

    # Pedestrian crossings.
    for pc in static_map.vector_pedestrian_crossings.values():
        poly = pc.polygon
        ax.add_patch(
            MplPolygon(
                poly[:, :2], closed=True, facecolor=CROSSWALK, edgecolor=CROSSWALK,
                alpha=0.18, lw=1.0, zorder=2,
            )
        )


# ----------------------------------------------------------------------------
# Hero figure
# ----------------------------------------------------------------------------
def render_hero(scenario: ArgoverseScenario, static_map: ArgoverseStaticMap, out_path: Path) -> None:
    """Render a dark-themed overview of the whole scenario to ``out_path``."""
    fig, ax = plt.subplots(figsize=(14, 14), dpi=130)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    draw_map(ax, static_map)

    focal_id = scenario.focal_track_id
    n_ts = len(scenario.timestamps_ns)
    counts: Dict[str, int] = {}
    all_xy: List[np.ndarray] = []

    # --- context agents -----------------------------------------------------
    for track in scenario.tracks:
        xy = track_xy(track)
        if len(xy) == 0:
            continue
        all_xy.append(xy)
        counts[track.object_type.value] = counts.get(track.object_type.value, 0) + 1

        if track.track_id == focal_id or track.track_id == "AV":
            continue  # drawn specially below

        color = TYPE_COLORS.get(track.object_type, DEFAULT_OBJ)
        ax.plot(xy[:, 0], xy[:, 1], color=color, lw=1.4, alpha=0.55, zorder=4)
        ax.scatter(*xy[-1], color=color, s=14, alpha=0.9, zorder=5,
                   edgecolors=BG, linewidths=0.4)

    # --- ego / autonomous vehicle ------------------------------------------
    av = next((t for t in scenario.tracks if t.track_id == "AV"), None)
    if av is not None:
        xy = track_xy(av)
        ax.plot(xy[:, 0], xy[:, 1], color=AV_COLOR, lw=2.2, alpha=0.9, zorder=6)
        ax.scatter(*xy[-1], marker="*", s=320, color=AV_COLOR, edgecolors=BG,
                   linewidths=1.0, zorder=7)

    # --- focal agent: time-coloured trail + oriented boxes ------------------
    focal = next((t for t in scenario.tracks if t.track_id == focal_id), None)
    if focal is not None:
        states = sorted(focal.object_states, key=lambda s: s.timestep)
        xy = np.array([s.position for s in states])
        # Gradient line coloured by normalised time.
        pts = xy.reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        t = np.array([s.timestep for s in states]) / max(n_ts - 1, 1)
        lc = LineCollection(segs, cmap=FOCAL_CMAP, zorder=8, linewidths=4.0)
        lc.set_array(t[:-1])
        ax.add_collection(lc)
        # Oriented vehicle footprints sampled along the trajectory.
        for s in states[:: max(len(states) // 12, 1)]:
            box = oriented_box(s.position[0], s.position[1], s.heading, 4.6, 2.0)
            col = FOCAL_CMAP(s.timestep / max(n_ts - 1, 1))
            ax.add_patch(MplPolygon(box, closed=True, facecolor=col, edgecolor=INK,
                                    lw=0.6, alpha=0.85, zorder=9))
        ax.scatter(*xy[0], marker="o", s=120, facecolor="none", edgecolors=INK,
                   linewidths=1.6, zorder=10)
        ax.scatter(*xy[-1], marker="X", s=200, color=INK, zorder=10)

    # --- framing ------------------------------------------------------------
    # Frame on the focal agent (plus ego) so the action fills the view, with a
    # generous margin for context. Fall back to all agents if no focal track.
    frame_xy = [xy for xy in (
        track_xy(focal) if focal is not None else None,
        track_xy(av) if av is not None else None,
    ) if xy is not None and len(xy)]
    pts = np.concatenate(frame_xy if frame_xy else all_xy, axis=0)
    cx, cy = (pts.max(0) + pts.min(0)) / 2
    half = max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1])) / 2 + 45
    half = max(half, 60.0)  # keep a sensible minimum window
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_aspect("equal")
    ax.axis("off")

    # --- titles & annotations ----------------------------------------------
    fig.text(0.5, 0.95, "ARGOVERSE 2  ·  Motion Forecasting Scenario",
             ha="center", color=INK, fontsize=22, fontweight="bold")
    fig.text(0.5, 0.925,
             f"{scenario.city_name.title()}   |   {len(scenario.tracks)} agents   "
             f"|   {n_ts} timesteps @ 10 Hz  ({(n_ts - 1) / 10:.0f} s)   "
             f"|   id {scenario.scenario_id[:8]}",
             ha="center", color=MUTED, fontsize=12)

    legend_items = [
        Line2D([0], [0], color=FOCAL_CMAP(0.6), lw=4, label="Focal agent (time-coloured trail)"),
        Line2D([0], [0], color=AV_COLOR, lw=2.5, label="Autonomous vehicle (ego)"),
        Line2D([0], [0], color=TYPE_COLORS[ObjectType.VEHICLE], lw=2, label="Vehicles"),
        Line2D([0], [0], color=TYPE_COLORS[ObjectType.PEDESTRIAN], lw=2, label="Pedestrians"),
        Line2D([0], [0], color=TYPE_COLORS[ObjectType.CYCLIST], lw=2, label="Cyclists / bikes"),
        Line2D([0], [0], color=LANE_LINE, lw=1.5, label="HD map lanes"),
    ]
    leg = ax.legend(handles=legend_items, loc="upper left", frameon=True,
                    facecolor=PANEL, edgecolor=LANE_LINE, labelcolor=INK, fontsize=10)
    leg.get_frame().set_alpha(0.9)

    # Composition breakdown box.
    top = sorted(counts.items(), key=lambda kv: -kv[1])[:6]
    txt = "Agent composition\n" + "\n".join(f"  {k:<18}{v:>3}" for k, v in top)
    ax.text(0.985, 0.015, txt, transform=ax.transAxes, ha="right", va="bottom",
            color=MUTED, fontsize=9, family="monospace",
            bbox=dict(boxstyle="round", facecolor=PANEL, edgecolor=LANE_LINE, alpha=0.9))

    fig.savefig(out_path, facecolor=BG, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    log.info("Saved hero figure -> %s", out_path)


# ----------------------------------------------------------------------------
# Animation
# ----------------------------------------------------------------------------
def render_animation(
    scenario: ArgoverseScenario,
    static_map: ArgoverseStaticMap,
    out_path: Path,
    fps: int = 10,
    stride: int = 2,
) -> None:
    """Animate all agents moving over the HD map and save an animated GIF.

    Self-contained replacement for the (matplotlib-incompatible) bundled viz.
    Each agent is an oriented box at its current pose; the focal agent keeps a
    fading tail. ``stride`` decimates timesteps to keep the GIF light.
    """
    from matplotlib.animation import FuncAnimation, PillowWriter

    n_ts = len(scenario.timestamps_ns)
    frames = list(range(0, n_ts, stride))
    focal_id = scenario.focal_track_id

    # Pre-index every track's state by timestep for O(1) frame lookups.
    indexed: List[dict] = []
    for tr in scenario.tracks:
        states = {s.timestep: s for s in tr.object_states}
        indexed.append({"track": tr, "states": states})

    fig, ax = plt.subplots(figsize=(11, 11), dpi=110)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    draw_map(ax, static_map)

    # Fixed frame on the focal agent.
    focal = next((t for t in scenario.tracks if t.track_id == focal_id), None)
    fxy = track_xy(focal) if focal is not None else np.concatenate(
        [track_xy(t) for t in scenario.tracks if len(t.object_states)], axis=0
    )
    cx, cy = (fxy.max(0) + fxy.min(0)) / 2
    half = max(np.ptp(fxy[:, 0]), np.ptp(fxy[:, 1])) / 2 + 45
    half = max(half, 55.0)
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_aspect("equal")
    ax.axis("off")

    fig.text(0.5, 0.955, f"Argoverse 2 · {scenario.city_name.title()} · {scenario.scenario_id[:8]}",
             ha="center", color=INK, fontsize=15, fontweight="bold")
    time_txt = fig.text(0.5, 0.925, "", ha="center", color=MUTED, fontsize=11)

    dynamic: List = []  # artists cleared every frame

    def draw_frame(ts: int):
        for art in dynamic:
            art.remove()
        dynamic.clear()

        for entry in indexed:
            tr = entry["track"]
            st = entry["states"].get(ts)
            if st is None:
                continue
            is_focal = tr.track_id == focal_id
            is_av = tr.track_id == "AV"
            if is_focal:
                color = FOCAL_CMAP(0.6)
                length, width = 4.6, 2.0
            elif is_av:
                color = AV_COLOR
                length, width = 4.6, 2.0
            else:
                color = TYPE_COLORS.get(tr.object_type, DEFAULT_OBJ)
                length, width = (4.4, 1.9) if tr.object_type == ObjectType.VEHICLE else (1.2, 1.2)
            box = oriented_box(st.position[0], st.position[1], st.heading, length, width)
            patch = MplPolygon(box, closed=True, facecolor=color,
                               edgecolor=INK if (is_focal or is_av) else "none",
                               lw=0.6, alpha=0.95, zorder=8 + int(is_focal))
            ax.add_patch(patch)
            dynamic.append(patch)

            # Fading tail for the focal agent.
            if is_focal:
                tail = [entry["states"][t].position for t in range(max(0, ts - 30), ts + 1)
                        if t in entry["states"]]
                if len(tail) > 1:
                    tail = np.array(tail)
                    (ln,) = ax.plot(tail[:, 0], tail[:, 1], color=FOCAL_CMAP(0.85),
                                    lw=3.0, alpha=0.8, zorder=7)
                    dynamic.append(ln)

        phase = "observed" if ts < 50 else "prediction horizon"
        time_txt.set_text(f"t = {ts / 10:4.1f} s   (frame {ts:3d}/{n_ts - 1})   ·   {phase}")
        return dynamic

    log.info("Animating %d frames (stride=%d) -> %s", len(frames), stride, out_path)
    anim = FuncAnimation(fig, draw_frame, frames=frames, interval=1000 / fps, blit=False)
    anim.save(out_path, writer=PillowWriter(fps=fps), dpi=110)
    plt.close(fig)
    log.info("Saved animation -> %s", out_path)


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------
def pick_scenarios(data_root: Path, limit: Optional[int]) -> List[Path]:
    dirs = sorted(p for p in data_root.iterdir() if p.is_dir())
    return dirs if limit is None else dirs[:limit]


def main() -> None:
    ap = argparse.ArgumentParser(description="Visualize Argoverse 2 motion-forecasting scenarios.")
    ap.add_argument("--data-root", type=Path, default=Path("data/motion-forecasting"))
    ap.add_argument("--out", type=Path, default=Path("viz"))
    ap.add_argument("--limit", type=int, default=None, help="max scenarios to render")
    ap.add_argument("--gif", action="store_true", help="also render the official animated GIF")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    scenarios = pick_scenarios(args.data_root, args.limit)
    log.info("Found %d scenario(s) under %s", len(scenarios), args.data_root)

    for scenario_dir in scenarios:
        scenario, static_map = load_scenario(scenario_dir)
        sid = scenario.scenario_id
        render_hero(scenario, static_map, args.out / f"{sid[:8]}_hero.png")
        if args.gif:
            render_animation(scenario, static_map, args.out / f"{sid[:8]}_anim.gif")

    log.info("Done. Outputs in %s", args.out.resolve())


if __name__ == "__main__":
    main()
