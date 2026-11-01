"""OpenDRIVE / CARLA map literacy: topology dump + optional route overlay."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from pipeline.opendrive_sample import SAMPLE_XODR, parse_opendrive_stats


def map_topology_stats(world_map) -> dict:
    """Summarize OpenDRIVE-backed lane topology CARLA exposes."""
    waypoints = world_map.generate_waypoints(2.0)
    road_ids = {wp.road_id for wp in waypoints}
    lane_keys = {(wp.road_id, wp.lane_id) for wp in waypoints}
    junctions = {wp.junction_id for wp in waypoints if wp.is_junction}
    lane_types = Counter(str(wp.lane_type) for wp in waypoints)
    spawn_n = len(world_map.get_spawn_points())
    xodr = world_map.to_opendrive()
    return {
        "map_name": world_map.name,
        "n_waypoints_2m": len(waypoints),
        "n_roads": len(road_ids),
        "n_lanes": len(lane_keys),
        "n_junction_ids": len(junctions),
        "n_spawn_points": spawn_n,
        "lane_types": dict(lane_types),
        "opendrive_bytes": len(xodr.encode("utf-8")),
        "opendrive_has_header": xodr.lstrip().startswith("<?xml")
        or "<OpenDRIVE" in xodr[:200],
    }


def dump_map_offline(*, out: Path | None = None) -> dict:
    """OpenDRIVE schema literacy without a live CARLA server."""
    stats = parse_opendrive_stats(SAMPLE_XODR)
    payload = {
        "mode": "offline",
        "note": (
            "Synthetic OpenDRIVE sample for schema literacy. "
            "Re-run without --offline while CARLA is up for Town01 topology + GRP route."
        ),
        "topology": stats,
        "carla_map_api": {
            "generate_waypoints": "lane center samples from OpenDRIVE",
            "get_spawn_points": "recommended start poses",
            "to_opendrive": "raw .xodr string for the loaded town",
            "get_waypoint": "project a world location onto a lane",
        },
    }
    if out is not None:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        xodr_path = out.with_suffix(".xodr.txt")
        xodr_path.write_text(SAMPLE_XODR, encoding="utf-8")
        print(f"Wrote {out} and {xodr_path}")
    print(json.dumps(payload, indent=2))
    return payload


def dump_map(
    *,
    map_name: str = "Town01",
    out: Path | None = None,
    spawn_a: int = 0,
    spawn_b: int | None = None,
    offline: bool = False,
) -> dict:
    if offline:
        return dump_map_offline(out=out)

    from utils.connection import connect_client
    from utils.routing import build_planner, plan_route
    from utils.run_meta import ensure_map

    client = connect_client(timeout=30.0)
    world = ensure_map(client, map_name)
    world_map = world.get_map()
    stats = map_topology_stats(world_map)
    spawn_pts = world_map.get_spawn_points()
    if spawn_b is None:
        spawn_b = min(len(spawn_pts) - 1, max(1, spawn_a + 20))
    spawn_b = max(0, min(spawn_b, len(spawn_pts) - 1))
    spawn_a = max(0, min(spawn_a, len(spawn_pts) - 1))

    grp = build_planner(world_map)
    origin = spawn_pts[spawn_a].location
    dest = spawn_pts[spawn_b].location
    route = plan_route(grp, origin, dest)
    route_summary = {
        "spawn_a": spawn_a,
        "spawn_b": spawn_b,
        "n_waypoints": len(route),
        "options": dict(Counter(str(int(opt)) for _, opt in route)),
    }
    payload = {"mode": "live", "topology": stats, "sample_route": route_summary}
    if out is not None:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        xodr_path = out.with_suffix(".xodr.txt")
        xodr = world_map.to_opendrive()
        xodr_path.write_text(xodr[:4000], encoding="utf-8")
        print(f"Wrote {out} and {xodr_path}")
    print(json.dumps(payload, indent=2))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", dest="map_name", default="Town01")
    parser.add_argument("--out", type=Path, default=Path("models/pipeline_preview/map_stats.json"))
    parser.add_argument("--spawn-a", type=int, default=0)
    parser.add_argument("--spawn-b", type=int, default=None)
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Parse the bundled OpenDRIVE sample (no CARLA server)",
    )
    args = parser.parse_args()
    dump_map(
        map_name=args.map_name,
        out=args.out,
        spawn_a=args.spawn_a,
        spawn_b=args.spawn_b,
        offline=args.offline,
    )


if __name__ == "__main__":
    main()
