"""
Drive planned GlobalRoutePlanner routes with a pure-pursuit waypoint follower.

Draws the planned path as colored world-debug lines (by RoadOption).

Examples:
  python -m expert.waypoint_drive --seconds 60
  python -m expert.waypoint_drive --map Town01 --spawn-index 42 --seconds 90
  python -m expert.waypoint_drive --from-run data/run_20260903_013829
"""
from __future__ import annotations

import argparse
from pathlib import Path

import carla

from utils.connection import connect_client
from utils.routing import (
    build_planner,
    draw_route,
    draw_target,
    needs_replan,
    pick_junction_heavy_destination,
)
from utils.run_meta import apply_run_meta, ensure_map, load_run_meta
from utils.spectator import follow_vehicle
from utils.state import speed_kmh
from utils.sync_mode import SynchronousMode
from utils.vehicles import spawn_vehicle
from utils.waypoint_follower import WaypointFollower


def _cleanup_actors(world: carla.World, actors: list[carla.Actor]) -> None:
    for actor in actors:
        try:
            if actor.is_alive:
                actor.destroy()
        except Exception:
            pass


def waypoint_drive_run(
    *,
    seconds: float = 60.0,
    fps: float = 20.0,
    map_name: str | None = "Town01",
    spawn_index: int | None = None,
    from_run: Path | None = None,
    cruise_kmh: float = 25.0,
    turn_kmh: float = 14.0,
    draw: bool = True,
) -> None:
    client = connect_client(timeout=30.0)
    drive_fps = fps

    if from_run is not None:
        meta = load_run_meta(from_run)
        world = apply_run_meta(client, meta)
        spawn_index = meta["spawn_index"]
        drive_fps = float(meta.get("fps", drive_fps))
        print(
            f"Matched spawn {spawn_index} on "
            f"{meta.get('map_load_name', meta['map'])}"
        )
    elif map_name is not None:
        world = ensure_map(client, map_name)
    else:
        world = client.get_world()

    world = client.get_world()
    world_map = world.get_map()
    spawn_pts = world_map.get_spawn_points()
    if spawn_index is not None:
        if not 0 <= spawn_index < len(spawn_pts):
            raise ValueError(
                f"spawn_index must be 0..{len(spawn_pts) - 1}, got {spawn_index}"
            )
        origin = spawn_pts[spawn_index].location
    else:
        origin = spawn_pts[0].location

    print("Building route planner (async, before sync mode) ...")
    grp = build_planner(world_map)
    dest, route, score = pick_junction_heavy_destination(grp, origin, spawn_pts)
    print(
        f"Route ready: {len(route)} wps, junction score={score:.1%}, "
        f"dest=({dest.x:.1f},{dest.y:.1f})"
    )

    follower = WaypointFollower(cruise_kmh=cruise_kmh, turn_kmh=turn_kmh)
    actors: list[carla.Actor] = []
    line_life = max(1.0 / drive_fps * 2.5, 0.12)

    try:
        with SynchronousMode(client, world, fps=drive_fps) as (world, _tm):
            vehicle = spawn_vehicle(world, spawn_index=spawn_index)
            actors.append(vehicle)
            if spawn_index is not None:
                print(f"Spawned at index {spawn_index}")
            print(
                f"Waypoint drive for {seconds:.0f}s @ {drive_fps:.0f} FPS "
                f"(cruise={cruise_kmh:.0f} turn={turn_kmh:.0f} km/h)"
            )
            print("Ctrl+C to stop.")

            frames = int(seconds * drive_fps)
            for frame_idx in range(frames):
                world.tick()
                follow_vehicle(world, vehicle)

                loc = vehicle.get_location()
                if needs_replan(loc, route, follower.route_index):
                    dest, route, score = pick_junction_heavy_destination(
                        grp, loc, spawn_pts
                    )
                    follower.reset()
                    print(
                        f"  replan @ frame {frame_idx}: "
                        f"{len(route)} wps, junction={score:.1%}"
                    )

                control, target, command = follower.control(vehicle, route)
                vehicle.apply_control(control)

                if draw:
                    draw_route(
                        world,
                        route,
                        life_time=line_life,
                        start_index=max(0, follower.route_index - 2),
                    )
                    draw_target(world, target, life_time=line_life)

                if frame_idx % int(drive_fps) == 0:
                    print(
                        f"  t={frame_idx / drive_fps:4.1f}s  "
                        f"speed={speed_kmh(vehicle):4.1f} km/h  "
                        f"cmd={command:12s}  "
                        f"steer={control.steer:+.2f} "
                        f"thr={control.throttle:.2f} brk={control.brake:.2f}  "
                        f"wp={follower.route_index}/{len(route)}"
                    )
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        _cleanup_actors(world, actors)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument(
        "--map",
        dest="map_name",
        default="Town01",
        help="CARLA map to load (default: Town01; skipped if already on that map)",
    )
    parser.add_argument("--spawn-index", type=int, default=None)
    parser.add_argument(
        "--from-run",
        type=Path,
        default=None,
        help="Use map + spawn from a recorded run's meta.json",
    )
    parser.add_argument("--cruise-kmh", type=float, default=28.0)
    parser.add_argument("--turn-kmh", type=float, default=22.0)
    parser.add_argument(
        "--no-draw",
        action="store_true",
        help="Disable debug route lines / target point",
    )
    args = parser.parse_args()
    waypoint_drive_run(
        seconds=args.seconds,
        fps=args.fps,
        map_name=None if args.from_run else args.map_name,
        spawn_index=args.spawn_index,
        from_run=args.from_run,
        cruise_kmh=args.cruise_kmh,
        turn_kmh=args.turn_kmh,
        draw=not args.no_draw,
    )


if __name__ == "__main__":
    main()
