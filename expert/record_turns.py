"""
Record one short clip per intersection exit.

Places the car a few meters before each Town junction and drives a single
left, right, or straight-through. The command is that maneuver for the whole
clip, including the frames where a wall is already straight ahead. Cruise
between junctions is not saved.

    python -m expert.record_turns --map Town01
    python -m expert.record_turns --map Town01 --only TURN_LEFT,TURN_RIGHT
"""
from __future__ import annotations

import argparse
import csv
import json
import queue
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import carla
from PIL import Image

from expert.model import IMAGE_HEIGHT, IMAGE_WIDTH
from expert.record import _cleanup_actors, _get_camera_images
from utils.balance import (
    CONTROLS_HEADER,
    warn_if_follow_lane_heavy,
    write_balanced_controls,
    write_command_distribution,
)
from utils.commands import (
    COMMAND_GO_STRAIGHT,
    COMMAND_TURN_LEFT,
    COMMAND_TURN_RIGHT,
    JUNCTION_COMMANDS,
)
from utils.connection import connect_client
from utils.images import image_to_array
from utils.lane_metrics import format_lane_metric_fields, lane_metrics_at
from utils.radar import format_radar_m, pop_radar_range, try_attach_radar
from utils.routing import (
    _wrap_yaw_delta_deg,
    build_planner,
    geometric_command_from_route,
    plan_route,
    remaining_route_distance_m,
    route_length_m,
)
from utils.run_meta import ensure_map, find_spawn_index, save_run_meta
from utils.spectator import follow_vehicle
from utils.state import speed_kmh
from utils.sync_mode import SynchronousMode
from utils.traffic import clear_other_vehicles
from utils.traffic_signs import (
    apply_traffic_light_hold,
    maybe_force_ego_light_green,
    near_stop_sign,
    speed_up_traffic_lights,
    traffic_light_fields,
)
from utils.vehicles import attach_rgb_cameras, recovery_steer, spawn_vehicle
from utils.waypoint_follower import WaypointFollower


@dataclass(frozen=True)
class JunctionManeuver:
    junction_id: int
    command: str
    entry: carla.Waypoint
    exit: carla.Waypoint


def maneuver_command(entry_yaw: float, exit_yaw: float) -> str | None:
    """Left, right, or straight from the yaw change through the junction.

    Large reversals are U-turns and are skipped. Sign matches the router:
    positive yaw change is a right turn.
    """
    dyaw = _wrap_yaw_delta_deg(exit_yaw - entry_yaw)
    if abs(dyaw) > 120.0:
        return None
    if dyaw >= 20.0:
        return COMMAND_TURN_RIGHT
    if dyaw <= -20.0:
        return COMMAND_TURN_LEFT
    return COMMAND_GO_STRAIGHT


def _along(waypoint: carla.Waypoint, dist_m: float) -> carla.Waypoint | None:
    """Move along the lane. Negative distance goes backward, toward the approach."""
    if abs(dist_m) < 0.5:
        return waypoint
    try:
        found = (
            waypoint.previous(abs(dist_m))
            if dist_m < 0
            else waypoint.next(abs(dist_m))
        )
    except Exception:
        return None
    if not found:
        return None
    yaw = waypoint.transform.rotation.yaw

    def _delta(other: carla.Waypoint) -> float:
        return abs(_wrap_yaw_delta_deg(other.transform.rotation.yaw - yaw))

    return min(found, key=_delta)


def list_junction_maneuvers(
    world_map: carla.Map,
    *,
    only: set[str] | None = None,
) -> list[JunctionManeuver]:
    """One maneuver per incoming lane and exit. Duplicates are dropped."""
    junctions: dict[int, carla.Junction] = {}
    for waypoint in world_map.generate_waypoints(2.0):
        if not waypoint.is_junction:
            continue
        junction = waypoint.get_junction()
        junctions[junction.id] = junction

    seen: set[tuple[int, int, int, int, int]] = set()
    maneuvers: list[JunctionManeuver] = []
    for junction_id, junction in junctions.items():
        try:
            pairs = junction.get_waypoints(carla.LaneType.Driving)
        except Exception:
            continue
        for entry, exit_wp in pairs:
            key = (
                junction_id,
                int(entry.road_id),
                int(entry.lane_id),
                int(exit_wp.road_id),
                int(exit_wp.lane_id),
            )
            if key in seen:
                continue
            seen.add(key)
            command = maneuver_command(
                entry.transform.rotation.yaw,
                exit_wp.transform.rotation.yaw,
            )
            if command is None:
                continue
            if only is not None and command not in only:
                continue
            maneuvers.append(
                JunctionManeuver(junction_id, command, entry, exit_wp)
            )
    maneuvers.sort(key=lambda m: (m.command, m.junction_id))
    return maneuvers


def _place(vehicle: carla.Vehicle, waypoint: carla.Waypoint) -> None:
    transform = waypoint.transform
    transform.location.z += 0.5
    vehicle.set_simulate_physics(False)
    vehicle.set_transform(transform)
    vehicle.set_simulate_physics(True)
    vehicle.set_target_velocity(carla.Vector3D())
    vehicle.set_target_angular_velocity(carla.Vector3D())
    vehicle.apply_control(carla.VehicleControl(brake=1.0))


def record_turns(
    *,
    map_name: str = "Town01",
    seconds_per_clip: float = 12.0,
    before_m: float = 8.0,
    after_m: float = 14.0,
    fps: float = 20.0,
    cameras: tuple[str, ...] = ("center",),
    only: set[str] | None = None,
    limit: int | None = None,
    out: Path = Path("data"),
    tm_port: int = 8000,
) -> Path:
    client = connect_client(timeout=30.0)
    world = ensure_map(client, map_name)
    world_map = world.get_map()
    maneuvers = list_junction_maneuvers(world_map, only=only)
    if limit is not None:
        maneuvers = maneuvers[: max(0, int(limit))]
    if not maneuvers:
        raise SystemExit("No junction maneuvers found.")
    counts: dict[str, int] = {name: 0 for name in JUNCTION_COMMANDS}
    for maneuver in maneuvers:
        counts[maneuver.command] = counts.get(maneuver.command, 0) + 1
    print(
        f"{len(maneuvers)} junction clips on {world_map.name}: "
        + "  ".join(f"{name}={counts.get(name, 0)}" for name in JUNCTION_COMMANDS)
    )

    print("Building route planner ...")
    grp = build_planner(world_map)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(out) / f"turns_{stamp}"
    image_dirs = {name: run_dir / "images" / name for name in cameras}
    for directory in image_dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    controls_path = run_dir / "controls.csv"

    actors: list[carla.Actor] = []
    cam_queues: dict[str, queue.Queue] = {name: queue.Queue() for name in cameras}
    recorded_rows: list[dict[str, str]] = []
    clip_meta: list[dict] = []
    follower = WaypointFollower(
        cruise_kmh=18.0,
        turn_kmh=12.0,
        min_lookahead_m=3.5,
        max_lookahead_m=8.0,
        steer_gain=1.65,
        max_steer_rate=0.22,
    )
    max_frames = max(1, int(seconds_per_clip * fps))
    frame_idx = 0
    spawn_index = 0

    with SynchronousMode(client, world, fps=fps, tm_port=tm_port) as (
        world,
        _tm,
    ):
        try:
            vehicle = spawn_vehicle(world)
            actors.append(vehicle)
            n_clear = clear_other_vehicles(world, ego=vehicle)
            if n_clear:
                print(f"Cleared {n_clear} leftover vehicle(s)")
            cams = attach_rgb_cameras(
                world,
                vehicle,
                names=cameras,
                width=IMAGE_WIDTH,
                height=IMAGE_HEIGHT,
                stagger_ticks=0,
            )
            for name, camera in cams.items():
                actors.append(camera)
                camera.listen(cam_queues[name].put)
            radar_queue: queue.Queue = queue.Queue()
            radar_on = try_attach_radar(world, vehicle, actors, radar_queue)
            radar_range_m: float | None = None
            n_lights = speed_up_traffic_lights(world, red_s=2.0, yellow_s=0.5, green_s=6.0)
            print(f"Fast lights ON ({n_lights} poles)")
            world.tick()
            _get_camera_images(world, cam_queues, cameras, timeout=10.0)
            stop_actors = list(world.get_actors().filter("traffic.stop"))
            spawn_index = find_spawn_index(vehicle, world)

            with controls_path.open("w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=CONTROLS_HEADER)
                writer.writeheader()

                for clip_i, maneuver in enumerate(maneuvers, start=1):
                    start_wp = _along(maneuver.entry, -before_m)
                    end_wp = _along(maneuver.exit, after_m)
                    if start_wp is None or end_wp is None:
                        print(f"  skip junction {maneuver.junction_id} {maneuver.command}: no approach")
                        continue
                    try:
                        route = plan_route(
                            grp,
                            start_wp.transform.location,
                            end_wp.transform.location,
                        )
                    except Exception as exc:
                        print(f"  skip junction {maneuver.junction_id}: {exc}")
                        continue
                    if len(route) < 3:
                        continue
                    length_m = route_length_m(route)
                    if length_m > before_m + after_m + 30.0:
                        print(
                            f"  skip junction {maneuver.junction_id} "
                            f"{maneuver.command}: detour {length_m:.0f}m"
                        )
                        continue
                    command = geometric_command_from_route(
                        route, lookahead_m=max(length_m, 20.0)
                    )
                    if command not in JUNCTION_COMMANDS:
                        print(
                            f"  skip junction {maneuver.junction_id}: "
                            f"route is {command}"
                        )
                        continue

                    _place(vehicle, start_wp)
                    follower.reset()
                    radar_range_m = None
                    red_wait_frames = 0
                    moved = False
                    finished = False
                    clip_start = frame_idx
                    row_start = len(recorded_rows)
                    print(
                        f"Clip {clip_i}/{len(maneuvers)}  junction {maneuver.junction_id}  "
                        f"{command}  ({length_m:.0f}m)"
                    )
                    for _step in range(max_frames):
                        world.tick()
                        if radar_on:
                            radar_range_m = pop_radar_range(radar_queue, radar_range_m)
                        follow_vehicle(world, vehicle)
                        loc = vehicle.get_location()
                        yaw = vehicle.get_transform().rotation.yaw
                        red_wait_frames, _forced = maybe_force_ego_light_green(
                            vehicle,
                            red_wait_frames=red_wait_frames,
                            max_wait_frames=int(0.4 * fps),
                        )
                        control, _target, _cmd = follower.control(vehicle, route)
                        control = apply_traffic_light_hold(control, vehicle)
                        vehicle.apply_control(control)
                        speed = speed_kmh(vehicle)
                        if speed > 2.0:
                            moved = True
                        at_light, light_state = traffic_light_fields(vehicle)
                        lane = lane_metrics_at(
                            loc, yaw, route, follower.route_index
                        )
                        images = _get_camera_images(
                            world,
                            cam_queues,
                            cameras,
                            timeout=5.0,
                            retries=3,
                            tick_on_retry=False,
                        )
                        for cam_name, image in images.items():
                            rgb = image_to_array(image)
                            Image.fromarray(rgb).save(
                                image_dirs[cam_name] / f"{frame_idx:06d}.png"
                            )
                            steer = recovery_steer(cam_name, float(control.steer))
                            row = {
                                "frame": str(frame_idx),
                                "camera": cam_name,
                                "steer": f"{steer:.6f}",
                                "throttle": f"{control.throttle:.6f}",
                                "brake": f"{control.brake:.6f}",
                                "speed_kmh": f"{speed:.3f}",
                                "command": command,
                                "at_traffic_light": str(at_light),
                                "light_state": light_state,
                                "near_stop_sign": str(
                                    near_stop_sign(
                                        world, vehicle, stop_actors=stop_actors
                                    )
                                ),
                                "radar_m": format_radar_m(radar_range_m),
                                **format_lane_metric_fields(lane),
                            }
                            writer.writerow(row)
                            recorded_rows.append(row)
                        frame_idx += 1
                        remaining = remaining_route_distance_m(
                            loc, route, start_index=follower.route_index
                        )
                        if moved and remaining < 4.0:
                            finished = True
                            break
                    if not finished:
                        del recorded_rows[row_start:]
                        for cam_name in cameras:
                            for fid in range(clip_start, frame_idx):
                                path = image_dirs[cam_name] / f"{fid:06d}.png"
                                path.unlink(missing_ok=True)
                        print(
                            f"  dropped unfinished clip "
                            f"(frames {clip_start}-{frame_idx - 1})"
                        )
                        continue
                    clip_meta.append(
                        {
                            "junction_id": maneuver.junction_id,
                            "command": command,
                            "frame_start": clip_start,
                            "frame_end": frame_idx - 1,
                            "length_m": round(length_m, 1),
                        }
                    )
        finally:
            _cleanup_actors(world, actors, allow_tick=False)

    if not recorded_rows:
        raise SystemExit(f"No clips saved. Empty dir: {run_dir}")

    # Rewrite csv so dropped clips are not left in the file.
    with controls_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CONTROLS_HEADER)
        writer.writeheader()
        writer.writerows(recorded_rows)

    save_run_meta(
        run_dir,
        map_name=world.get_map().name,
        spawn_index=spawn_index,
        fps=fps,
        lane_change=False,
        traffic=0,
    )
    meta_path = run_dir / "meta.json"
    meta = json.loads(meta_path.read_text())
    meta["turn_clips"] = True
    meta["clips"] = len(clip_meta)
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    (run_dir / "maneuvers.json").write_text(json.dumps(clip_meta, indent=2) + "\n")
    warn_if_follow_lane_heavy(recorded_rows)
    write_command_distribution(run_dir, recorded_rows)
    write_balanced_controls(run_dir, recorded_rows)
    done = {name: 0 for name in JUNCTION_COMMANDS}
    for clip in clip_meta:
        done[clip["command"]] = done.get(clip["command"], 0) + 1
    print(
        f"Saved {len(clip_meta)} clips, {len(recorded_rows)} frames → {run_dir}"
    )
    print("  " + "  ".join(f"{name}={done.get(name, 0)}" for name in JUNCTION_COMMANDS))
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", dest="map_name", default="Town01")
    parser.add_argument("--seconds-per-clip", type=float, default=12.0)
    parser.add_argument(
        "--before-m",
        type=float,
        default=8.0,
        help="How far before the junction to start (default 8)",
    )
    parser.add_argument(
        "--after-m",
        type=float,
        default=14.0,
        help="How far past the exit to stop (default 14)",
    )
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--cameras", default="center")
    parser.add_argument(
        "--only",
        default="",
        help="Comma list of TURN_LEFT,TURN_RIGHT,GO_STRAIGHT. Empty = all three.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Stop after N clips")
    parser.add_argument("--out", type=Path, default=Path("data"))
    parser.add_argument("--tm-port", type=int, default=8000)
    args = parser.parse_args()
    only = None
    if args.only.strip():
        only = {part.strip() for part in args.only.split(",") if part.strip()}
        unknown = only - set(JUNCTION_COMMANDS)
        if unknown:
            raise SystemExit(f"Unknown command in --only: {sorted(unknown)}")
    names = tuple(part.strip() for part in args.cameras.split(",") if part.strip())
    record_turns(
        map_name=args.map_name,
        seconds_per_clip=args.seconds_per_clip,
        before_m=args.before_m,
        after_m=args.after_m,
        fps=args.fps,
        cameras=names or ("center",),
        only=only,
        limit=args.limit,
        out=args.out,
        tm_port=args.tm_port,
    )


if __name__ == "__main__":
    main()
