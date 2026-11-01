"""
Drive with a trained behavioral cloning model.

The network owns steer, throttle, and brake for the whole drive. The route
and the light are inputs to that network, not a second controller.

Examples:
  python -m expert.drive --model models/bc.pt --seconds 30
  python -m expert.drive --from-run data/run_20260831_120000
  python -m expert.drive --from-run data/run_... --steer-smooth 0.5
"""
from __future__ import annotations

import argparse
import csv
import json
import queue
from collections import deque
from datetime import datetime
from pathlib import Path

import carla
import numpy as np
import torch

from expert.model import IMAGE_HEIGHT, IMAGE_WIDTH, exclusive_controls, load_checkpoint
from utils.commands import (
    COMMAND_FOLLOW_LANE,
    COMMAND_TURN_LEFT,
    COMMAND_TURN_RIGHT,
    command_to_onehot,
)
from utils.connection import connect_client
from utils.device import describe_device, resolve_device
from utils.drive_score import (
    EventCounter,
    collisions_per_km,
    path_length_m,
    route_completion_pct,
)
from utils.images import image_to_array
from utils.lane_metrics import format_lane_metric_fields, lane_metrics_at
from utils.lights import light_to_onehot, normalize_light_state
from utils.routing import (
    build_planner,
    draw_route,
    draw_target,
    draw_destination,
    hold_turn_command,
    lookahead_target,
    farthest_spawn_index,
    needs_replan,
    pick_junction_heavy_destination,
    pick_next_destination,
    plan_spawn_to_spawn,
)
from utils.radar import (
    encode_radar,
    format_radar_m,
    format_radar_status,
    pop_radar_range,
    try_attach_radar,
)
from utils.run_meta import apply_run_meta, ensure_map, find_spawn_index, load_run_meta
from utils.spectator import follow_vehicle
from utils.state import normalize_speed, speed_kmh
from utils.sync_mode import SynchronousMode
from utils.traffic import (
    _safe_destroy,
    clear_other_vehicles,
    destroy_wrecked_vehicles,
    prune_actor_list,
    spawn_npc_traffic,
)
from utils.traffic_signs import traffic_light_fields
from utils.vehicles import attach_rgb_camera, spawn_vehicle

_LANE_KEEP_COMMANDS = (COMMAND_FOLLOW_LANE, "GO_STRAIGHT")
_TURN_COMMANDS = (COMMAND_TURN_LEFT, COMMAND_TURN_RIGHT)
# Matches rl.env.ROUTE_DONE_PCT so --fixed-route scorecards line up with RL.
FIXED_ROUTE_DONE_PCT = 95.0

_DRIVE_LOG_HEADER = [
    "frame",
    "steer",
    "throttle",
    "brake",
    "speed_kmh",
    "command",
    "cte_m",
    "lane_offset_frac",
    "heading_err_deg",
    "lane_invasions",
    "collisions",
    "radar_m",
]


def _cleanup_actors(
    world: carla.World,
    actors: list[carla.Actor],
    *,
    allow_tick: bool = True,
) -> None:
    del world, allow_tick
    sensors = [a for a in actors if isinstance(a, carla.Sensor)]
    others = [a for a in actors if not isinstance(a, carla.Sensor)]
    # Do not call is_alive / is_listening. On a dead handle those abort the
    # process instead of raising a Python exception.
    for sensor in sensors:
        try:
            sensor.stop()
        except Exception:
            pass
    for sensor in sensors:
        try:
            sensor.destroy()
        except Exception:
            pass
    for actor in others:
        _safe_destroy(actor)


def _image_for_frame(
    q: queue.Queue[carla.Image], frame: int, *, timeout: float = 2.0
) -> carla.Image:
    """Return the camera image for this sim frame, discarding stale backlog."""
    while True:
        image = q.get(timeout=timeout)
        if image.frame >= frame:
            return image


def smooth_steer(prev: float, value: float, alpha: float) -> float:
    """Adaptive EMA: damp small twitch, pass through turn intent."""
    alpha = max(0.0, min(1.0, alpha))
    if alpha <= 0.0:
        return value
    delta = abs(value - prev)
    turn_factor = min(1.0, delta / 0.25)
    mag_factor = min(1.0, abs(value) / 0.4)
    alpha_eff = alpha * (1.0 - 0.85 * turn_factor) * (1.0 - 0.5 * mag_factor)
    return alpha_eff * prev + (1.0 - alpha_eff) * value


def _rate_limit_steer(prev: float, value: float, max_delta: float) -> float:
    if max_delta <= 0.0:
        return value
    delta = value - prev
    if delta > max_delta:
        return prev + max_delta
    if delta < -max_delta:
        return prev - max_delta
    return value


def _speed_steer_limit(speed_km: float, max_steer: float) -> float:
    """
    Cap |steer| by speed so the net can't full-lock at cruise into a curve.

    ~full authority below 12 km/h; ~45% of max_steer by 30 km/h.
    """
    if speed_km <= 12.0:
        return max_steer
    if speed_km >= 30.0:
        return max_steer * 0.45
    t = (speed_km - 12.0) / 18.0
    return max_steer * (1.0 - 0.55 * t)


def _limit_steer(
    steer: float,
    *,
    command: str,
    max_steer: float,
    follow_scale: float,
    speed_km: float,
) -> float:
    limit = _speed_steer_limit(speed_km, max_steer)
    if command in _LANE_KEEP_COMMANDS:
        limit = min(limit, max_steer * follow_scale)
    return max(-limit, min(limit, steer))


def drive_run(
    *,
    model_path: Path = Path("models/bc.pt"),
    seconds: float = 30.0,
    fps: float = 20.0,
    from_run: Path | None = None,
    map_name: str | None = None,
    spawn_index: int | None = None,
    device_name: str = "cuda",
    steer_smooth: float = 0.55,
    max_steer: float = 0.55,
    follow_steer_scale: float = 0.25,
    steer_rate: float = 0.04,
    draw_path: bool = True,
    raw_steer: bool = False,
    turn_speed_kmh: float = 12.0,
    log_dir: Path | None = None,
    traffic: int | None = None,
    tm_port: int = 8000,
    max_collisions_per_km: float = 1.0,
    fixed_route: bool = False,
) -> Path | None:
    device = resolve_device(device_name)
    print(f"Inference device: {describe_device(device)}")
    model = load_checkpoint(model_path, device)
    use_command = bool(model.use_command)
    if fixed_route and not use_command:
        raise SystemExit("--fixed-route needs a command-conditioned model (route progress comes from the command tracker)")
    use_light = bool(getattr(model, "use_light", False))
    use_radar = bool(getattr(model, "use_radar", False))
    branched = bool(getattr(model, "branched", False))
    stack_frames = int(getattr(model, "stack_frames", 1))
    if use_command:
        kind = "branched CIL" if branched else "concat command"
        print(f"Command-conditioned model ({model.n_commands}-way, {kind})")
        if not branched:
            print(
                "  note: this checkpoint concatenates command weakly — "
                "retrain for branched heads so FOLLOW_LANE ≠ turn yank"
            )
    else:
        print("Legacy speed-only model (no command input)")
    if use_light:
        print(
            f"Light-conditioned model ({model.n_lights}-way GT light state; "
            "sim oracle — replace with a detector later)"
        )
    if use_radar:
        print("Radar model: range feeds throttle/brake only, not steer")
    else:
        print(
            "Radar will be logged. This checkpoint does not use it yet "
            "(retrain with --radar)"
        )
    if stack_frames > 1:
        print(
            f"Frame stack: {stack_frames} "
            f"({3 * stack_frames} input channels)"
        )

    client = connect_client()
    drive_fps = fps

    if from_run is not None:
        meta = load_run_meta(from_run)
        world = apply_run_meta(client, meta)
        spawn_index = meta["spawn_index"]
        drive_fps = meta.get("fps", drive_fps)
        if traffic is None:
            traffic = 15
        print(
            f"Matched eval: spawn {spawn_index} on "
            f"{meta.get('map_load_name', meta['map'])}"
        )
    else:
        if map_name is not None:
            world = ensure_map(client, map_name)
        else:
            world = client.get_world()
        if traffic is None:
            traffic = 15

    world = client.get_world()
    world_map = world.get_map()
    spawn_pts = world_map.get_spawn_points()
    route: list = []
    grp = None
    route_index = 0
    if use_command:
        print("Building route planner for command labels only (not path control) ...")
        grp = build_planner(world_map)

    if log_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = Path("data") / f"drive_log_{stamp}"
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    drive_log_path = log_dir / "drive_log.csv"
    print(f"Lane metrics log: {drive_log_path}")

    actors: list[carla.Actor] = []
    image_queue: queue.Queue[carla.Image] = queue.Queue()
    prev_steer = 0.0
    sticky_turn: str | None = None
    sticky_turn_start: int | None = None
    drive_rows: list[dict[str, str]] = []
    route_pts: list[tuple[float, float, float]] = []
    planned_m = 0.0
    credit_m = 0.0
    max_route_index = 0
    route_done = False
    odom_m = 0.0
    prev_odom: carla.Location | None = None
    events = EventCounter(collision_gap_frames=max(1, int(fps)))

    with SynchronousMode(client, world, fps=drive_fps, tm_port=tm_port) as (
        world,
        tm,
    ):
        try:
            vehicle = spawn_vehicle(world, spawn_index=spawn_index)
            camera = attach_rgb_camera(
                world, vehicle, width=IMAGE_WIDTH, height=IMAGE_HEIGHT
            )
            actors.extend([vehicle, camera])
            camera.listen(image_queue.put)
            radar_queue: queue.Queue = queue.Queue()
            radar_on = try_attach_radar(world, vehicle, actors, radar_queue)
            radar_range_m: float | None = None
            _attach_score_sensors(world, vehicle, events, actors)

            if spawn_index is None:
                spawn_index = find_spawn_index(vehicle, world)
            print(f"Spawned at index {spawn_index}")

            traffic_n = max(0, int(traffic or 0))
            npcs: list = []
            if traffic_n > 0:
                exclude = {spawn_index}
                npcs = spawn_npc_traffic(
                    world,
                    tm,
                    count=traffic_n,
                    tm_port=tm_port,
                    ego=vehicle,
                    exclude_spawn_indices=exclude,
                )
                actors.extend(npcs)
                print(f"Traffic ON: spawned {len(npcs)}/{traffic_n} NPC vehicles")
                print(
                    "Model owns steer, throttle, and brake "
                    "(image, speed, command, and light are inputs)"
                )
                for _ in range(max(1, int(drive_fps))):
                    world.tick()
            else:
                n_clear = clear_other_vehicles(world, ego=vehicle)
                if n_clear:
                    print(f"Cleared {n_clear} leftover vehicle(s)")
                    world.tick()

            if fixed_route and grp is not None:
                # Same route as rl.env.CarlaLaneKeepEnv._build_route: spawn ->
                # farthest spawn, no replans, done at FIXED_ROUTE_DONE_PCT.
                dest_index = farthest_spawn_index(spawn_pts, spawn_index)
                try:
                    _dest, route, _score = plan_spawn_to_spawn(
                        grp, spawn_pts, spawn_index, dest_index
                    )
                except (ValueError, RuntimeError):
                    _dest, route, _score = pick_junction_heavy_destination(
                        grp, spawn_pts[spawn_index].location, spawn_pts
                    )
                route_pts = _route_xyz(route)
                planned_m = path_length_m(route_pts, 0, len(route_pts) - 1)
                print(
                    f"Fixed route (same as RL env): spawn {spawn_index} -> "
                    f"spawn {dest_index}, {len(route)} wps, {planned_m:.0f} m"
                )
            elif use_command and grp is not None:
                spawned = vehicle.get_transform()
                dest, route, score, stats = pick_next_destination(
                    grp,
                    spawned.location,
                    spawn_pts,
                    route_mode="balanced",
                    heading_yaw=spawned.rotation.yaw,
                    world_map=world_map,
                )
                print(
                    f"Command route from this spawn: →{stats.get('dest')}  "
                    f"{len(route)} wps  L={stats.get('left')} R={stats.get('right')}  "
                    f"align={stats.get('forward_align', 0):+.2f}  "
                    f"len={stats.get('length_m', 0):.0f}m"
                )
                route_pts = _route_xyz(route)
                planned_m = path_length_m(route_pts, 0, len(route_pts) - 1)
            events.collision_gap_frames = max(1, int(drive_fps))
            print(f"Driving with {model_path} for {seconds:.0f}s (model controls) ...")
            print(
                f"Scorecard gate: <{max_collisions_per_km:.2f} collisions/km "
                "on this route"
            )
            if raw_steer:
                print("Steer stabilize OFF (--raw-steer): no smooth/rate/FOLLOW clamps")
            else:
                print(
                    f"Steer smooth={steer_smooth:.2f} rate≤{steer_rate:.3f}/frame  "
                    f"FOLLOW_LANE max≈{max_steer * follow_steer_scale:.2f}  "
                    f"+ speed-scaled turn cap"
                )
            print(
                "Command is next-maneuver intent (held through the exit), "
                "not a steer-now trigger"
            )
            if draw_path and route:
                print(
                    "Path overlay ON (HUD only — not in RGB cameras); "
                    "--no-draw-path to hide"
                )
            line_life = max(1.0 / drive_fps * 2.5, 0.12)
            frames = int(seconds * drive_fps)
            frame_buf: deque[np.ndarray] = deque(maxlen=stack_frames)
            for frame_idx in range(frames):
                sim_frame = world.tick()
                if radar_on:
                    radar_range_m = pop_radar_range(radar_queue, radar_range_m)
                follow_vehicle(world, vehicle)
                image = _image_for_frame(image_queue, sim_frame)
                rgb = image_to_array(image).astype(np.float32) / 255.0
                frame_buf.append(rgb)
                while len(frame_buf) < stack_frames:
                    frame_buf.appendleft(rgb)
                stacked = np.concatenate(list(frame_buf), axis=2)
                image_tensor = (
                    torch.from_numpy(stacked)
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                    .to(device)
                )
                speed = normalize_speed(speed_kmh(vehicle))
                speed_tensor = torch.tensor(
                    [[speed]], dtype=torch.float32, device=device
                )
                speed_km = speed * 120.0

                command_name = COMMAND_FOLLOW_LANE
                command_tensor = None
                loc = vehicle.get_location()
                yaw = vehicle.get_transform().rotation.yaw
                if use_command and grp is not None and route:
                    if not fixed_route and needs_replan(loc, route, route_index):
                        yaw = vehicle.get_transform().rotation.yaw
                        dest, route_new, _score, stats = pick_next_destination(
                            grp,
                            loc,
                            spawn_pts,
                            route_mode="balanced",
                            heading_yaw=yaw,
                            world_map=world_map,
                        )
                        align = float(stats.get("forward_align", 0.0))
                        if align < 0.15:
                            print(
                                f"  skip reverse replan @ frame {frame_idx}: "
                                f"→{stats.get('dest')} align={align:+.2f}"
                            )
                        else:
                            if route_pts:
                                credit_m += path_length_m(
                                    route_pts, 0, max_route_index
                                )
                            route = route_new
                            route_pts = _route_xyz(route)
                            planned_m += path_length_m(
                                route_pts, 0, len(route_pts) - 1
                            )
                            route_index = 0
                            max_route_index = 0
                            sticky_turn = None
                            sticky_turn_start = None
                            print(
                                f"  replan @ frame {frame_idx}: "
                                f"→{stats.get('dest')} "
                                f"align={align:+.2f} "
                                f"L={stats.get('left')} R={stats.get('right')}"
                            )
                    (
                        command_name,
                        route_index,
                        sticky_turn,
                        sticky_turn_start,
                    ) = hold_turn_command(
                        loc,
                        route,
                        start_hint=route_index,
                        sticky_command=sticky_turn,
                        sticky_start_index=sticky_turn_start,
                        vehicle_yaw=yaw,
                    )
                    command_tensor = torch.tensor(
                        [command_to_onehot(command_name)],
                        dtype=torch.float32,
                        device=device,
                    )

                light_name = "Off"
                light_tensor = None
                if use_light:
                    try:
                        _at, raw_light = traffic_light_fields(vehicle)
                    except Exception:
                        raw_light = "Off"
                    light_name = normalize_light_state(raw_light)
                    light_tensor = torch.tensor(
                        [light_to_onehot(light_name)],
                        dtype=torch.float32,
                        device=device,
                    )

                radar_tensor = None
                if use_radar:
                    range_norm, detected = encode_radar(radar_range_m)
                    radar_tensor = torch.tensor(
                        [[range_norm, detected]],
                        dtype=torch.float32,
                        device=device,
                    )

                with torch.no_grad():
                    if use_command and use_light:
                        out = model(
                            image_tensor,
                            speed_tensor,
                            command_tensor,
                            light_tensor,
                            radar_tensor,
                        )
                    elif use_command:
                        out = model(
                            image_tensor, speed_tensor, command_tensor, None, radar_tensor
                        )
                    elif use_light:
                        out = model(
                            image_tensor, speed_tensor, None, light_tensor, radar_tensor
                        )
                    else:
                        out = model(image_tensor, speed_tensor, None, None, radar_tensor)
                    steer, throttle, brake = out[0].tolist()
                steer, throttle, brake = exclusive_controls(steer, throttle, brake)

                if not raw_steer:
                    # Soft safety only — model still chooses the action.
                    steer = _limit_steer(
                        steer,
                        command=command_name,
                        max_steer=max_steer,
                        follow_scale=follow_steer_scale,
                        speed_km=speed_km,
                    )
                    steer = _rate_limit_steer(prev_steer, steer, steer_rate)
                    steer = smooth_steer(prev_steer, steer, steer_smooth)
                    steer = _limit_steer(
                        steer,
                        command=command_name,
                        max_steer=max_steer,
                        follow_scale=follow_steer_scale,
                        speed_km=speed_km,
                    )
                prev_steer = steer

                control = carla.VehicleControl(
                    steer=float(steer),
                    throttle=float(throttle),
                    brake=float(brake),
                )
                vehicle.apply_control(control)

                if route_pts:
                    max_route_index = max(max_route_index, route_index)
                    if fixed_route and route_completion_pct(
                        path_length_m(route_pts, 0, max_route_index), planned_m
                    ) >= FIXED_ROUTE_DONE_PCT:
                        route_done = True
                if prev_odom is not None:
                    step_m = prev_odom.distance(loc)
                    if step_m < 8.0:
                        odom_m += step_m
                prev_odom = loc
                n_col, n_inv = events.snapshot()

                lane = lane_metrics_at(loc, yaw, route, route_index)
                lane_fields = format_lane_metric_fields(lane)
                drive_rows.append(
                    {
                        "frame": str(frame_idx),
                        "steer": f"{steer:.6f}",
                        "throttle": f"{throttle:.6f}",
                        "brake": f"{brake:.6f}",
                        "speed_kmh": f"{speed_km:.3f}",
                        "command": command_name,
                        **lane_fields,
                        "lane_invasions": str(n_inv),
                        "collisions": str(n_col),
                        "radar_m": format_radar_m(radar_range_m),
                    }
                )

                if draw_path and route:
                    target, _ = lookahead_target(
                        loc, route, start_hint=route_index
                    )
                    draw_route(
                        world,
                        route,
                        life_time=line_life,
                        start_index=max(0, route_index - 2),
                    )
                    draw_target(world, target, life_time=line_life)
                    dest_loc = route[-1][0].transform.location
                    draw_destination(world, dest_loc, life_time=line_life)

                if frame_idx > 0 and frame_idx % int(drive_fps) == 0:
                    if npcs:
                        npcs, n_wreck = destroy_wrecked_vehicles(
                            world, ego=vehicle, tracked=npcs
                        )
                        if n_wreck:
                            prune_actor_list(actors, ego=vehicle, npcs=npcs)
                            print(f"  removed {n_wreck} wrecked NPC(s)")
                    cmd_bit = f"  cmd={command_name}" if use_command else ""
                    light_bit = f"  light={light_name}" if use_light else ""
                    radar_bit = f"  radar={format_radar_status(radar_range_m)}"
                    print(
                        f"  t={frame_idx / drive_fps:4.1f}s  "
                        f"speed={speed_km:.0f} km/h  "
                        f"steer={steer:+.2f} throttle={throttle:.2f} "
                        f"brake={brake:.2f}  "
                        f"|cte|={abs(lane['cte_m']):.2f}m  "
                        f"inv={n_inv} col={n_col}{cmd_bit}{light_bit}{radar_bit}"
                    )
                elif frame_idx % int(drive_fps) == 0:
                    cmd_bit = f"  cmd={command_name}" if use_command else ""
                    light_bit = f"  light={light_name}" if use_light else ""
                    radar_bit = f"  radar={format_radar_status(radar_range_m)}"
                    print(
                        f"  t={frame_idx / drive_fps:4.1f}s  "
                        f"speed={speed_km:.0f} km/h  "
                        f"steer={steer:+.2f} throttle={throttle:.2f} "
                        f"brake={brake:.2f}  "
                        f"|cte|={abs(lane['cte_m']):.2f}m  "
                        f"inv={n_inv} col={n_col}{cmd_bit}{light_bit}{radar_bit}"
                    )
                if route_done:
                    print(f"Route done ({FIXED_ROUTE_DONE_PCT:.0f}%) at t={frame_idx / drive_fps:.1f}s")
                    break
        except KeyboardInterrupt:
            print("\nStopped.")
        finally:
            _finish_drive_log(
                log_dir,
                drive_log_path,
                drive_rows,
                events=events,
                odom_m=odom_m,
                credit_m=credit_m,
                route_pts=route_pts,
                max_route_index=max_route_index,
                planned_m=planned_m,
                max_collisions_per_km=max_collisions_per_km,
            )
            _cleanup_actors(world, actors, allow_tick=False)
    return drive_log_path if drive_rows else None


def _finish_drive_log(
    log_dir: Path,
    drive_log_path: Path,
    drive_rows: list[dict[str, str]],
    *,
    events: EventCounter,
    odom_m: float,
    credit_m: float,
    route_pts: list[tuple[float, float, float]],
    max_route_index: int,
    planned_m: float,
    max_collisions_per_km: float,
) -> None:
    """Write the csv and scorecard before actors are destroyed."""
    if not drive_rows:
        return
    progress_m = credit_m
    if route_pts:
        progress_m += path_length_m(route_pts, 0, max_route_index)
    with drive_log_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_DRIVE_LOG_HEADER)
        writer.writeheader()
        writer.writerows(drive_rows)
    _print_drive_cte_summary(drive_rows)
    _write_drive_scorecard(
        log_dir,
        rows=drive_rows,
        events=events,
        odom_m=odom_m,
        progress_m=progress_m,
        planned_m=planned_m,
        max_collisions_per_km=max_collisions_per_km,
    )
    print(f"Wrote {drive_log_path} ({len(drive_rows)} frames)")


def _route_xyz(route: list) -> list[tuple[float, float, float]]:
    points: list[tuple[float, float, float]] = []
    for wp, _opt in route:
        loc = wp.transform.location
        points.append((float(loc.x), float(loc.y), float(loc.z)))
    return points


def _attach_score_sensors(
    world: carla.World,
    vehicle: carla.Vehicle,
    events: EventCounter,
    actors: list,
) -> None:
    """Collision and lane-invasion sensors. Counts are not fed to the model."""
    try:
        library = world.get_blueprint_library()
        pairs = (
            ("sensor.other.collision", events.on_collision),
            ("sensor.other.lane_invasion", events.on_lane_invasion),
        )
        for blueprint_id, callback in pairs:
            sensor = world.spawn_actor(
                library.find(blueprint_id),
                carla.Transform(),
                attach_to=vehicle,
            )
            sensor.listen(callback)
            actors.append(sensor)
    except Exception as exc:
        print(f"Score sensors unavailable ({exc})")
        return
    print("Score sensors ON: collision, lane invasion (not model inputs)")


def _write_drive_scorecard(
    log_dir: Path,
    *,
    rows: list[dict[str, str]],
    events: EventCounter,
    odom_m: float,
    progress_m: float,
    planned_m: float,
    max_collisions_per_km: float,
) -> None:
    collisions, invasions = events.snapshot()
    speeds = [float(row["speed_kmh"]) for row in rows]
    avg_speed = sum(speeds) / len(speeds) if speeds else 0.0
    completion = route_completion_pct(progress_m, planned_m)
    per_km = collisions_per_km(collisions, odom_m)
    passed = per_km < max_collisions_per_km
    summary = {
        "lane_invasions": invasions,
        "collisions": collisions,
        "distance_m": round(odom_m, 2),
        "route_progress_m": round(progress_m, 2),
        "route_length_m": round(planned_m, 2),
        "route_completion_pct": round(completion, 2),
        "avg_speed_kmh": round(avg_speed, 2),
        "collisions_per_km": round(per_km, 3),
        "max_collisions_per_km": max_collisions_per_km,
        "milestone": "PASS" if passed else "FAIL",
    }
    path = log_dir / "drive_metrics.json"
    path.write_text(json.dumps(summary, indent=2) + "\n")
    print("Drive scorecard:")
    print(
        f"  route completion: {completion:.1f}%  "
        f"({progress_m:.0f} m / {planned_m:.0f} m)"
    )
    print(f"  avg speed: {avg_speed:.1f} km/h over {odom_m:.0f} m")
    print(f"  lane invasions: {invasions}")
    print(f"  collisions: {collisions}")
    print(
        f"  collisions/km: {per_km:.2f}  "
        f"(limit <{max_collisions_per_km:.2f})  "
        f"{'PASS' if passed else 'FAIL'}"
    )
    print(f"Wrote {path}")


def _print_drive_cte_summary(rows: list[dict[str, str]]) -> None:
    """Primary success metric: |cte| under TURN_* stays in-lane."""
    def _stats(subset: list[dict[str, str]]) -> str:
        if not subset:
            return "n=0"
        abs_cte = [abs(float(r["cte_m"])) for r in subset]
        abs_cte.sort()
        n = len(abs_cte)
        mean = sum(abs_cte) / n
        p90 = abs_cte[min(n - 1, int(0.9 * n))]
        mx = abs_cte[-1]
        edge = sum(1 for r in subset if abs(float(r["lane_offset_frac"])) > 0.85)
        wide = sum(1 for r in subset if abs(float(r["lane_offset_frac"])) > 0.7)
        return (
            f"n={n} mean|cte|={mean:.2f} p90={p90:.2f} max={mx:.2f}  "
            f"|frac|>0.7={100 * wide / n:.1f}%  |frac|>0.85={100 * edge / n:.1f}%"
        )

    turns = [r for r in rows if r.get("command") in _TURN_COMMANDS]
    follow = [r for r in rows if r.get("command") in _LANE_KEEP_COMMANDS]
    print("Drive lane summary (eval-only CTE; not model input):")
    print(f"  TURN_*:     {_stats(turns)}")
    print(f"  FOLLOW/GS:  {_stats(follow)}")
    if turns:
        p90 = sorted(abs(float(r["cte_m"])) for r in turns)[
            min(len(turns) - 1, int(0.9 * len(turns)))
        ]
        ok = p90 < 0.6
        print(
            f"  gate: TURN_* p90 |cte|={'PASS' if ok else 'FAIL'} "
            f"({p90:.2f}m; want <0.6m)"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("models/bc.pt"))
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument(
        "--from-run",
        type=Path,
        default=None,
        help="Use map + spawn from a recorded run's meta.json (matched eval)",
    )
    parser.add_argument("--map", dest="map_name", default=None)
    parser.add_argument("--spawn-index", type=int, default=None)
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu", "auto"],
        default="cuda",
    )
    parser.add_argument(
        "--steer-smooth",
        type=float,
        default=0.55,
        help="Adaptive EMA on steer: 0=raw, ~0.55=default",
    )
    parser.add_argument(
        "--max-steer",
        type=float,
        default=0.55,
        help="Hard clamp on |steer| before speed scaling",
    )
    parser.add_argument(
        "--follow-steer-scale",
        type=float,
        default=0.50,
        help="Max steer scale while cmd is FOLLOW_LANE/GO_STRAIGHT (room to recover)",
    )
    parser.add_argument(
        "--steer-rate",
        type=float,
        default=0.04,
        help="Max |Δsteer| per frame (eases turn snap)",
    )
    parser.add_argument(
        "--no-draw-path",
        action="store_true",
        help="Hide route / lookahead / destination debug overlay",
    )
    parser.add_argument(
        "--raw-steer",
        action="store_true",
        help="Disable steer smooth/rate/FOLLOW+speed clamps (pure network steer)",
    )
    parser.add_argument(
        "--turn-speed-kmh",
        type=float,
        default=16.0,
        help=(
            "Max speed while turning with |steer|>=0.12 (0=off). "
            "Only applies after turn-in, not on the straight approach."
        ),
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="Write drive_log.csv with eval-only cte_m here (default: data/drive_log_*)",
    )
    parser.add_argument(
        "--traffic",
        type=int,
        default=None,
        help="Spawn N NPC vehicles (default: 15; --no-traffic for empty road)",
    )
    parser.add_argument(
        "--no-traffic",
        action="store_true",
        help="Force empty road (no NPCs)",
    )
    parser.add_argument(
        "--max-collisions-per-km",
        type=float,
        default=1.0,
        help="Town01 milestone: PASS when collisions/km stay below this",
    )
    parser.add_argument("--tm-port", type=int, default=8000)
    parser.add_argument(
        "--fixed-route",
        action="store_true",
        help="Drive the RL env's route (--spawn-index -> farthest spawn, no "
        "replans) and stop at 95%%, so the scorecard is comparable with "
        "rl.eval_drive. Default spawn 0.",
    )
    args = parser.parse_args()
    if args.fixed_route and args.spawn_index is None and args.from_run is None:
        args.spawn_index = 0
    if args.no_traffic:
        traffic_arg: int | None = 0
    else:
        traffic_arg = args.traffic  # None → 15 inside drive_run
    drive_run(
        model_path=args.model,
        seconds=args.seconds,
        fps=args.fps,
        from_run=args.from_run,
        map_name=args.map_name,
        spawn_index=args.spawn_index,
        device_name=args.device,
        steer_smooth=args.steer_smooth,
        max_steer=args.max_steer,
        follow_steer_scale=args.follow_steer_scale,
        steer_rate=args.steer_rate,
        draw_path=not args.no_draw_path,
        raw_steer=args.raw_steer,
        turn_speed_kmh=args.turn_speed_kmh,
        log_dir=args.log_dir,
        traffic=traffic_arg,
        tm_port=args.tm_port,
        max_collisions_per_km=args.max_collisions_per_km,
        fixed_route=args.fixed_route,
    )


if __name__ == "__main__":
    main()
