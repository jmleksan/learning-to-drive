"""
Record expert driving data for behavioral cloning.

Default expert is a waypoint path follower (stable, tracks the drawn route).
Commands for CIL still come from that route (FOLLOW_LANE / TURN_* / …).

Optional --cil uses Traffic Manager autopilot with a sparse dest hint — more
“natural” driving, but TM can be erratic (U-turns, ignoring the path).

Examples:
  python -m expert.record --seconds 300
  python -m expert.record --map Town01 --spawn-index 42
  python -m expert.record --cil --seconds 180
  python -m expert.record --semantic --cameras center --seconds 60
  python -m expert.record --semantic --lidar --walkers 40 --cameras center --seconds 300
  python -m expert.record --semantic --depth --lidar --weather SoftRainNoon --seconds 300
  python -m expert.record --semantic --depth --lidar --weather ClearNight --seconds 300
  python -m expert.record --semantic --depth --lidar --weather night --seconds 300
  python -m expert.record --gnss --imu --no-traffic --seconds 60
"""
from __future__ import annotations

import argparse
import csv
import queue
import random
from collections import deque
from datetime import datetime
from pathlib import Path

import carla
from PIL import Image

from expert.model import IMAGE_HEIGHT, IMAGE_WIDTH
from localization.geo import geo_to_enu
from localization.log import open_localization_writer
from localization.pose_enu import (
    location_to_enu,
    origin_from_location,
    yaw_enu_rad,
)
from localization.sensors import (
    DEFAULT_GNSS_STDDEV_DEG,
    pop_gnss,
    pop_imu,
    try_attach_gnss,
    try_attach_imu,
)
from utils.balance import (
    CONTROLS_HEADER,
    warn_if_follow_lane_heavy,
    write_balanced_controls,
    write_command_distribution,
)
from utils.commands import (
    COMMAND_TURN_LEFT,
    COMMAND_TURN_RIGHT,
    JUNCTION_COMMANDS,
    label_command,
)
from utils.connection import connect_client
from utils.images import image_to_array
from perception.depth import depth_to_meters, save_depth_meters
from perception.lidar import (
    attach_semantic_lidar,
    points_from_measurement,
    save_lidar_points,
)
from segmentation.labels import semantic_to_labels
from utils.lane_metrics import format_lane_metric_fields, lane_metrics_at
from utils.routing import (
    analyze_route_turns,
    apply_tm_path,
    build_planner,
    draw_route,
    draw_target,
    draw_destination,
    hold_turn_command,
    lookahead_target,
    needs_replan,
    pick_balanced_route,
    pick_next_destination,
    plan_origin_along_heading,
    plan_route,
    plan_spawn_to_spawn,
    route_forward_alignment,
)
from utils.run_meta import find_spawn_index, save_run_meta, ensure_map
from utils.weather import WEATHER_PRESETS, apply_weather
from utils.spectator import follow_vehicle
from utils.state import speed_kmh
from utils.sync_mode import SynchronousMode
from utils.traffic import (
    _safe_destroy,
    apply_lead_vehicle_hold,
    clear_other_vehicles,
    destroy_wrecked_vehicles,
    lead_vehicle_gap_m,
    prune_actor_list,
    spawn_npc_traffic,
)
from utils.traffic_signs import (
    apply_traffic_light_hold,
    maybe_force_ego_light_green,
    near_stop_sign,
    speed_up_traffic_lights,
    traffic_light_fields,
)
from utils.radar import format_radar_m, format_radar_status, pop_radar_range, try_attach_radar
from utils.vehicles import (
    CAMERA_SPECS,
    attach_depth_camera,
    attach_rgb_cameras,
    attach_semantic_camera,
    recovery_steer,
    spawn_vehicle,
)
from utils.walkers import (
    apply_pedestrian_hold,
    destroy_walkers,
    partition_walker_actors,
    spawn_walkers,
)
from utils.waypoint_follower import WaypointFollower


def _cleanup_actors(
    world: carla.World,
    actors: list[carla.Actor],
    *,
    allow_tick: bool = True,
) -> None:
    """Stop sensors, let the render thread finish, then destroy.

    Do not call is_alive or is_listening: on a dead handle those abort the
    process instead of raising. The semantic camera's shader pass has to
    drain with a tick before destroy, or the server ignores the destroy RPC
    until the client aborts. Walker controllers are destroyed before walkers.
    """
    sensors: list[carla.Sensor] = []
    non_sensors: list[carla.Actor] = []
    for actor in actors:
        if isinstance(actor, carla.Sensor):
            sensors.append(actor)
        else:
            non_sensors.append(actor)

    for sensor in sensors:
        try:
            sensor.stop()
        except Exception:
            pass

    if allow_tick:
        for _ in range(3):
            try:
                world.tick()
            except Exception:
                break

    for sensor in sensors:
        try:
            sensor.destroy()
        except Exception:
            pass

    if allow_tick:
        try:
            world.tick()
        except Exception:
            pass

    controllers, walkers, others = partition_walker_actors(non_sensors)
    destroy_walkers(controllers, walkers)
    for actor in others:
        _safe_destroy(actor)


def print_turn_summary(controls_path: Path, *, steer_threshold: float = 0.15) -> None:
    rows = list(csv.DictReader(controls_path.open()))
    if not rows:
        return
    center = [r for r in rows if r.get("camera", "center") == "center"]
    use = center or rows
    turns = sum(1 for r in use if abs(float(r["steer"])) >= steer_threshold)
    sharp = sum(1 for r in use if abs(float(r["steer"])) >= 0.3)
    print(
        f"Turn frames (|steer|>={steer_threshold}, center): {turns}/{len(use)} "
        f"({100 * turns / len(use):.1f}%)  sharp (>0.3): {sharp}"
    )


def _clear_queue(q: queue.Queue) -> None:
    while True:
        try:
            q.get_nowait()
        except queue.Empty:
            break


def _drain_queue(q: queue.Queue, *, timeout: float = 2.0) -> carla.Image:
    """Wait for a frame, then drop any backlog so the newest one wins."""
    image = q.get(timeout=timeout)
    while True:
        try:
            image = q.get_nowait()
        except queue.Empty:
            break
    return image


def _get_camera_images(
    world: carla.World,
    cam_queues: dict[str, queue.Queue[carla.Image]],
    camera_names: tuple[str, ...],
    *,
    timeout: float = 5.0,
    retries: int = 15,
    tick_on_retry: bool = True,
) -> dict[str, carla.Image]:
    """Wait until every camera has delivered a frame."""
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            return {
                name: _drain_queue(cam_queues[name], timeout=timeout)
                for name in camera_names
            }
        except queue.Empty as exc:
            last_err = exc
            for q in cam_queues.values():
                _clear_queue(q)
            if not tick_on_retry or attempt + 1 >= retries:
                break
            world.tick()
    raise RuntimeError(
        f"No camera frames after {retries} tries ({last_err!r}). "
        "If you just switched Town10HD→Town01, restart Carla already on Town01 "
        "and use --keep-map."
    )


def parse_camera_names(value: str) -> tuple[str, ...]:
    """Parse 'center' / 'all' / 'center,left,right' into ordered camera names."""
    raw = value.strip().lower()
    order = ("center", "left", "right", "left_wide", "right_wide")
    if raw in {"all", "5", "multi"}:
        return order
    if raw in {"3", "center,left,right"}:
        return ("center", "left", "right")
    names = tuple(part.strip() for part in raw.split(",") if part.strip())
    allowed = set(order)
    bad = [n for n in names if n not in allowed]
    if bad:
        raise ValueError(
            f"Unknown camera(s) {bad}; use center,left,right,left_wide,right_wide or all"
        )
    if not names:
        raise ValueError("Need at least one camera name")
    return tuple(n for n in order if n in names)


def record_run(
    *,
    seconds: float = 60.0,
    fps: float = 20.0,
    out: Path = Path("data"),
    tm_port: int = 8000,
    map_name: str | None = None,
    spawn_index: int | None = None,
    dest_index: int | None = None,
    shuttle: bool = True,
    route_mode: str = "balanced",
    lane_change: bool = False,
    ignore_lights: bool = False,
    cameras: tuple[str, ...] = (
        "center",
        "left",
        "right",
        "left_wide",
        "right_wide",
    ),
    follow_spectator: bool = True,
    draw_path: bool = True,
    expert: str = "path",
    map_settle_seconds: float | None = None,
    mid_turn_nudge: bool = True,
    mid_turn_nudge_mag: float = 0.08,
    fast_lights: bool = True,
    light_red_s: float = 5.0,
    light_yellow_s: float = 1.5,
    light_green_s: float = 7.0,
    max_red_wait_s: float = 6.0,
    traffic: int = 15,
    semantic: bool = False,
    depth: bool = False,
    lidar: bool = False,
    gnss: bool = False,
    imu: bool = False,
    gnss_noise_deg: float = DEFAULT_GNSS_STDDEV_DEG,
    walkers: int = 0,
    weather: str = "ClearNoon",
) -> Path:
    client = connect_client(timeout=30.0)
    if map_name is not None:
        ensure_map(client, map_name, settle_seconds=map_settle_seconds)
    # Always re-fetch after map ops.
    world = client.get_world()
    client.set_timeout(30.0)
    weather_name = apply_weather(world, weather)
    print(f"Weather: {weather_name}")

    # Plan route BEFORE sync mode — GRP graph build can take a while and must
    # not run while synchronous_mode is on without ticking (that hung cameras).
    world_map = world.get_map()
    spawn_pts = world_map.get_spawn_points()
    if spawn_index is not None:
        if not 0 <= spawn_index < len(spawn_pts):
            raise ValueError(
                f"spawn_index must be 0..{len(spawn_pts) - 1}, got {spawn_index}"
            )
        start_index = spawn_index
    else:
        start_index = 0

    print("Building route planner (async, before sync mode) ...")
    grp = build_planner(world_map)
    route_rng = random.Random()

    # Fixed A→B only when user passes --dest-index. Otherwise each leg picks a
    # new varied destination (avoids looping the same 2–3 courses).
    point_to_point = dest_index is not None
    recent_dests: deque[int] = deque(maxlen=12)

    if point_to_point:
        assert dest_index is not None
        if dest_index < 0:
            print("Selecting balanced A→B from start spawn ...")
            oi, di, dest, route, stats = pick_balanced_route(
                grp,
                spawn_pts,
                start_index=start_index,
                rng=route_rng,
                top_k=8,
            )
            dest_index = di
            print(
                f"--dest-index auto (balanced): {oi}→{di}  "
                f"L={stats['left']} R={stats['right']} S={stats['straight']}  "
                f"len={stats['length_m']:.0f}m score={stats['score']:.2f}"
            )
        else:
            dest, route, _junc = plan_spawn_to_spawn(
                grp, spawn_pts, start_index, dest_index
            )
            stats = analyze_route_turns(route)
            print(
                f"Point-to-point: spawn {start_index} → {dest_index}  "
                f"L={stats['left']} R={stats['right']} S={stats['straight']}  "
                f"len={stats['length_m']:.0f}m"
            )
        home_index = start_index
        away_index = dest_index
        going_to_away = True
        recent_dests.append(away_index)
        print(f"shuttle={'ON' if shuttle else 'OFF'}")
    else:
        home_index = start_index
        away_index = start_index
        going_to_away = True
        origin = spawn_pts[start_index].location
        dest, route, score, stats = pick_next_destination(
            grp,
            origin,
            spawn_pts,
            route_mode=route_mode,
            rng=route_rng,
            top_k=8,
            heading_yaw=spawn_pts[start_index].rotation.yaw,
            world_map=world_map,
        )
        if "dest" in stats:
            away_index = int(stats["dest"])
            home_index = int(stats.get("start", start_index))
            start_index = home_index
            recent_dests.append(away_index)
        print(
            f"Route mode={stats.get('mode', route_mode)}: "
            f"spawn {home_index}→{away_index}  "
            f"L={stats.get('left', '?')} R={stats.get('right', '?')} "
            f"S={stats.get('straight', '?')}  "
            f"align={stats.get('forward_align', 0):+.2f}  "
            f"len={stats.get('length_m', 0):.0f}m score={score:.2f}"
        )
        print("Variety ON: each leg picks a new destination (not fixed shuttle)")

    actors: list[carla.Actor] = []
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = out / f"run_{stamp}"
    camera_names = tuple(cameras) if cameras else ("center", "left", "right")
    image_dirs = {name: run_dir / "images" / name for name in camera_names}
    for d in image_dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    controls_path = run_dir / "controls.csv"
    cam_queues: dict[str, queue.Queue[carla.Image]] = {
        name: queue.Queue() for name in image_dirs
    }
    recorded_rows: list[dict[str, str]] = []
    semantic_queues: dict[str, queue.Queue[carla.Image]] = {}
    seg_dirs: dict[str, Path] = {}
    depth_queue: queue.Queue[carla.Image] | None = None
    depth_dir: Path | None = None
    lidar_queue: queue.Queue | None = None
    lidar_dir: Path | None = None
    gnss_queue: queue.Queue | None = None
    imu_queue: queue.Queue | None = None
    loc_fh = None
    loc_writer = None
    geo_origin = None
    last_gnss = None
    last_imu = None
    if semantic and "center" not in camera_names:
        raise ValueError(
            "Semantic labels require the center camera "
            "(plus any other --cameras you want tags for); "
            "pass --cameras center or center,left,right"
        )
    if depth and "center" not in camera_names:
        raise ValueError(
            "Depth labels are aligned to the center camera; "
            "pass --cameras center (add right_wide only if training stereo)"
        )
    if semantic:
        for name in camera_names:
            seg_dirs[name] = run_dir / "seg" / name
            seg_dirs[name].mkdir(parents=True, exist_ok=True)
    if depth:
        depth_dir = run_dir / "depth" / "center"
        depth_dir.mkdir(parents=True, exist_ok=True)
    if lidar:
        lidar_dir = run_dir / "lidar"
        lidar_dir.mkdir(parents=True, exist_ok=True)
    if gnss or imu:
        loc_fh, loc_writer = open_localization_writer(run_dir)

    actors_released = False
    try:
        world = client.get_world()
        with SynchronousMode(client, world, fps=fps, tm_port=tm_port) as (
            world,
            tm,
        ):
            try:
                vehicle = spawn_vehicle(world, spawn_index=start_index)
                used_spawn = (
                    start_index
                    if spawn_index is not None or point_to_point
                    else find_spawn_index(vehicle, world)
                )
                if gnss or imu:
                    geo_origin = origin_from_location(
                        world.get_map(), vehicle.get_location()
                    )
                save_run_meta(
                    run_dir,
                    map_name=world.get_map().name,
                    spawn_index=used_spawn,
                    fps=fps,
                    lane_change=lane_change,
                    traffic=max(0, int(traffic)),
                    walkers=max(0, int(walkers)),
                    semantic=semantic,
                    depth=depth,
                    lidar=lidar,
                    gnss=gnss,
                    imu=imu,
                    weather=weather_name,
                    geo_origin=(
                        {
                            "lat_deg": geo_origin.lat_deg,
                            "lon_deg": geo_origin.lon_deg,
                            "alt_m": geo_origin.alt_m,
                        }
                        if geo_origin is not None
                        else None
                    ),
                    gnss_noise_lat_lon_deg=gnss_noise_deg if gnss else None,
                )
                actors.append(vehicle)

                npcs: list = []
                if traffic > 0:
                    npcs = spawn_npc_traffic(
                        world,
                        tm,
                        count=int(traffic),
                        tm_port=tm_port,
                        ego=vehicle,
                        exclude_spawn_indices={used_spawn},
                        ignore_lights=ignore_lights,
                    )
                    actors.extend(npcs)
                    print(
                        f"Traffic ON: spawned {len(npcs)}/{traffic} NPC vehicles "
                        "(TM autopilot; path expert brakes for lead cars)"
                    )
                    # Let TM settle before cameras / first control frame.
                    for _ in range(max(1, int(fps))):
                        world.tick()
                else:
                    n_clear = clear_other_vehicles(world, ego=vehicle)
                    if n_clear:
                        print(f"Cleared {n_clear} leftover vehicle(s)")
                        world.tick()

                if walkers > 0:
                    ped_walkers, ped_controllers = spawn_walkers(
                        client, world, count=int(walkers)
                    )
                    actors.extend(ped_walkers)
                    actors.extend(ped_controllers)
                    print(
                        f"Walkers ON: spawned {len(ped_walkers)}/{walkers} pedestrians "
                        f"({len(ped_controllers)} AI controllers)"
                    )
                    for _ in range(max(1, int(fps))):
                        world.tick()

                print(f"Attaching cameras: {', '.join(camera_names)} ...")
                cams = attach_rgb_cameras(
                    world,
                    vehicle,
                    names=camera_names,
                    width=IMAGE_WIDTH,
                    height=IMAGE_HEIGHT,
                    stagger_ticks=0,
                )
                for name, camera in cams.items():
                    actors.append(camera)
                    camera.listen(cam_queues[name].put)
                if semantic:
                    for name in camera_names:
                        _n, loc, yaw = next(
                            spec for spec in CAMERA_SPECS if spec[0] == name
                        )
                        sem_cam = attach_semantic_camera(
                            world,
                            vehicle,
                            width=IMAGE_WIDTH,
                            height=IMAGE_HEIGHT,
                            location=loc,
                            yaw=yaw,
                        )
                        actors.append(sem_cam)
                        q: queue.Queue[carla.Image] = queue.Queue()
                        sem_cam.listen(q.put)
                        semantic_queues[name] = q
                    print(
                        "Semantic ON: tag ids → "
                        + ", ".join(f"seg/{n}" for n in camera_names)
                    )
                if depth:
                    _name, center_loc, center_yaw = next(
                        spec for spec in CAMERA_SPECS if spec[0] == "center"
                    )
                    depth_cam = attach_depth_camera(
                        world,
                        vehicle,
                        width=IMAGE_WIDTH,
                        height=IMAGE_HEIGHT,
                        location=center_loc,
                        yaw=center_yaw,
                    )
                    actors.append(depth_cam)
                    depth_queue = queue.Queue()
                    depth_cam.listen(depth_queue.put)
                    print("Depth camera ON: center meters → depth/center")
                if lidar:
                    lidar_sensor = attach_semantic_lidar(world, vehicle, fps=fps)
                    actors.append(lidar_sensor)
                    lidar_queue = queue.Queue()
                    lidar_sensor.listen(lidar_queue.put)
                    print("Lidar ON: roof semantic points → lidar/")
                if gnss:
                    gnss_queue = queue.Queue()
                    if not try_attach_gnss(
                        world,
                        vehicle,
                        actors,
                        gnss_queue,
                        fps=fps,
                        noise_lat_lon_deg=gnss_noise_deg,
                    ):
                        gnss_queue = None
                if imu:
                    imu_queue = queue.Queue()
                    if not try_attach_imu(
                        world, vehicle, actors, imu_queue, fps=fps
                    ):
                        imu_queue = None
                if gnss or imu:
                    print("Localization log → localization.csv (ENU meters + GT)")
                radar_queue: queue.Queue = queue.Queue()
                radar_on = try_attach_radar(world, vehicle, actors, radar_queue)
                radar_range_m: float | None = None

                print(f"Spawn index: {used_spawn} / {len(spawn_pts) - 1}")
                print(f"Map: {world.get_map().name}")

                # Prove the sensor stream works BEFORE autopilot / set_path.
                print("Waiting for first camera frame ...")
                world.tick()
                first = {
                    name: _drain_queue(cam_queues[name], timeout=10.0)
                    for name in camera_names
                }
                del first
                for q in semantic_queues.values():
                    _drain_queue(q, timeout=10.0)
                if depth_queue is not None:
                    _drain_queue(depth_queue, timeout=10.0)
                if lidar_queue is not None:
                    _drain_queue(lidar_queue, timeout=10.0)
                if gnss_queue is not None:
                    pop_gnss(gnss_queue)
                if imu_queue is not None:
                    pop_imu(imu_queue)
                for q in cam_queues.values():
                    _clear_queue(q)
                for q in semantic_queues.values():
                    _clear_queue(q)
                if depth_queue is not None:
                    _clear_queue(depth_queue)
                if lidar_queue is not None:
                    _clear_queue(lidar_queue)
                print("Camera OK.")

                # path (default): stable pure-pursuit; route also → CIL commands.
                # cil / tm: TM drives; can U-turn / ignore path — opt-in only.
                expert_mode = (expert or "path").lower()
                use_tm = expert_mode in {
                    "cil",
                    "tm",
                    "traffic_manager",
                    "autopilot",
                }
                # Middle ground: tighter than soft ~0.13 arcs, smoother than snap turns.
                # Middle ground: tighter than soft ~0.13 arcs, smoother than snap turns.
                follower = WaypointFollower(
                    cruise_kmh=24.0,
                    turn_kmh=16.0,
                    min_lookahead_m=3.5,
                    max_lookahead_m=9.0,
                    steer_gain=1.65,
                    max_steer_rate=0.22,
                )
                if use_tm:
                    vehicle.set_autopilot(True, tm_port)
                    light_pct = 100.0 if ignore_lights else 0.0
                    tm.ignore_lights_percentage(vehicle, light_pct)
                    tm.ignore_signs_percentage(vehicle, light_pct)
                    tm.auto_lane_change(vehicle, lane_change)
                    # Denser than before — sparse 30m paths made TM spin/U-turn.
                    apply_tm_path(tm, vehicle, route, min_spacing_m=8.0)
                    print(
                        "Expert: CIL/TM (autopilot; route = commands + path hint). "
                        "Warning: TM can still U-turn or leave the drawn route. "
                        f"lights={'IGNORE' if ignore_lights else 'OBEY'}"
                    )
                else:
                    vehicle.set_autopilot(False)
                    print(
                        "Expert: path follower (stable; CIL commands from route; "
                        "use --cil for TM autopilot). "
                        f"lights={'IGNORE' if ignore_lights else 'OBEY'}"
                    )

                if not ignore_lights and fast_lights:
                    n_lights = speed_up_traffic_lights(
                        world,
                        red_s=light_red_s,
                        yellow_s=light_yellow_s,
                        green_s=light_green_s,
                    )
                    print(
                        f"Fast lights ON: {n_lights} lights  "
                        f"R={light_red_s:.1f}s Y={light_yellow_s:.1f}s "
                        f"G={light_green_s:.1f}s  "
                        f"max ego red wait={max_red_wait_s:.1f}s"
                    )

                route_index = 0
                sticky_turn: str | None = None
                sticky_turn_start: int | None = None
                red_wait_frames = 0
                max_red_wait_frames = (
                    int(max_red_wait_s * fps) if max_red_wait_s > 0 else 0
                )
                nudge_left = 0
                nudge_steer = 0.0
                nudge_kind = "follow"  # "follow" | "mid_turn"
                # Deterministic FOLLOW poke so recovery actually appears in CSV.
                nudge_cooldown = 0
                mid_turn_nudge_cooldown = int(6.0 * fps)
                follow_nudge_period = int(10.0 * fps)  # ~10s between FOLLOW pokes
                stop_actors = list(world.get_actors().filter("traffic.stop"))

                print(f"Recording expert to {run_dir} for {seconds:.0f}s @ {fps:.0f} FPS ...")
                if draw_path:
                    print(
                        "Path overlay ON (HUD only — not baked into RGB cameras). "
                        "Green=follow, cyan≈left, orange≈right, yellow=straight; "
                        "red=lookahead. Use UE spectator view."
                    )
                if not use_tm:
                    print(
                        f"FOLLOW recovery: |nudge|~0.05–0.22 every ~"
                        f"{follow_nudge_period / fps:.0f}s while FOLLOW "
                        "(biased small; prints 'nudge FOLLOW')"
                    )
                if mid_turn_nudge and not use_tm:
                    print(
                        "Mid-turn recovery ON: |nudge|~0.05–0.10 "
                        "during sticky TURN_* (varying; labels stay TURN_*)"
                    )
                print("Ctrl+C to stop early.")

                frames = int(seconds * fps)
                line_life = max(1.0 / fps * 2.5, 0.12)

                with controls_path.open("w", newline="") as fh:
                    writer = csv.DictWriter(fh, fieldnames=CONTROLS_HEADER)
                    writer.writeheader()

                    for frame_idx in range(frames):
                        world.tick()
                        if radar_on:
                            radar_range_m = pop_radar_range(radar_queue, radar_range_m)
                        if follow_spectator:
                            follow_vehicle(world, vehicle)

                        loc = vehicle.get_location()
                        yaw = vehicle.get_transform().rotation.yaw
                        wmap = world.get_map()

                        def _set_route(new_route: list) -> None:
                            nonlocal route, route_index, sticky_turn, sticky_turn_start
                            route = new_route
                            route_index = 0
                            sticky_turn = None
                            sticky_turn_start = None
                            follower.reset()
                            if use_tm:
                                apply_tm_path(tm, vehicle, route, min_spacing_m=8.0)

                        if needs_replan(loc, route, route_index):
                            if point_to_point:
                                if not shuttle:
                                    print(
                                        f"  reached dest @ frame {frame_idx}; "
                                        "holding path (shuttle off)"
                                    )
                                else:
                                    going_to_away = not going_to_away
                                    b = away_index if going_to_away else home_index
                                    dest_loc = spawn_pts[b].location
                                    plan_from = plan_origin_along_heading(wmap, loc, yaw)
                                    new_route = plan_route(grp, plan_from, dest_loc)
                                    align = route_forward_alignment(new_route, loc, yaw)
                                    # If A* would leave backward toward B, pick any
                                    # forward-leaving dest instead (B can wait).
                                    if align < -0.15 or len(new_route) < 5:
                                        dest_loc, new_route, score, stats = (
                                            pick_next_destination(
                                                grp,
                                                loc,
                                                spawn_pts,
                                                route_mode="farthest",
                                                rng=route_rng,
                                                heading_yaw=yaw,
                                                world_map=wmap,
                                                top_k=8,
                                            )
                                        )
                                        align = float(stats.get("forward_align", 0.0))
                                        print(
                                            f"  shuttle skip U-turn → alt dest "
                                            f"{stats.get('dest')} align={align:+.2f}"
                                        )
                                    else:
                                        score = analyze_route_turns(new_route)["score"]
                                    _set_route(new_route)
                                    print(
                                        f"  shuttle @ frame {frame_idx}: "
                                        f"toward {b} align={align:+.2f} "
                                        f"({len(route)} wps)"
                                    )
                            else:
                                dest, new_route, score, stats = pick_next_destination(
                                    grp,
                                    loc,
                                    spawn_pts,
                                    route_mode=route_mode,
                                    rng=route_rng,
                                    exclude_dests=list(recent_dests),
                                    top_k=8,
                                    heading_yaw=yaw,
                                    world_map=wmap,
                                )
                                align = float(stats.get("forward_align", 0.0))
                                if align < 0.15:
                                    print(
                                        f"  skip reverse replan @ frame {frame_idx}: "
                                        f"→{stats.get('dest')} align={align:+.2f} "
                                        f"(keeping current route)"
                                    )
                                else:
                                    _set_route(new_route)
                                    if "dest" in stats and int(stats["dest"]) >= 0:
                                        recent_dests.append(int(stats["dest"]))
                                    print(
                                        f"  new leg @ frame {frame_idx}: "
                                        f"→{stats.get('dest')} "
                                        f"align={align:+.2f} "
                                        f"mode={stats.get('mode')} "
                                        f"L={stats.get('left')} R={stats.get('right')} "
                                        f"len={stats.get('length_m', 0):.0f}m"
                                    )

                        if use_tm:
                            control = vehicle.get_control()
                            target, _ = lookahead_target(
                                loc, route, start_hint=route_index
                            )
                            applied = control
                            log_steer = float(control.steer)
                        else:
                            control, target, command = follower.control(vehicle, route)
                            route_index = follower.route_index
                            if not ignore_lights:
                                red_wait_frames, forced = maybe_force_ego_light_green(
                                    vehicle,
                                    red_wait_frames=red_wait_frames,
                                    max_wait_frames=max_red_wait_frames,
                                )
                                if forced:
                                    print(
                                        f"  force light GREEN (siblings → red) "
                                        f"@ frame {frame_idx} "
                                        f"(ego red wait ≥{max_red_wait_s:.0f}s)"
                                    )
                                control = apply_traffic_light_hold(control, vehicle)
                            if traffic > 0:
                                control = apply_lead_vehicle_hold(
                                    control, world, vehicle
                                )
                            if walkers > 0:
                                control = apply_pedestrian_hold(
                                    control, world, vehicle, world_map=wmap
                                )
                            # Recovery nudges (minority of frames):
                            # - FOLLOW: mild drift on straights (not blocked by sticky
                            #   TURN_* held early on the approach — that hid all nudges).
                            # - TURN_*: brief outward drift only once actually turning.
                            applied = control
                            log_steer = float(control.steer)
                            in_sticky_turn = sticky_turn in (
                                COMMAND_TURN_LEFT,
                                COMMAND_TURN_RIGHT,
                            )
                            turning_now = abs(control.steer) >= 0.12
                            holding_light = float(control.brake) >= 0.5
                            speed_now = speed_kmh(vehicle)
                            speed_ok = speed_now > 10.0 and not holding_light
                            # A car in the center image must appear with BOTH
                            # "steer ~0" and "steer back to the lane" labels.
                            # Otherwise the car itself becomes the steer signal.
                            # Skip only when the lead is close enough to hit.
                            lead_gap = None
                            if traffic > 0:
                                lead_gap = lead_vehicle_gap_m(
                                    world,
                                    vehicle,
                                    max_dist_m=22.0,
                                    max_lateral_m=7.5,
                                    max_angle_deg=50.0,
                                )
                            car_in_view = lead_gap is not None and lead_gap >= 9.0
                            if car_in_view:
                                speed_ok = speed_now > 6.0 and float(control.brake) < 0.85
                            # FOLLOW nudges only when the label will stay FOLLOW_LANE.
                            # (If sticky TURN_* is already set, recovery lands in TURN_*
                            # CSV rows — FOLLOW looks like steer≈0 even when nudging.)
                            allow_follow = (
                                not in_sticky_turn
                                and not turning_now
                                and speed_ok
                            )
                            allow_mid_turn = (
                                mid_turn_nudge
                                and in_sticky_turn
                                and turning_now
                                and speed_ok
                            )

                            def _apply_nudge(delta: float) -> carla.VehicleControl:
                                steer = control.steer + delta
                                # Keep turn direction if we're mid-corner.
                                if in_sticky_turn and abs(control.steer) >= 0.08:
                                    if sticky_turn == COMMAND_TURN_LEFT:
                                        steer = min(steer, -0.05)
                                    elif sticky_turn == COMMAND_TURN_RIGHT:
                                        steer = max(steer, 0.05)
                                return carla.VehicleControl(
                                    throttle=control.throttle,
                                    steer=max(-1.0, min(1.0, steer)),
                                    brake=control.brake,
                                )

                            if nudge_left > 0:
                                if nudge_kind == "follow" and not allow_follow:
                                    nudge_left = 0
                                elif nudge_kind == "mid_turn" and not allow_mid_turn:
                                    nudge_left = 0
                                else:
                                    applied = _apply_nudge(nudge_steer)
                                    nudge_left -= 1
                            else:
                                nudge_left = 0
                                if allow_follow and nudge_cooldown <= 0:
                                    nudge_kind = "follow"
                                    # Bias small: most recoveries are gentle so the
                                    # net doesn't learn to hunt ±0.2 every offset.
                                    # ~70% in [0.05, 0.12], ~30% in [0.12, 0.22].
                                    if car_in_view:
                                        # Stay in the lane. Do not swing toward the lead.
                                        mag = random.uniform(0.05, 0.10)
                                    elif random.random() < 0.70:
                                        mag = random.uniform(0.05, 0.12)
                                    else:
                                        mag = random.uniform(0.12, 0.22)
                                    nudge_steer = random.choice((-mag, mag))
                                    nudge_left = max(6, int(0.45 * fps))
                                    nudge_cooldown = follow_nudge_period
                                    applied = _apply_nudge(nudge_steer)
                                    nudge_left -= 1
                                    print(
                                        f"  nudge FOLLOW {nudge_steer:+.2f} "
                                        f"@ frame {frame_idx}"
                                    )
                                elif (
                                    allow_mid_turn
                                    and mid_turn_nudge_cooldown <= 0
                                    and random.random() < 0.28
                                ):
                                    mag = random.uniform(
                                        max(0.04, abs(mid_turn_nudge_mag) * 0.6),
                                        max(0.06, abs(mid_turn_nudge_mag) * 1.25),
                                    )
                                    if sticky_turn == COMMAND_TURN_LEFT:
                                        nudge_steer = mag
                                    else:
                                        nudge_steer = -mag
                                    nudge_kind = "mid_turn"
                                    nudge_left = max(4, int(0.28 * fps))
                                    mid_turn_nudge_cooldown = int(12.0 * fps)
                                    applied = _apply_nudge(nudge_steer)
                                    nudge_left -= 1
                                    print(
                                        f"  nudge mid-turn {nudge_steer:+.2f} "
                                        f"@ frame {frame_idx} ({sticky_turn})"
                                    )
                                else:
                                    if allow_follow:
                                        nudge_cooldown = max(0, nudge_cooldown - 1)
                                    if mid_turn_nudge_cooldown > 0:
                                        mid_turn_nudge_cooldown = max(
                                            0, mid_turn_nudge_cooldown - 1
                                        )

                            # CSV label = centering action. While we poke the car off
                            # lane, log the opposite of the poke (recovery), not 0.
                            if abs(float(applied.steer) - float(control.steer)) > 1e-4:
                                log_steer = max(
                                    -1.0,
                                    min(1.0, float(control.steer) - float(nudge_steer)),
                                )
                        vehicle.apply_control(applied)

                        # Same sticky turn timing as drive — labels match inference.
                        (
                            command,
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
                        # Keep next-maneuver intent. Only remap if the expert
                        # is clearly driving the opposite direction.
                        command = label_command(command, float(control.steer))

                        if draw_path and route:
                            draw_route(
                                world,
                                route,
                                life_time=line_life,
                                start_index=max(0, route_index - 2),
                            )
                            draw_target(world, target, life_time=line_life)
                            dest_loc = route[-1][0].transform.location
                            draw_destination(world, dest_loc, life_time=line_life)

                        speed = speed_kmh(vehicle)
                        at_light, light_state = traffic_light_fields(vehicle)
                        stop_flag = near_stop_sign(
                            world, vehicle, stop_actors=stop_actors
                        )
                        lane = lane_metrics_at(loc, yaw, route, route_index)
                        lane_fields = format_lane_metric_fields(lane)

                        # Do not extra-tick here — we already stepped once this frame.
                        images = _get_camera_images(
                            world,
                            cam_queues,
                            camera_names,
                            timeout=5.0,
                            retries=3,
                            tick_on_retry=False,
                        )

                        for cam_name, sem_q in semantic_queues.items():
                            sem_image = _drain_queue(sem_q, timeout=5.0)
                            labels = semantic_to_labels(sem_image)
                            Image.fromarray(labels, mode="L").save(
                                seg_dirs[cam_name] / f"{frame_idx:06d}.png"
                            )
                        if depth_queue is not None and depth_dir is not None:
                            depth_image = _drain_queue(depth_queue, timeout=5.0)
                            save_depth_meters(
                                depth_dir / f"{frame_idx:06d}.npy",
                                depth_to_meters(depth_image),
                            )
                        if lidar_queue is not None and lidar_dir is not None:
                            sweep = _drain_queue(lidar_queue, timeout=5.0)
                            save_lidar_points(
                                lidar_dir / f"{frame_idx:06d}.npy",
                                points_from_measurement(sweep),
                            )

                        if loc_writer is not None and geo_origin is not None:
                            if gnss_queue is not None:
                                sample = pop_gnss(gnss_queue)
                                if sample is not None:
                                    last_gnss = sample
                            if imu_queue is not None:
                                sample = pop_imu(imu_queue)
                                if sample is not None:
                                    last_imu = sample
                            transform = vehicle.get_transform()
                            gt_e, gt_n, _gt_u = location_to_enu(
                                wmap, transform.location, geo_origin
                            )
                            gt_yaw = yaw_enu_rad(wmap, transform, geo_origin)
                            speed_mps = float(speed) / 3.6
                            if last_gnss is not None:
                                gnss_e, gnss_n, gnss_alt = geo_to_enu(
                                    float(last_gnss.latitude),
                                    float(last_gnss.longitude),
                                    float(last_gnss.altitude),
                                    geo_origin,
                                )
                            else:
                                gnss_e = gnss_n = gnss_alt = float("nan")
                            if last_imu is not None:
                                imu_ax = float(last_imu.accelerometer.x)
                                imu_ay = float(last_imu.accelerometer.y)
                                imu_az = float(last_imu.accelerometer.z)
                                imu_gx = float(last_imu.gyroscope.x)
                                imu_gy = float(last_imu.gyroscope.y)
                                imu_gz = float(last_imu.gyroscope.z)
                                imu_compass = float(last_imu.compass)
                            else:
                                imu_ax = imu_ay = imu_az = float("nan")
                                imu_gx = imu_gy = imu_gz = float("nan")
                                imu_compass = float("nan")
                            loc_writer.writerow(
                                {
                                    "frame": str(frame_idx),
                                    "t_s": f"{frame_idx / fps:.4f}",
                                    "gt_e_m": f"{gt_e:.6f}",
                                    "gt_n_m": f"{gt_n:.6f}",
                                    "gt_yaw_rad": f"{gt_yaw:.6f}",
                                    "gt_speed_mps": f"{speed_mps:.6f}",
                                    "gnss_e_m": f"{gnss_e:.6f}",
                                    "gnss_n_m": f"{gnss_n:.6f}",
                                    "gnss_alt_m": f"{gnss_alt:.6f}",
                                    "imu_ax": f"{imu_ax:.6f}",
                                    "imu_ay": f"{imu_ay:.6f}",
                                    "imu_az": f"{imu_az:.6f}",
                                    "imu_gx": f"{imu_gx:.6f}",
                                    "imu_gy": f"{imu_gy:.6f}",
                                    "imu_gz": f"{imu_gz:.6f}",
                                    "imu_compass": f"{imu_compass:.6f}",
                                    "gt_x_m": f"{transform.location.x:.6f}",
                                    "gt_y_m": f"{transform.location.y:.6f}",
                                    "gt_z_m": f"{transform.location.z:.6f}",
                                }
                            )

                        for cam_name, image in images.items():
                            rgb = image_to_array(image)
                            Image.fromarray(rgb).save(
                                image_dirs[cam_name] / f"{frame_idx:06d}.png"
                            )
                            steer = recovery_steer(cam_name, log_steer)
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
                                "near_stop_sign": str(stop_flag),
                                "radar_m": format_radar_m(radar_range_m),
                                **lane_fields,
                            }
                            writer.writerow(row)
                            recorded_rows.append(row)

                        # Skip wreck sweep on frame 0 — freshly spawned NPCs + TM
                        # settle; touching dead handles here aborted the process.
                        if frame_idx > 0 and frame_idx % max(1, int(fps)) == 0:
                            if npcs:
                                before = len(npcs)
                                npcs, n_wreck = destroy_wrecked_vehicles(
                                    world, ego=vehicle, tracked=npcs
                                )
                                if n_wreck:
                                    prune_actor_list(
                                        actors, ego=vehicle, npcs=npcs
                                    )
                                    print(
                                        f"  removed {n_wreck} wrecked NPC(s) "
                                        f"({before}→{len(npcs)}) @ frame {frame_idx}"
                                    )
                            print(
                                f"  frame {frame_idx}/{frames}  "
                                f"speed={speed:4.1f} km/h  "
                                f"steer={log_steer:+.2f}  "
                                f"app={float(applied.steer):+.2f}  "
                                f"cmd={command}  "
                                f"cte={lane['cte_m']:+.2f}m  "
                                f"light={light_state}  "
                                f"radar={format_radar_status(radar_range_m)}"
                            )
                        elif frame_idx < 5:
                            print(f"  frame {frame_idx}/{frames} ...")
                        elif frame_idx % max(1, int(fps)) == 0:
                            print(
                                f"  frame {frame_idx}/{frames}  "
                                f"speed={speed:4.1f} km/h  "
                                f"steer={log_steer:+.2f}  "
                                f"app={float(applied.steer):+.2f}  "
                                f"cmd={command}  "
                                f"cte={lane['cte_m']:+.2f}m  "
                                f"light={light_state}  "
                                f"radar={format_radar_status(radar_range_m)}"
                            )
            finally:
                # Still in sync mode, so a few ticks can finish the
                # semantic camera's shader pass before destroy.
                _cleanup_actors(world, actors, allow_tick=True)
                actors_released = True
    except KeyboardInterrupt:
        print("\nStopped early.")
    finally:
        # Fallback if we never entered the sync block. Ticking here is too
        # late: sync mode is already off, and a second destroy aborts.
        if not actors_released:
            _cleanup_actors(world, actors, allow_tick=False)
        if loc_fh is not None:
            loc_fh.close()

    if recorded_rows:
        warn_if_follow_lane_heavy(recorded_rows)
        dist_path = write_command_distribution(run_dir, recorded_rows)
        bal_path = write_balanced_controls(run_dir, recorded_rows)
        print(f"Wrote {dist_path.name} and {bal_path.name}")

        center = [r for r in recorded_rows if r["camera"] == "center"]
        if center:
            n = len(center)
            follow = sum(1 for r in center if r["command"] == "FOLLOW_LANE")
            junction = sum(1 for r in center if r["command"] in JUNCTION_COMMANDS)
            print(
                f"Center commands: FOLLOW_LANE={100 * follow / n:.1f}%  "
                f"junction(TURN_*|GO_STRAIGHT)={100 * junction / n:.1f}%"
            )

    if controls_path.exists():
        print_turn_summary(controls_path)
    for name, path in seg_dirs.items():
        n_seg = len(list(path.glob("*.png")))
        print(f"Semantic labels ({name}): {n_seg} frames in {path}")
    print(f"Done. Dataset: {run_dir}")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--out", type=Path, default=Path("data"))
    parser.add_argument("--tm-port", type=int, default=8000)
    parser.add_argument(
        "--map-settle-seconds",
        type=float,
        default=None,
        help="Wait after load_world (default: 5s; only when map actually changes)",
    )
    parser.add_argument(
        "--map",
        dest="map_name",
        default="Town01",
        help="CARLA map to load (default: Town01; skipped if already on that map)",
    )
    parser.add_argument(
        "--spawn-index",
        type=int,
        default=None,
        help="Start spawn point index",
    )
    parser.add_argument(
        "--dest-index",
        type=int,
        default=None,
        help=(
            "End spawn for fixed A→B. Use -1 to auto-pick a left/right-balanced "
            "long route from --spawn-index."
        ),
    )
    parser.add_argument(
        "--route-mode",
        choices=["balanced", "junction", "farthest"],
        default="balanced",
        help=(
            "How to choose destinations when --dest-index is omitted "
            "(default: balanced = sample many A→B pairs for L/R mix)"
        ),
    )
    parser.add_argument(
        "--no-shuttle",
        action="store_true",
        help="With --dest-index, do not reverse A↔B when the destination is reached",
    )
    parser.add_argument(
        "--lane-change",
        action="store_true",
        help="Allow TM lane changes (default: stay in lane)",
    )
    parser.add_argument(
        "--ignore-lights",
        action="store_true",
        help="Ignore traffic lights (default: obey red/yellow)",
    )
    parser.add_argument(
        "--cameras",
        default="all",
        help="Cameras: 'all' (default: center + positional left/right and wide), 'center', or a comma list",
    )
    parser.add_argument(
        "--no-follow-spectator",
        action="store_true",
        help="Disable spectator chase cam (on by default with path overlay)",
    )
    parser.add_argument(
        "--no-draw-path",
        action="store_true",
        help="Hide route HUD overlay in the UE spectator (on by default; HUD only, not in RGB)",
    )
    parser.add_argument(
        "--cil",
        action="store_true",
        help=(
            "Use Traffic Manager as expert (CIL: route mainly for commands). "
            "Can be erratic (U-turns). Default is stable path follower."
        ),
    )
    parser.add_argument(
        "--tm-expert",
        action="store_true",
        help="Same as --cil",
    )
    parser.add_argument(
        "--no-mid-turn-nudge",
        action="store_true",
        help="Disable mild outward drift during sticky TURN_* (default: on)",
    )
    parser.add_argument(
        "--mid-turn-nudge-mag",
        type=float,
        default=0.08,
        help="Steer delta for mid-turn drift (default 0.08; FOLLOW uses ±0.16)",
    )
    parser.add_argument(
        "--no-fast-lights",
        action="store_true",
        help="Keep default CARLA light timings (default: shorten cycles)",
    )
    parser.add_argument("--light-red-s", type=float, default=5.0)
    parser.add_argument("--light-yellow-s", type=float, default=1.5)
    parser.add_argument("--light-green-s", type=float, default=7.0)
    parser.add_argument(
        "--max-red-wait-s",
        type=float,
        default=6.0,
        help="Force ego's light green after this many seconds on red (0=off)",
    )
    parser.add_argument(
        "--traffic",
        type=int,
        default=15,
        help="Spawn N NPC vehicles with TM autopilot (default: 15)",
    )
    parser.add_argument(
        "--no-traffic",
        action="store_true",
        help="Force empty road (same as --traffic 0)",
    )
    parser.add_argument(
        "--semantic",
        action="store_true",
        help=(
            "Save CARLA semantic tag ids for each RGB camera in --cameras "
            "(seg/<cam>/*.png, aligned with images/<cam>). "
            "Use center,left,right to retrain SegNet for multi-cam BEV."
        ),
    )
    parser.add_argument(
        "--depth",
        action="store_true",
        help=(
            "Save the center depth camera (depth/center/*.npy) as the BEV "
            "answer key. Requires --cameras center. Add right_wide only if "
            "you still want to train stereo."
        ),
    )
    parser.add_argument(
        "--lidar",
        action="store_true",
        help="Save a roof semantic lidar sweep per frame (lidar/*.npy) for the BEV comparison.",
    )
    parser.add_argument(
        "--gnss",
        action="store_true",
        help="Attach noisy GNSS and log ENU meters + GT into localization.csv",
    )
    parser.add_argument(
        "--imu",
        action="store_true",
        help="Attach IMU and log accel/gyro/compass into localization.csv",
    )
    parser.add_argument(
        "--gnss-noise-deg",
        type=float,
        default=DEFAULT_GNSS_STDDEV_DEG,
        help=(
            "GNSS lat/lon noise stddev in degrees "
            f"(default {DEFAULT_GNSS_STDDEV_DEG:.1e} ≈ few meters)"
        ),
    )
    parser.add_argument(
        "--walkers",
        type=int,
        default=0,
        help="Spawn N AI pedestrians on the navmesh (default: 0)",
    )
    parser.add_argument(
        "--weather",
        type=str,
        default="ClearNoon",
        help=(
            "Preset (ClearNoon, SoftRainNight, …), time group "
            "(noon / sunset / night = random condition at that time), "
            "or 'random'. "
            f"Presets: {', '.join(WEATHER_PRESETS)}"
        ),
    )
    args = parser.parse_args()
    if args.cil or args.tm_expert:
        expert_name = "cil"
    else:
        expert_name = "path"
    record_run(
        seconds=args.seconds,
        fps=args.fps,
        out=args.out,
        tm_port=args.tm_port,
        map_name=args.map_name,
        spawn_index=args.spawn_index,
        dest_index=args.dest_index,
        shuttle=not args.no_shuttle,
        route_mode=args.route_mode,
        lane_change=args.lane_change,
        ignore_lights=args.ignore_lights,
        cameras=parse_camera_names(args.cameras),
        follow_spectator=not args.no_follow_spectator,
        draw_path=not args.no_draw_path,
        expert=expert_name,
        map_settle_seconds=args.map_settle_seconds,
        mid_turn_nudge=not args.no_mid_turn_nudge,
        mid_turn_nudge_mag=args.mid_turn_nudge_mag,
        fast_lights=not args.no_fast_lights,
        light_red_s=args.light_red_s,
        light_yellow_s=args.light_yellow_s,
        light_green_s=args.light_green_s,
        max_red_wait_s=args.max_red_wait_s,
        traffic=0 if args.no_traffic else max(0, int(args.traffic)),
        semantic=args.semantic,
        depth=args.depth,
        lidar=args.lidar,
        gnss=args.gnss,
        imu=args.imu,
        gnss_noise_deg=args.gnss_noise_deg,
        walkers=max(0, int(args.walkers)),
        weather=args.weather,
    )


if __name__ == "__main__":
    main()
