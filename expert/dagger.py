"""
Automated DAgger pass: policy drives, path expert takes over on drift.

Uses the matched map/spawn from a prior KEEP run. On high |cte| (or large
steer disagreement while already off-center), the path expert applies and
those (observation, expert-action) frames are saved for retraining.

Traffic is on by default (15 NPCs). The expert takes the controls when the
car leaves the lane, or when a lead car is inside 14 m and the policy is
not braking. Radar range is fed to throttle/brake when the checkpoint was
trained with --radar, and it is always logged. --no-traffic keeps the
empty-road lane corrections.

Examples:
  python -m expert.dagger --from-run data/run_20260920_134818 --model models/bc.pt
  python -m expert.dagger --from-run data/run_20260920_134818 --seconds 300 --no-traffic
  # then:
  python -m expert.train --data data/run_20260920_134818 data/dagger_... --stack-frames 3
"""
from __future__ import annotations

import argparse
import csv
import queue
from collections import deque
from datetime import datetime
from pathlib import Path

import carla
import numpy as np
import torch
from PIL import Image

from expert.model import IMAGE_HEIGHT, IMAGE_WIDTH, exclusive_controls, load_checkpoint
from expert.record import parse_camera_names, _cleanup_actors, _get_camera_images
from utils.balance import (
    CONTROLS_HEADER,
    write_balanced_controls,
    write_command_distribution,
)
from utils.commands import command_to_onehot
from utils.radar import (
    encode_radar,
    format_radar_m,
    format_radar_status,
    pop_radar_range,
    try_attach_radar,
)
from utils.connection import connect_client
from utils.device import describe_device, resolve_device
from utils.images import image_to_array
from utils.lane_metrics import format_lane_metric_fields, lane_metrics_at
from utils.lights import light_to_onehot, normalize_light_state
from utils.routing import (
    build_planner,
    draw_route,
    draw_target,
    hold_turn_command,
    needs_replan,
    pick_next_destination,
)
from utils.run_meta import apply_run_meta, load_run_meta, save_run_meta
from utils.spectator import follow_vehicle
from utils.state import normalize_speed, speed_kmh
from utils.sync_mode import SynchronousMode
from utils.traffic import (
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
from utils.vehicles import attach_rgb_cameras, recovery_steer, spawn_vehicle
from utils.waypoint_follower import WaypointFollower


def dagger_run(
    *,
    from_run: Path,
    model_path: Path = Path("models/bc.pt"),
    seconds: float = 300.0,
    device_name: str = "cuda",
    cameras: tuple[str, ...] = ("center", "left", "right"),
    enter_cte_m: float = 0.55,
    exit_cte_m: float = 0.28,
    enter_frac: float = 0.70,
    disagree_steer: float = 0.22,
    min_intervene_s: float = 1.2,
    draw_path: bool = True,
    fast_lights: bool = True,
    max_red_wait_s: float = 6.0,
    out: Path = Path("data"),
    tm_port: int = 8000,
    traffic: int = 15,
) -> Path:
    """
    Drive ``model_path`` on ``from_run``'s map/spawn; save expert corrections.
    """
    device = resolve_device(device_name)
    print(f"Inference device: {describe_device(device)}")
    model = load_checkpoint(model_path, device)
    use_command = bool(model.use_command)
    use_light = bool(getattr(model, "use_light", False))
    use_radar = bool(getattr(model, "use_radar", False))
    branched = bool(getattr(model, "branched", False))
    stack_frames = int(getattr(model, "stack_frames", 1))
    if not use_command:
        raise SystemExit("DAgger expects a command-conditioned checkpoint")
    print(
        f"Model: {model_path}  "
        f"({'branched CIL' if branched else 'concat'})  stack={stack_frames}"
    )

    meta = load_run_meta(from_run)
    client = connect_client(timeout=30.0)
    world = apply_run_meta(client, meta)
    world = client.get_world()
    spawn_index = int(meta["spawn_index"])
    fps = float(meta.get("fps", 20.0))
    print(
        f"DAgger matched: spawn {spawn_index} on "
        f"{meta.get('map_load_name', meta['map'])}  "
        f"parent={from_run.name}"
    )
    traffic_n = max(0, int(traffic))
    print(
        f"Intervene when |cte|≥{enter_cte_m:.2f}m or |frac|≥{enter_frac:.2f} "
        f"(or |Δsteer|≥{disagree_steer:.2f} while |cte|≥{exit_cte_m:.2f})"
        + (
            "; or a lead inside 14m while the policy is not braking"
            if traffic_n
            else ""
        )
        + f"; hand back once recovered, after ≥{min_intervene_s:.1f}s"
    )
    if use_radar:
        print("Radar fed to throttle/brake (not steer)")
    else:
        print("Radar logged only; this checkpoint does not read it (train with --radar)")

    world_map = world.get_map()
    spawn_pts = world_map.get_spawn_points()
    print("Building route planner ...")
    grp = build_planner(world_map)
    origin = spawn_pts[spawn_index].location
    heading = spawn_pts[spawn_index].rotation.yaw
    dest, route, _score, stats = pick_next_destination(
        grp,
        origin,
        spawn_pts,
        route_mode="balanced",
        heading_yaw=heading,
        world_map=world_map,
    )
    print(
        f"Route →{stats.get('dest')}  L={stats.get('left')} R={stats.get('right')}  "
        f"len={stats.get('length_m', 0):.0f}m"
    )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(out) / f"dagger_{stamp}"
    camera_names = tuple(cameras) if cameras else ("center",)
    image_dirs = {name: run_dir / "images" / name for name in camera_names}
    for d in image_dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    controls_path = run_dir / "controls.csv"
    save_run_meta(
        run_dir,
        map_name=world.get_map().name,
        spawn_index=spawn_index,
        fps=fps,
        lane_change=False,
        traffic=traffic_n,
    )
    # Extra dagger provenance (meta.json already written; append fields).
    extra = {
        **__import__("json").loads((run_dir / "meta.json").read_text()),
        "dagger": True,
        "parent_run": str(from_run),
        "policy_model": str(model_path),
        "enter_cte_m": enter_cte_m,
        "exit_cte_m": exit_cte_m,
    }
    (run_dir / "meta.json").write_text(
        __import__("json").dumps(extra, indent=2) + "\n"
    )

    actors: list[carla.Actor] = []
    cam_queues: dict[str, queue.Queue] = {
        name: queue.Queue() for name in camera_names
    }
    recorded_rows: list[dict[str, str]] = []
    follower = WaypointFollower(
        cruise_kmh=24.0,
        turn_kmh=16.0,
        min_lookahead_m=3.5,
        max_lookahead_m=9.0,
        steer_gain=1.65,
        max_steer_rate=0.22,
    )
    min_intervene_frames = max(1, int(min_intervene_s * fps))
    intervening = False
    intervene_left = 0
    n_intervene_events = 0
    n_saved = 0
    sticky_turn: str | None = None
    sticky_turn_start: int | None = None
    route_index = 0
    red_wait_frames = 0
    max_red_wait_frames = (
        int(max_red_wait_s * fps) if max_red_wait_s > 0 else 0
    )
    stop_actors = list(world.get_actors().filter("traffic.stop"))
    frame_buf: deque[np.ndarray] = deque(maxlen=stack_frames)
    prev_steer = 0.0

    try:
        with SynchronousMode(client, world, fps=fps, tm_port=tm_port) as (
            world,
            _tm,
        ):
            vehicle = spawn_vehicle(world, spawn_index=spawn_index)
            actors.append(vehicle)
            npcs: list = []
            if traffic_n > 0:
                npcs = spawn_npc_traffic(
                    world,
                    _tm,
                    count=traffic_n,
                    tm_port=tm_port,
                    ego=vehicle,
                    exclude_spawn_indices={spawn_index},
                )
                actors.extend(npcs)
                print(
                    f"Traffic ON: spawned {len(npcs)}/{traffic_n} NPC vehicles"
                )
                for _ in range(max(1, int(fps))):
                    world.tick()
            else:
                n_clear = clear_other_vehicles(world, ego=vehicle)
                if n_clear:
                    print(f"Cleared {n_clear} leftover vehicle(s)")
                    world.tick()
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
            radar_queue: queue.Queue = queue.Queue()
            radar_on = try_attach_radar(world, vehicle, actors, radar_queue)
            radar_range_m: float | None = None

            if fast_lights:
                n_lights = speed_up_traffic_lights(world)
                print(f"Fast lights ON ({n_lights} poles)")

            world.tick()
            _get_camera_images(world, cam_queues, camera_names, timeout=10.0)
            print(
                f"DAgger recording corrections → {run_dir} for {seconds:.0f}s "
                "(empty road; Ctrl+C to stop)"
            )

            frames = int(seconds * fps)
            line_life = max(1.0 / fps * 2.5, 0.12)
            with controls_path.open("w", newline="") as fh:
                writer = csv.DictWriter(
                    fh, fieldnames=CONTROLS_HEADER, extrasaction="ignore"
                )
                writer.writeheader()

                for frame_idx in range(frames):
                    world.tick()
                    if radar_on:
                        radar_range_m = pop_radar_range(radar_queue, radar_range_m)
                    follow_vehicle(world, vehicle)
                    loc = vehicle.get_location()
                    yaw = vehicle.get_transform().rotation.yaw

                    if needs_replan(loc, route, route_index):
                        dest, route_new, _score, stats = pick_next_destination(
                            grp,
                            loc,
                            spawn_pts,
                            route_mode="balanced",
                            heading_yaw=yaw,
                            world_map=world.get_map(),
                        )
                        align = float(stats.get("forward_align", 0.0))
                        if align >= 0.15:
                            route = route_new
                            route_index = 0
                            follower.reset()
                            sticky_turn = None
                            sticky_turn_start = None
                            print(
                                f"  replan @ frame {frame_idx}: "
                                f"→{stats.get('dest')} align={align:+.2f}"
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

                    expert_ctrl, target, _cmd = follower.control(vehicle, route)
                    route_index = follower.route_index
                    red_wait_frames, forced = maybe_force_ego_light_green(
                        vehicle,
                        red_wait_frames=red_wait_frames,
                        max_wait_frames=max_red_wait_frames,
                    )
                    if forced:
                        print(
                            f"  force light GREEN (siblings → red) "
                            f"@ frame {frame_idx}"
                        )
                    expert_ctrl = apply_traffic_light_hold(expert_ctrl, vehicle)
                    lead_gap = None
                    if traffic_n > 0:
                        expert_ctrl = apply_lead_vehicle_hold(
                            expert_ctrl, world, vehicle
                        )
                        lead_gap = lead_vehicle_gap_m(
                            world, vehicle, max_dist_m=16.0
                        )

                    # --- policy inference (for drive + disagreement) ---
                    images = _get_camera_images(
                        world,
                        cam_queues,
                        camera_names,
                        timeout=5.0,
                        retries=3,
                        tick_on_retry=False,
                    )
                    center_rgb = image_to_array(images["center"])
                    rgb_f = center_rgb.astype(np.float32) / 255.0
                    frame_buf.append(rgb_f)
                    while len(frame_buf) < stack_frames:
                        frame_buf.appendleft(rgb_f)
                    stacked = np.concatenate(list(frame_buf), axis=2)
                    image_tensor = (
                        torch.from_numpy(stacked)
                        .permute(2, 0, 1)
                        .unsqueeze(0)
                        .to(device)
                    )
                    speed = speed_kmh(vehicle)
                    speed_tensor = torch.tensor(
                        [[normalize_speed(speed)]],
                        dtype=torch.float32,
                        device=device,
                    )
                    command_tensor = torch.tensor(
                        [command_to_onehot(command_name)],
                        dtype=torch.float32,
                        device=device,
                    )
                    at_light, light_state = traffic_light_fields(vehicle)
                    light_tensor = None
                    if use_light:
                        light_tensor = torch.tensor(
                            [
                                light_to_onehot(
                                    normalize_light_state(light_state)
                                )
                            ],
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
                        if use_light:
                            out_t = model(
                                image_tensor,
                                speed_tensor,
                                command_tensor,
                                light_tensor,
                                radar_tensor,
                            )
                        else:
                            out_t = model(
                                image_tensor,
                                speed_tensor,
                                command_tensor,
                                None,
                                radar_tensor,
                            )
                    m_steer, m_thr, m_brk = exclusive_controls(
                        *out_t[0].tolist()
                    )
                    # Mild rate limit so intervene edges aren't extra violent.
                    max_d = 0.08
                    m_steer = max(
                        prev_steer - max_d,
                        min(prev_steer + max_d, m_steer),
                    )

                    lane = lane_metrics_at(loc, yaw, route, route_index)
                    abs_cte = abs(lane["cte_m"])
                    abs_frac = abs(lane["lane_offset_frac"])
                    d_steer = abs(float(expert_ctrl.steer) - float(m_steer))
                    d_brake = float(expert_ctrl.brake) - float(m_brk)
                    want_enter = (
                        abs_cte >= enter_cte_m
                        or abs_frac >= enter_frac
                        or (
                            d_steer >= disagree_steer
                            and abs_cte >= exit_cte_m
                        )
                    )
                    # Policy is about to drive into the lead. Expert brake
                    # label is the correction; steer stays the lane command.
                    want_lead = (
                        lead_gap is not None
                        and lead_gap <= 14.0
                        and d_brake >= 0.20
                    )
                    lead_blocking = (
                        lead_gap is not None
                        and lead_gap <= 14.0
                        and float(expert_ctrl.brake) >= 0.15
                    )
                    if intervening:
                        intervene_left -= 1
                        lane_clear = (
                            abs_cte < exit_cte_m
                            and abs_frac < enter_frac * 0.85
                        )
                        if intervene_left <= 0 and lane_clear and not lead_blocking:
                            intervening = False
                            gap_bit = (
                                f"  lead={lead_gap:.1f}m"
                                if lead_gap is not None
                                else ""
                            )
                            print(
                                f"  hand back to policy @ frame {frame_idx} "
                                f"cte={lane['cte_m']:+.2f}m{gap_bit}"
                            )
                    elif want_lead or (want_enter and speed > 3.0):
                        intervening = True
                        intervene_left = min_intervene_frames
                        n_intervene_events += 1
                        why = "lead" if want_lead else "lane"
                        gap_bit = (
                            f"  lead={lead_gap:.1f}m Δbrake={d_brake:.2f}"
                            if want_lead and lead_gap is not None
                            else ""
                        )
                        print(
                            f"  INTERVENE #{n_intervene_events} ({why}) @ frame "
                            f"{frame_idx}  cte={lane['cte_m']:+.2f}m  "
                            f"Δsteer={d_steer:.2f}  cmd={command_name}{gap_bit}"
                        )

                    if intervening:
                        applied = expert_ctrl
                        log_steer = float(expert_ctrl.steer)
                        log_thr = float(expert_ctrl.throttle)
                        log_brk = float(expert_ctrl.brake)
                        save_frame = True
                        prev_steer = log_steer
                    else:
                        applied = carla.VehicleControl(
                            throttle=float(m_thr),
                            steer=float(m_steer),
                            brake=float(m_brk),
                        )
                        log_steer = float(m_steer)
                        log_thr = float(m_thr)
                        log_brk = float(m_brk)
                        save_frame = False
                        prev_steer = m_steer

                    vehicle.apply_control(applied)

                    if draw_path and route:
                        draw_route(
                            world,
                            route,
                            life_time=line_life,
                            start_index=max(0, route_index - 2),
                        )
                        draw_target(world, target, life_time=line_life)

                    if save_frame:
                        stop_flag = near_stop_sign(
                            world, vehicle, stop_actors=stop_actors
                        )
                        lane_fields = format_lane_metric_fields(lane)
                        for cam_name, image in images.items():
                            rgb = image_to_array(image)
                            Image.fromarray(rgb).save(
                                image_dirs[cam_name] / f"{frame_idx:06d}.png"
                            )
                            steer = recovery_steer(
                                cam_name, log_steer, command=command_name
                            )
                            row = {
                                "frame": str(frame_idx),
                                "camera": cam_name,
                                "steer": f"{steer:.6f}",
                                "throttle": f"{log_thr:.6f}",
                                "brake": f"{log_brk:.6f}",
                                "speed_kmh": f"{speed:.3f}",
                                "command": command_name,
                                "at_traffic_light": str(at_light),
                                "light_state": light_state,
                                "near_stop_sign": str(stop_flag),
                                "radar_m": format_radar_m(radar_range_m),
                                **lane_fields,
                            }
                            writer.writerow(row)
                            recorded_rows.append(row)
                        n_saved += 1

                    if frame_idx > 0 and frame_idx % max(1, int(fps)) == 0 and npcs:
                        npcs, n_wreck = destroy_wrecked_vehicles(
                            world, ego=vehicle, tracked=npcs
                        )
                        if n_wreck:
                            prune_actor_list(actors, ego=vehicle, npcs=npcs)
                            print(f"  removed {n_wreck} wrecked NPC(s)")
                    if frame_idx % max(1, int(fps)) == 0:
                        mode = "EXPERT" if intervening else "policy"
                        gap_bit = (
                            f"  lead={lead_gap:.0f}m"
                            if lead_gap is not None
                            else ""
                        )
                        print(
                            f"  t={frame_idx / fps:5.1f}s  {mode}  "
                            f"speed={speed:4.1f}  "
                            f"steer={float(applied.steer):+.2f}  "
                            f"brake={float(applied.brake):.2f}  "
                            f"cte={lane['cte_m']:+.2f}m  "
                            f"radar={format_radar_status(radar_range_m)}"
                            f"{gap_bit}  "
                            f"cmd={command_name}  saved={n_saved}"
                        )
    except KeyboardInterrupt:
        print("\nStopped early.")
    finally:
        _cleanup_actors(world, actors, allow_tick=False)

    if recorded_rows:
        write_command_distribution(run_dir, recorded_rows)
        write_balanced_controls(run_dir, recorded_rows)
        center = [r for r in recorded_rows if r.get("camera") == "center"]
        print(
            f"DAgger done: {n_intervene_events} interventions, "
            f"{len(center)} center correction frames → {run_dir}"
        )
        print(
            "Retrain mix (DAgger frames auto-boost ×4 when present):\n"
            f"  python -m expert.train --data {from_run} {run_dir} "
            f"--center-only --radar --stack-frames 3 --device cuda"
        )
    else:
        print(
            f"No corrections saved (policy stayed inside thresholds). "
            f"Empty run dir: {run_dir}"
        )
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--from-run",
        type=Path,
        required=True,
        help="Parent KEEP run (map/spawn); also mixed back in at train time",
    )
    parser.add_argument("--model", type=Path, default=Path("models/bc.pt"))
    parser.add_argument("--seconds", type=float, default=300.0)
    parser.add_argument("--device", choices=["cuda", "cpu", "auto"], default="cuda")
    parser.add_argument("--cameras", default="center,left,right")
    parser.add_argument("--enter-cte-m", type=float, default=0.55)
    parser.add_argument("--exit-cte-m", type=float, default=0.28)
    parser.add_argument("--enter-frac", type=float, default=0.70)
    parser.add_argument("--disagree-steer", type=float, default=0.22)
    parser.add_argument("--min-intervene-s", type=float, default=1.2)
    parser.add_argument(
        "--no-draw-path",
        action="store_true",
        help="Hide route HUD overlay (on by default; spectator only, not in RGB)",
    )
    parser.add_argument("--draw-path", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--no-fast-lights", action="store_true")
    parser.add_argument("--max-red-wait-s", type=float, default=6.0)
    parser.add_argument("--out", type=Path, default=Path("data"))
    parser.add_argument("--tm-port", type=int, default=8000)
    parser.add_argument(
        "--traffic",
        type=int,
        default=15,
        help="NPC vehicles (default 15). Expert brakes for the lead car.",
    )
    parser.add_argument(
        "--no-traffic",
        action="store_true",
        help="Empty road. Lane-centering corrections only.",
    )
    args = parser.parse_args()

    dagger_run(
        from_run=args.from_run,
        model_path=args.model,
        seconds=args.seconds,
        device_name=args.device,
        cameras=parse_camera_names(args.cameras),
        enter_cte_m=args.enter_cte_m,
        exit_cte_m=args.exit_cte_m,
        enter_frac=args.enter_frac,
        disagree_steer=args.disagree_steer,
        min_intervene_s=args.min_intervene_s,
        draw_path=not args.no_draw_path,
        fast_lights=not args.no_fast_lights,
        max_red_wait_s=args.max_red_wait_s,
        out=args.out,
        tm_port=args.tm_port,
        traffic=0 if args.no_traffic else args.traffic,
    )


if __name__ == "__main__":
    main()
