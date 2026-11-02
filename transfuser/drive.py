"""
Closed-loop Town01 drive with TransFuser-lite and the Step 1 scorecard.

The net sees the RGB strip, the lidar BEV, speed, and two target points on
the planned route (TARGET_M and TARGET_FAR_M ahead), also drawn into the BEV
as goal channels. It outputs waypoints. The PID in
transfuser/controller.py turns them into controls. No expert, no map
lane-keeping, no light or lead-car override.

Examples:
  python -m transfuser.drive --model models/transfuser.pt --seconds 300
  python -m transfuser.drive --model models/transfuser.pt --traffic 30 --spawn-index 42
  python -m transfuser.drive --model models/transfuser.pt --traffic 30 --walkers 40
"""
from __future__ import annotations

import argparse
import csv
import json
import queue
from datetime import datetime
from pathlib import Path

import carla
import numpy as np
import torch
from PIL import Image

from perception.lidar import attach_semantic_lidar, points_from_measurement
from transfuser.controller import WaypointController
from transfuser.data import CAMERAS, TARGET_FAR_M, TARGET_M, image_strip, lidar_input
from transfuser.model import load_checkpoint
from utils.connection import connect_client
from utils.device import resolve_device
from utils.drive_score import (
    EventCounter,
    collisions_per_km,
    path_length_m,
    route_completion_pct,
)
from utils.images import image_to_array
from utils.lane_metrics import format_lane_metric_fields, lane_metrics_at
from utils.routing import (
    build_planner,
    draw_route,
    draw_target,
    lookahead_target,
    needs_replan,
    pick_junction_heavy_destination,
    route_xyz,
)
from utils.run_meta import ensure_map
from utils.spectator import follow_vehicle
from utils.state import speed_kmh
from utils.sync_mode import SynchronousMode
from utils.traffic import _safe_destroy, clear_other_vehicles, spawn_npc_traffic
from utils.traffic_signs import speed_up_traffic_lights, traffic_light_fields
from utils.vehicles import attach_rgb_cameras, spawn_vehicle
from utils.walkers import destroy_walkers, spawn_walkers
from utils.weather import apply_weather

_LOG_HEADER = [
    "frame",
    "steer",
    "throttle",
    "brake",
    "speed_kmh",
    "desired_kmh",
    "target_x",
    "target_y",
    "wp_x",
    "wp_y",
    "light_truth",
    "red_light_p",
    "brake_p",
    "cte_m",
    "lane_offset_frac",
    "heading_err_deg",
    "lane_invasions",
    "collisions",
]


def _cleanup(actors: list[carla.Actor]) -> None:
    sensors = [a for a in actors if isinstance(a, carla.Sensor)]
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
    for actor in actors:
        if not isinstance(actor, carla.Sensor):
            _safe_destroy(actor)


def _for_frame(q: queue.Queue, frame: int, timeout: float = 5.0):
    """Sensor data for this sim frame, dropping stale backlog."""
    while True:
        data = q.get(timeout=timeout)
        if data.frame >= frame:
            return data


def _ego_xy(transform: carla.Transform, target: carla.Location) -> np.ndarray:
    """World location → ego frame (x forward, y right)."""
    fwd = transform.get_forward_vector()
    right = transform.get_right_vector()
    dx = target.x - transform.location.x
    dy = target.y - transform.location.y
    return np.array([dx * fwd.x + dy * fwd.y, dx * right.x + dy * right.y], dtype=np.float32)


def transfuser_drive(
    *,
    model_path: Path,
    seconds: float = 300.0,
    fps: float = 20.0,
    map_name: str = "Town01",
    spawn_index: int = 0,
    traffic: int = 0,
    walkers: int = 0,
    fast_lights: bool = True,
    device_name: str = "auto",
    out: Path | None = None,
    draw: bool = True,
    save_stops: bool = False,
    weather: str = "ClearNoon",
) -> Path:
    device = resolve_device(device_name)
    model, ckpt = load_checkpoint(model_path, device)
    print(f"Loaded {model_path} (modality={ckpt['modality']}, aux={ckpt['aux']}, epoch={ckpt.get('epoch')})")

    client = connect_client(timeout=30.0)
    ensure_map(client, map_name)
    world = client.get_world()
    print(f"Weather: {apply_weather(world, weather)}")
    world_map = world.get_map()
    spawn_pts = world_map.get_spawn_points()
    print("Building GRP ...")
    grp = build_planner(world_map)
    dest, route, score = pick_junction_heavy_destination(grp, spawn_pts[spawn_index].location, spawn_pts)
    route_pts = route_xyz(route)
    planned_m = path_length_m(route_pts, 0, len(route_pts) - 1)
    print(f"Route: {len(route)} wps, junction={score:.1%}, len={planned_m:.0f}m")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(out) if out else Path(f"data/transfuser_drive_{stamp}")
    out_dir.mkdir(parents=True, exist_ok=True)
    line_life = max(2.5 / fps, 0.12)

    actors: list[carla.Actor] = []
    rows: list[dict[str, str]] = []
    events = EventCounter()
    hit_types: list[str] = []
    ped_walkers: list[carla.Actor] = []
    ped_controllers: list[carla.Actor] = []
    controller = WaypointController()
    odom_m = 0.0
    prev_loc: carla.Location | None = None
    route_index = 0
    max_route_index = 0
    credit_m = 0.0
    stopped_s = 0.0
    stop_frames = 0
    if save_stops:
        (out_dir / "stops").mkdir(exist_ok=True)

    try:
        with SynchronousMode(client, world, fps=fps) as (world, tm):
            clear_other_vehicles(world)
            if fast_lights:
                speed_up_traffic_lights(world)
            vehicle = spawn_vehicle(world, spawn_index=spawn_index)
            actors.append(vehicle)

            def on_collision(event) -> None:
                before, _ = events.snapshot()
                events.on_collision(event)
                if events.snapshot()[0] > before:
                    hit_types.append(event.other_actor.type_id)

            library = world.get_blueprint_library()
            for bp_id, cb in (
                ("sensor.other.collision", on_collision),
                ("sensor.other.lane_invasion", events.on_lane_invasion),
            ):
                sensor = world.spawn_actor(library.find(bp_id), carla.Transform(), attach_to=vehicle)
                sensor.listen(cb)
                actors.append(sensor)

            cams = attach_rgb_cameras(world, vehicle, names=CAMERAS)
            cam_q: dict[str, queue.Queue] = {}
            for name, cam in cams.items():
                actors.append(cam)
                cam_q[name] = queue.Queue()
                cam.listen(cam_q[name].put)
            lidar = attach_semantic_lidar(world, vehicle, fps=fps)
            actors.append(lidar)
            lidar_q: queue.Queue = queue.Queue()
            lidar.listen(lidar_q.put)

            if traffic > 0:
                npcs = spawn_npc_traffic(world, tm, count=traffic, ego=vehicle)
                actors.extend(npcs)
                print(f"Spawned {len(npcs)} NPC vehicles")
            if walkers > 0:
                ped_walkers, ped_controllers = spawn_walkers(client, world, count=walkers)
                print(f"Spawned {len(ped_walkers)} pedestrians")

            frames = int(seconds * fps)
            print(f"TransFuser drive {seconds:.0f}s @ {fps:.0f} FPS")
            for frame_idx in range(frames):
                sim_frame = world.tick()
                follow_vehicle(world, vehicle)
                transform = vehicle.get_transform()
                loc = transform.location

                # Replan early enough that the far target stays on the route.
                if needs_replan(loc, route, route_index, threshold_m=TARGET_FAR_M + 5.0):
                    credit_m += path_length_m(route_pts, 0, max_route_index)
                    dest, route, score = pick_junction_heavy_destination(grp, loc, spawn_pts)
                    route_pts = route_xyz(route)
                    planned_m += path_length_m(route_pts, 0, len(route_pts) - 1)
                    route_index = max_route_index = 0
                    print(f"  replan @ {frame_idx}: {len(route)} wps junction={score:.1%}")

                target_loc, route_index = lookahead_target(
                    loc, route, start_hint=route_index, lookahead_m=TARGET_M
                )
                max_route_index = max(max_route_index, route_index)
                far_loc, _ = lookahead_target(
                    loc, route, start_hint=route_index, lookahead_m=TARGET_FAR_M
                )
                target = np.concatenate([_ego_xy(transform, target_loc), _ego_xy(transform, far_loc)])

                rgb = [image_to_array(_for_frame(cam_q[c], sim_frame)) for c in CAMERAS]
                points = points_from_measurement(_for_frame(lidar_q, sim_frame))
                speed_mps = speed_kmh(vehicle) / 3.6

                with torch.no_grad():
                    pred = model(
                        torch.from_numpy(image_strip(rgb))[None].to(device) if model.use_camera else None,
                        torch.from_numpy(lidar_input(points, target))[None].to(device) if model.use_lidar else None,
                        torch.tensor([[speed_mps]], dtype=torch.float32, device=device),
                        torch.from_numpy(target)[None].to(device),
                    )
                wps = pred["waypoints"][0].cpu().numpy()
                aff = torch.sigmoid(pred["affordance"][0]).cpu().numpy() if "affordance" in pred else None

                steer, throttle, brake, dbg = controller.step(wps, speed_mps)
                vehicle.apply_control(carla.VehicleControl(steer=steer, throttle=throttle, brake=brake))

                if prev_loc is not None:
                    step_m = prev_loc.distance(loc)
                    if step_m < 8.0:
                        odom_m += step_m
                prev_loc = loc
                stopped_s += (1.0 / fps) if speed_mps < 0.5 else 0.0
                stop_frames = stop_frames + 1 if speed_mps < 0.5 else 0
                _, light_truth = traffic_light_fields(vehicle)
                # Every 2 s of a stop: what the net sees, for debugging stalls.
                if save_stops and stop_frames % int(2 * fps) == 1:
                    name = f"{frame_idx:06d}_{light_truth}.png"
                    Image.fromarray(np.concatenate(rgb, axis=1)).save(out_dir / "stops" / name)
                n_col, n_inv = events.snapshot()
                lane = lane_metrics_at(loc, transform.rotation.yaw, route, route_index)
                rows.append(
                    {
                        "frame": str(frame_idx),
                        "steer": f"{steer:.4f}",
                        "throttle": f"{throttle:.4f}",
                        "brake": f"{brake:.1f}",
                        "speed_kmh": f"{speed_mps * 3.6:.3f}",
                        "desired_kmh": f"{dbg['desired_mps'] * 3.6:.3f}",
                        "target_x": f"{target[0]:.2f}",
                        "target_y": f"{target[1]:.2f}",
                        "wp_x": " ".join(f"{v:.2f}" for v in wps[:, 0]),
                        "wp_y": " ".join(f"{v:.2f}" for v in wps[:, 1]),
                        "light_truth": light_truth,
                        "red_light_p": "" if aff is None else f"{aff[0]:.3f}",
                        "brake_p": "" if aff is None else f"{aff[1]:.3f}",
                        **format_lane_metric_fields(lane),
                        "lane_invasions": str(n_inv),
                        "collisions": str(n_col),
                    }
                )

                if draw:
                    draw_route(world, route, life_time=line_life, start_index=max(0, route_index - 2))
                    draw_target(world, target_loc, life_time=line_life)
                    for x, y in wps:
                        p = transform.transform(carla.Location(x=float(x), y=float(y), z=0.5))
                        world.debug.draw_point(p, size=0.1, color=carla.Color(255, 40, 40), life_time=line_life)
                if frame_idx % int(fps) == 0:
                    print(
                        f"  t={frame_idx / fps:5.1f}s  speed={speed_mps * 3.6:4.1f}  "
                        f"want={dbg['desired_mps'] * 3.6:4.1f}  steer={steer:+.2f}  "
                        f"tp=({target[0]:+5.1f},{target[1]:+5.1f})  "
                        f"|cte|={abs(lane['cte_m']):.2f}  inv={n_inv} col={n_col}"
                        + ("" if aff is None else f"  red={aff[0]:.2f}")
                        + f"  light={light_truth}"
                    )
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        destroy_walkers(ped_controllers, ped_walkers)
        _cleanup(actors)

    progress_m = credit_m + path_length_m(route_pts, 0, max_route_index)
    # Read the log, not the counters: after sync mode ends the car rolls on with
    # its last throttle, and those hits are not part of the drive.
    n_col = int(rows[-1]["collisions"]) if rows else 0
    n_inv = int(rows[-1]["lane_invasions"]) if rows else 0
    hit_types = hit_types[:n_col]
    ctes = sorted(abs(float(r["cte_m"])) for r in rows)
    metrics = {
        "agent": "transfuser",
        "model": str(model_path),
        "modality": ckpt["modality"],
        "aux": ckpt["aux"],
        "lane_invasions": n_inv,
        "collisions": n_col,
        "hit_types": hit_types,
        "distance_m": round(odom_m, 2),
        "route_progress_m": round(progress_m, 2),
        "route_length_m": round(planned_m, 2),
        "route_completion_pct": round(route_completion_pct(progress_m, planned_m), 2),
        "avg_speed_kmh": round(sum(float(r["speed_kmh"]) for r in rows) / max(1, len(rows)), 2),
        "stopped_pct": round(100.0 * stopped_s / max(1e-6, len(rows) / fps), 1),
        "collisions_per_km": round(collisions_per_km(n_col, odom_m), 3),
        "cte_p50_m": round(ctes[len(ctes) // 2], 3) if ctes else None,
        "cte_p90_m": round(ctes[int(0.9 * (len(ctes) - 1))], 3) if ctes else None,
        "traffic": traffic,
        "walkers": walkers,
        "weather": weather,
        "spawn_index": spawn_index,
        "map": world_map.name,
    }
    metrics["milestone"] = "PASS" if metrics["collisions_per_km"] <= 1.0 else "FAIL"

    with (out_dir / "drive_log.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_LOG_HEADER)
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "drive_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    print(f"Wrote {out_dir}")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, default=Path("models/transfuser.pt"))
    parser.add_argument("--seconds", type=float, default=300.0)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--map", dest="map_name", default="Town01")
    parser.add_argument("--spawn-index", type=int, default=0)
    parser.add_argument("--traffic", type=int, default=0)
    parser.add_argument("--walkers", type=int, default=0, help="Spawn N AI pedestrians")
    parser.add_argument("--no-fast-lights", action="store_true", help="Keep CARLA's default light timings")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--no-draw", action="store_true")
    parser.add_argument("--save-stops", action="store_true", help="Save the camera strip every 2 s while stopped")
    parser.add_argument("--weather", default="ClearNoon", help="Preset from utils/weather.py")
    args = parser.parse_args()
    transfuser_drive(
        model_path=args.model,
        seconds=args.seconds,
        fps=args.fps,
        map_name=args.map_name,
        spawn_index=args.spawn_index,
        traffic=args.traffic,
        walkers=args.walkers,
        fast_lights=not args.no_fast_lights,
        device_name=args.device,
        out=args.out,
        draw=not args.no_draw,
        save_stops=args.save_stops,
        weather=args.weather,
    )


if __name__ == "__main__":
    main()
