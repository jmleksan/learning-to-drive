"""
Classical modular Town01 drive with the Step 1 scorecard.

No neural net for planning/control. Optional live EKF pose + SegNet/Occupancy
perception. Defaults to CARLA actor GT for other agents (oracle), like BasicAgent.

Examples:
  python -m pipeline.drive --seconds 120 --controller hybrid --debug-decisions
  python -m pipeline.drive --traffic 15 --perceive --debug-decisions
  python -m pipeline.drive --ekf --plan-from-ekf --debug-decisions
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import queue
from datetime import datetime
from pathlib import Path

import carla

from localization.ekf import EgoEKF
from localization.geo import geo_to_enu
from localization.pose_enu import location_to_enu, origin_from_location, yaw_enu_rad
from localization.sensors import (
    attach_gnss,
    attach_imu,
    pop_gnss,
    pop_imu,
)
from pipeline.agent import ClassicalAgent
from pipeline.perceive import LivePerception, enu_to_world_location
from pipeline.viz import draw_decisions, draw_ekf_pose
from utils.connection import connect_client
from utils.drive_score import (
    EventCounter,
    collisions_per_km,
    path_length_m,
    route_completion_pct,
)
from utils.lane_metrics import format_lane_metric_fields, lane_metrics_at
from utils.routing import (
    build_planner,
    draw_route,
    draw_target,
    needs_replan,
    pick_junction_heavy_destination,
    route_xyz,
)
from utils.run_meta import apply_run_meta, ensure_map, load_run_meta
from utils.spectator import follow_vehicle
from utils.state import speed_kmh
from utils.sync_mode import SynchronousMode
from utils.traffic import (
    _safe_destroy,
    clear_other_vehicles,
    spawn_npc_traffic,
)
from utils.vehicles import spawn_vehicle

_DRIVE_LOG_HEADER = [
    "frame",
    "steer",
    "throttle",
    "brake",
    "speed_kmh",
    "command",
    "behavior",
    "reason",
    "light",
    "sense",
    "cte_m",
    "lane_offset_frac",
    "heading_err_deg",
    "lane_invasions",
    "collisions",
    "lead_m",
    "n_pred",
    "ekf_e",
    "ekf_n",
]


def _cleanup(actors: list[carla.Actor]) -> None:
    sensors = [a for a in actors if isinstance(a, carla.Sensor)]
    others = [a for a in actors if not isinstance(a, carla.Sensor)]
    for sensor in sensors:
        try:
            sensor.stop()
        except Exception:
            pass
        try:
            sensor.destroy()
        except Exception:
            pass
    for actor in others:
        _safe_destroy(actor)


def classical_drive(
    *,
    seconds: float = 120.0,
    fps: float = 20.0,
    map_name: str | None = "Town01",
    spawn_index: int | None = None,
    from_run: Path | None = None,
    controller: str = "hybrid",
    cruise_kmh: float = 28.0,
    turn_kmh: float = 16.0,
    traffic: int = 0,
    ekf: bool = False,
    plan_from_ekf: bool = False,
    perceive: bool = False,
    seg_model: Path | None = None,
    occupancy_model: Path | None = None,
    device: str = "auto",
    debug_decisions: bool = False,
    out: Path | None = None,
    draw: bool = True,
    use_frenet: bool = True,
) -> Path:
    client = connect_client(timeout=30.0)
    drive_fps = fps

    if from_run is not None:
        meta = load_run_meta(from_run)
        world = apply_run_meta(client, meta)
        spawn_index = meta["spawn_index"]
        drive_fps = float(meta.get("fps", drive_fps))
    elif map_name is not None:
        world = ensure_map(client, map_name)
    else:
        world = client.get_world()

    world = client.get_world()
    world_map = world.get_map()
    spawn_pts = world_map.get_spawn_points()
    if spawn_index is not None:
        if not 0 <= spawn_index < len(spawn_pts):
            raise ValueError(f"spawn_index out of range: {spawn_index}")
        origin = spawn_pts[spawn_index].location
    else:
        spawn_index = 0
        origin = spawn_pts[0].location

    print("Building GRP ...")
    grp = build_planner(world_map)
    dest, route, score = pick_junction_heavy_destination(grp, origin, spawn_pts)
    route_pts = route_xyz(route)
    planned_m = path_length_m(route_pts, 0, len(route_pts) - 1)
    print(
        f"Route: {len(route)} wps, junction={score:.1%}, "
        f"len={planned_m:.0f}m, controller={controller}"
    )

    agent = ClassicalAgent(
        controller=controller,
        cruise_kmh=cruise_kmh,
        turn_kmh=turn_kmh,
        use_frenet=use_frenet,
    )
    actors: list[carla.Actor] = []
    npcs: list[carla.Actor] = []
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(out) if out else Path(f"data/classical_drive_{stamp}")
    out_dir.mkdir(parents=True, exist_ok=True)
    line_life = max(1.0 / drive_fps * 2.5, 0.12)

    gnss_q: queue.Queue | None = None
    imu_q: queue.Queue | None = None
    ekf_filter: EgoEKF | None = None
    geo_origin = None
    live_perc: LivePerception | None = None

    if plan_from_ekf and not ekf:
        raise ValueError("--plan-from-ekf requires --ekf")

    drive_rows: list[dict[str, str]] = []
    events = EventCounter()
    odom_m = 0.0
    prev_odom: carla.Location | None = None
    max_route_index = 0
    credit_m = 0.0

    try:
        with SynchronousMode(client, world, fps=drive_fps) as (world, tm):
            clear_other_vehicles(world)
            vehicle = spawn_vehicle(world, spawn_index=spawn_index)
            actors.append(vehicle)

            col_bp = world.get_blueprint_library().find("sensor.other.collision")
            inv_bp = world.get_blueprint_library().find("sensor.other.lane_invasion")
            col_sensor = world.spawn_actor(
                col_bp, carla.Transform(), attach_to=vehicle
            )
            inv_sensor = world.spawn_actor(
                inv_bp, carla.Transform(), attach_to=vehicle
            )
            actors.extend([col_sensor, inv_sensor])
            col_sensor.listen(lambda e: events.on_collision(e))
            inv_sensor.listen(lambda e: events.on_lane_invasion(e))

            if traffic > 0:
                npcs = spawn_npc_traffic(
                    world, tm, count=traffic, ego=vehicle
                )
                actors.extend(npcs)
                print(f"Spawned {len(npcs)} NPC vehicles")

            if perceive:
                seg_path = Path(seg_model or "models/seg.pt")
                occ_path = Path(occupancy_model or "models/occupancy.pt")
                if not seg_path.is_file() or not occ_path.is_file():
                    raise FileNotFoundError(
                        f"--perceive needs {seg_path} and {occ_path}"
                    )
                live_perc = LivePerception.attach(
                    world,
                    vehicle,
                    actors,
                    seg_path=seg_path,
                    occ_path=occ_path,
                    fps=drive_fps,
                    device=device,
                )

            if ekf:
                import numpy as np

                gnss_q = queue.Queue()
                imu_q = queue.Queue()
                gnss = attach_gnss(world, vehicle, fps=drive_fps)
                imu = attach_imu(world, vehicle, fps=drive_fps)
                actors.extend([gnss, imu])
                gnss.listen(gnss_q.put)
                imu.listen(imu_q.put)
                tf0 = vehicle.get_transform()
                geo_origin = origin_from_location(world_map, tf0.location)
                e0, n0, _ = location_to_enu(world_map, tf0.location, geo_origin)
                yaw0 = yaw_enu_rad(world_map, tf0, geo_origin)
                ekf_filter = EgoEKF(x0=np.array([e0, n0, yaw0, 0.0]))
                print(
                    "EKF enabled (GNSS+IMU)"
                    + (" — planning from EKF pose" if plan_from_ekf else "")
                )

            frames = int(seconds * drive_fps)
            sense_label = "nets" if perceive else "oracle"
            print(
                f"Classical drive {seconds:.0f}s @ {drive_fps:.0f} FPS "
                f"(controller={controller}, sense={sense_label}, "
                f"debug={debug_decisions})"
            )
            dt = 1.0 / drive_fps
            for frame_idx in range(frames):
                world.tick()
                follow_vehicle(world, vehicle)
                loc = vehicle.get_location()
                yaw = vehicle.get_transform().rotation.yaw

                ekf_e = ekf_n = float("nan")
                ekf_loc: carla.Location | None = None
                if ekf_filter is not None and gnss_q is not None and imu_q is not None:
                    imu_meas = pop_imu(imu_q)
                    gnss_meas = pop_gnss(gnss_q)
                    yaw_rate = 0.0
                    if imu_meas is not None:
                        yaw_rate = -float(imu_meas.gyroscope.z)
                    speed = speed_kmh(vehicle) / 3.6
                    ekf_filter.predict(dt, yaw_rate=yaw_rate, speed=speed)
                    if gnss_meas is not None and geo_origin is not None:
                        e_m, n_m, _ = geo_to_enu(
                            float(gnss_meas.latitude),
                            float(gnss_meas.longitude),
                            float(gnss_meas.altitude),
                            geo_origin,
                        )
                        ekf_filter.update_gnss(e_m, n_m)
                    if imu_meas is not None:
                        ekf_filter.update_compass(float(imu_meas.compass))
                    ekf_e, ekf_n = ekf_filter.e, ekf_filter.n
                    if geo_origin is not None:
                        ekf_loc = enu_to_world_location(
                            world_map,
                            ekf_filter.e,
                            ekf_filter.n,
                            0.0,
                            geo_origin,
                            loc,
                        )

                plan_loc = (
                    ekf_loc if (plan_from_ekf and ekf_loc is not None) else loc
                )

                if needs_replan(plan_loc, route, agent.route_index):
                    if route_pts:
                        credit_m += path_length_m(route_pts, 0, max_route_index)
                    dest, route, score = pick_junction_heavy_destination(
                        grp, plan_loc, spawn_pts
                    )
                    route_pts = route_xyz(route)
                    planned_m += path_length_m(route_pts, 0, len(route_pts) - 1)
                    agent.reset()
                    if live_perc is not None:
                        live_perc.reset()
                    max_route_index = 0
                    print(
                        f"  replan @ {frame_idx}: {len(route)} wps "
                        f"junction={score:.1%}"
                    )

                tracks = None
                sense_mode = "oracle"
                if live_perc is not None:
                    tracks = live_perc.update()
                    sense_mode = "nets"

                step = agent.step(
                    vehicle,
                    route,
                    world=None if tracks is not None else world,
                    tracks=tracks,
                    pose_location=plan_loc if plan_from_ekf else None,
                    sense_mode=sense_mode,
                )
                vehicle.apply_control(step.control)

                max_route_index = max(max_route_index, step.route_index)
                if prev_odom is not None:
                    step_m = prev_odom.distance(loc)
                    if step_m < 8.0:
                        odom_m += step_m
                prev_odom = loc
                n_col, n_inv = events.snapshot()
                lane = lane_metrics_at(loc, yaw, route, step.route_index)
                lane_fields = format_lane_metric_fields(lane)
                drive_rows.append(
                    {
                        "frame": str(frame_idx),
                        "steer": f"{step.control.steer:.6f}",
                        "throttle": f"{step.control.throttle:.6f}",
                        "brake": f"{step.control.brake:.6f}",
                        "speed_kmh": f"{speed_kmh(vehicle):.3f}",
                        "command": step.decision.command,
                        "behavior": step.decision.state.value,
                        "reason": step.decision.reason,
                        "light": step.light,
                        "sense": step.sense_mode,
                        **lane_fields,
                        "lane_invasions": str(n_inv),
                        "collisions": str(n_col),
                        "lead_m": "" if step.lead_m is None else f"{step.lead_m:.2f}",
                        "n_pred": str(step.n_predictions),
                        "ekf_e": "" if math.isnan(ekf_e) else f"{ekf_e:.3f}",
                        "ekf_n": "" if math.isnan(ekf_n) else f"{ekf_n:.3f}",
                    }
                )

                if draw:
                    draw_route(
                        world,
                        route,
                        life_time=line_life,
                        start_index=max(0, step.route_index - 2),
                    )
                    draw_target(world, step.target, life_time=line_life)
                if debug_decisions:
                    draw_decisions(
                        world, vehicle, step, life_time=line_life
                    )
                if ekf_loc is not None and (debug_decisions or draw):
                    draw_ekf_pose(world, ekf_loc, life_time=line_life)

                if frame_idx % int(drive_fps) == 0:
                    print(
                        f"  t={frame_idx / drive_fps:4.1f}s  "
                        f"speed={speed_kmh(vehicle):4.1f}  "
                        f"beh={step.decision.state.value:14s}  "
                        f"why={step.decision.reason:18s}  "
                        f"cmd={step.decision.command:12s}  "
                        f"light={step.light:6s}  "
                        f"sense={step.sense_mode:6s}  "
                        f"|cte|={abs(lane['cte_m']):.2f}  "
                        f"inv={n_inv} col={n_col}  "
                        f"pred={step.n_predictions}"
                    )
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        _cleanup(actors)

    progress_m = credit_m + path_length_m(route_pts, 0, max_route_index)
    n_col, n_inv = events.snapshot()
    metrics = {
        "agent": "classical",
        "controller": controller,
        "lane_invasions": n_inv,
        "collisions": n_col,
        "distance_m": round(odom_m, 2),
        "route_progress_m": round(progress_m, 2),
        "route_length_m": round(planned_m, 2),
        "route_completion_pct": round(
            route_completion_pct(progress_m, planned_m), 2
        ),
        "avg_speed_kmh": round(
            (sum(float(r["speed_kmh"]) for r in drive_rows) / max(1, len(drive_rows))),
            2,
        ),
        "collisions_per_km": round(collisions_per_km(n_col, odom_m), 3),
        "max_collisions_per_km": 1.0,
        "ekf": ekf,
        "plan_from_ekf": plan_from_ekf,
        "perceive": perceive,
        "use_frenet": use_frenet,
        "traffic": traffic,
        "spawn_index": spawn_index,
        "map": world_map.name,
        "debug_decisions": debug_decisions,
    }
    metrics["milestone"] = (
        "PASS" if metrics["collisions_per_km"] <= 1.0 else "FAIL"
    )
    # CTE stats
    ctes = [abs(float(r["cte_m"])) for r in drive_rows]
    if ctes:
        ctes_sorted = sorted(ctes)
        metrics["cte_p50_m"] = round(ctes_sorted[len(ctes_sorted) // 2], 3)
        metrics["cte_p90_m"] = round(
            ctes_sorted[int(0.9 * (len(ctes_sorted) - 1))], 3
        )

    log_path = out_dir / "drive_log.csv"
    with log_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_DRIVE_LOG_HEADER)
        writer.writeheader()
        writer.writerows(drive_rows)
    metrics_path = out_dir / "drive_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Wrote {log_path} and {metrics_path}")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=120.0)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--map", dest="map_name", default="Town01")
    parser.add_argument("--spawn-index", type=int, default=None)
    parser.add_argument("--from-run", type=Path, default=None)
    parser.add_argument(
        "--controller",
        default="hybrid",
        choices=["pid", "pure_pursuit", "stanley", "hybrid", "mpc"],
    )
    parser.add_argument("--cruise-kmh", type=float, default=28.0)
    parser.add_argument("--turn-kmh", type=float, default=16.0)
    parser.add_argument("--traffic", type=int, default=0)
    parser.add_argument("--ekf", action="store_true")
    parser.add_argument(
        "--plan-from-ekf",
        action="store_true",
        help="Use EKF ego pose for route indexing / upcoming command (needs --ekf)",
    )
    parser.add_argument(
        "--perceive",
        action="store_true",
        help="Use SegNet+OccupancyNet+tracker instead of CARLA actor GT",
    )
    parser.add_argument("--seg", type=Path, default=Path("models/seg.pt"))
    parser.add_argument(
        "--occupancy", type=Path, default=Path("models/occupancy.pt")
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--debug-decisions",
        action="store_true",
        help="HUD: behavior reason, predicted futures, Frenet path",
    )
    parser.add_argument("--no-frenet", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--no-draw", action="store_true")
    args = parser.parse_args()
    classical_drive(
        seconds=args.seconds,
        fps=args.fps,
        map_name=None if args.from_run else args.map_name,
        spawn_index=args.spawn_index,
        from_run=args.from_run,
        controller=args.controller,
        cruise_kmh=args.cruise_kmh,
        turn_kmh=args.turn_kmh,
        traffic=args.traffic,
        ekf=args.ekf,
        plan_from_ekf=args.plan_from_ekf,
        perceive=args.perceive,
        seg_model=args.seg,
        occupancy_model=args.occupancy,
        device=args.device,
        debug_decisions=args.debug_decisions,
        out=args.out,
        draw=not args.no_draw,
        use_frenet=not args.no_frenet,
    )


if __name__ == "__main__":
    main()
