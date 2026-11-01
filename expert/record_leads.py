"""
Record stops behind a car, on straights and at turn commands.

Straight clips: FOLLOW_LANE, one parked car ahead, brake to a gap.
Turn clips: the command is TURN_LEFT, TURN_RIGHT, or GO_STRAIGHT, a car is
waiting at the light on that approach, and the expert stops behind it.
The turn is not driven. The label is "this command, car ahead, brake."

Distances are between vehicle centers. The stop is at 8 m, which leaves a
few meters between bumpers.

    python -m expert.record_leads --map Town01
    python -m expert.record_leads --map Town01 --clips 8 --turn-clips 6
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import queue
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import carla
from PIL import Image

from expert.model import IMAGE_HEIGHT, IMAGE_WIDTH
from expert.record import _cleanup_actors, _get_camera_images
from expert.record_turns import _along, list_junction_maneuvers
from utils.balance import CONTROLS_HEADER, write_balanced_controls, write_command_distribution
from utils.commands import (
    COMMAND_FOLLOW_LANE,
    COMMAND_GO_STRAIGHT,
    COMMAND_TURN_LEFT,
    COMMAND_TURN_RIGHT,
    JUNCTION_COMMANDS,
    RoadOption,
)
from utils.connection import connect_client
from utils.images import image_to_array
from utils.lane_metrics import format_lane_metric_fields, lane_metrics_at
from utils.radar import format_radar_m, format_radar_status, pop_radar_range, try_attach_radar
from utils.routing import _wrap_yaw_delta_deg, build_planner, plan_route
from utils.run_meta import ensure_map, find_spawn_index, save_run_meta
from utils.spectator import follow_vehicle
from utils.state import speed_kmh
from utils.sync_mode import SynchronousMode
from utils.traffic import clear_other_vehicles, lead_vehicle_gap_m
from utils.traffic_signs import (
    apply_traffic_light_hold,
    maybe_force_ego_light_green,
    near_stop_sign,
    speed_up_traffic_lights,
    traffic_light_fields,
)
from utils.vehicles import attach_rgb_cameras, choose_vehicle_blueprint, spawn_vehicle
from utils.waypoint_follower import WaypointFollower

# Center-to-center. 8 m leaves roughly 3 m between bumpers.
STOP_M = 8.0
SOFT_M = 18.0
GAPS_M = (22.0, 34.0)
LEAD_FILTERS = (
    "vehicle.audi.tt",
    "vehicle.nissan.patrol",
    "vehicle.dodge.charger_2020",
    "vehicle.toyota.prius",
    "vehicle.mini.cooper_s",
    "vehicle.lincoln.mkz_2017",
    "vehicle.mercedes.coupe",
    "vehicle.bmw.grandtourer",
)


@dataclass(frozen=True)
class LeadApproach:
    start: carla.Waypoint
    lead: carla.Waypoint
    end: carla.Waypoint
    gap_m: float
    chain: tuple[carla.Waypoint, ...]
    command: str = COMMAND_FOLLOW_LANE
    hold_red: bool = False
    junction_id: int = -1


def _chain_forward(start: carla.Waypoint, max_m: float, step: float = 2.0) -> list[carla.Waypoint]:
    """Walk a straight driving lane until a junction or a real bend."""
    current = start
    traveled = 0.0
    chain = [start]
    yaw0 = start.transform.rotation.yaw
    while traveled < max_m:
        try:
            nxt = current.next(step)
        except Exception:
            break
        if not nxt:
            break
        current = min(
            nxt,
            key=lambda wp: abs(
                _wrap_yaw_delta_deg(wp.transform.rotation.yaw - current.transform.rotation.yaw)
            ),
        )
        if current.is_junction:
            break
        if abs(_wrap_yaw_delta_deg(current.transform.rotation.yaw - yaw0)) > 12.0:
            break
        traveled += step
        chain.append(current)
    return chain


def _at_distance(chain: list[carla.Waypoint], dist_m: float, step: float = 2.0) -> carla.Waypoint:
    index = int(round(dist_m / step))
    return chain[max(0, min(index, len(chain) - 1))]


def list_lead_approaches(
    world_map: carla.Map,
    *,
    gaps_m: tuple[float, ...] = GAPS_M,
    clips: int = 20,
) -> list[LeadApproach]:
    """Straight approaches with a parked car at each gap. One lane is used once."""
    seen: set[tuple[int, int]] = set()
    approaches: list[LeadApproach] = []
    need = max(1, int(clips))
    for waypoint in world_map.generate_waypoints(8.0):
        if len(approaches) >= need:
            break
        if waypoint.is_junction or waypoint.lane_type != carla.LaneType.Driving:
            continue
        key = (int(waypoint.road_id), int(waypoint.lane_id))
        if key in seen:
            continue
        chain = _chain_forward(waypoint, max_m=70.0)
        length_m = 2.0 * (len(chain) - 1)
        if length_m < 50.0:
            continue
        seen.add(key)
        for gap in gaps_m:
            if length_m < gap + 16.0:
                continue
            approaches.append(
                LeadApproach(
                    start=chain[0],
                    lead=_at_distance(chain, gap),
                    end=_at_distance(chain, gap + 16.0),
                    gap_m=gap,
                    chain=tuple(chain),
                )
            )
            if len(approaches) >= need:
                break
    return approaches


def _route_length_m(route: list) -> float:
    total = 0.0
    for i in range(1, len(route)):
        a = route[i - 1][0].transform.location
        b = route[i][0].transform.location
        total += a.distance(b)
    return total


def list_turn_queues(
    world_map: carla.Map,
    grp,
    *,
    clips: int = 12,
    gap_m: float = 20.0,
) -> list[LeadApproach]:
    """One queued stop per junction exit, command set to that exit.

    The lead waits just before the junction. The route continues through the
    turn so the command is real, and a red light holds both cars there.
    """
    buckets: dict[str, list] = {name: [] for name in JUNCTION_COMMANDS}
    for maneuver in list_junction_maneuvers(world_map):
        buckets.setdefault(maneuver.command, []).append(maneuver)
    ordered = []
    index = 0
    names = (COMMAND_TURN_LEFT, COMMAND_TURN_RIGHT, COMMAND_GO_STRAIGHT)
    while any(buckets.values()) and len(ordered) < max(1, clips) * 4:
        name = names[index % len(names)]
        index += 1
        if not buckets.get(name):
            continue
        ordered.append(buckets[name].pop(0))

    approaches: list[LeadApproach] = []
    for maneuver in ordered:
        if len(approaches) >= clips:
            break
        lead = _along(maneuver.entry, -8.0)
        start = _along(maneuver.entry, -(8.0 + gap_m))
        end = _along(maneuver.exit, 14.0)
        if start is None or lead is None or end is None:
            continue
        try:
            route = plan_route(grp, start.transform.location, end.transform.location)
        except Exception:
            continue
        if len(route) < 3 or _route_length_m(route) > gap_m + 55.0:
            continue
        approaches.append(
            LeadApproach(
                start=start,
                lead=lead,
                end=end,
                gap_m=gap_m,
                chain=tuple(wp for wp, _opt in route),
                command=maneuver.command,
                hold_red=True,
                junction_id=maneuver.junction_id,
            )
        )
    return approaches


def _freeze_approach_red(world: carla.World, entry: carla.Waypoint) -> list[carla.Actor]:
    """Hold the light for this approach on red. Returns the lights to unfreeze."""
    frozen: list[carla.Actor] = []
    red = carla.TrafficLightState.Red
    entry_loc = entry.transform.location
    entry_yaw = entry.transform.rotation.yaw
    for light in world.get_actors().filter("traffic.traffic_light"):
        try:
            stops = light.get_stop_waypoints()
        except Exception:
            continue
        matched = False
        for stop in stops:
            dist = stop.transform.location.distance(entry_loc)
            dyaw = abs(_wrap_yaw_delta_deg(stop.transform.rotation.yaw - entry_yaw))
            if dist < 15.0 and dyaw < 30.0:
                matched = True
                break
        if not matched:
            continue
        try:
            light.set_state(red)
            light.freeze(True)
            frozen.append(light)
        except Exception:
            pass
    return frozen


def _unfreeze(lights: list[carla.Actor]) -> None:
    for light in lights:
        try:
            light.freeze(False)
        except Exception:
            pass


def _spawn_lead(
    world: carla.World,
    pattern: str,
    waypoint: carla.Waypoint,
) -> carla.Vehicle | None:
    """Spawn off the road mesh, then the caller drops the car onto the lane.

    A waypoint sits on the road surface. Most vehicle boxes intersect that
    mesh, so try_spawn_actor returns None. One taller body (the patrol) fits,
    which is why only every 8th clip started.
    """
    blueprint = _lead_blueprint(world, pattern)
    loc = waypoint.transform.location
    lifted = carla.Transform(
        carla.Location(x=loc.x, y=loc.y, z=loc.z + 2.0),
        waypoint.transform.rotation,
    )
    lead = world.try_spawn_actor(blueprint, lifted)
    if lead is not None:
        return lead
    for transform in world.get_map().get_spawn_points():
        lead = world.try_spawn_actor(blueprint, transform)
        if lead is not None:
            return lead
    return None


def _brake_for_gap(
    control: carla.VehicleControl,
    gap: float | None,
    speed: float,
) -> carla.VehicleControl:
    """Ease speed down so the car arrives stopped at STOP_M.

    Target speed falls with the distance left to the stop. Near the lead the
    throttle stays off, so it does not stop short and creep the rest of the way.
    """
    steer = float(control.steer)
    if gap is None or gap > SOFT_M + 6.0:
        return control
    if gap <= STOP_M or (speed < 1.0 and gap <= STOP_M + 3.0):
        return carla.VehicleControl(steer=steer, throttle=0.0, brake=1.0)
    remaining = max(gap - STOP_M, 0.0)
    # ~1 m/s^2: a 14 km/h approach needs about 8 m to roll to a stop.
    desired = 3.6 * math.sqrt(2.0 * 1.0 * remaining)
    if gap <= STOP_M + 6.0:
        throttle = 0.0
    elif speed + 2.0 < desired:
        throttle = min(float(control.throttle), 0.25)
    else:
        throttle = 0.0
    if speed > desired:
        brake = min(0.45, 0.07 * (speed - desired))
    else:
        brake = 0.0
    if brake > 0.04:
        throttle = 0.0
    return carla.VehicleControl(steer=steer, throttle=throttle, brake=brake)


def _place(vehicle: carla.Vehicle, waypoint: carla.Waypoint, *, brake: float) -> None:
    transform = waypoint.transform
    transform.location.z += 0.5
    vehicle.set_simulate_physics(False)
    vehicle.set_transform(transform)
    vehicle.set_simulate_physics(True)
    vehicle.set_target_velocity(carla.Vector3D())
    vehicle.set_target_angular_velocity(carla.Vector3D())
    vehicle.apply_control(carla.VehicleControl(brake=brake, hand_brake=brake > 0.5))


def _drop_actor(actor: carla.Actor, actors: list) -> None:
    try:
        if hasattr(actor, "set_autopilot"):
            actor.set_autopilot(False)
    except Exception:
        pass
    try:
        actor.destroy()
    except Exception:
        pass
    try:
        actors.remove(actor)
    except ValueError:
        pass


def _lead_blueprint(world: carla.World, pattern: str) -> carla.ActorBlueprint:
    library = world.get_blueprint_library()
    found = library.filter(pattern)
    if not found:
        return choose_vehicle_blueprint(world, "vehicle.*")
    blueprint = found[0]
    if blueprint.has_attribute("color"):
        blueprint.set_attribute("color", blueprint.get_attribute("color").recommended_values[0])
    if blueprint.has_attribute("role_name"):
        blueprint.set_attribute("role_name", "lead")
    return blueprint


def record_leads(
    *,
    map_name: str = "Town01",
    clips: int = 20,
    turn_clips: int = 12,
    seconds_per_clip: float = 16.0,
    fps: float = 20.0,
    out: Path = Path("data"),
    tm_port: int = 8000,
) -> Path:
    client = connect_client(timeout=30.0)
    world = ensure_map(client, map_name)
    world_map = world.get_map()
    print("Building route planner ...")
    grp = build_planner(world_map)
    straights = list_lead_approaches(world_map, clips=clips)
    turns = list_turn_queues(world_map, grp, clips=turn_clips)
    approaches = straights + turns
    if not approaches:
        raise SystemExit("No lead approaches found.")
    turn_counts = {name: 0 for name in JUNCTION_COMMANDS}
    for approach in turns:
        turn_counts[approach.command] = turn_counts.get(approach.command, 0) + 1
    print(
        f"{len(straights)} straight stops and {len(turns)} turn-queue stops "
        f"on {world_map.name} (stop at {STOP_M:.0f} m center-to-center)"
    )
    if turns:
        print(
            "  turn commands: "
            + "  ".join(f"{name}={turn_counts.get(name, 0)}" for name in JUNCTION_COMMANDS)
        )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(out) / f"leads_{stamp}"
    image_dir = run_dir / "images" / "center"
    image_dir.mkdir(parents=True, exist_ok=True)
    controls_path = run_dir / "controls.csv"
    cameras = ("center",)
    actors: list[carla.Actor] = []
    cam_queues: dict[str, queue.Queue] = {"center": queue.Queue()}
    recorded_rows: list[dict[str, str]] = []
    clip_meta: list[dict] = []
    follower = WaypointFollower(
        cruise_kmh=14.0,
        turn_kmh=12.0,
        min_lookahead_m=3.5,
        max_lookahead_m=8.0,
        steer_gain=1.65,
        max_steer_rate=0.22,
    )
    max_frames = max(1, int(seconds_per_clip * fps))
    frame_idx = 0
    spawn_index = 0

    with SynchronousMode(client, world, fps=fps, tm_port=tm_port) as (world, _tm):
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
            n_lights = speed_up_traffic_lights(world, red_s=2.0, yellow_s=0.5, green_s=6.0)
            print(f"Fast lights ON ({n_lights} poles)")
            world.tick()
            _get_camera_images(world, cam_queues, cameras, timeout=10.0)
            stop_actors = list(world.get_actors().filter("traffic.stop"))
            spawn_index = find_spawn_index(vehicle, world)

            with controls_path.open("w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=CONTROLS_HEADER)
                writer.writeheader()
                for clip_i, approach in enumerate(approaches, start=1):
                    pattern = LEAD_FILTERS[(clip_i - 1) % len(LEAD_FILTERS)]
                    lead = _spawn_lead(world, pattern, approach.lead)
                    if lead is None:
                        print(f"  skip clip {clip_i}: lead did not spawn ({pattern})")
                        continue
                    actors.append(lead)
                    _place(lead, approach.lead, brake=1.0)
                    _place(vehicle, approach.start, brake=1.0)
                    frozen: list[carla.Actor] = []
                    if approach.hold_red:
                        frozen = _freeze_approach_red(world, approach.lead)
                    for _ in range(int(0.4 * fps)):
                        lead.apply_control(carla.VehicleControl(brake=1.0, hand_brake=True))
                        vehicle.apply_control(carla.VehicleControl(brake=1.0))
                        world.tick()
                    follower.reset()
                    route = [(wp, RoadOption.LANEFOLLOW) for wp in approach.chain]
                    radar_range_m: float | None = None
                    red_wait_frames = 0
                    stopped = 0
                    saw_radar = False
                    min_gap = 999.0
                    finished = False
                    hit = False
                    clip_start = frame_idx
                    row_start = len(recorded_rows)
                    where = (
                        f"junction {approach.junction_id} {approach.command}, "
                        f"car waiting at the light"
                        if approach.hold_red
                        else f"parked car {approach.gap_m:.0f} m ahead"
                    )
                    print(f"Clip {clip_i}/{len(approaches)}  {where}")
                    try:
                        for _step in range(max_frames):
                            lead.apply_control(
                                carla.VehicleControl(brake=1.0, hand_brake=True)
                            )
                            world.tick()
                            if radar_on:
                                radar_range_m = pop_radar_range(radar_queue, radar_range_m)
                            follow_vehicle(world, vehicle)
                            if not approach.hold_red:
                                red_wait_frames, _forced = maybe_force_ego_light_green(
                                    vehicle,
                                    red_wait_frames=red_wait_frames,
                                    max_wait_frames=int(0.4 * fps),
                                )
                            control, _target, _cmd = follower.control(vehicle, route)
                            if not approach.hold_red:
                                control = apply_traffic_light_hold(control, vehicle)
                            speed = speed_kmh(vehicle)
                            gap = lead_vehicle_gap_m(world, vehicle, max_dist_m=45.0)
                            control = _brake_for_gap(control, gap, speed)
                            vehicle.apply_control(control)
                            loc = vehicle.get_location()
                            yaw = vehicle.get_transform().rotation.yaw
                            if gap is not None:
                                min_gap = min(min_gap, gap)
                            if (
                                radar_range_m is not None
                                and gap is not None
                                and gap < SOFT_M
                            ):
                                saw_radar = True
                            if gap is not None and gap < 3.5 and speed > 2.0:
                                hit = True
                                break
                            if (
                                speed < 1.0
                                and gap is not None
                                and STOP_M - 1.5 <= gap <= STOP_M + 3.0
                            ):
                                stopped += 1
                            else:
                                stopped = 0
                            lane = lane_metrics_at(loc, yaw, route, follower.route_index)
                            at_light, light_state = traffic_light_fields(vehicle)
                            images = _get_camera_images(
                                world,
                                cam_queues,
                                cameras,
                                timeout=5.0,
                                retries=3,
                                tick_on_retry=False,
                            )
                            rgb = image_to_array(images["center"])
                            Image.fromarray(rgb).save(image_dir / f"{frame_idx:06d}.png")
                            row = {
                                "frame": str(frame_idx),
                                "camera": "center",
                                "steer": f"{float(control.steer):.6f}",
                                "throttle": f"{float(control.throttle):.6f}",
                                "brake": f"{float(control.brake):.6f}",
                                "speed_kmh": f"{speed:.3f}",
                                "command": approach.command,
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
                            if frame_idx % int(fps) == 0:
                                gap_txt = f"{gap:.1f}m" if gap is not None else "none"
                                print(
                                    f"  {approach.command}  speed={speed:4.1f}  "
                                    f"brake={float(control.brake):.2f}  "
                                    f"gap={gap_txt}  radar={format_radar_status(radar_range_m)}  "
                                    f"light={light_state}"
                                )
                            frame_idx += 1
                            if stopped >= int(0.8 * fps):
                                finished = True
                                break
                    finally:
                        _unfreeze(frozen)
                        _drop_actor(lead, actors)
                    keep = finished and saw_radar and not hit
                    if not keep:
                        del recorded_rows[row_start:]
                        for fid in range(clip_start, frame_idx):
                            (image_dir / f"{fid:06d}.png").unlink(missing_ok=True)
                        if hit:
                            why = "hit the lead"
                        elif not saw_radar:
                            why = "radar never saw the car"
                        else:
                            why = "did not settle near the stop gap"
                        print(f"  dropped ({why}, closest gap {min_gap:.1f} m)")
                        continue
                    clip_meta.append(
                        {
                            "gap_m": approach.gap_m,
                            "command": approach.command,
                            "junction_id": approach.junction_id,
                            "hold_red": approach.hold_red,
                            "stop_gap_m": round(min_gap, 2),
                            "frame_start": clip_start,
                            "frame_end": frame_idx - 1,
                        }
                    )
                    print(f"  kept, stopped near {min_gap:.1f} m")
        finally:
            _cleanup_actors(world, actors, allow_tick=False)

    if not recorded_rows:
        raise SystemExit(f"No clips saved. Empty dir: {run_dir}")

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
    meta["lead_clips"] = True
    meta["clips"] = len(clip_meta)
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    (run_dir / "leads.json").write_text(json.dumps(clip_meta, indent=2) + "\n")
    write_command_distribution(run_dir, recorded_rows)
    write_balanced_controls(run_dir, recorded_rows)
    print(f"Saved {len(clip_meta)} clips, {len(recorded_rows)} frames → {run_dir}")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", dest="map_name", default="Town01")
    parser.add_argument("--clips", type=int, default=20, help="Straight FOLLOW_LANE stops")
    parser.add_argument(
        "--turn-clips",
        type=int,
        default=12,
        help="Stops at a red light behind a car, with a turn command",
    )
    parser.add_argument("--seconds-per-clip", type=float, default=16.0)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--out", type=Path, default=Path("data"))
    parser.add_argument("--tm-port", type=int, default=8000)
    args = parser.parse_args()
    record_leads(
        map_name=args.map_name,
        clips=args.clips,
        turn_clips=args.turn_clips,
        seconds_per_clip=args.seconds_per_clip,
        fps=args.fps,
        out=args.out,
        tm_port=args.tm_port,
    )


if __name__ == "__main__":
    main()
