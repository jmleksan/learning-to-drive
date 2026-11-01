"""Background NPC traffic helpers (Traffic Manager autopilot)."""
from __future__ import annotations

import math
import random

import carla

from utils.vehicles import choose_vehicle_blueprint


def _actor_alive(actor: carla.Actor) -> bool:
    """Safe alive check — never let a dead handle abort the process."""
    try:
        return bool(actor.is_alive)
    except Exception:
        return False


def _safe_destroy(actor: carla.Actor) -> bool:
    """Disable autopilot if needed and destroy. True if destroy was attempted."""
    if not _actor_alive(actor):
        return False
    try:
        if hasattr(actor, "set_autopilot"):
            try:
                actor.set_autopilot(False)
            except Exception:
                pass
        actor.destroy()
        return True
    except Exception:
        return False


def clear_other_vehicles(
    world: carla.World,
    ego: carla.Vehicle | None = None,
) -> int:
    """Destroy every vehicle except ego (clears leftover wrecks from prior runs)."""
    ego_id = None
    if ego is not None:
        try:
            ego_id = ego.id
        except Exception:
            ego_id = None
    removed = 0
    try:
        vehicles = list(world.get_actors().filter("vehicle.*"))
    except Exception:
        return 0
    for actor in vehicles:
        try:
            if ego_id is not None and actor.id == ego_id:
                continue
        except Exception:
            continue
        if _safe_destroy(actor):
            removed += 1
    return removed


def is_wrecked_vehicle(vehicle: carla.Vehicle) -> bool:
    """True if flipped / on its side / likely blocking debris."""
    if not _actor_alive(vehicle):
        return True
    try:
        rot = vehicle.get_transform().rotation
        if abs(rot.roll) > 40.0 or abs(rot.pitch) > 40.0:
            return True
        vel = vehicle.get_velocity()
        speed = 3.6 * math.sqrt(vel.x * vel.x + vel.y * vel.y + vel.z * vel.z)
        if abs(rot.roll) > 25.0 and speed < 1.5:
            return True
        if abs(vel.z) > 8.0 and speed < 3.0:
            return True
    except Exception:
        return True
    return False


def destroy_wrecked_vehicles(
    world: carla.World,
    *,
    ego: carla.Vehicle | None = None,
    tracked: list[carla.Vehicle] | None = None,
) -> tuple[list[carla.Vehicle], int]:
    """
    Remove flipped/dead NPCs.

    Returns (surviving tracked vehicles, how many destroyed).
    If ``tracked`` is None, scans all vehicles except ego.
    """
    ego_id = None
    if ego is not None:
        try:
            ego_id = ego.id
        except Exception:
            ego_id = None
    destroyed = 0
    survivors: list[carla.Vehicle] = []
    if tracked is not None:
        pool: list[carla.Actor] = list(tracked)
    else:
        try:
            pool = list(world.get_actors().filter("vehicle.*"))
        except Exception:
            return [], 0

    for actor in pool:
        try:
            if ego_id is not None and actor.id == ego_id:
                continue
        except Exception:
            destroyed += 1
            continue
        try:
            if not _actor_alive(actor) or is_wrecked_vehicle(actor):
                _safe_destroy(actor)
                destroyed += 1
                continue
        except Exception:
            destroyed += 1
            continue
        if tracked is not None:
            survivors.append(actor)  # type: ignore[arg-type]

    if tracked is None:
        return [], destroyed
    return survivors, destroyed


def prune_actor_list(
    actors: list[carla.Actor],
    *,
    ego: carla.Actor,
    npcs: list[carla.Vehicle],
) -> None:
    """Drop destroyed NPC handles from ``actors`` without probing dead refs."""
    keep_ids: set[int] = set()
    try:
        keep_ids.add(ego.id)
    except Exception:
        pass
    for n in npcs:
        try:
            keep_ids.add(n.id)
        except Exception:
            pass
    kept: list[carla.Actor] = []
    for a in actors:
        try:
            aid = a.id
            tid = a.type_id
        except Exception:
            continue
        if tid.startswith("sensor.") or aid in keep_ids:
            kept.append(a)
    actors[:] = kept


def spawn_npc_traffic(
    world: carla.World,
    tm: carla.TrafficManager,
    *,
    count: int,
    tm_port: int = 8000,
    ego: carla.Vehicle | None = None,
    exclude_spawn_indices: set[int] | None = None,
    ignore_lights: bool = False,
    seed: int | None = None,
    clear_existing: bool = True,
) -> list[carla.Vehicle]:
    """
    Spawn up to ``count`` autopilot vehicles at free spawn points.

    NPCs obey lights/signs by default so red-stop demos stay coherent with ego.
    """
    if clear_existing:
        n = clear_other_vehicles(world, ego=ego)
        if n:
            print(f"Cleared {n} leftover vehicle(s) before spawning traffic")

    if count <= 0:
        return []

    spawn_pts = list(world.get_map().get_spawn_points())
    if not spawn_pts:
        return []

    blocked: set[int] = set(exclude_spawn_indices or ())
    if ego is not None:
        ego_loc = ego.get_location()
        for i, tf in enumerate(spawn_pts):
            if tf.location.distance(ego_loc) < 12.0:
                blocked.add(i)

    free = [i for i in range(len(spawn_pts)) if i not in blocked]
    rng = random.Random(seed)
    rng.shuffle(free)

    light_pct = 100.0 if ignore_lights else 0.0
    vehicles: list[carla.Vehicle] = []
    for idx in free:
        if len(vehicles) >= count:
            break
        bp = choose_vehicle_blueprint(world)
        if bp.has_attribute("role_name"):
            bp.set_attribute("role_name", "autopilot")
        actor = world.try_spawn_actor(bp, spawn_pts[idx])
        if actor is None:
            continue
        vehicle = actor  # type: ignore[assignment]
        vehicle.set_autopilot(True, tm_port)
        tm.ignore_lights_percentage(vehicle, light_pct)
        tm.ignore_signs_percentage(vehicle, light_pct)
        tm.auto_lane_change(vehicle, False)
        tm.distance_to_leading_vehicle(vehicle, 6.0)
        vehicles.append(vehicle)

    return vehicles


def lead_vehicle_gap_m(
    world: carla.World,
    ego: carla.Vehicle,
    *,
    max_dist_m: float = 18.0,
    max_angle_deg: float = 28.0,
    max_lateral_m: float = 2.2,
) -> float | None:
    """Distance (m) to nearest vehicle roughly ahead in-lane, or None."""
    try:
        ego_tf = ego.get_transform()
        ego_id = ego.id
    except Exception:
        return None
    ego_loc = ego_tf.location
    forward = ego_tf.get_forward_vector()
    right = ego_tf.get_right_vector()
    max_angle = math.radians(max_angle_deg)

    best: float | None = None
    try:
        others = list(world.get_actors().filter("vehicle.*"))
    except Exception:
        return None
    for actor in others:
        try:
            if actor.id == ego_id or not _actor_alive(actor):
                continue
            other = actor.get_location()
        except Exception:
            continue
        dx = other.x - ego_loc.x
        dy = other.y - ego_loc.y
        dz = other.z - ego_loc.z
        dist = math.sqrt(dx * dx + dy * dy + dz * dz)
        if dist < 0.5 or dist > max_dist_m:
            continue
        ahead = dx * forward.x + dy * forward.y + dz * forward.z
        if ahead < 1.0:
            continue
        lateral = abs(dx * right.x + dy * right.y + dz * right.z)
        if lateral > max_lateral_m:
            continue
        flat = math.sqrt(max(dx * dx + dy * dy, 1e-6))
        cosang = ahead / flat
        if cosang < math.cos(max_angle):
            continue
        if best is None or dist < best:
            best = dist
    return best


def apply_lead_vehicle_hold(
    control: carla.VehicleControl,
    world: carla.World,
    ego: carla.Vehicle,
    *,
    soft_m: float = 14.0,
    hard_m: float = 7.0,
    stop_m: float = 4.5,
) -> carla.VehicleControl:
    """
    Coast/brake for a lead vehicle so the path expert does not plow into NPCs.

    Composes with light holds: take the stronger brake / lower throttle.
    """
    gap = lead_vehicle_gap_m(world, ego, max_dist_m=soft_m + 2.0)
    if gap is None:
        return control

    throttle = float(control.throttle)
    brake = float(control.brake)
    if gap <= stop_m:
        throttle = 0.0
        brake = max(brake, 0.95)
    elif gap <= hard_m:
        t = (gap - stop_m) / max(hard_m - stop_m, 1e-3)
        throttle = 0.0
        brake = max(brake, 0.95 - 0.60 * t)
    elif gap <= soft_m:
        t = (gap - hard_m) / max(soft_m - hard_m, 1e-3)
        throttle = min(throttle, 0.25 * t)
        brake = max(brake, 0.12 * (1.0 - t))
        if brake > 0.05:
            throttle = 0.0

    return carla.VehicleControl(
        throttle=throttle,
        steer=float(control.steer),
        brake=brake,
        hand_brake=control.hand_brake,
        reverse=control.reverse,
        manual_gear_shift=control.manual_gear_shift,
        gear=control.gear,
    )
