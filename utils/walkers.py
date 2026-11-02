"""AI pedestrians for perception recordings."""
from __future__ import annotations

import math
import random

import carla

from utils.traffic import _actor_alive


def _is_walker_controller(actor: carla.Actor) -> bool:
    try:
        return "controller.ai.walker" in actor.type_id
    except Exception:
        return False


def _is_walker(actor: carla.Actor) -> bool:
    try:
        return actor.type_id.startswith("walker.pedestrian")
    except Exception:
        return False


def spawn_walkers(
    client: carla.Client,
    world: carla.World,
    *,
    count: int,
    cross_factor: float = 0.2,
    seed: int | None = None,
) -> tuple[list[carla.Walker], list[carla.WalkerAIController]]:
    """Spawn ``count`` walking pedestrians with AI controllers.

    Controllers must be destroyed before the walkers. In sync mode the batch
    spawn uses ``do_tick=False`` and an explicit ``world.tick()`` before
    ``controller.start()``. Batch commands go through the client.
    """
    if count <= 0:
        return [], []

    library = world.get_blueprint_library()
    walker_bps = list(library.filter("walker.pedestrian.*"))
    if not walker_bps:
        print("No walker blueprints in this CARLA build")
        return [], []
    controller_bp = library.find("controller.ai.walker")
    if controller_bp is None:
        print("No controller.ai.walker blueprint")
        return [], []

    world.set_pedestrians_cross_factor(float(cross_factor))
    if seed is not None:
        world.set_pedestrians_seed(int(seed))

    rng = random.Random(seed)
    spawn_batch: list[carla.command.SpawnActor] = []
    for _ in range(count):
        bp = rng.choice(walker_bps)
        if bp.has_attribute("is_invincible"):
            bp.set_attribute("is_invincible", "false")
        location = world.get_random_location_from_navigation()
        if location is None:
            continue
        transform = carla.Transform(location)
        spawn_batch.append(carla.command.SpawnActor(bp, transform))

    if not spawn_batch:
        print("No navigation locations for walkers (navmesh not ready?)")
        return [], []

    walker_results = client.apply_batch_sync(spawn_batch, False)
    world.tick()

    walkers: list[carla.Walker] = []
    controller_batch: list[carla.command.SpawnActor] = []
    for result in walker_results:
        if result.error:
            continue
        walker = world.get_actor(result.actor_id)
        if walker is None:
            continue
        walkers.append(walker)  # type: ignore[arg-type]
        controller_batch.append(
            carla.command.SpawnActor(controller_bp, carla.Transform(), result.actor_id)
        )

    controller_results = client.apply_batch_sync(controller_batch, False)
    world.tick()

    controllers: list[carla.WalkerAIController] = []
    for result in controller_results:
        if result.error:
            continue
        controller = world.get_actor(result.actor_id)
        if controller is None:
            continue
        controllers.append(controller)  # type: ignore[arg-type]
        controller.start()
        destination = world.get_random_location_from_navigation()
        if destination is not None:
            controller.go_to_location(destination)
        controller.set_max_speed(rng.uniform(1.0, 1.8))

    return walkers, controllers


def destroy_walkers(
    controllers: list[carla.Actor],
    walkers: list[carla.Actor],
) -> None:
    """Stop and destroy controllers first, then walkers."""
    for controller in controllers:
        try:
            if hasattr(controller, "stop"):
                controller.stop()
        except Exception:
            pass
        try:
            controller.destroy()
        except Exception:
            pass
    for walker in walkers:
        try:
            walker.destroy()
        except Exception:
            pass


def partition_walker_actors(
    actors: list[carla.Actor],
) -> tuple[list[carla.Actor], list[carla.Actor], list[carla.Actor]]:
    """Split into (controllers, walkers, everyone else)."""
    controllers: list[carla.Actor] = []
    walkers: list[carla.Actor] = []
    others: list[carla.Actor] = []
    for actor in actors:
        if _is_walker_controller(actor):
            controllers.append(actor)
        elif _is_walker(actor):
            walkers.append(actor)
        else:
            others.append(actor)
    return controllers, walkers, others


def pedestrian_gap_m(
    world: carla.World,
    ego: carla.Vehicle,
    *,
    max_dist_m: float = 16.0,
    max_angle_deg: float = 40.0,
    max_lateral_m: float = 2.8,
    world_map: carla.Map | None = None,
) -> float | None:
    """Distance (m) to the nearest walker roughly ahead, or None.

    With ``world_map``, only walkers standing on a driving lane count. A
    walker waiting at the curb is inside ``max_lateral_m`` and would
    otherwise hold the car forever.
    """
    try:
        ego_tf = ego.get_transform()
    except Exception:
        return None
    ego_loc = ego_tf.location
    forward = ego_tf.get_forward_vector()
    right = ego_tf.get_right_vector()
    max_angle = math.radians(max_angle_deg)

    best: float | None = None
    try:
        walkers = list(world.get_actors().filter("walker.pedestrian.*"))
    except Exception:
        return None
    for actor in walkers:
        try:
            if not _actor_alive(actor):
                continue
            other = actor.get_location()
        except Exception:
            continue
        dx = other.x - ego_loc.x
        dy = other.y - ego_loc.y
        dz = other.z - ego_loc.z
        dist = math.sqrt(dx * dx + dy * dy + dz * dz)
        if dist < 0.3 or dist > max_dist_m:
            continue
        ahead = dx * forward.x + dy * forward.y + dz * forward.z
        if ahead < 0.5:
            continue
        lateral = abs(dx * right.x + dy * right.y + dz * right.z)
        if lateral > max_lateral_m:
            continue
        flat = math.sqrt(max(dx * dx + dy * dy, 1e-6))
        cosang = ahead / flat
        if cosang < math.cos(max_angle):
            continue
        if world_map is not None and world_map.get_waypoint(
            other, project_to_road=False, lane_type=carla.LaneType.Driving
        ) is None:
            continue
        if best is None or dist < best:
            best = dist
    return best


def apply_pedestrian_hold(
    control: carla.VehicleControl,
    world: carla.World,
    ego: carla.Vehicle,
    *,
    soft_m: float = 12.0,
    hard_m: float = 6.0,
    stop_m: float = 3.5,
    world_map: carla.Map | None = None,
) -> carla.VehicleControl:
    """Coast/brake for a walker ahead so the path expert does not hit people.

    Tighter than the lead-car hold: stop earlier and harder. Pass
    ``world_map`` to ignore walkers on the sidewalk.
    """
    gap = pedestrian_gap_m(world, ego, max_dist_m=soft_m + 2.0, world_map=world_map)
    if gap is None:
        return control

    throttle = float(control.throttle)
    brake = float(control.brake)
    if gap <= stop_m:
        throttle = 0.0
        brake = max(brake, 1.0)
    elif gap <= hard_m:
        t = (gap - stop_m) / max(hard_m - stop_m, 1e-3)
        throttle = 0.0
        brake = max(brake, 1.0 - 0.45 * t)
    elif gap <= soft_m:
        t = (gap - hard_m) / max(soft_m - hard_m, 1e-3)
        throttle = min(throttle, 0.15 * t)
        brake = max(brake, 0.20 * (1.0 - t))
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
