"""Traffic light and stop-sign state for logging."""
from __future__ import annotations

import math

import carla


def _forward_unit(vehicle: carla.Vehicle) -> carla.Vector3D:
    return vehicle.get_transform().get_forward_vector()


def traffic_light_fields(vehicle: carla.Vehicle) -> tuple[int, str]:
    """
    Returns (at_traffic_light as 0/1, light_state name).

    light_state is Red|Yellow|Green|Off|None.
    """
    at = 1 if vehicle.is_at_traffic_light() else 0
    light = vehicle.get_traffic_light()
    if light is None:
        return at, "None"
    state = light.get_state()
    # carla.TrafficLightState: Red, Yellow, Green, Off, Unknown
    name = str(state)
    if "." in name:
        name = name.split(".")[-1]
    if name == "Unknown":
        name = "Off"
    return at, name


def should_stop_for_light(vehicle: carla.Vehicle) -> bool:
    """True when the vehicle is affected by a red or yellow light."""
    at, state = traffic_light_fields(vehicle)
    return bool(at) and state in ("Red", "Yellow")


def apply_traffic_light_hold(
    control: carla.VehicleControl,
    vehicle: carla.Vehicle,
) -> carla.VehicleControl:
    """Full stop at red/yellow; keep steer so the car doesn't wander in queue."""
    if not should_stop_for_light(vehicle):
        return control
    return carla.VehicleControl(
        steer=float(control.steer),
        throttle=0.0,
        brake=1.0,
        hand_brake=False,
        reverse=False,
    )


def speed_up_traffic_lights(
    world: carla.World,
    *,
    red_s: float = 5.0,
    yellow_s: float = 1.5,
    green_s: float = 7.0,
) -> int:
    """Shorten all light cycles so recording spends less time waiting on red.

    Affects every vehicle at those lights (including future traffic). Keep values
    mild — tiny reds with dense cars get messy at junctions.
    """
    lights = list(world.get_actors().filter("traffic.traffic_light"))
    for light in lights:
        try:
            light.set_red_time(float(red_s))
            light.set_yellow_time(float(yellow_s))
            light.set_green_time(float(green_s))
        except Exception:
            pass
    return len(lights)


# Lights we froze during an exclusive force-green; cleared on next force or manually.
_frozen_for_force: list[carla.Actor] = []


def _unfreeze_forced_lights() -> None:
    global _frozen_for_force
    for actor in _frozen_for_force:
        try:
            if actor.is_alive:
                actor.freeze(False)
        except Exception:
            pass
    _frozen_for_force = []


def _force_exclusive_green(light: carla.Actor, *, green_s: float = 4.0) -> int:
    """
    Set ``light`` green and every other light in its junction group to red.

    Returns how many sibling lights were forced red. Freezes the group briefly
    so the CARLA controller cannot immediately re-green cross traffic.
    """
    global _frozen_for_force
    _unfreeze_forced_lights()

    red = carla.TrafficLightState.Red
    green = carla.TrafficLightState.Green
    siblings_red = 0
    try:
        group = list(light.get_group_traffic_lights())
    except Exception:
        group = [light]

    for other in group:
        try:
            if other.id == light.id:
                continue
            other.set_state(red)
            other.freeze(True)
            _frozen_for_force.append(other)
            siblings_red += 1
        except Exception:
            pass

    light.set_state(green)
    try:
        light.set_green_time(max(green_s, float(light.get_green_time())))
    except Exception:
        pass
    try:
        # Hold ego green so the group cycle does not yank it back to red mid-clear.
        light.freeze(True)
        _frozen_for_force.append(light)
    except Exception:
        pass
    return siblings_red


def maybe_force_ego_light_green(
    vehicle: carla.Vehicle,
    *,
    red_wait_frames: int,
    max_wait_frames: int,
) -> tuple[int, bool]:
    """If ego has been held on red too long, force that light green.

    Sibling lights in the same junction group are set to red so only ego's
    approach is green (avoids forced-green into still-green cross traffic).

    Returns (updated_wait_frames, forced). Prefer mild cycle speedup first;
    this is a record-time cap so one stuck light doesn't burn the whole run.
    """
    if max_wait_frames <= 0:
        return 0, False
    if not should_stop_for_light(vehicle):
        # Still at the pole on forced green — keep freeze until we leave.
        at, _state = traffic_light_fields(vehicle)
        if not at and _frozen_for_force:
            _unfreeze_forced_lights()
        return 0, False
    at, state = traffic_light_fields(vehicle)
    if not at or state != "Red":
        return 0, False
    wait = red_wait_frames + 1
    if wait < max_wait_frames:
        return wait, False
    light = vehicle.get_traffic_light()
    if light is None:
        return 0, False
    try:
        _force_exclusive_green(light, green_s=4.0)
    except Exception:
        return wait, False
    return 0, True


def near_stop_sign(
    world: carla.World,
    vehicle: carla.Vehicle,
    *,
    max_distance_m: float = 12.0,
    stop_actors: list[carla.Actor] | None = None,
) -> int:
    """1 if any traffic.stop actor is within max_distance_m ahead of the vehicle."""
    vehicle_tf = vehicle.get_transform()
    vehicle_loc = vehicle_tf.location
    forward = _forward_unit(vehicle)
    fx, fy, fz = forward.x, forward.y, forward.z

    actors = stop_actors
    if actors is None:
        actors = list(world.get_actors().filter("traffic.stop"))

    for actor in actors:
        if not actor.is_alive:
            continue
        loc = actor.get_location()
        dx = loc.x - vehicle_loc.x
        dy = loc.y - vehicle_loc.y
        dz = loc.z - vehicle_loc.z
        dist = math.sqrt(dx * dx + dy * dy + dz * dz)
        if dist > max_distance_m:
            continue
        if dx * fx + dy * fy + dz * fz > 0.0:
            return 1
    return 0
