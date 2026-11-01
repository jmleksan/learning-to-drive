"""Front bumper radar. The range is a throttle/brake input, not a steer input."""

from __future__ import annotations

import math
import queue

import carla

# Matches the lead-vehicle window the expert already brakes inside, plus margin.
RADAR_RANGE_M = 30.0
RADAR_HFOV_DEG = 30.0
RADAR_VFOV_DEG = 10.0
# Detections outside this cone are shoulders and walls, not the car ahead.
RADAR_AZIMUTH_DEG = 12.0
# The beam also hits the pavement. From a ~1 m mount, road inside 30 m
# sits below about -2°. A lead car's body is near the horizon.
RADAR_MIN_ALTITUDE_DEG = -1.5
RADAR_MAX_ALTITUDE_DEG = 5.0


def attach_radar(
    world: carla.World,
    vehicle: carla.Vehicle,
    *,
    range_m: float = RADAR_RANGE_M,
) -> carla.Sensor:
    """Mount a forward radar on the bumper."""
    bp = world.get_blueprint_library().find("sensor.other.radar")
    bp.set_attribute("horizontal_fov", str(RADAR_HFOV_DEG))
    bp.set_attribute("vertical_fov", str(RADAR_VFOV_DEG))
    bp.set_attribute("range", str(range_m))
    bp.set_attribute("points_per_second", "1500")
    transform = carla.Transform(carla.Location(x=2.0, z=1.0))
    return world.spawn_actor(bp, transform, attach_to=vehicle)


def try_attach_radar(
    world: carla.World,
    vehicle: carla.Vehicle,
    actors: list,
    radar_queue: queue.Queue,
) -> bool:
    """Spawn and listen. Returns False if this CARLA build has no radar."""
    try:
        radar = attach_radar(world, vehicle)
    except Exception as exc:
        print(f"Radar unavailable ({exc})")
        return False
    actors.append(radar)
    radar.listen(radar_queue.put)
    print(
        f"Radar ON: bumper, {RADAR_RANGE_M:.0f} m forward, "
        f"±{RADAR_AZIMUTH_DEG:.0f}° (throttle/brake input only)"
    )
    return True


def nearest_forward_depth(detections) -> float | None:
    """Closest return in the forward cone, or None when the lane ahead is clear.

    Negative altitude is downward. Those returns are the road and are ignored,
    so an empty lane does not look like a car at a fixed distance.
    """
    az_lim = math.radians(RADAR_AZIMUTH_DEG)
    min_alt = math.radians(RADAR_MIN_ALTITUDE_DEG)
    max_alt = math.radians(RADAR_MAX_ALTITUDE_DEG)
    best: float | None = None
    for det in detections:
        azimuth = abs(float(det.azimuth))
        altitude = float(det.altitude)
        depth = float(det.depth)
        if azimuth > az_lim or altitude < min_alt or altitude > max_alt or depth < 0.5:
            continue
        if best is None or depth < best:
            best = depth
    return best


def pop_radar_range(
    radar_queue: queue.Queue,
    previous: float | None,
) -> float | None:
    """Newest measurement after a tick. A missing callback keeps the last range."""
    measurement = None
    got = False
    while True:
        try:
            measurement = radar_queue.get_nowait()
            got = True
        except queue.Empty:
            break
    if not got or measurement is None:
        return previous
    return nearest_forward_depth(measurement)


def format_radar_m(range_m: float | None) -> str:
    """CSV cell. Empty means no forward return."""
    if range_m is None:
        return ""
    return f"{range_m:.3f}"


def format_radar_status(range_m: float | None) -> str:
    if range_m is None:
        return "clear"
    return f"{range_m:.1f}m"


def encode_radar(
    range_m: float | None,
    *,
    max_range_m: float = RADAR_RANGE_M,
) -> tuple[float, float]:
    """(range_norm, detected). A clear lane is (1, 0)."""
    if range_m is None:
        return 1.0, 0.0
    span = max(float(max_range_m), 1e-3)
    norm = max(0.0, min(1.0, float(range_m) / span))
    return norm, 1.0


def encode_radar_cell(
    text: str | None,
    *,
    max_range_m: float = RADAR_RANGE_M,
) -> tuple[float, float]:
    if text is None or str(text).strip() == "":
        return 1.0, 0.0
    try:
        return encode_radar(float(text), max_range_m=max_range_m)
    except ValueError:
        return 1.0, 0.0
