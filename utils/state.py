"""Vehicle state helpers."""
import carla

MAX_SPEED_KMH = 120.0


def speed_kmh(vehicle: carla.Vehicle) -> float:
    vel = vehicle.get_velocity()
    return 3.6 * (vel.x**2 + vel.y**2 + vel.z**2) ** 0.5


def normalize_speed(speed_kmh: float) -> float:
    return min(max(speed_kmh / MAX_SPEED_KMH, 0.0), 1.0)
