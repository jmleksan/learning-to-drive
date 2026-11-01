"""Spectator camera helpers."""
import carla


def follow_vehicle(
    world: carla.World,
    vehicle: carla.Vehicle,
    *,
    distance: float = 8.0,
    height: float = 4.0,
    pitch: float = -15.0,
) -> None:
    """Place the spectator behind and above the vehicle, looking forward."""
    try:
        if not vehicle.is_alive:
            return
        transform = vehicle.get_transform()
    except Exception:
        return
    forward = transform.get_forward_vector()
    location = transform.location - distance * forward
    location.z += height

    try:
        world.get_spectator().set_transform(
            carla.Transform(
                location,
                carla.Rotation(pitch=pitch, yaw=transform.rotation.yaw),
            )
        )
    except Exception:
        return
