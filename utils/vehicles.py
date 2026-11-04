"""Vehicle spawning helpers."""
import random

import carla

# Lateral offset still drives recovery labels. Left/right also yaw outward so
# Step 3 multi-cam BEV gets real side coverage (not just parallel stereo).
# Wide cams stay yaw=0 for stereo depth experiments.

CAMERA_SPECS = (
    ("center", carla.Location(x=1.5, y=0.0, z=2.4), 0.0),
    ("left", carla.Location(x=1.5, y=-0.40, z=2.4), -55.0),
    ("right", carla.Location(x=1.5, y=0.40, z=2.4), 55.0),
    ("left_wide", carla.Location(x=1.5, y=-1.00, z=2.4), 0.0),
    ("right_wide", carla.Location(x=1.5, y=1.00, z=2.4), 0.0),
)


# CARLA +y is right. Left-shifted camera → positive (right) recovery steer.
RECOVERY_BY_CAMERA = {
    "left": 0.08,
    "right": -0.08,
    "left_wide": 0.16,
    "right_wide": -0.16,
}


def choose_vehicle_blueprint(
    world: carla.World,
    filter_pattern: str = "vehicle.*",
) -> carla.ActorBlueprint:
    blueprints = world.get_blueprint_library().filter(filter_pattern)
    if not blueprints:
        raise RuntimeError(f"No blueprints matched {filter_pattern!r}")
    bp = random.choice(blueprints)
    if bp.has_attribute("color"):
        color = random.choice(bp.get_attribute("color").recommended_values)
        bp.set_attribute("color", color)
    return bp


def spawn_vehicle(
    world: carla.World,
    filter_pattern: str = "vehicle.tesla.model3",
    *,
    spawn_index: int | None = None,
) -> carla.Vehicle:
    blueprint = choose_vehicle_blueprint(world, filter_pattern)
    # Lets other clients, like scripts/chase_cam.py, tell the ego car from traffic.
    if blueprint.has_attribute("role_name"):
        blueprint.set_attribute("role_name", "hero")
    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        raise RuntimeError("Map has no spawn points.")

    if spawn_index is not None:
        if not 0 <= spawn_index < len(spawn_points):
            raise ValueError(
                f"spawn_index must be 0..{len(spawn_points) - 1}, got {spawn_index}"
            )
        ordered = [spawn_points[spawn_index]]
    else:
        ordered = spawn_points.copy()
        random.shuffle(ordered)

    for transform in ordered:
        vehicle = world.try_spawn_actor(blueprint, transform)
        if vehicle is not None:
            return vehicle

    raise RuntimeError("Could not spawn a vehicle at any spawn point.")


def attach_rgb_camera(
    world: carla.World,
    vehicle: carla.Vehicle,
    *,
    width: int = 160,
    height: int = 120,
    fov: float = 90.0,
    location: carla.Location | None = None,
    yaw: float = 0.0,
) -> carla.Sensor:
    """Attach an RGB camera on the hood (default: forward center)."""
    bp = world.get_blueprint_library().find("sensor.camera.rgb")
    bp.set_attribute("image_size_x", str(width))
    bp.set_attribute("image_size_y", str(height))
    bp.set_attribute("fov", str(fov))
    # Prefer the simple/default camera blueprint; post-process off can break
    # some shipping builds and isn't required for 160x120 BC frames.
    if location is None:
        location = carla.Location(x=1.5, z=2.4)
    transform = carla.Transform(location, carla.Rotation(yaw=yaw))
    return world.spawn_actor(bp, transform, attach_to=vehicle)


def attach_semantic_camera(
    world: carla.World,
    vehicle: carla.Vehicle,
    *,
    width: int = 160,
    height: int = 120,
    fov: float = 90.0,
    location: carla.Location | None = None,
    yaw: float = 0.0,
) -> carla.Sensor:
    """Semantic camera on the same mount as the center RGB camera.

    Each pixel's red channel is the CARLA tag id (see segmentation.labels).
    """
    bp = world.get_blueprint_library().find("sensor.camera.semantic_segmentation")
    bp.set_attribute("image_size_x", str(width))
    bp.set_attribute("image_size_y", str(height))
    bp.set_attribute("fov", str(fov))
    if location is None:
        location = carla.Location(x=1.5, y=0.0, z=2.4)
    transform = carla.Transform(location, carla.Rotation(yaw=yaw))
    return world.spawn_actor(bp, transform, attach_to=vehicle)


def attach_depth_camera(
    world: carla.World,
    vehicle: carla.Vehicle,
    *,
    width: int = 160,
    height: int = 120,
    fov: float = 90.0,
    location: carla.Location | None = None,
    yaw: float = 0.0,
) -> carla.Sensor:
    """Depth camera on the same mount as the center RGB camera.

    This is the stereo teacher's label, not a replacement for the RGB camera.
    Decode with perception.depth.depth_to_meters.
    """
    bp = world.get_blueprint_library().find("sensor.camera.depth")
    bp.set_attribute("image_size_x", str(width))
    bp.set_attribute("image_size_y", str(height))
    bp.set_attribute("fov", str(fov))
    if location is None:
        location = carla.Location(x=1.5, y=0.0, z=2.4)
    transform = carla.Transform(location, carla.Rotation(yaw=yaw))
    return world.spawn_actor(bp, transform, attach_to=vehicle)


def attach_rgb_cameras(
    world: carla.World,
    vehicle: carla.Vehicle,
    *,
    names: tuple[str, ...] | list[str] = ("center", "left", "right"),
    width: int = 160,
    height: int = 120,
    fov: float = 90.0,
    stagger_ticks: int = 2,
) -> dict[str, carla.Sensor]:
    """
    Spawn named RGB cameras (subset of CAMERA_SPECS).

    stagger_ticks: world.tick() between each spawn so a weak GPU can settle
    before the next sensor render target is allocated.
    """
    wanted = {n.lower() for n in names}
    specs = [(n, loc, yaw) for n, loc, yaw in CAMERA_SPECS if n in wanted]
    if not specs:
        raise ValueError(f"No valid cameras in {names!r}; expected center/left/right")

    cameras: dict[str, carla.Sensor] = {}
    for i, (name, location, yaw) in enumerate(specs):
        cameras[name] = attach_rgb_camera(
            world,
            vehicle,
            width=width,
            height=height,
            fov=fov,
            location=location,
            yaw=yaw,
        )
        if stagger_ticks > 0 and i < len(specs) - 1:
            for _ in range(stagger_ticks):
                world.tick()
    return cameras


def recovery_steer(camera: str, steer: float) -> float:
    """Add a small centering nudge for a laterally shifted camera.

    Skipped once the expert is already turning (|steer|>=0.12) so a corner
    label does not flip sign. On a straight approach the nudge stays, including
    under TURN_* intent, so the branch learns "stay in lane until the corner."
    """
    extra = RECOVERY_BY_CAMERA.get(camera, 0.0)
    if extra == 0.0 or abs(steer) >= 0.12:
        return max(-1.0, min(1.0, steer))
    return max(-1.0, min(1.0, steer + extra))
