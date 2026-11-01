"""Semantic lidar on the roof. Each point is a measured hit plus a tag.

Points are saved in the vehicle frame: x forward, y right, z up.
The sensor sits at the roof center, 2.4 m up, the same height as the cameras.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

LIDAR_X = 0.0
LIDAR_Y = 0.0
LIDAR_Z = 2.4
# Hits closer than this are the ego body.
LIDAR_EGO_RADIUS_M = 2.5

_RAW_DTYPE = np.dtype(
    [
        ("x", np.float32),
        ("y", np.float32),
        ("z", np.float32),
        ("cos", np.float32),
        ("obj_idx", np.uint32),
        ("obj_tag", np.uint32),
    ]
)

__all__ = [
    "LIDAR_EGO_RADIUS_M",
    "LIDAR_X",
    "LIDAR_Y",
    "LIDAR_Z",
    "attach_semantic_lidar",
    "load_lidar_points",
    "points_from_measurement",
    "save_lidar_points",
]


def attach_semantic_lidar(world, vehicle, *, fps: float = 10.0):
    """Roof semantic lidar. One spin per recorded frame."""
    import carla

    bp = world.get_blueprint_library().find("sensor.lidar.ray_cast_semantic")
    settings = {
        "channels": "32",
        "range": "50",
        "points_per_second": "200000",
        "rotation_frequency": str(max(1.0, float(fps))),
        "upper_fov": "10",
        "lower_fov": "-30",
        "horizontal_fov": "360",
    }
    for key, value in settings.items():
        if bp.has_attribute(key):
            bp.set_attribute(key, value)
    transform = carla.Transform(carla.Location(x=LIDAR_X, y=LIDAR_Y, z=LIDAR_Z))
    return world.spawn_actor(bp, transform, attach_to=vehicle)


def points_from_measurement(measurement) -> np.ndarray:
    """Semantic lidar sweep → Nx4 float32 (x, y, z, tag) in the vehicle frame."""
    raw = np.frombuffer(measurement.raw_data, dtype=_RAW_DTYPE)
    points = np.zeros((len(raw), 4), dtype=np.float32)
    if len(raw) == 0:
        return points
    points[:, 0] = raw["x"] + LIDAR_X
    points[:, 1] = raw["y"] + LIDAR_Y
    points[:, 2] = raw["z"] + LIDAR_Z
    points[:, 3] = raw["obj_tag"].astype(np.float32)
    return points


def save_lidar_points(path: Path, points: np.ndarray) -> None:
    np.save(path, np.ascontiguousarray(points, dtype=np.float32))


def load_lidar_points(path: Path) -> np.ndarray:
    points = np.load(path).astype(np.float32)
    if points.ndim != 2 or points.shape[1] != 4:
        raise ValueError(f"{path} is {points.shape}, expected Nx4")
    return points
