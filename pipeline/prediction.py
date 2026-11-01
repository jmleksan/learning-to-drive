"""Other-agent trajectory prediction (constant velocity → optional learned stub)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PredictedAgent:
    """One agent with a future polyline in ego frame (x forward, y right)."""

    track_id: int
    class_id: int
    name: str
    # (T+1, 2) including t=0 current position
    trajectory_xy: np.ndarray
    length_m: float = 4.5
    width_m: float = 2.0


def constant_velocity_predict(
    tracks: list,
    *,
    horizon_s: float = 3.0,
    dt_s: float = 0.1,
) -> list[PredictedAgent]:
    """Roll each track forward with its ego-frame velocity.

    ``tracks`` items need ``track_id``, ``class_id``, ``name``, ``x_m``, ``y_m``,
    ``vx_mps``, ``vy_mps``, and optionally ``length_m`` / ``width_m``
    (compatible with ``perception.track.TrackedObject``).
    """
    steps = max(1, int(round(horizon_s / max(dt_s, 1e-3))))
    out: list[PredictedAgent] = []
    for track in tracks:
        x0 = float(getattr(track, "x_m", 0.0))
        y0 = float(getattr(track, "y_m", 0.0))
        vx = float(getattr(track, "vx_mps", 0.0))
        vy = float(getattr(track, "vy_mps", 0.0))
        traj = np.zeros((steps + 1, 2), dtype=np.float64)
        for i in range(steps + 1):
            t = i * dt_s
            traj[i, 0] = x0 + vx * t
            traj[i, 1] = y0 + vy * t
        out.append(
            PredictedAgent(
                track_id=int(getattr(track, "track_id", 0)),
                class_id=int(getattr(track, "class_id", 0)),
                name=str(getattr(track, "name", "agent")),
                trajectory_xy=traj,
                length_m=float(getattr(track, "length_m", 4.5)),
                width_m=float(getattr(track, "width_m", 2.0)),
            )
        )
    return out


def actors_to_ego_tracks(
    ego: object,
    actors: list,
    *,
    max_range_m: float = 50.0,
) -> list:
    """Build track-like objects from CARLA vehicle/walker actors (oracle sensors).

    Used when running the classical stack without SegNet/OccupancyNet.
    """
    import math

    from perception.track import TrackedObject

    ego_tf = ego.get_transform()
    ego_loc = ego_tf.location
    yaw = math.radians(ego_tf.rotation.yaw)
    c, s = math.cos(yaw), math.sin(yaw)
    tracks: list[TrackedObject] = []
    tid = 1
    for actor in actors:
        if actor.id == ego.id:
            continue
        type_id = getattr(actor, "type_id", "")
        if not (
            type_id.startswith("vehicle.") or type_id.startswith("walker.")
        ):
            continue
        loc = actor.get_location()
        dx = loc.x - ego_loc.x
        dy = loc.y - ego_loc.y
        # Ego frame: +x forward, +y right (CARLA left-handed → match BEV).
        x_m = c * dx + s * dy
        y_m = -s * dx + c * dy
        if x_m < -5.0 or x_m > max_range_m or abs(y_m) > 15.0:
            continue
        vel = actor.get_velocity()
        vx_w, vy_w = float(vel.x), float(vel.y)
        vx = c * vx_w + s * vy_w
        vy = -s * vx_w + c * vy_w
        is_walker = type_id.startswith("walker.")
        extent = actor.bounding_box.extent
        tracks.append(
            TrackedObject(
                track_id=tid,
                class_id=12 if is_walker else 14,
                name="pedestrian" if is_walker else "vehicle",
                x_m=x_m,
                y_m=y_m,
                length_m=max(0.5, 2.0 * float(extent.x)),
                width_m=max(0.4, 2.0 * float(extent.y)),
                cells=1,
                vx_mps=vx,
                vy_mps=vy,
                age=3,
            )
        )
        tid += 1
    return tracks
