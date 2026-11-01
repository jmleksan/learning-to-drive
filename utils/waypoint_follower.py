"""Tight path tracking for GRP waypoint routes (short lookahead).

Stage 4 also exposes the same ideas as swappable controllers in
``pipeline.control`` (``pid`` / ``pure_pursuit`` / ``stanley`` / ``hybrid`` /
``mpc``). This module remains the original blended follower used by
``expert.waypoint_drive``.
"""
from __future__ import annotations

import math

import carla

from utils.commands import RoadOption
from utils.routing import lookahead_target, nearest_route_index, upcoming_command
from utils.state import speed_kmh

_WHEELBASE_M = 2.5
_MAX_STEER_RAD = math.radians(70.0)


def _to_vehicle_frame(
    transform: carla.Transform,
    target: carla.Location,
) -> tuple[float, float]:
    dx = target.x - transform.location.x
    dy = target.y - transform.location.y
    yaw = math.radians(transform.rotation.yaw)
    local_x = math.cos(yaw) * dx + math.sin(yaw) * dy
    local_y = -math.sin(yaw) * dx + math.cos(yaw) * dy
    return local_x, local_y


def _wrap_deg(delta: float) -> float:
    return (delta + 180.0) % 360.0 - 180.0


def _pure_pursuit_steer(
    transform: carla.Transform,
    target: carla.Location,
    *,
    steer_gain: float,
) -> float:
    local_x, local_y = _to_vehicle_frame(transform, target)
    if local_x < 0.3:
        return max(-1.0, min(1.0, math.copysign(0.9, local_y) * steer_gain))
    ld_sq = local_x * local_x + local_y * local_y
    if ld_sq < 1e-4:
        return 0.0
    curvature = 2.0 * local_y / ld_sq
    steer = (_WHEELBASE_M * curvature) / _MAX_STEER_RAD
    return max(-1.0, min(1.0, steer * steer_gain))


def _crosstrack_m(path_tf: carla.Transform, vehicle_loc: carla.Location) -> float:
    """+ = vehicle is right of the path heading."""
    dx = vehicle_loc.x - path_tf.location.x
    dy = vehicle_loc.y - path_tf.location.y
    yaw = math.radians(path_tf.rotation.yaw)
    return -math.sin(yaw) * dx + math.cos(yaw) * dy


def _heading_error_rad(path_yaw_deg: float, vehicle_yaw_deg: float) -> float:
    return math.radians(_wrap_deg(path_yaw_deg - vehicle_yaw_deg))


def _path_curvature_ahead(
    route: list,
    start_index: int,
    *,
    window_m: float = 10.0,
) -> float:
    if not route or start_index >= len(route) - 1:
        return 0.0
    i = start_index
    traveled = 0.0
    yaw0 = route[i][0].transform.rotation.yaw
    prev = route[i][0].transform.location
    yaw1 = yaw0
    while i + 1 < len(route) and traveled < window_m:
        i += 1
        loc = route[i][0].transform.location
        traveled += prev.distance(loc)
        prev = loc
        yaw1 = route[i][0].transform.rotation.yaw
    if traveled < 1.0:
        return 0.0
    return abs(_wrap_deg(yaw1 - yaw0)) * math.pi / 180.0 / traveled


def _meters_until_bend(
    route: list,
    start_index: int,
    *,
    bend_curvature: float = 0.04,
    horizon_m: float = 32.0,
) -> float | None:
    """How far until the path actually bends. None if the horizon stays straight."""
    if not route:
        return None
    i = max(0, min(start_index, len(route) - 1))
    traveled = 0.0
    prev = route[i][0].transform.location
    while i < len(route) and traveled < horizon_m:
        if _path_curvature_ahead(route, i, window_m=8.0) >= bend_curvature:
            return traveled
        if i + 1 >= len(route):
            break
        i += 1
        loc = route[i][0].transform.location
        traveled += prev.distance(loc)
        prev = loc
    return None


class WaypointFollower:
    """Hug the polyline: short lookahead + strong crosstrack correction."""

    def __init__(
        self,
        *,
        cruise_kmh: float = 26.0,
        turn_kmh: float = 18.0,
        # Keep the pursuit point almost on top of the car.
        min_lookahead_m: float = 2.5,
        lookahead_speed_gain: float = 0.12,
        max_lookahead_m: float = 5.5,
        steer_gain: float = 1.35,
        cte_gain: float = 0.72,
        heading_gain: float = 0.95,
        throttle_gain: float = 0.1,
        brake_gain: float = 0.08,
        max_steer: float = 0.95,
        steer_smooth: float = 0.18,
        max_steer_rate: float = 0.20,
    ) -> None:
        self.cruise_kmh = cruise_kmh
        self.turn_kmh = turn_kmh
        self.min_lookahead_m = min_lookahead_m
        self.lookahead_speed_gain = lookahead_speed_gain
        self.max_lookahead_m = max_lookahead_m
        self.steer_gain = steer_gain
        self.cte_gain = cte_gain
        self.heading_gain = heading_gain
        self.throttle_gain = throttle_gain
        self.brake_gain = brake_gain
        self.max_steer = max_steer
        self.steer_smooth = steer_smooth
        self.max_steer_rate = max_steer_rate
        self.route_index = 0
        self._prev_steer = 0.0
        self._prev_desired = cruise_kmh

    def reset(self) -> None:
        self.route_index = 0
        self._prev_steer = 0.0
        self._prev_desired = self.cruise_kmh

    def _desired_speed_kmh(self, route: list, index: int, curvature: float) -> float:
        """Ramp cruise → turn speed from road shape only. The command is not used."""
        if curvature > 0.05:
            target = self.turn_kmh
        else:
            dist = _meters_until_bend(route, index)
            if dist is None:
                target = self.cruise_kmh
            else:
                # Straight at 32 m, turn speed as the bend arrives. No step.
                span = 32.0
                blend = min(1.0, max(0.0, dist / span))
                target = self.turn_kmh + blend * (self.cruise_kmh - self.turn_kmh)
        # Ease the setpoint so the brake stays a coast, not a stop.
        self._prev_desired = 0.9 * self._prev_desired + 0.1 * target
        return self._prev_desired

    def control(
        self,
        vehicle: carla.Vehicle,
        route: list[tuple[carla.Waypoint, RoadOption]] | list,
    ) -> tuple[carla.VehicleControl, carla.Location, str]:
        if not route:
            return carla.VehicleControl(brake=1.0), vehicle.get_location(), "FOLLOW_LANE"

        transform = vehicle.get_transform()
        loc = transform.location
        speed = speed_kmh(vehicle)
        command, self.route_index = upcoming_command(
            loc, route, start_hint=self.route_index
        )
        idx = nearest_route_index(loc, route, start_hint=self.route_index)
        self.route_index = idx

        curvature = _path_curvature_ahead(route, idx)
        # Command is the next maneuver, not "we are in the bend."
        turning = curvature > 0.04
        # Short lookahead always; modestly shorter in bends (not snap-turn).
        lookahead = min(
            self.max_lookahead_m,
            max(
                self.min_lookahead_m,
                self.min_lookahead_m + self.lookahead_speed_gain * speed,
            ),
        )
        if curvature > 0.03 or turning:
            lookahead = max(self.min_lookahead_m, lookahead * 0.55)

        target, self.route_index = lookahead_target(
            loc,
            route,
            start_hint=idx,
            lookahead_m=lookahead,
        )

        nearest_wp = route[idx][0]
        # Prefer the next waypoint's pose for heading when available.
        heading_wp = route[min(idx + 1, len(route) - 1)][0]
        path_tf = nearest_wp.transform
        cte = _crosstrack_m(path_tf, loc)
        heading_err = _heading_error_rad(
            heading_wp.transform.rotation.yaw, transform.rotation.yaw
        )

        # Strong Stanley: stay glued to the lane centerline through bends.
        speed_m_s = max(speed / 3.6, 0.8)
        cte_gain = self.cte_gain * (1.35 if turning else 1.0)
        stanley = (
            self.heading_gain * heading_err
            + cte_gain * math.atan(-cte / speed_m_s)
        )
        # Hard pull if off the ribbon — escalate earlier in bends.
        off_soft = 0.25 if turning else 0.35
        off_hard = 0.55 if turning else 0.8
        if abs(cte) > off_soft:
            stanley += -0.30 * math.copysign(min(abs(cte), 2.0), cte)
        if abs(cte) > off_hard:
            stanley += -0.28 * math.copysign(1.0, cte)

        pp = _pure_pursuit_steer(transform, target, steer_gain=self.steer_gain)
        # In bends prioritize Stanley/CTE so the demo never floats wide.
        if turning:
            steer = 0.45 * pp + 0.55 * stanley
            steer *= 1.15
        else:
            steer = 0.35 * pp + 0.65 * stanley
        steer = max(-self.max_steer, min(self.max_steer, steer))

        rate_lim = self.max_steer_rate * (1.2 if turning else 1.0)
        delta = max(
            -rate_lim,
            min(rate_lim, steer - self._prev_steer),
        )
        steer = self._prev_steer + delta
        # Less smoothing in bends so CTE corrections actually land.
        alpha = self.steer_smooth * (0.65 if turning else 1.0)
        steer = alpha * self._prev_steer + (1.0 - alpha) * steer
        steer = max(-self.max_steer, min(self.max_steer, steer))
        self._prev_steer = steer

        desired = self._desired_speed_kmh(route, idx, curvature)
        # If far off path, slow down so CTE can recover instead of plowing barriers.
        if abs(cte) > 0.9:
            desired = min(desired, 12.0)

        err = desired - speed
        if err >= 0:
            throttle = min(0.7, self.throttle_gain * err)
            brake = 0.0
        else:
            throttle = 0.0
            brake = min(0.2, self.brake_gain * (-err))
        if brake > 0.04:
            throttle = 0.0

        return (
            carla.VehicleControl(
                steer=float(steer),
                throttle=float(throttle),
                brake=float(brake),
            ),
            target,
            command,
        )
