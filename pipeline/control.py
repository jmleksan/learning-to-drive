"""Swappable classical controllers: PID, Pure Pursuit, Stanley, MPC, hybrid."""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

import carla
import numpy as np

from utils.state import speed_kmh
from utils.waypoint_follower import _crosstrack_m, _to_vehicle_frame, _wrap_deg

_WHEELBASE_M = 2.5
_MAX_STEER_RAD = math.radians(70.0)


@dataclass
class ControlTarget:
    """What the motion layer asks control to track."""

    target: carla.Location
    path_transform: carla.Transform
    desired_speed_kmh: float
    heading_yaw_deg: float | None = None


class Controller(ABC):
    name: str = "base"

    def reset(self) -> None:
        return None

    @abstractmethod
    def compute(
        self, vehicle: carla.Vehicle, goal: ControlTarget
    ) -> carla.VehicleControl:
        raise NotImplementedError


class PIDController(Controller):
    """Lateral CTE/heading PID + longitudinal speed PID (CARLA-style)."""

    name = "pid"

    def __init__(
        self,
        *,
        kp_lat: float = 0.85,
        ki_lat: float = 0.05,
        kd_lat: float = 0.15,
        kp_lon: float = 0.25,
        ki_lon: float = 0.02,
        kd_lon: float = 0.05,
        max_steer: float = 0.9,
        max_throttle: float = 0.75,
        max_brake: float = 0.4,
    ) -> None:
        self.kp_lat, self.ki_lat, self.kd_lat = kp_lat, ki_lat, kd_lat
        self.kp_lon, self.ki_lon, self.kd_lon = kp_lon, ki_lon, kd_lon
        self.max_steer = max_steer
        self.max_throttle = max_throttle
        self.max_brake = max_brake
        self._i_lat = 0.0
        self._i_lon = 0.0
        self._prev_lat = 0.0
        self._prev_lon = 0.0
        self._prev_steer = 0.0

    def reset(self) -> None:
        self._i_lat = self._i_lon = 0.0
        self._prev_lat = self._prev_lon = 0.0
        self._prev_steer = 0.0

    def compute(
        self, vehicle: carla.Vehicle, goal: ControlTarget
    ) -> carla.VehicleControl:
        tf = vehicle.get_transform()
        cte = _crosstrack_m(goal.path_transform, tf.location)
        path_yaw = (
            goal.heading_yaw_deg
            if goal.heading_yaw_deg is not None
            else goal.path_transform.rotation.yaw
        )
        heading_err = math.radians(_wrap_deg(path_yaw - tf.rotation.yaw))
        # Combined lateral error for PID (CTE in meters + heading).
        lat_err = heading_err + math.atan2(-cte, 8.0)
        self._i_lat = max(-1.0, min(1.0, self._i_lat + lat_err * 0.05))
        d_lat = lat_err - self._prev_lat
        self._prev_lat = lat_err
        steer = (
            self.kp_lat * lat_err
            + self.ki_lat * self._i_lat
            + self.kd_lat * d_lat
        )
        steer = max(-self.max_steer, min(self.max_steer, steer))
        # Rate limit
        steer = max(self._prev_steer - 0.12, min(self._prev_steer + 0.12, steer))
        self._prev_steer = steer

        speed = speed_kmh(vehicle)
        lon_err = goal.desired_speed_kmh - speed
        self._i_lon = max(-5.0, min(5.0, self._i_lon + lon_err * 0.05))
        d_lon = lon_err - self._prev_lon
        self._prev_lon = lon_err
        accel = (
            self.kp_lon * lon_err
            + self.ki_lon * self._i_lon
            + self.kd_lon * d_lon
        )
        if accel >= 0.0:
            throttle = min(self.max_throttle, accel)
            brake = 0.0
        else:
            throttle = 0.0
            brake = min(self.max_brake, -accel)
        return carla.VehicleControl(
            steer=float(steer), throttle=float(throttle), brake=float(brake)
        )


class PurePursuitController(Controller):
    name = "pure_pursuit"

    def __init__(
        self,
        *,
        steer_gain: float = 1.2,
        throttle_gain: float = 0.1,
        brake_gain: float = 0.08,
        max_steer: float = 0.95,
    ) -> None:
        self.steer_gain = steer_gain
        self.throttle_gain = throttle_gain
        self.brake_gain = brake_gain
        self.max_steer = max_steer
        self._prev_steer = 0.0

    def reset(self) -> None:
        self._prev_steer = 0.0

    def compute(
        self, vehicle: carla.Vehicle, goal: ControlTarget
    ) -> carla.VehicleControl:
        tf = vehicle.get_transform()
        local_x, local_y = _to_vehicle_frame(tf, goal.target)
        if local_x < 0.3:
            steer = math.copysign(0.9, local_y) * self.steer_gain
        else:
            ld_sq = local_x * local_x + local_y * local_y
            curvature = 2.0 * local_y / max(ld_sq, 1e-4)
            steer = (_WHEELBASE_M * curvature) / _MAX_STEER_RAD * self.steer_gain
        steer = max(-self.max_steer, min(self.max_steer, steer))
        steer = 0.3 * self._prev_steer + 0.7 * steer
        self._prev_steer = steer
        return _speed_control(
            vehicle,
            goal.desired_speed_kmh,
            steer,
            self.throttle_gain,
            self.brake_gain,
        )


class StanleyController(Controller):
    name = "stanley"

    def __init__(
        self,
        *,
        cte_gain: float = 0.72,
        heading_gain: float = 0.95,
        throttle_gain: float = 0.1,
        brake_gain: float = 0.08,
        max_steer: float = 0.95,
    ) -> None:
        self.cte_gain = cte_gain
        self.heading_gain = heading_gain
        self.throttle_gain = throttle_gain
        self.brake_gain = brake_gain
        self.max_steer = max_steer
        self._prev_steer = 0.0

    def reset(self) -> None:
        self._prev_steer = 0.0

    def compute(
        self, vehicle: carla.Vehicle, goal: ControlTarget
    ) -> carla.VehicleControl:
        tf = vehicle.get_transform()
        cte = _crosstrack_m(goal.path_transform, tf.location)
        path_yaw = (
            goal.heading_yaw_deg
            if goal.heading_yaw_deg is not None
            else goal.path_transform.rotation.yaw
        )
        heading_err = math.radians(_wrap_deg(path_yaw - tf.rotation.yaw))
        speed_m_s = max(speed_kmh(vehicle) / 3.6, 0.8)
        steer = self.heading_gain * heading_err + self.cte_gain * math.atan(
            -cte / speed_m_s
        )
        steer = max(-self.max_steer, min(self.max_steer, steer))
        steer = 0.25 * self._prev_steer + 0.75 * steer
        self._prev_steer = steer
        return _speed_control(
            vehicle,
            goal.desired_speed_kmh,
            steer,
            self.throttle_gain,
            self.brake_gain,
        )


class HybridController(Controller):
    """Blended Pure Pursuit + Stanley (matches utils.waypoint_follower spirit)."""

    name = "hybrid"

    def __init__(self, **kwargs) -> None:
        self._pp = PurePursuitController(**{
            k: kwargs[k]
            for k in ("steer_gain", "throttle_gain", "brake_gain", "max_steer")
            if k in kwargs
        })
        self._stanley = StanleyController(**{
            k: kwargs[k]
            for k in (
                "cte_gain",
                "heading_gain",
                "throttle_gain",
                "brake_gain",
                "max_steer",
            )
            if k in kwargs
        })
        self._prev_steer = 0.0
        self.max_steer = float(kwargs.get("max_steer", 0.95))

    def reset(self) -> None:
        self._pp.reset()
        self._stanley.reset()
        self._prev_steer = 0.0

    def compute(
        self, vehicle: carla.Vehicle, goal: ControlTarget
    ) -> carla.VehicleControl:
        pp = self._pp.compute(vehicle, goal)
        st = self._stanley.compute(vehicle, goal)
        cte = abs(_crosstrack_m(goal.path_transform, vehicle.get_location()))
        # More Stanley when off path.
        w = 0.55 if cte > 0.35 else 0.65
        steer = (1.0 - w) * pp.steer + w * st.steer
        steer = max(-self.max_steer, min(self.max_steer, steer))
        steer = 0.2 * self._prev_steer + 0.8 * steer
        self._prev_steer = steer
        throttle = max(pp.throttle, st.throttle) if pp.brake == 0 and st.brake == 0 else 0.0
        brake = max(pp.brake, st.brake)
        if goal.desired_speed_kmh <= 0.5:
            throttle, brake = 0.0, max(brake, 0.55)
        return carla.VehicleControl(steer=float(steer), throttle=float(throttle), brake=float(brake))


class MPCController(Controller):
    """Simple kinematic bicycle MPC over a short horizon."""

    name = "mpc"

    def __init__(
        self,
        *,
        horizon: int = 8,
        dt_s: float = 0.1,
        max_steer: float = 0.85,
        max_accel: float = 2.5,
        w_cte: float = 2.5,
        w_heading: float = 1.5,
        w_speed: float = 0.4,
        w_steer: float = 0.3,
        w_accel: float = 0.05,
    ) -> None:
        self.horizon = horizon
        self.dt_s = dt_s
        self.max_steer = max_steer
        self.max_accel = max_accel
        self.w_cte = w_cte
        self.w_heading = w_heading
        self.w_speed = w_speed
        self.w_steer = w_steer
        self.w_accel = w_accel
        self._prev_steer = 0.0

    def reset(self) -> None:
        self._prev_steer = 0.0

    def compute(
        self, vehicle: carla.Vehicle, goal: ControlTarget
    ) -> carla.VehicleControl:
        tf = vehicle.get_transform()
        v0 = max(speed_kmh(vehicle) / 3.6, 0.1)
        path_yaw = math.radians(
            goal.heading_yaw_deg
            if goal.heading_yaw_deg is not None
            else goal.path_transform.rotation.yaw
        )
        yaw0 = math.radians(tf.rotation.yaw)
        # Local frame at ego: path point relative.
        local_x, local_y = _to_vehicle_frame(tf, goal.target)
        cte0 = _crosstrack_m(goal.path_transform, tf.location)
        v_des = max(goal.desired_speed_kmh / 3.6, 0.0)

        best_cost = float("inf")
        best_steer = 0.0
        best_accel = 0.0
        # Coarse open-loop search (no external QP solver).
        steer_samples = np.linspace(-self.max_steer, self.max_steer, 9)
        accel_samples = np.linspace(-self.max_accel, self.max_accel, 7)
        for steer_cmd in steer_samples:
            for accel in accel_samples:
                x, y, yaw, v = 0.0, 0.0, 0.0, v0
                cost = 0.0
                for _ in range(self.horizon):
                    # Bicycle in ego-at-t0 frame.
                    delta = steer_cmd * _MAX_STEER_RAD
                    yaw += (v / _WHEELBASE_M) * math.tan(delta) * self.dt_s
                    x += v * math.cos(yaw) * self.dt_s
                    y += v * math.sin(yaw) * self.dt_s
                    v = max(0.0, v + accel * self.dt_s)
                    # Approximate CTE as lateral vs target bearing / path.
                    heading_err = _wrap_rad(yaw - (path_yaw - yaw0))
                    # Soft target: progress toward local_x/local_y.
                    cte = (y - local_y * (x / max(local_x, 1.0))) if local_x > 1.0 else y + cte0
                    cost += (
                        self.w_cte * cte * cte
                        + self.w_heading * heading_err * heading_err
                        + self.w_speed * (v - v_des) ** 2
                    )
                cost += self.w_steer * steer_cmd * steer_cmd + self.w_accel * accel * accel
                cost += 0.15 * (steer_cmd - self._prev_steer) ** 2
                if cost < best_cost:
                    best_cost = cost
                    best_steer = float(steer_cmd)
                    best_accel = float(accel)

        steer = max(self._prev_steer - 0.15, min(self._prev_steer + 0.15, best_steer))
        self._prev_steer = steer
        if v_des <= 0.15:
            return carla.VehicleControl(steer=steer, throttle=0.0, brake=0.6)
        if best_accel >= 0.0:
            throttle = min(0.75, best_accel / self.max_accel)
            brake = 0.0
        else:
            throttle = 0.0
            brake = min(0.5, -best_accel / self.max_accel)
        return carla.VehicleControl(
            steer=float(steer), throttle=float(throttle), brake=float(brake)
        )


def _wrap_rad(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _speed_control(
    vehicle: carla.Vehicle,
    desired_kmh: float,
    steer: float,
    throttle_gain: float,
    brake_gain: float,
) -> carla.VehicleControl:
    if desired_kmh <= 0.5:
        return carla.VehicleControl(steer=float(steer), throttle=0.0, brake=0.65)
    speed = speed_kmh(vehicle)
    err = desired_kmh - speed
    if err >= 0:
        throttle = min(0.7, throttle_gain * err)
        brake = 0.0
    else:
        throttle = 0.0
        brake = min(0.35, brake_gain * (-err))
    if brake > 0.04:
        throttle = 0.0
    return carla.VehicleControl(
        steer=float(steer), throttle=float(throttle), brake=float(brake)
    )


CONTROLLERS = {
    "pid": PIDController,
    "pure_pursuit": PurePursuitController,
    "stanley": StanleyController,
    "hybrid": HybridController,
    "mpc": MPCController,
}


def make_controller(name: str, **kwargs) -> Controller:
    key = (name or "hybrid").strip().lower()
    if key not in CONTROLLERS:
        raise ValueError(f"Unknown controller {name!r}; choose from {sorted(CONTROLLERS)}")
    return CONTROLLERS[key](**kwargs)
