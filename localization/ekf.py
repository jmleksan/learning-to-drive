"""Extended Kalman Filter for planar ego pose.

State: [e, n, yaw, v]
  e, n — local east / north meters
  yaw  — heading radians, 0 = north, increasing toward east (ENU compass)
  v    — forward speed m/s

Predict from wheel speed + IMU yaw rate.
Update from GNSS (e, n) and optional compass heading.
"""

from __future__ import annotations

import math

import numpy as np


def wrap_angle(angle: float) -> float:
    """Wrap to (-pi, pi]."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def compass_to_yaw_enu(compass_rad: float) -> float:
    """CARLA IMU compass → ENU yaw (0 = north, + toward east).

    Empirically ``yaw = π - compass`` matches GT heading on Town maps.
    """
    return wrap_angle(math.pi - float(compass_rad))


class EgoEKF:
    """4-state bicycle / constant-turn EKF."""

    def __init__(
        self,
        *,
        x0: np.ndarray | None = None,
        p0_diag: tuple[float, float, float, float] = (25.0, 25.0, 0.5, 4.0),
        q_diag: tuple[float, float, float, float] = (0.05, 0.05, 0.05, 0.5),
        r_gnss_diag: tuple[float, float] = (9.0, 9.0),
        r_yaw: float = 0.05,
    ) -> None:
        self.x = (
            np.zeros(4, dtype=np.float64)
            if x0 is None
            else np.asarray(x0, dtype=np.float64).copy()
        )
        self.P = np.diag(np.asarray(p0_diag, dtype=np.float64))
        self.Q = np.diag(np.asarray(q_diag, dtype=np.float64))
        self.R = np.diag(np.asarray(r_gnss_diag, dtype=np.float64))
        self.r_yaw = float(r_yaw)

    @property
    def e(self) -> float:
        return float(self.x[0])

    @property
    def n(self) -> float:
        return float(self.x[1])

    @property
    def yaw(self) -> float:
        return float(self.x[2])

    @property
    def v(self) -> float:
        return float(self.x[3])

    def predict(self, dt: float, *, yaw_rate: float, speed: float | None = None) -> None:
        """Propagate with yaw_rate (rad/s) and optional wheel-speed override."""
        if dt <= 0.0:
            return
        e, n, yaw, v = self.x
        if speed is not None and np.isfinite(speed):
            v = float(speed)

        # ENU: 0 yaw = north, +yaw toward east → de = v sin ψ, dn = v cos ψ.
        yaw_n = wrap_angle(yaw + yaw_rate * dt)
        e_n = e + v * math.sin(yaw) * dt
        n_n = n + v * math.cos(yaw) * dt
        # Speed held from odometry measurement in predict; mild process noise.
        v_n = v

        # Jacobian of f wrt state (speed treated as measured input when given).
        F = np.eye(4)
        F[0, 2] = v * math.cos(yaw) * dt
        F[0, 3] = math.sin(yaw) * dt
        F[1, 2] = -v * math.sin(yaw) * dt
        F[1, 3] = math.cos(yaw) * dt
        F[2, 2] = 1.0

        self.x = np.array([e_n, n_n, yaw_n, v_n], dtype=np.float64)
        self.P = F @ self.P @ F.T + self.Q * max(dt, 1e-3)

    def update_gnss(self, e_m: float, n_m: float) -> None:
        """GNSS position update in the same ENU frame."""
        if not (np.isfinite(e_m) and np.isfinite(n_m)):
            return
        H = np.zeros((2, 4), dtype=np.float64)
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        z = np.array([e_m, n_m], dtype=np.float64)
        y = z - H @ self.x
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.x[2] = wrap_angle(float(self.x[2]))
        self.P = (np.eye(4) - K @ H) @ self.P

    def update_speed(self, speed_mps: float, *, r: float = 0.5) -> None:
        """Soft update on forward speed from wheel odometry."""
        if not np.isfinite(speed_mps):
            return
        H = np.array([[0.0, 0.0, 0.0, 1.0]])
        innov = np.array([speed_mps - self.x[3]])
        S = H @ self.P @ H.T + np.array([[r]])
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + (K @ innov).ravel()
        self.P = (np.eye(4) - K @ H) @ self.P

    def update_yaw(self, yaw_rad: float, *, r: float | None = None) -> None:
        """Absolute heading update (compass / track angle), angle-wrapped."""
        if not np.isfinite(yaw_rad):
            return
        meas_r = self.r_yaw if r is None else float(r)
        H = np.array([[0.0, 0.0, 1.0, 0.0]])
        innov = np.array([wrap_angle(float(yaw_rad) - float(self.x[2]))])
        S = H @ self.P @ H.T + np.array([[meas_r]])
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + (K @ innov).ravel()
        self.x[2] = wrap_angle(float(self.x[2]))
        self.P = (np.eye(4) - K @ H) @ self.P

    def update_compass(self, compass_rad: float, *, r: float | None = None) -> None:
        """CARLA IMU compass → ENU yaw, then soft heading update."""
        if not np.isfinite(compass_rad):
            return
        self.update_yaw(compass_to_yaw_enu(compass_rad), r=r)
