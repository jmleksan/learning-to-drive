"""Waypoints → steer / throttle / brake. TransFuser's PID controller.

Steer aims at the midpoint of the first two waypoints. Target speed is the
spacing of those two waypoints (they are 0.5 s apart), so the net slows
down or stops by bunching its waypoints, not through a brake output.

Unlike TransFuser, braking is proportional. A full brake on any overspeed
turned one-frame target-speed wobbles into hard stops: the expert almost never
slows on an open road, so at low speed the net then predicts a stop for real.
Firm braking needs the target to stay far below the current speed for several
frames in a row, so a wobble stays a light touch but closing on a stopped car
does not.
"""
from __future__ import annotations

import math
from collections import deque

import numpy as np

from transfuser.data import WAYPOINT_DT_S


class PID:
    def __init__(self, kp: float, ki: float, kd: float, n: int) -> None:
        self.kp, self.ki, self.kd = kp, ki, kd
        self.window: deque[float] = deque([0.0] * n, maxlen=n)

    def step(self, error: float) -> float:
        self.window.append(error)
        integral = float(np.mean(self.window))
        derivative = self.window[-1] - self.window[-2]
        return self.kp * error + self.ki * integral + self.kd * derivative


class WaypointController:
    # TransFuser defaults (tuned at 20 FPS).
    BRAKE_SPEED_MPS = 0.4
    BRAKE_RATIO = 1.1
    CLIP_DELTA = 0.25
    MAX_THROTTLE = 0.75
    # Added here for proportional braking.
    BRAKE_GAIN = 0.15  # brake per m/s over target
    SOFT_BRAKE_MAX = 0.3
    FIRM_RATIO = 0.6  # target under this share of current speed
    FIRM_CONFIRM_FRAMES = 3
    HOLD_SPEED_MPS = 0.3

    def __init__(self) -> None:
        self.turn_pid = PID(1.25, 0.75, 0.3, 40)
        self.speed_pid = PID(5.0, 0.5, 1.0, 40)
        self.firm_frames = 0

    def step(self, waypoints: np.ndarray, speed_mps: float) -> tuple[float, float, float, dict]:
        """waypoints: [4, 2] ego frame (x fwd, y right). Returns steer, throttle, brake, debug."""
        desired = float(np.linalg.norm(waypoints[0] - waypoints[1])) / WAYPOINT_DT_S
        stopping = desired < self.BRAKE_SPEED_MPS
        braking = stopping or (speed_mps / max(desired, 1e-3)) > self.BRAKE_RATIO
        firm = stopping or desired < self.FIRM_RATIO * speed_mps
        self.firm_frames = self.firm_frames + 1 if firm else 0

        over = max(0.0, speed_mps - desired)
        brake = 0.0
        if self.firm_frames >= self.FIRM_CONFIRM_FRAMES:
            hold = stopping and speed_mps < self.HOLD_SPEED_MPS
            brake = 1.0 if hold else float(np.clip(self.BRAKE_GAIN * over, self.SOFT_BRAKE_MAX, 1.0))
        elif braking:
            brake = min(self.SOFT_BRAKE_MAX, self.BRAKE_GAIN * over)

        delta = float(np.clip(desired - speed_mps, 0.0, self.CLIP_DELTA))
        throttle = float(np.clip(self.speed_pid.step(delta), 0.0, self.MAX_THROTTLE))
        if braking:
            throttle = 0.0

        aim = 0.5 * (waypoints[0] + waypoints[1])
        angle = math.degrees(math.atan2(float(aim[1]), float(aim[0]))) / 90.0
        # Stopped with bunched waypoints: the aim angle is noise.
        if stopping:
            angle = 0.0
        steer = float(np.clip(self.turn_pid.step(angle), -1.0, 1.0))
        debug = {"desired_mps": desired, "aim_angle": angle, "braking": braking}
        return steer, throttle, brake, debug
