"""TransFuser-lite as a Leaderboard / Bench2Drive agent.

Same net and controller as transfuser/drive.py, but with only what a benchmark
agent may use: sensors plus the route as GPS points. Three things differ from
the training setup and are bridged here:

  Lidar     The Leaderboard fixes it at 64 channels, 600k points/s, 10 Hz. At
            20 FPS that is half a spin per frame, so two frames make one spin
            (the older one moved into the current ego frame), thinned to the
            density of the 32-channel, 200k/s, 20 Hz training lidar. It also
            only allows ray_cast, not semantic; the net ignores tags anyway.
  Position  GPS noise is about 0.5 m. A filter dead-reckons on speed and
            compass and leans slowly toward GPS, so the targets don't jitter.
  Route     Given as GPS points. Past the end it is extended straight on, so
            targets keep the distances the net was trained on.

Config string: path to the checkpoint (the evaluator appends "+<route name>").
"""
from __future__ import annotations

import math
import os

import carla
import numpy as np
import torch

from leaderboard.autoagents import autonomous_agent
from perception.lidar import LIDAR_X, LIDAR_Y, LIDAR_Z
from transfuser.controller import WaypointController
from transfuser.data import CAMERAS, TARGET_FAR_M, TARGET_M, image_strip, lidar_input
from transfuser.model import load_checkpoint

FPS = 20.0
DT_S = 1.0 / FPS
CAMERA_WIDTH = 160
CAMERA_HEIGHT = 120
CAMERA_FOV = 90.0
# Same mounts as utils/vehicles.py CAMERA_SPECS.
CAMERA_MOUNTS = {
    "left": (1.5, -0.40, 2.4, -55.0),
    "center": (1.5, 0.0, 2.4, 0.0),
    "right": (1.5, 0.40, 2.4, 55.0),
}
# Share of points to keep. Measured, not the 1/6 ratio of rays per spin: 1/6 gave
# 4,950 points per frame on Town05 against about 8,000 in training.
LIDAR_KEEP = 0.27
METERS_PER_DEGREE = 111319.49
GPS_GAIN = 0.05
ROUTE_SEARCH_POINTS = 60
DEBUG_EVERY_STEPS = 100


def get_entry_point() -> str:
    return "TransfuserAgent"


def gps_to_xy(lat: float, lon: float) -> np.ndarray:
    """Lat/lon → world meters (x east, y south, like CARLA)."""
    return np.array([lon * METERS_PER_DEGREE, -lat * METERS_PER_DEGREE])


def heading_axes(compass: float) -> tuple[np.ndarray, np.ndarray]:
    """Compass (0 = north, clockwise) → forward and right unit vectors."""
    fwd = np.array([math.sin(compass), -math.cos(compass)])
    right = np.array([math.cos(compass), math.sin(compass)])
    return fwd, right


class RouteTargets:
    """Tracks progress along the route and reads points ahead of the car."""

    def __init__(self, route_xy: np.ndarray) -> None:
        keep = np.r_[True, np.linalg.norm(np.diff(route_xy, axis=0), axis=1) > 1e-3]
        self.xy = route_xy[keep]
        self.s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(self.xy, axis=0), axis=1))]
        self.index = 0

    def point_at(self, s: float) -> np.ndarray:
        if s >= self.s[-1]:
            tail = self.xy[-1] - self.xy[-2]
            return self.xy[-1] + tail / np.linalg.norm(tail) * (s - self.s[-1])
        return np.array([np.interp(s, self.s, self.xy[:, 0]), np.interp(s, self.s, self.xy[:, 1])])

    def ahead(self, pos: np.ndarray, distances: tuple[float, ...]) -> list[np.ndarray]:
        # Only search forward, so a route that doubles back can't jump ahead.
        window = self.xy[self.index : self.index + ROUTE_SEARCH_POINTS]
        self.index += int(np.argmin(np.linalg.norm(window - pos, axis=1)))
        return [self.point_at(self.s[self.index] + d) for d in distances]


class TransfuserAgent(autonomous_agent.AutonomousAgent):
    def setup(self, path_to_conf_file: str) -> None:
        self.track = autonomous_agent.Track.SENSORS
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, _ = load_checkpoint(path_to_conf_file.split("+")[0], self.device)
        self.controller = WaypointController()
        self.targets = RouteTargets(
            np.array([gps_to_xy(p["lat"], p["lon"]) for p, _ in self._plan_gps_HACK])
        )
        self.rng = np.random.default_rng(0)
        self.pos: np.ndarray | None = None
        self.compass = 0.0
        self.prev_sweep: np.ndarray | None = None
        self.step = 0
        self.debug_csv = None
        debug_path = os.environ.get("TF_AGENT_DEBUG")
        if debug_path:
            self.debug_csv = open(debug_path, "a", encoding="utf-8")

    def sensors(self) -> list[dict]:
        specs = [
            {
                "type": "sensor.camera.rgb",
                "id": name,
                "x": x, "y": y, "z": z, "roll": 0.0, "pitch": 0.0, "yaw": yaw,
                "width": CAMERA_WIDTH, "height": CAMERA_HEIGHT, "fov": CAMERA_FOV,
            }
            for name, (x, y, z, yaw) in CAMERA_MOUNTS.items()
        ]
        pose = {"x": 0.0, "y": 0.0, "z": 0.0, "roll": 0.0, "pitch": 0.0, "yaw": 0.0}
        return specs + [
            {
                "type": "sensor.lidar.ray_cast", "id": "lidar",
                "x": LIDAR_X, "y": LIDAR_Y, "z": LIDAR_Z, "roll": 0.0, "pitch": 0.0, "yaw": 0.0,
            },
            {"type": "sensor.other.gnss", "id": "gnss", **pose},
            {"type": "sensor.other.imu", "id": "imu", **pose},
            {"type": "sensor.speedometer", "id": "speed"},
        ]

    def _update_pose(self, lat: float, lon: float, compass: float, speed: float) -> float:
        """Returns the yaw change this frame (rad, clockwise positive)."""
        # The compass goes NaN for a frame or two at spawn.
        if math.isnan(compass):
            compass = self.compass
        turned = (compass - self.compass + math.pi) % (2 * math.pi) - math.pi
        self.compass = compass
        gps = gps_to_xy(lat, lon)
        if self.pos is None:
            self.pos = gps
            return 0.0
        self.pos = self.pos + heading_axes(compass)[0] * speed * DT_S
        self.pos = self.pos + GPS_GAIN * (gps - self.pos)
        return turned

    def _full_sweep(self, raw: np.ndarray, speed: float, turned: float) -> np.ndarray:
        """This half spin + the last one, in today's ego frame, at training density."""
        points = raw[:, :3].astype(np.float32).copy()
        points[:, 2] += LIDAR_Z  # sensor frame → the vehicle frame the net was trained on
        sweep = points
        if self.prev_sweep is not None:
            old = self.prev_sweep.copy()
            old[:, 0] -= speed * DT_S
            c, s = math.cos(turned), math.sin(turned)
            x, y = old[:, 0].copy(), old[:, 1].copy()
            old[:, 0] = c * x + s * y
            old[:, 1] = -s * x + c * y
            sweep = np.concatenate([old, points])
        self.prev_sweep = points
        keep = self.rng.random(len(sweep)) < LIDAR_KEEP
        # lidar_input expects x, y, z, tag and ignores the tag.
        return np.concatenate([sweep[keep], np.zeros((int(keep.sum()), 1), dtype=np.float32)], axis=1)

    @torch.no_grad()
    def run_step(self, input_data: dict, timestamp: float) -> carla.VehicleControl:
        # Signed here, and about -1 m/s just after spawn; training speed was never negative.
        speed = abs(float(input_data["speed"][1]["speed"]))
        lat, lon = input_data["gnss"][1][:2]
        compass = float(input_data["imu"][1][-1])
        turned = self._update_pose(float(lat), float(lon), compass, speed)

        fwd, right = heading_axes(self.compass)
        near, far = self.targets.ahead(self.pos, (TARGET_M, TARGET_FAR_M))
        target = np.array(
            [
                (near - self.pos) @ fwd, (near - self.pos) @ right,
                (far - self.pos) @ fwd, (far - self.pos) @ right,
            ],
            dtype=np.float32,
        )

        points = self._full_sweep(input_data["lidar"][1], speed, turned)
        rgb = [input_data[c][1][:, :, :3][:, :, ::-1] for c in CAMERAS]
        pred = self.model(
            torch.from_numpy(image_strip(rgb))[None].to(self.device) if self.model.use_camera else None,
            torch.from_numpy(lidar_input(points, target))[None].to(self.device) if self.model.use_lidar else None,
            torch.tensor([[speed]], dtype=torch.float32, device=self.device),
            torch.from_numpy(target)[None].to(self.device),
        )
        waypoints = pred["waypoints"][0].cpu().numpy()
        aff = torch.sigmoid(pred["affordance"][0]).cpu().numpy()
        self.step += 1
        steer, throttle, brake, dbg = self.controller.step(waypoints, speed)

        if self.debug_csv:
            t = self.hero_actor.get_transform()
            f, r = t.get_forward_vector(), t.get_right_vector()
            true_near = [
                (near[0] - t.location.x) * f.x + (near[1] - t.location.y) * f.y,
                (near[0] - t.location.x) * r.x + (near[1] - t.location.y) * r.y,
            ]
            self.debug_csv.write(
                f"{timestamp:.2f},{self.pos[0] - t.location.x:.3f},{self.pos[1] - t.location.y:.3f},"
                f"{target[0]:.3f},{target[1]:.3f},{true_near[0]:.3f},{true_near[1]:.3f},"
                f"{speed:.2f},{steer:.3f},{throttle:.3f},{brake:.3f},{len(points)},"
                f"{dbg['desired_mps']:.3f},{aff[0]:.3f},{aff[1]:.3f}\n"
            )
            self.debug_csv.flush()
            if self.step % DEBUG_EVERY_STEPS == 0:
                np.savez_compressed(
                    self.debug_csv.name.replace(".csv", f"_{self.step:05d}.npz"),
                    rgb=np.concatenate(rgb, axis=1), points=points, waypoints=waypoints, target=target,
                )
        return carla.VehicleControl(steer=steer, throttle=throttle, brake=brake)

    def destroy(self) -> None:
        if self.debug_csv:
            self.debug_csv.close()
