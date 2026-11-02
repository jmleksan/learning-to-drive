"""Gymnasium CARLA env: stay in lane.

Default observation is privileged GPS/CTE. With camera=True the policy
sees stacked center RGB (same sensor as Step 1 IL) plus speed and a
turn command. With bev=True it sees a bird's-eye road + route grid
(rl/bev_obs.py) plus the same vec. The road channel is SegNet road from the
car's own cameras by default; bev_source="map" uses map truth instead. The
route always comes from the planner. Reward and termination always use the
map. speed_control=True makes the action steer-only; a P controller holds
the follower's speed profile. speed_setpoint=True makes it (steer, target
speed) instead, so the agent can stop.

Traffic (lights / npc_vehicles / walkers) needs bev + speed_setpoint. The BEV
gains vehicle / pedestrian / red-stop-line channels (true actor boxes), and the
reward caps speed for hazards on the route ahead (sim truth, like the map).
"""
from __future__ import annotations

import math
import queue
from collections import deque
from pathlib import Path

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from PIL import Image as PILImage

import carla

from rl.bev_obs import (
    BEV_CHANNELS,
    BEV_H,
    BEV_OBJECT_CHANNELS,
    BEV_W,
    MIN_WALKER_HALF_M,
    PERCEIVED_CAMERAS,
    BevObjects,
    BevRasterizer,
    PerceivedRoad,
    bev_to_rgb,
    road_agreement,
)
from rl.rewards import step_reward
from utils.commands import (
    COMMAND_DIM,
    COMMAND_TURN_LEFT,
    COMMAND_TURN_RIGHT,
    RoadOption,
    command_to_onehot,
)
from utils.connection import connect_client
from utils.drive_score import EventCounter, path_length_m, route_completion_pct
from utils.images import image_to_array
from utils.lane_metrics import lane_metrics_at
from utils.routing import (
    build_planner,
    farthest_spawn_index,
    geometric_command_from_route,
    hold_turn_command,
    nearest_route_index,
    pick_junction_heavy_destination,
    plan_spawn_to_spawn,
    route_xyz,
)
from utils.run_meta import ensure_map
from utils.spectator import follow_vehicle
from utils.state import speed_kmh
from utils.sync_mode import SynchronousMode
from utils.traffic import _actor_alive, clear_other_vehicles, spawn_npc_traffic
from utils.traffic_signs import speed_up_traffic_lights
from utils.vehicles import attach_rgb_camera, attach_rgb_cameras, spawn_vehicle
from utils.walkers import destroy_walkers, spawn_walkers

CTE_FAIL_M = 2.5
ROUTE_DONE_PCT = 95.0
WARMUP_TICKS = 5
# EMA mix for the old steer filter, only used with steer_smooth=True (models
# trained before it went away: rl_sac, rl_sac_cam). Default is raw steer; r_jerk
# and r_flip carry smoothness instead of lagging every command.
STEER_SMOOTH = 0.35
STEER_TURN_SMOOTH = 0.3
STEER_DEADBAND = 0.04
TURN_START_PROB = 0.7
# Camera teach-turns: most train episodes approach a real corner (left /
# right / go-straight). Approach is 18–40 m so the command is still "ahead".
# Turn episodes cap sooner so we reset into a new corner instead of grinding
# the same exit straight. Eval keeps spawn 0.
CAM_TURN_START_PROB = 0.85
CAM_APPROACH_M = (18.0, 40.0)
CAM_TURN_EPISODE_STEPS = 800
TURN_APPROACH_WPS = 12
# Stalled = under STALL_MIN_M of route progress in the last STALL_STEPS
# (5 s at 20 fps). Ends the episode with the stall penalty.
STALL_STEPS = 100
STALL_MIN_M = 3.0
# A camera turn episode counts as cleared once the car is this far past
# the corner waypoint it was spawned to approach (logged, not rewarded).
TURN_CLEAR_M = 15.0
# A missed turn that fails past this point (m before the corner waypoint)
# is labelled wide/tight; earlier failures count as "approach".
TURN_ENTRY_M = 5.0
TURN_LEFT_BIAS = 0.6

# NatureCNN-style: crop sky/hood, grayscale 84x84, stack 4. Same center camera as IL.
CAM_W = 84
CAM_H = 84
CAM_STACK = 4
CAM_CAPTURE_W = 160
CAM_CAPTURE_H = 120
CAM_CROP_TOP = 0.35
CAM_CROP_BOTTOM = 0.10
# speed, command one-hot, applied steer, turning flag
VEC_DIM = 1 + COMMAND_DIM + 2
_CAM_PREVIEW = Path("data/rl_cam_preview.png")
_CAM_PREVIEW_WRITTEN = False
_BEV_PREVIEW = Path("data/rl_bev_preview.png")
_BEV_PREVIEW_WRITTEN = False

# speed_control: same profile and gains as utils.waypoint_follower, which
# clears every turn in eval_turns. Turn speed kicks in with the turning flag.
SPEED_CRUISE_KMH = 18.0
SPEED_TURN_KMH = 14.0
SPEED_EASE = 0.1
SPEED_THROTTLE_GAIN = 0.1
SPEED_MAX_THROTTLE = 0.7
SPEED_BRAKE_GAIN = 0.08
SPEED_MAX_BRAKE = 0.2

# speed_setpoint: action[1] in [-1, 1] -> 0..SETPOINT_MAX_KMH target tracked by
# the same P throttle. The agent decides stop / slow / go; braking is firmer
# than the follower's coast so it can actually stop for a red.
SETPOINT_MAX_KMH = 25.0
SETPOINT_BRAKE_GAIN = 0.12

# Traffic hazards on the route ahead -> reward speed cap. The cap ramps from 0
# at the stop point up to cruise, by *_KMH_PER_M for each metre of room.
EGO_HALF_LENGTH_M = 2.4
LIGHT_LOOKAHEAD_M = 30.0
LIGHT_STOP_M = 1.5
LIGHT_KMH_PER_M = 1.2
# A yellow this close is a commit, not a stop.
YELLOW_COMMIT_M = 4.0
LEAD_LOOKAHEAD_M = 30.0
LEAD_LATERAL_M = 2.0
LEAD_GAP_M = 4.0
LEAD_KMH_PER_M = 1.5
WALKER_LOOKAHEAD_M = 20.0
WALKER_LATERAL_M = 2.5
WALKER_GAP_M = 5.0
WALKER_KMH_PER_M = 1.5
# Under this cap a stop is legitimate: no stall counting.
STOP_OK_KMH = 3.0
# Shorter light cycles so training spends less time waiting (server-wide).
LIGHT_RED_S, LIGHT_YELLOW_S, LIGHT_GREEN_S = 6.0, 2.0, 8.0
BEV_OBJECT_RADIUS_M = 45.0


def _rgb_to_gray84(image: carla.Image) -> np.ndarray:
    rgb = image_to_array(image)
    h, _w = rgb.shape[:2]
    y0 = int(h * CAM_CROP_TOP)
    y1 = max(y0 + 1, int(h * (1.0 - CAM_CROP_BOTTOM)))
    crop = rgb[y0:y1, :, :]
    gray = np.asarray(
        PILImage.fromarray(crop).convert("L").resize((CAM_W, CAM_H), PILImage.BILINEAR),
        dtype=np.uint8,
    )
    return gray[None, ...]


def env_kwargs_for_model(model) -> dict:
    """camera / bev / action flags matching a saved policy's spaces.

    Traffic itself (lights / npc_vehicles / walkers) is an eval choice, not
    stored in the checkpoint; pass those separately.
    """
    act = model.action_space
    kwargs = {
        "speed_control": act.shape[0] == 1,
        # Setpoint and throttle actions are both 2-D; setpoint's low is -1.
        "speed_setpoint": act.shape[0] == 2 and float(act.low[1]) < -0.5,
    }
    if isinstance(model.observation_space, spaces.Dict):
        shape = tuple(model.observation_space["image"].shape)
        bev = shape[1:] == (BEV_H, BEV_W) and shape[0] in (BEV_CHANNELS, BEV_OBJECT_CHANNELS)
        kwargs["bev"] = bev
        kwargs["camera"] = not bev
        kwargs["bev_objects"] = bev and shape[0] == BEV_OBJECT_CHANNELS
    return kwargs


class CarlaLaneKeepEnv(gym.Env):
    """Single-vehicle Town01 lane-keep. GPS obs by default; RGB+vec if camera."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        host: str | None = None,
        port: int = 2000,
        map_name: str = "Town01",
        spawn_index: int = 0,
        fps: float = 20.0,
        max_steps: int = 600,
        follow: bool | None = None,
        no_rendering: bool = False,
        turn_start_prob: float | None = None,
        camera: bool = False,
        bev: bool = False,
        bev_source: str = "perceived",
        seg_path: str | Path = "models/seg.pt",
        speed_control: bool = False,
        speed_setpoint: bool = False,
        lights: bool = False,
        npc_vehicles: int = 0,
        walkers: int = 0,
        bev_objects: bool | None = None,
        fixed_route: bool = False,
        steer_smooth: bool = False,
    ) -> None:
        super().__init__()
        self.map_name = map_name
        self.spawn_index = spawn_index
        self.fps = float(fps)
        self.max_steps = int(max_steps)
        self.camera = bool(camera)
        self.bev = bool(bev)
        if self.camera and self.bev:
            raise ValueError("pick one of camera or bev")
        if bev_source not in ("map", "perceived"):
            raise ValueError(f"bev_source must be map or perceived, not {bev_source!r}")
        if bev_source == "perceived" and not self.bev:
            raise ValueError("bev_source='perceived' needs bev=True")
        self.bev_source = bev_source
        # Image-style obs share the turn-teaching episode setup.
        self.image_obs = self.camera or self.bev
        self.speed_control = bool(speed_control)
        self.speed_setpoint = bool(speed_setpoint)
        if self.speed_control and self.speed_setpoint:
            raise ValueError("pick one of speed_control or speed_setpoint")
        self.lights = bool(lights)
        self.npc_vehicles = max(0, int(npc_vehicles))
        self.n_walkers = max(0, int(walkers))
        self.traffic = self.lights or self.npc_vehicles > 0 or self.n_walkers > 0
        if self.traffic and not (self.bev and self.speed_setpoint):
            raise ValueError(
                "traffic (lights / npc_vehicles / walkers) needs bev=True and "
                "speed_setpoint=True"
            )
        self.bev_objects = self.traffic if bev_objects is None else bool(bev_objects)
        if self.bev_objects and not self.bev:
            raise ValueError("bev_objects needs bev=True")
        self.fixed_route = bool(fixed_route)
        self.steer_smooth = bool(steer_smooth)
        if (self.camera or self.bev_source == "perceived") and no_rendering:
            print("camera obs needs the renderer; ignoring --no-render")
            no_rendering = False
        self.no_rendering = bool(no_rendering)
        self.turn_start_prob = (
            TURN_START_PROB if turn_start_prob is None else float(turn_start_prob)
        )
        self.follow = (not self.no_rendering) if follow is None else bool(follow)

        if self.speed_control:
            self.action_space = spaces.Box(
                low=np.array([-1.0], dtype=np.float32),
                high=np.array([1.0], dtype=np.float32),
                dtype=np.float32,
            )
        elif self.speed_setpoint:
            self.action_space = spaces.Box(
                low=np.array([-1.0, -1.0], dtype=np.float32),
                high=np.array([1.0, 1.0], dtype=np.float32),
                dtype=np.float32,
            )
        else:
            self.action_space = spaces.Box(
                low=np.array([-1.0, 0.0], dtype=np.float32),
                high=np.array([1.0, 1.0], dtype=np.float32),
                dtype=np.float32,
            )
        if self.image_obs:
            image_shape = (
                (
                    BEV_OBJECT_CHANNELS if self.bev_objects else BEV_CHANNELS,
                    BEV_H,
                    BEV_W,
                )
                if self.bev
                else (CAM_STACK, CAM_H, CAM_W)
            )
            self.observation_space = spaces.Dict(
                {
                    "image": spaces.Box(
                        low=0,
                        high=255,
                        shape=image_shape,
                        dtype=np.uint8,
                    ),
                    "vec": spaces.Box(
                        low=-1.0,
                        high=2.0,
                        shape=(VEC_DIM,),
                        dtype=np.float32,
                    ),
                }
            )
        else:
            self.observation_space = spaces.Box(
                low=np.array([-10.0, -1.0, 0.0, -5.0], dtype=np.float32),
                high=np.array([10.0, 1.0, 3.0, 5.0], dtype=np.float32),
                dtype=np.float32,
            )

        self.client = connect_client(host=host, port=port, timeout=30.0)
        world = ensure_map(self.client, map_name)
        self._sync = SynchronousMode(
            self.client, world, fps=self.fps, no_rendering=self.no_rendering
        )
        self.world, self.tm = self._sync.__enter__()
        try:
            self.world_map = self.world.get_map()
            self.grp = build_planner(self.world_map)
            self.spawn_pts = self.world_map.get_spawn_points()
            if not 0 <= self.spawn_index < len(self.spawn_pts):
                raise ValueError(
                    f"spawn_index {self.spawn_index} out of range "
                    f"0..{len(self.spawn_pts) - 1}"
                )
        except Exception:
            self._sync.__exit__(None, None, None)
            self._sync = None
            raise

        self._vehicle: carla.Vehicle | None = None
        self._sensors: list[carla.Actor] = []
        self._events = EventCounter()
        self._route: list = []
        self._route_pts: list[tuple[float, float, float]] = []
        self._planned_m = 0.0
        self._route_index = 0
        self._prev_route_index = 0
        self._step_count = 0
        self._last_steer = 0.0
        self._applied_steer = 0.0
        self._prev_loc: carla.Location | None = None
        self._odom_m = 0.0
        self._closed = False
        self._cam_latest: carla.Image | None = None
        self._cam_queue: queue.Queue = queue.Queue()
        self._frame_stack: deque[np.ndarray] = deque(maxlen=CAM_STACK)
        self._sticky_command: str | None = None
        self._sticky_start: int | None = None
        self._cam_warned = False
        self._turn_episode = False
        self._desired_kmh = SPEED_CRUISE_KMH
        self._bev: BevRasterizer | None = (
            BevRasterizer.from_map(self.world_map) if self.bev else None
        )
        self._perceived: PerceivedRoad | None = (
            PerceivedRoad(seg_path) if self.bev_source == "perceived" else None
        )
        self._bev_cam_queues: dict[str, queue.Queue] = {}
        self._road_agreement: tuple[float, float] | None = None
        self._last_map_road: np.ndarray | None = None
        self._recent_progress: deque[float] = deque(maxlen=STALL_STEPS)
        self._turn_kind: str | None = None
        self._turn_exit_i: int | None = None
        self._turn_entry_i: int | None = None
        self._speed_sum = 0.0
        self._npcs: list[carla.Actor] = []
        self._walkers: list[carla.Actor] = []
        self._walker_ctrls: list[carla.Actor] = []
        self._route_xy = np.zeros((0, 2))
        self._route_s = np.zeros(0)
        self._route_yaw = np.zeros(0)
        self._route_stops: list[tuple[int, carla.Actor]] = []
        self._last_cap: float | None = None
        # Drain leftover Town10 actors after load_world before the first reset.
        for _ in range(5):
            self.world.tick()
        clear_other_vehicles(self.world)
        self._stop_lines: list[tuple[carla.Actor, list[tuple[float, float, float, float]]]] = []
        if self.lights:
            speed_up_traffic_lights(
                self.world,
                red_s=LIGHT_RED_S,
                yellow_s=LIGHT_YELLOW_S,
                green_s=LIGHT_GREEN_S,
            )
            self._stop_lines = self._collect_stop_lines()

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._teardown_episode()
        if self.npc_vehicles:
            # Last episode's NPCs could sit on the ego spawn point.
            clear_other_vehicles(self.world)
            self._npcs = []

        self._vehicle = spawn_vehicle(self.world, spawn_index=self.spawn_index)
        self._events = EventCounter()
        self._build_route()
        self._turn_kind = None
        self._turn_exit_i = None
        self._turn_entry_i = None
        start_i = self._choose_start_index()
        self._turn_episode = bool(self.image_obs and start_i > 0)
        if self._bev is not None:
            self._bev.set_route(self._route_pts)
        self._route_stops = self._match_stops_to_route() if self.lights else []
        self._last_cap = None
        self._desired_kmh = SPEED_CRUISE_KMH
        # Spawn points sit off-center; always snap to the route/lane middle.
        self._place_on_route(start_i)
        self.world.tick()
        self._place_on_route(start_i)
        self._sticky_command = None
        self._sticky_start = None
        self._cam_latest = None
        self._cam_queue = queue.Queue()
        self._frame_stack.clear()
        if self.camera:
            self._attach_camera()
        if self._perceived is not None:
            self._attach_bev_cameras()

        for _ in range(WARMUP_TICKS):
            self._vehicle.apply_control(
                carla.VehicleControl(throttle=0.0, brake=1.0, steer=0.0)
            )
            self.world.tick()
            if self.camera:
                self._ingest_camera()
        self._place_on_route(start_i)
        if self.traffic:
            self._spawn_traffic()
        self._attach_score_sensors()
        if self.follow:
            follow_vehicle(self.world, self._vehicle)

        self._route_index = nearest_route_index(
            self._vehicle.get_location(),
            self._route,
            start_hint=start_i,
        )
        self._prev_route_index = self._route_index
        self._step_count = 0
        self._recent_progress.clear()
        self._speed_sum = 0.0
        self._last_steer = 0.0
        self._applied_steer = 0.0
        self._prev_loc = self._vehicle.get_location()
        self._odom_m = 0.0
        obs = self._observe()
        if self.camera:
            self._maybe_warn_camera(obs["image"])
        if self.bev:
            self._maybe_write_bev_preview(obs["image"])
        return obs, self._info(reward_parts={}, terminated=False, truncated=False)

    def step(self, action):
        if self._vehicle is None:
            raise RuntimeError("step() before reset()")

        turning = self._upcoming_turn()
        steer_cmd = float(np.clip(action[0], -1.0, 1.0))
        if not turning and abs(steer_cmd) < STEER_DEADBAND:
            steer_cmd = 0.0
        if self.steer_smooth:
            # Heavier filter in corners: keep a held radius, don't follow twitches.
            mix = STEER_TURN_SMOOTH if turning else STEER_SMOOTH
            steer = (1.0 - mix) * self._applied_steer + mix * steer_cmd
        else:
            steer = steer_cmd
        self._applied_steer = steer
        target_kmh = None
        if self.speed_control:
            throttle, brake = self._speed_command(turning)
        elif self.speed_setpoint:
            target_kmh = (float(np.clip(action[1], -1.0, 1.0)) + 1.0) * 0.5 * SETPOINT_MAX_KMH
            throttle, brake = self._setpoint_command(target_kmh)
        else:
            throttle, brake = float(np.clip(action[1], 0.0, 1.0)), 0.0
        d_steer = steer - self._last_steer
        self._last_steer = steer

        col_before, inv_before = self._events.snapshot()
        self._vehicle.apply_control(
            carla.VehicleControl(steer=steer, throttle=throttle, brake=brake)
        )
        self.world.tick()
        if self.camera:
            self._ingest_camera()
        if self.follow:
            follow_vehicle(self.world, self._vehicle)

        loc = self._vehicle.get_location()
        if self._prev_loc is not None:
            self._odom_m += loc.distance(self._prev_loc)
        self._prev_loc = loc

        self._route_index = nearest_route_index(
            loc, self._route, start_hint=self._route_index
        )
        if self._route_index > self._prev_route_index:
            progress_m = path_length_m(
                self._route_pts, self._prev_route_index, self._route_index
            )
        else:
            progress_m = 0.0
        ran_red = self._ran_red_light(self._prev_route_index, self._route_index)
        self._prev_route_index = self._route_index

        col_after, inv_after = self._events.snapshot()
        collided = col_after > col_before
        n_invasions = max(0, inv_after - inv_before)

        obs = self._observe()
        metrics = lane_metrics_at(
            loc,
            self._vehicle.get_transform().rotation.yaw,
            self._route,
            self._route_index,
        )
        off_lane = abs(metrics["cte_m"]) > CTE_FAIL_M
        cap, hazard = self._safe_speed() if self.traffic else (None, None)
        self._last_cap = cap
        if cap is not None and cap < STOP_OK_KMH:
            # Waiting for a hazard is not stalling.
            self._recent_progress.clear()
        else:
            self._recent_progress.append(progress_m)
        stalled = (
            len(self._recent_progress) == STALL_STEPS
            and sum(self._recent_progress) < STALL_MIN_M
        )
        speed = speed_kmh(self._vehicle)
        reward, parts = step_reward(
            progress_m=progress_m,
            cte_m=metrics["cte_m"],
            heading_err_deg=metrics["heading_err_deg"],
            n_invasions=n_invasions,
            collided=collided,
            d_steer=d_steer,
            off_lane=off_lane,
            stalled=stalled,
            ran_red=ran_red,
            speed_cap_kmh=cap,
            steer=steer,
            speed_kmh=speed,
            turning=turning,
        )

        self._step_count += 1
        self._speed_sum += speed
        done_route = self._route_pct() >= ROUTE_DONE_PCT
        terminated = bool(
            collided or off_lane or ran_red or stalled or done_route
        )
        horizon = self.max_steps
        if self._turn_episode:
            horizon = min(horizon, CAM_TURN_EPISODE_STEPS)
        truncated = (not terminated) and self._step_count >= horizon
        info = self._info(
            reward_parts=parts,
            terminated=terminated,
            truncated=truncated,
            steer=steer,
            throttle=throttle,
            brake=brake,
            progress_m=progress_m,
        )
        info["cte_m"] = metrics["cte_m"]
        info["lane_offset_frac"] = metrics["lane_offset_frac"]
        info["heading_err_deg"] = metrics["heading_err_deg"]
        info["stalled"] = stalled
        if self.traffic:
            info["speed_cap_kmh"] = cap
            info["hazard"] = hazard
            info["ran_red"] = ran_red
        if target_kmh is not None:
            info["target_kmh"] = target_kmh
        if self._road_agreement is not None:
            info["bev_road_precision"], info["bev_road_recall"] = self._road_agreement
        if terminated or truncated:
            if collided:
                info["end_reason"] = "collision"
            elif off_lane:
                info["end_reason"] = "off_lane"
            elif ran_red:
                info["end_reason"] = "red_light"
            elif stalled:
                info["end_reason"] = "stalled"
            elif done_route:
                info["end_reason"] = "route_done"
            else:
                info["end_reason"] = "timeout"
            info["ep_mean_speed_kmh"] = self._speed_sum / max(1, self._step_count)
        if (terminated or truncated) and self._turn_kind is not None:
            cleared = self._route_index >= int(self._turn_exit_i)
            info["turn_kind"] = self._turn_kind
            info["turn_cleared"] = cleared
            if not cleared and self._turn_kind in ("left", "right") and (
                collided or off_lane
            ):
                info["turn_miss"] = self._turn_miss(metrics["cte_m"])
        return obs, float(reward), terminated, truncated, info

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._teardown_episode()
        if self._npcs:
            clear_other_vehicles(self.world)
            self._npcs = []
        if self._walkers or self._walker_ctrls:
            destroy_walkers(self._walker_ctrls, self._walkers)
            self._walkers, self._walker_ctrls = [], []
        if self.traffic:
            try:
                self.world.tick()
            except Exception:
                pass
        if self._sync is not None:
            self._sync.__exit__(None, None, None)
            self._sync = None

    def _build_route(self) -> None:
        origin_i = self.spawn_index
        # Train (not eval): pick a random Town01 spawn so turns aren't always
        # the same two corners after spawn 0. fixed_route keeps spawn_index.
        if (
            self.image_obs
            and not self.fixed_route
            and self.turn_start_prob > 0.0
            and self.spawn_pts
        ):
            origin_i = int(self.np_random.integers(0, len(self.spawn_pts)))
        dest_index = farthest_spawn_index(self.spawn_pts, origin_i)
        try:
            _dest, route, _score = plan_spawn_to_spawn(
                self.grp, self.spawn_pts, origin_i, dest_index
            )
        except (ValueError, RuntimeError):
            origin = self.spawn_pts[origin_i].location
            _dest, route, _score = pick_junction_heavy_destination(
                self.grp, origin, self.spawn_pts
            )
        self._route = route
        self._route_pts = route_xyz(route)
        self._planned_m = path_length_m(
            self._route_pts, 0, max(0, len(self._route_pts) - 1)
        )
        self._route_xy = np.asarray(self._route_pts, dtype=np.float64).reshape(-1, 3)[:, :2]
        seg = np.linalg.norm(np.diff(self._route_xy, axis=0), axis=1)
        self._route_s = np.concatenate([[0.0], np.cumsum(seg)])
        self._route_yaw = np.array([wp.transform.rotation.yaw for wp, _ in route])

    def _choose_start_index(self) -> int:
        """GPS: ~25 m before left-biased turns. Camera: see _choose_camera_start."""
        if (
            not self._route
            or self.turn_start_prob <= 0.0
            or float(self.np_random.random()) > self.turn_start_prob
        ):
            return 0
        if self.image_obs:
            return self._choose_camera_start()
        lefts: list[int] = []
        turns: list[int] = []
        for i, (_, opt) in enumerate(self._route):
            code = int(opt)
            if code == int(RoadOption.LEFT):
                lefts.append(i)
                turns.append(i)
            elif code == int(RoadOption.RIGHT):
                turns.append(i)
        if not turns:
            return 0
        pool = (
            lefts
            if lefts and float(self.np_random.random()) < TURN_LEFT_BIAS
            else turns
        )
        if not pool:
            pool = turns
        turn_i = int(pool[int(self.np_random.integers(0, len(pool)))])
        return max(0, turn_i - TURN_APPROACH_WPS)

    def _choose_camera_start(self) -> int:
        """Approach a left, right, or go-straight with the IL-style command."""
        buckets: dict[int, list[int]] = {
            int(RoadOption.LEFT): [],
            int(RoadOption.RIGHT): [],
            int(RoadOption.STRAIGHT): [],
        }
        for i, (_, opt) in enumerate(self._route):
            code = int(opt)
            if code in buckets:
                buckets[code].append(i)
        kinds = [k for k, idxs in buckets.items() if idxs]
        if not kinds:
            return 0
        kind = int(kinds[int(self.np_random.integers(0, len(kinds)))])
        pool = buckets[kind]
        turn_i = int(pool[int(self.np_random.integers(0, len(pool)))])
        self._turn_kind = {
            int(RoadOption.LEFT): "left",
            int(RoadOption.RIGHT): "right",
            int(RoadOption.STRAIGHT): "straight",
        }[kind]
        self._turn_exit_i = self._index_meters_after(turn_i, TURN_CLEAR_M)
        self._turn_entry_i = self._index_meters_before(turn_i, TURN_ENTRY_M)
        lo, hi = CAM_APPROACH_M
        approach_m = float(self.np_random.uniform(lo, hi))
        return self._index_meters_before(turn_i, approach_m)

    def _index_meters_before(self, turn_i: int, meters: float) -> int:
        if not self._route_pts or turn_i <= 0:
            return 0
        i = int(turn_i)
        while i > 0 and path_length_m(self._route_pts, i, turn_i) < meters:
            i -= 1
        return i

    def _turn_miss(self, cte_m: float) -> str:
        """Why a left/right was missed: wide (too little steer), tight, or approach.

        cte + = right of path. Running wide on a left leaves the car right of
        the path; running wide on a right leaves it left.
        """
        if self._turn_entry_i is None or self._route_index < self._turn_entry_i:
            return "approach"
        outside = cte_m > 0.0 if self._turn_kind == "left" else cte_m < 0.0
        return "wide" if outside else "tight"

    def _index_meters_after(self, turn_i: int, meters: float) -> int:
        last = max(0, len(self._route_pts) - 1)
        i = int(turn_i)
        while i < last and path_length_m(self._route_pts, turn_i, i) < meters:
            i += 1
        return i

    def _upcoming_turn(self) -> bool:
        """True through the whole bend, not just the 1–2 GRP LEFT flags."""
        if not self._route:
            return False
        geo = geometric_command_from_route(
            self._route,
            start_index=self._route_index,
            lookahead_m=18.0,
            turn_yaw_deg=18.0,
        )
        if geo in (COMMAND_TURN_LEFT, COMMAND_TURN_RIGHT):
            return True
        start = max(0, self._route_index - 5)
        end = min(len(self._route), self._route_index + 8)
        turn = {int(RoadOption.LEFT), int(RoadOption.RIGHT)}
        return any(int(self._route[i][1]) in turn for i in range(start, end))

    def _place_on_route(self, index: int) -> None:
        assert self._vehicle is not None
        if not self._route:
            return
        idx = max(0, min(int(index), len(self._route) - 1))
        path_wp = self._route[idx][0]
        tf = path_wp.transform
        try:
            road_wp = self.world_map.get_waypoint(
                tf.location,
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )
            if road_wp is not None:
                placed = road_wp.transform
                placed.rotation.yaw = path_wp.transform.rotation.yaw
                tf = placed
        except Exception:
            pass
        self._vehicle.set_transform(tf)
        try:
            self._vehicle.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
            self._vehicle.set_target_angular_velocity(carla.Vector3D(0.0, 0.0, 0.0))
        except Exception:
            pass

    def _attach_score_sensors(self) -> None:
        assert self._vehicle is not None
        library = self.world.get_blueprint_library()
        col = self.world.spawn_actor(
            library.find("sensor.other.collision"),
            carla.Transform(),
            attach_to=self._vehicle,
        )
        inv = self.world.spawn_actor(
            library.find("sensor.other.lane_invasion"),
            carla.Transform(),
            attach_to=self._vehicle,
        )
        col.listen(lambda e: self._events.on_collision(e))
        inv.listen(lambda e: self._events.on_lane_invasion(e))
        self._sensors.extend([col, inv])

    def _attach_camera(self) -> None:
        assert self._vehicle is not None
        cam = attach_rgb_camera(
            self.world,
            self._vehicle,
            width=CAM_CAPTURE_W,
            height=CAM_CAPTURE_H,
        )
        cam.listen(self._on_camera)
        self._sensors.append(cam)

    def _on_camera(self, image: carla.Image) -> None:
        self._cam_queue.put(image)

    def _sync_camera(self) -> None:
        """Block until the image for the tick that just ran has arrived.

        The sensor callback runs on another thread and can lag tick() by a
        frame, which would hand the policy a stale image.
        """
        want = self.world.get_snapshot().frame
        while True:
            try:
                image = self._cam_queue.get(timeout=2.0)
            except queue.Empty:
                return
            self._cam_latest = image
            if image.frame >= want:
                return

    def _ingest_camera(self) -> None:
        self._sync_camera()
        if self._cam_latest is None:
            frame = np.zeros((1, CAM_H, CAM_W), dtype=np.uint8)
        else:
            frame = _rgb_to_gray84(self._cam_latest)
        self._frame_stack.append(frame)
        while len(self._frame_stack) < CAM_STACK:
            self._frame_stack.append(frame)

    def _maybe_warn_camera(self, image: np.ndarray) -> None:
        global _CAM_PREVIEW_WRITTEN
        mean = float(image.mean())
        if mean < 5.0 and not self._cam_warned:
            self._cam_warned = True
            print(
                f"camera frames look empty (mean={mean:.1f}); "
                "check that CARLA is rendering"
            )
        if not _CAM_PREVIEW_WRITTEN:
            _CAM_PREVIEW_WRITTEN = True
            _CAM_PREVIEW.parent.mkdir(parents=True, exist_ok=True)
            PILImage.fromarray(image[-1], mode="L").save(_CAM_PREVIEW)
            print(f"Wrote camera preview {_CAM_PREVIEW} (mean={mean:.1f})")

    def _setpoint_command(self, target_kmh: float) -> tuple[float, float]:
        """Throttle/brake toward the agent's target speed; holds a full stop."""
        assert self._vehicle is not None
        speed = speed_kmh(self._vehicle)
        if target_kmh < 0.5 and speed < 1.0:
            return 0.0, 1.0
        err = target_kmh - speed
        if err >= 0.0:
            return min(SPEED_MAX_THROTTLE, SPEED_THROTTLE_GAIN * err), 0.0
        brake = min(1.0, SETPOINT_BRAKE_GAIN * -err)
        return (0.0, brake) if brake > 0.04 else (0.0, 0.0)

    def follower_action(self, ctrl: carla.VehicleControl) -> np.ndarray:
        """Waypoint-follower control as an action for this env's action space.

        In setpoint mode the follower's speed is its turn/cruise profile capped
        by the last hazard cap, so it doubles as a traffic-obeying baseline.
        """
        if self.speed_control:
            return np.array([ctrl.steer], dtype=np.float32)
        if self.speed_setpoint:
            target = SPEED_TURN_KMH if self._upcoming_turn() else SPEED_CRUISE_KMH
            if self._last_cap is not None:
                target = min(target, self._last_cap)
            return np.array(
                [ctrl.steer, 2.0 * target / SETPOINT_MAX_KMH - 1.0], dtype=np.float32
            )
        return np.array([ctrl.steer, ctrl.throttle], dtype=np.float32)

    def _spawn_traffic(self) -> None:
        assert self._vehicle is not None
        seed = int(self.np_random.integers(0, 2**31 - 1))
        if self.npc_vehicles:
            self._npcs = spawn_npc_traffic(
                self.world,
                self.tm,
                count=self.npc_vehicles,
                ego=self._vehicle,
                clear_existing=False,
                seed=seed,
            )
        if self.n_walkers:
            # Walkers persist across episodes (spawning them is slow); top up
            # only when most have gone.
            alive = [w for w in self._walkers if _actor_alive(w)]
            if len(alive) < self.n_walkers // 2:
                destroy_walkers(self._walker_ctrls, self._walkers)
                self._walkers, self._walker_ctrls = spawn_walkers(
                    self.client, self.world, count=self.n_walkers, seed=seed
                )

    def _collect_stop_lines(self) -> list[tuple[carla.Actor, list[tuple[float, float, float, float]]]]:
        """(light, [(x, y, yaw, half lane width)]) for every traffic light's stop line."""
        out = []
        for light in self.world.get_actors().filter("traffic.traffic_light"):
            stops = []
            for wp in light.get_stop_waypoints():
                tf = wp.transform
                stops.append(
                    (tf.location.x, tf.location.y, tf.rotation.yaw, 0.5 * float(wp.lane_width or 3.5))
                )
            if stops:
                out.append((light, stops))
        return out

    def _match_stops_to_route(self) -> list[tuple[int, carla.Actor]]:
        """Stop lines that sit on this episode's route, as (route index, light)."""
        found: dict[int, carla.Actor] = {}
        if not len(self._route_xy):
            return []
        for light, stops in self._stop_lines:
            for x, y, yaw, _hw in stops:
                d = np.hypot(self._route_xy[:, 0] - x, self._route_xy[:, 1] - y)
                i = int(np.argmin(d))
                dyaw = (yaw - self._route_yaw[i] + 180.0) % 360.0 - 180.0
                if d[i] < 2.5 and abs(dyaw) < 45.0:
                    found[i] = light
        return sorted(found.items())

    def _ran_red_light(self, prev_i: int, cur_i: int) -> bool:
        """Crossed a route stop line this step while its light was red."""
        for ri, light in self._route_stops:
            if prev_i < ri <= cur_i and light.get_state() == carla.TrafficLightState.Red:
                return True
        return False

    def _route_distance_to(self, loc: carla.Location, lookahead_m: float, lateral_m: float) -> float | None:
        """Path distance (m) from the ego's route position to an actor in the
        route corridor ahead, or None if it isn't in the corridor."""
        assert self._vehicle is not None
        ego = self._vehicle.get_location()
        if loc.distance(ego) > lookahead_m + 5.0:
            return None
        cur = self._route_index
        here = self._route_s[cur]
        end = int(np.searchsorted(self._route_s, here + lookahead_m, side="right"))
        pts = self._route_xy[cur:max(cur + 1, end)]
        d = np.hypot(pts[:, 0] - loc.x, pts[:, 1] - loc.y)
        k = int(np.argmin(d))
        if d[k] > lateral_m:
            return None
        if k == 0:
            # Same route point as the ego: use straight-line distance ahead.
            fwd = self._vehicle.get_transform().get_forward_vector()
            ahead = (loc.x - ego.x) * fwd.x + (loc.y - ego.y) * fwd.y
            return ahead if ahead > 0.0 else None
        return float(self._route_s[cur + k] - here)

    def _safe_speed(self) -> tuple[float, str | None]:
        """Speed cap (km/h) for hazards on the route ahead, and which one set it."""
        cap, why = SPEED_CRUISE_KMH, None
        here = self._route_s[self._route_index]
        for ri, light in self._route_stops:
            if ri <= self._route_index:
                continue
            d = self._route_s[ri] - here - EGO_HALF_LENGTH_M
            if d > LIGHT_LOOKAHEAD_M:
                continue
            state = light.get_state()
            stop = state == carla.TrafficLightState.Red or (
                state == carla.TrafficLightState.Yellow and d > YELLOW_COMMIT_M
            )
            if stop:
                c = float(np.clip((d - LIGHT_STOP_M) * LIGHT_KMH_PER_M, 0.0, SPEED_CRUISE_KMH))
                if c < cap:
                    cap, why = c, "light"
        for actors, look, lat, gap, per_m, kind in (
            (self._npcs, LEAD_LOOKAHEAD_M, LEAD_LATERAL_M, LEAD_GAP_M, LEAD_KMH_PER_M, "vehicle"),
            (self._walkers, WALKER_LOOKAHEAD_M, WALKER_LATERAL_M, WALKER_GAP_M, WALKER_KMH_PER_M, "walker"),
        ):
            for actor in actors:
                if not _actor_alive(actor):
                    continue
                d = self._route_distance_to(actor.get_location(), look, lat)
                if d is None:
                    continue
                room = d - EGO_HALF_LENGTH_M - actor.bounding_box.extent.x - gap
                c = float(np.clip(room * per_m, 0.0, SPEED_CRUISE_KMH))
                if c < cap:
                    cap, why = c, kind
        return cap, why

    def _bev_objects(self) -> BevObjects:
        """True boxes of nearby cars / pedestrians and red-yellow stop lines."""
        assert self._vehicle is not None
        ego = self._vehicle.get_location()
        objs = BevObjects()
        for actors, out, min_half in (
            (self._npcs, objs.vehicles, 0.0),
            (self._walkers, objs.walkers, MIN_WALKER_HALF_M),
        ):
            for actor in actors:
                if not _actor_alive(actor):
                    continue
                tf = actor.get_transform()
                if tf.location.distance(ego) > BEV_OBJECT_RADIUS_M:
                    continue
                e = actor.bounding_box.extent
                out.append(
                    (tf.location.x, tf.location.y, tf.rotation.yaw, max(min_half, e.x), max(min_half, e.y))
                )
        for light, stops in self._stop_lines:
            state = light.get_state()
            if state not in (carla.TrafficLightState.Red, carla.TrafficLightState.Yellow):
                continue
            for x, y, yaw, hw in stops:
                if math.hypot(x - ego.x, y - ego.y) <= BEV_OBJECT_RADIUS_M:
                    objs.stops.append((x, y, yaw, 0.75, hw))
        return objs

    def _speed_command(self, turning: bool) -> tuple[float, float]:
        """Throttle/brake toward cruise or turn speed (follower's P control)."""
        assert self._vehicle is not None
        target = SPEED_TURN_KMH if turning else SPEED_CRUISE_KMH
        # Ease the setpoint so slowing for a turn is a coast, not a stop.
        self._desired_kmh += SPEED_EASE * (target - self._desired_kmh)
        err = self._desired_kmh - speed_kmh(self._vehicle)
        if err >= 0.0:
            return min(SPEED_MAX_THROTTLE, SPEED_THROTTLE_GAIN * err), 0.0
        brake = min(SPEED_MAX_BRAKE, SPEED_BRAKE_GAIN * -err)
        if brake > 0.04:
            return 0.0, brake
        return 0.0, 0.0

    def _maybe_write_bev_preview(self, bev: np.ndarray) -> None:
        global _BEV_PREVIEW_WRITTEN
        if _BEV_PREVIEW_WRITTEN:
            return
        _BEV_PREVIEW_WRITTEN = True
        _BEV_PREVIEW.parent.mkdir(parents=True, exist_ok=True)
        rgb = bev_to_rgb(bev)
        if self._last_map_road is not None:
            # Perceived (left) next to what the map says (right).
            truth = bev.copy()
            truth[0] = self._last_map_road
            gap = np.zeros((rgb.shape[0], 8, 3), dtype=np.uint8)
            rgb = np.concatenate([rgb, gap, bev_to_rgb(truth)], axis=1)
        PILImage.fromarray(rgb).save(_BEV_PREVIEW)
        print(f"Wrote BEV preview {_BEV_PREVIEW}")

    def _attach_bev_cameras(self) -> None:
        assert self._vehicle is not None
        cams = attach_rgb_cameras(
            self.world,
            self._vehicle,
            names=PERCEIVED_CAMERAS,
            width=CAM_CAPTURE_W,
            height=CAM_CAPTURE_H,
        )
        self._bev_cam_queues = {}
        for name, cam in cams.items():
            q: queue.Queue = queue.Queue()
            cam.listen(q.put)
            self._bev_cam_queues[name] = q
            self._sensors.append(cam)

    def _bev_camera_rgb(self) -> dict[str, np.ndarray]:
        """Latest frame per camera for the tick that just ran (see _sync_camera)."""
        want = self.world.get_snapshot().frame
        out: dict[str, np.ndarray] = {}
        for name, q in self._bev_cam_queues.items():
            image = None
            while True:
                try:
                    image = q.get(timeout=2.0)
                except queue.Empty:
                    break
                if image.frame >= want:
                    break
            if image is not None:
                out[name] = image_to_array(image)
        if not out:
            raise RuntimeError("no BEV camera frames arrived; is CARLA rendering?")
        return out

    def _observe_vec(self) -> np.ndarray:
        assert self._vehicle is not None
        speed = speed_kmh(self._vehicle) / 50.0
        loc = self._vehicle.get_location()
        yaw = self._vehicle.get_transform().rotation.yaw
        command_name, _, self._sticky_command, self._sticky_start = hold_turn_command(
            loc,
            self._route,
            start_hint=self._route_index,
            sticky_command=self._sticky_command,
            sticky_start_index=self._sticky_start,
            vehicle_yaw=yaw,
        )
        onehot = np.asarray(command_to_onehot(command_name), dtype=np.float32)
        extra = np.array(
            [self._applied_steer, 1.0 if self._upcoming_turn() else 0.0],
            dtype=np.float32,
        )
        return np.concatenate(
            [np.array([speed], dtype=np.float32), onehot, extra]
        ).astype(np.float32)

    def _observe(self) -> np.ndarray | dict:
        assert self._vehicle is not None
        if self.camera:
            if len(self._frame_stack) < CAM_STACK:
                self._ingest_camera()
            image = np.concatenate(list(self._frame_stack), axis=0)
            return {"image": image, "vec": self._observe_vec()}
        if self._bev is not None:
            tf = self._vehicle.get_transform()
            image = self._bev.render(
                tf.location.x,
                tf.location.y,
                tf.rotation.yaw,
                self._route_index,
                objects=self._bev_objects() if self.bev_objects else None,
            )
            if self._perceived is not None:
                self._last_map_road = image[0].copy()
                image[0] = self._perceived.road(self._bev_camera_rgb())
                self._road_agreement = road_agreement(image[0], self._last_map_road)
            return {"image": image, "vec": self._observe_vec()}
        loc = self._vehicle.get_location()
        yaw = self._vehicle.get_transform().rotation.yaw
        metrics = lane_metrics_at(loc, yaw, self._route, self._route_index)
        speed = speed_kmh(self._vehicle)
        obs = np.array(
            [
                metrics["cte_m"],
                metrics["heading_err_deg"] / 180.0,
                speed / 50.0,
                metrics["lane_offset_frac"],
            ],
            dtype=np.float32,
        )
        return np.clip(obs, self.observation_space.low, self.observation_space.high)

    def _route_pct(self) -> float:
        progress_m = path_length_m(self._route_pts, 0, self._route_index)
        return route_completion_pct(progress_m, self._planned_m)

    def _info(
        self,
        *,
        reward_parts: dict,
        terminated: bool,
        truncated: bool,
        steer: float = 0.0,
        throttle: float = 0.0,
        brake: float = 0.0,
        progress_m: float = 0.0,
    ) -> dict:
        n_col, n_inv = self._events.snapshot()
        speed = speed_kmh(self._vehicle) if self._vehicle is not None else 0.0
        return {
            "speed_kmh": speed,
            "collisions": n_col,
            "lane_invasions": n_inv,
            "odom_m": self._odom_m,
            "route_progress_m": path_length_m(
                self._route_pts, 0, self._route_index
            ),
            "route_length_m": self._planned_m,
            "route_completion_pct": self._route_pct(),
            "steer": steer,
            "throttle": throttle,
            "brake": brake,
            "step_progress_m": progress_m,
            "terminated": terminated,
            "truncated": truncated,
            **reward_parts,
        }

    def _teardown_episode(self) -> None:
        """Stop sensors, tick so the server drains them, then destroy once.

        Do not call is_alive on a dead handle (can abort). Do not destroy
        sensors and then the parent again without a tick — CARLA prints
        'unable to destroy actor: not found' for the second RPC.
        """
        sensors = list(self._sensors)
        vehicle = self._vehicle
        self._sensors = []
        self._vehicle = None
        for sensor in sensors:
            try:
                sensor.stop()
            except Exception:
                pass
        for _ in range(3):
            try:
                self.world.tick()
            except Exception:
                break
        for actor in [*sensors, vehicle] if vehicle is not None else sensors:
            try:
                actor.destroy()
            except Exception:
                pass
        try:
            self.world.tick()
        except Exception:
            pass
