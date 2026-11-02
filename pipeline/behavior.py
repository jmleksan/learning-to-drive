"""Finite-state behavior planner: lane-follow / change / stop / yield."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from utils.commands import (
    COMMAND_CHANGE_LANE_LEFT,
    COMMAND_CHANGE_LANE_RIGHT,
    COMMAND_FOLLOW_LANE,
    COMMAND_GO_STRAIGHT,
    COMMAND_TURN_LEFT,
    COMMAND_TURN_RIGHT,
)
from utils.lights import LIGHT_GREEN, LIGHT_OFF, normalize_light_state


class BehaviorState(str, Enum):
    LANE_FOLLOW = "LANE_FOLLOW"
    LANE_CHANGE_LEFT = "LANE_CHANGE_LEFT"
    LANE_CHANGE_RIGHT = "LANE_CHANGE_RIGHT"
    STOP = "STOP"
    YIELD = "YIELD"
    TURN = "TURN"


@dataclass(frozen=True)
class BehaviorDecision:
    state: BehaviorState
    target_speed_kmh: float
    command: str
    reason: str


class BehaviorFSM:
    """Rule-based behavior over route command + light + lead clearance."""

    def __init__(
        self,
        *,
        cruise_kmh: float = 28.0,
        turn_kmh: float = 16.0,
        stop_distance_m: float = 8.0,
        yield_distance_m: float = 14.0,
    ) -> None:
        self.cruise_kmh = float(cruise_kmh)
        self.turn_kmh = float(turn_kmh)
        self.stop_distance_m = float(stop_distance_m)
        self.yield_distance_m = float(yield_distance_m)
        self.state = BehaviorState.LANE_FOLLOW

    def reset(self) -> None:
        self.state = BehaviorState.LANE_FOLLOW

    def step(
        self,
        *,
        route_command: str,
        light_state: str | None,
        lead_distance_m: float | None,
        crossing_agent_ahead: bool = False,
        junction_ahead_m: float | None = None,
    ) -> BehaviorDecision:
        command = (route_command or COMMAND_FOLLOW_LANE).strip() or COMMAND_FOLLOW_LANE
        light = normalize_light_state(light_state)

        # Hard stop: red/yellow light or bumper-close lead.
        if light not in (LIGHT_GREEN, LIGHT_OFF):
            self.state = BehaviorState.STOP
            return BehaviorDecision(
                self.state, 0.0, command, f"traffic_light={light}"
            )
        if lead_distance_m is not None and lead_distance_m < self.stop_distance_m:
            self.state = BehaviorState.STOP
            return BehaviorDecision(
                self.state, 0.0, command, f"lead={lead_distance_m:.1f}m"
            )

        # Yield: agent in corridor near junction, or lead in soft gap.
        if crossing_agent_ahead and (
            junction_ahead_m is None or junction_ahead_m < 25.0
        ):
            self.state = BehaviorState.YIELD
            return BehaviorDecision(
                self.state, min(8.0, self.turn_kmh), command, "crossing_agent"
            )
        if lead_distance_m is not None and lead_distance_m < self.yield_distance_m:
            self.state = BehaviorState.YIELD
            # Slow proportional to gap.
            span = max(1.0, self.yield_distance_m - self.stop_distance_m)
            blend = (lead_distance_m - self.stop_distance_m) / span
            speed = max(4.0, blend * self.cruise_kmh * 0.6)
            return BehaviorDecision(
                self.state, speed, command, f"follow_gap={lead_distance_m:.1f}m"
            )

        if command in (COMMAND_CHANGE_LANE_LEFT,):
            self.state = BehaviorState.LANE_CHANGE_LEFT
            return BehaviorDecision(
                self.state, self.cruise_kmh * 0.85, command, "route_lane_change"
            )
        if command in (COMMAND_CHANGE_LANE_RIGHT,):
            self.state = BehaviorState.LANE_CHANGE_RIGHT
            return BehaviorDecision(
                self.state, self.cruise_kmh * 0.85, command, "route_lane_change"
            )
        if command in (COMMAND_TURN_LEFT, COMMAND_TURN_RIGHT, COMMAND_GO_STRAIGHT):
            self.state = BehaviorState.TURN
            return BehaviorDecision(self.state, self.turn_kmh, command, "maneuver")

        self.state = BehaviorState.LANE_FOLLOW
        return BehaviorDecision(
            self.state, self.cruise_kmh, COMMAND_FOLLOW_LANE, "cruise"
        )


def lead_distance_from_predictions(
    predictions: list,
    *,
    lane_half_width_m: float = 2.0,
    max_forward_m: float = 40.0,
) -> float | None:
    """Nearest predicted agent roughly in-lane ahead (ego x>0)."""
    best: float | None = None
    for pred in predictions:
        traj = pred.trajectory_xy
        if traj is None or len(traj) == 0:
            continue
        x0, y0 = float(traj[0, 0]), float(traj[0, 1])
        if x0 < 1.0 or x0 > max_forward_m:
            continue
        if abs(y0) > lane_half_width_m + 0.5 * float(getattr(pred, "width_m", 2.0)):
            continue
        best = x0 if best is None else min(best, x0)
    return best


def crossing_threat(
    predictions: list,
    *,
    horizon_s: float = 2.5,
    corridor_x_m: float = 18.0,
    corridor_y_m: float = 3.5,
) -> bool:
    """True if any predicted sample enters the forward corridor soon."""
    for pred in predictions:
        traj = pred.trajectory_xy
        if traj is None:
            continue
        # subsample ~ every 0.2s assuming 0.1 dt
        for i in range(0, len(traj), 2):
            t = i * 0.1
            if t > horizon_s:
                break
            x, y = float(traj[i, 0]), float(traj[i, 1])
            if 2.0 < x < corridor_x_m and abs(y) < corridor_y_m:
                # Ignore agents already counted as lead (same lane, slow).
                if abs(y) < 1.5 and x < 12.0:
                    continue
                if abs(y) >= 1.2:
                    return True
    return False
