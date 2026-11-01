"""High-level driving commands derived from CARLA RoadOption."""
from __future__ import annotations

from enum import IntEnum


class RoadOption(IntEnum):
    """Topological maneuver between lane segments (CARLA agents.navigation)."""

    VOID = -1
    LEFT = 1
    RIGHT = 2
    STRAIGHT = 3
    LANEFOLLOW = 4
    CHANGELANELEFT = 5
    CHANGELANERIGHT = 6


COMMAND_FOLLOW_LANE = "FOLLOW_LANE"
COMMAND_TURN_LEFT = "TURN_LEFT"
COMMAND_TURN_RIGHT = "TURN_RIGHT"
COMMAND_GO_STRAIGHT = "GO_STRAIGHT"
COMMAND_CHANGE_LANE_LEFT = "CHANGE_LANE_LEFT"
COMMAND_CHANGE_LANE_RIGHT = "CHANGE_LANE_RIGHT"

ALL_COMMANDS = (
    COMMAND_FOLLOW_LANE,
    COMMAND_TURN_LEFT,
    COMMAND_TURN_RIGHT,
    COMMAND_GO_STRAIGHT,
    COMMAND_CHANGE_LANE_LEFT,
    COMMAND_CHANGE_LANE_RIGHT,
)

JUNCTION_COMMANDS = (
    COMMAND_TURN_LEFT,
    COMMAND_TURN_RIGHT,
    COMMAND_GO_STRAIGHT,
)

_ROAD_OPTION_TO_COMMAND = {
    RoadOption.VOID: COMMAND_FOLLOW_LANE,
    RoadOption.LEFT: COMMAND_TURN_LEFT,
    RoadOption.RIGHT: COMMAND_TURN_RIGHT,
    RoadOption.STRAIGHT: COMMAND_GO_STRAIGHT,
    RoadOption.LANEFOLLOW: COMMAND_FOLLOW_LANE,
    RoadOption.CHANGELANELEFT: COMMAND_CHANGE_LANE_LEFT,
    RoadOption.CHANGELANERIGHT: COMMAND_CHANGE_LANE_RIGHT,
}


def road_option_to_command(option: RoadOption | int) -> str:
    return _ROAD_OPTION_TO_COMMAND.get(RoadOption(int(option)), COMMAND_FOLLOW_LANE)


COMMAND_DIM = len(ALL_COMMANDS)
_COMMAND_INDEX = {name: i for i, name in enumerate(ALL_COMMANDS)}


def command_to_onehot(command: str) -> list[float]:
    """One-hot over ALL_COMMANDS; unknown → FOLLOW_LANE."""
    vec = [0.0] * COMMAND_DIM
    idx = _COMMAND_INDEX.get(command, _COMMAND_INDEX[COMMAND_FOLLOW_LANE])
    vec[idx] = 1.0
    return vec


def label_command(
    planner_command: str,
    steer: float,
    *,
    steer_turn: float = 0.40,
    steer_follow_max: float = 0.40,
) -> str:
    """Keep the route's next-maneuver command. Wheel angle must not rename it.

    A left can briefly steer right while recovering in the lane. Calling that
    TURN_RIGHT trains the wrong branch. Large steer on FOLLOW_LANE is still a
    missed-corner remap.
    """
    planner = (planner_command or COMMAND_FOLLOW_LANE).strip() or COMMAND_FOLLOW_LANE

    if planner == COMMAND_FOLLOW_LANE and abs(steer) > max(steer_follow_max, steer_turn):
        return COMMAND_TURN_RIGHT if steer > 0.0 else COMMAND_TURN_LEFT

    if planner in ALL_COMMANDS:
        return planner
    return COMMAND_FOLLOW_LANE

