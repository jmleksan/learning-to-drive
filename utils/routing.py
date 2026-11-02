"""Route selection, TM path, and debug visualization helpers."""
from __future__ import annotations

import math
import random
from typing import Sequence

import carla

from utils.commands import (
    COMMAND_FOLLOW_LANE,
    COMMAND_GO_STRAIGHT,
    COMMAND_TURN_LEFT,
    COMMAND_TURN_RIGHT,
    RoadOption,
    road_option_to_command,
)
from utils.route_planner import GlobalRoutePlanner

DEFAULT_SAMPLING = 2.0
LOOKAHEAD_METERS = 12.0
ROUTE_REPLAN_REMAINING_M = 25.0


def route_xyz(route: list) -> list[tuple[float, float, float]]:
    return [
        (wp.transform.location.x, wp.transform.location.y, wp.transform.location.z)
        for wp, _ in route
    ]


def route_length_m(route: list) -> float:
    total = 0.0
    for i in range(1, len(route)):
        a = route[i - 1][0].transform.location
        b = route[i][0].transform.location
        total += a.distance(b)
    return total


def build_planner(world_map: carla.Map, sampling: float = DEFAULT_SAMPLING) -> GlobalRoutePlanner:
    return GlobalRoutePlanner(world_map, sampling)


def route_junction_score(route: Sequence[tuple[carla.Waypoint, RoadOption]]) -> float:
    """Fraction of waypoints whose RoadOption is not LANEFOLLOW/VOID."""
    if not route:
        return 0.0
    non_follow = sum(
        1
        for _, opt in route
        if opt not in (RoadOption.LANEFOLLOW, RoadOption.VOID)
    )
    return non_follow / len(route)


def plan_route(
    grp: GlobalRoutePlanner,
    origin: carla.Location,
    destination: carla.Location,
) -> list[tuple[carla.Waypoint, RoadOption]]:
    return grp.trace_route(origin, destination)


def farthest_spawn_index(
    spawn_points: Sequence[carla.Transform],
    origin_index: int,
) -> int:
    """Spawn index farthest (Euclidean) from origin_index."""
    if not spawn_points:
        raise ValueError("No spawn points")
    origin = spawn_points[origin_index].location
    best_i = 0
    best_d = -1.0
    for i, sp in enumerate(spawn_points):
        d = origin.distance(sp.location)
        if d > best_d:
            best_d = d
            best_i = i
    return best_i


def plan_spawn_to_spawn(
    grp: GlobalRoutePlanner,
    spawn_points: Sequence[carla.Transform],
    start_index: int,
    dest_index: int,
) -> tuple[carla.Location, list[tuple[carla.Waypoint, RoadOption]], float]:
    """Plan a fixed route between two spawn indices."""
    if not 0 <= start_index < len(spawn_points):
        raise ValueError(f"start_index out of range: {start_index}")
    if not 0 <= dest_index < len(spawn_points):
        raise ValueError(f"dest_index out of range: {dest_index}")
    if start_index == dest_index:
        raise ValueError("start_index and dest_index must differ")
    origin = spawn_points[start_index].location
    dest = spawn_points[dest_index].location
    route = plan_route(grp, origin, dest)
    if len(route) < 5:
        raise RuntimeError(
            f"Route spawn {start_index}→{dest_index} too short ({len(route)} wps)"
        )
    return dest, route, route_junction_score(route)


def pick_junction_heavy_destination(
    grp: GlobalRoutePlanner,
    origin: carla.Location,
    spawn_points: Sequence[carla.Transform],
    *,
    candidates: int = 24,
    min_distance_m: float = 80.0,
    rng: random.Random | None = None,
) -> tuple[carla.Location, list[tuple[carla.Waypoint, RoadOption]], float]:
    """
    Sample spawn destinations, score by non-LANEFOLLOW fraction, return best.

    Returns (dest_location, route, score).
    """
    rng = rng or random.Random()
    far = [
        sp
        for sp in spawn_points
        if origin.distance(sp.location) >= min_distance_m
    ]
    pool = far if len(far) >= 3 else list(spawn_points)
    sample = pool if len(pool) <= candidates else rng.sample(list(pool), candidates)

    best_dest = sample[0].location
    best_route: list[tuple[carla.Waypoint, RoadOption]] = []
    best_score = -1.0

    for transform in sample:
        dest = transform.location
        if origin.distance(dest) < 5.0:
            continue
        try:
            route = plan_route(grp, origin, dest)
        except Exception:
            continue
        if len(route) < 5:
            continue
        score = route_junction_score(route)
        if score > best_score:
            best_score = score
            best_dest = dest
            best_route = route

    if not best_route:
        # Fallback: farthest spawn
        farthest = max(spawn_points, key=lambda sp: origin.distance(sp.location))
        best_dest = farthest.location
        best_route = plan_route(grp, origin, best_dest)
        best_score = route_junction_score(best_route)

    return best_dest, best_route, best_score


def route_locations(route: Sequence[tuple[carla.Waypoint, RoadOption]]) -> list[carla.Location]:
    return [wp.transform.location for wp, _ in route]


def sparsify_locations(
    locations: Sequence[carla.Location],
    *,
    min_spacing_m: float = 5.0,
) -> list[carla.Location]:
    """Drop near-duplicate path points (dense GRP paths confuse Traffic Manager)."""
    if not locations:
        return []
    out = [locations[0]]
    for loc in locations[1:]:
        if out[-1].distance(loc) >= min_spacing_m:
            out.append(loc)
    if len(locations) >= 2 and out[-1].distance(locations[-1]) > 1.0:
        out.append(locations[-1])
    return out


def apply_tm_path(
    tm: carla.TrafficManager,
    vehicle: carla.Vehicle,
    route: Sequence[tuple[carla.Waypoint, RoadOption]],
    *,
    min_spacing_m: float = 5.0,
) -> None:
    """Tell Traffic Manager to follow the planned waypoint path."""
    locations = sparsify_locations(route_locations(route), min_spacing_m=min_spacing_m)
    if len(locations) < 2:
        return
    tm.set_path(vehicle, locations)


def remaining_route_distance_m(
    vehicle_loc: carla.Location,
    route: Sequence[tuple[carla.Waypoint, RoadOption]],
    *,
    start_index: int = 0,
) -> float:
    if not route or start_index >= len(route):
        return 0.0
    total = vehicle_loc.distance(route[start_index][0].transform.location)
    for i in range(start_index, len(route) - 1):
        a = route[i][0].transform.location
        b = route[i + 1][0].transform.location
        total += a.distance(b)
    return total


def nearest_route_index(
    vehicle_loc: carla.Location,
    route: Sequence[tuple[carla.Waypoint, RoadOption]],
    *,
    start_hint: int = 0,
    search_ahead: int = 80,
) -> int:
    if not route:
        return 0
    start = max(0, min(start_hint, len(route) - 1))
    end = min(len(route), start + search_ahead)
    best_i = start
    best_d = float("inf")
    for i in range(start, end):
        d = vehicle_loc.distance(route[i][0].transform.location)
        if d < best_d:
            best_d = d
            best_i = i
    # Also allow a short look-back if we skipped ahead
    look_back = max(0, start - 5)
    for i in range(look_back, start):
        d = vehicle_loc.distance(route[i][0].transform.location)
        if d < best_d:
            best_d = d
            best_i = i
    return best_i


def _local_xy(
    origin: carla.Location,
    yaw_deg: float,
    target: carla.Location,
) -> tuple[float, float]:
    """World target → frame with +x forward, +y right (matches CARLA steer sign)."""
    dx = target.x - origin.x
    dy = target.y - origin.y
    yaw = math.radians(yaw_deg)
    local_x = math.cos(yaw) * dx + math.sin(yaw) * dy
    local_y = -math.sin(yaw) * dx + math.cos(yaw) * dy
    return local_x, local_y


def _wrap_yaw_delta_deg(delta: float) -> float:
    return (delta + 180.0) % 360.0 - 180.0


def geometric_command_from_route(
    route: Sequence[tuple[carla.Waypoint, RoadOption]],
    *,
    start_index: int = 0,
    lookahead_m: float = LOOKAHEAD_METERS,
    turn_lateral_m: float = 2.0,
    turn_yaw_deg: float = 20.0,
    straight_yaw_deg: float = 10.0,
) -> str:
    """
    Classify upcoming maneuver from route geometry (not RoadOption L/R).

    RoadOption LEFT/RIGHT from GRP can disagree with the path the car actually
    takes; lateral/yaw of the polyline matches VehicleControl steer sign:
    +y / +steer = right.
    """
    if not route:
        return COMMAND_FOLLOW_LANE

    idx = max(0, min(start_index, len(route) - 1))
    start_wp = route[idx][0]
    start_loc = start_wp.transform.location
    start_yaw = start_wp.transform.rotation.yaw

    traveled = 0.0
    prev = start_loc
    end_loc = start_loc
    end_yaw = start_yaw
    saw_lane_change: RoadOption | None = None
    saw_junction_option = False
    # First real bend inside the window, not the heading at the far end.
    # The far end can already be the next corner and flip left/right.
    first_turn: tuple[float, float] | None = None

    for i in range(idx, len(route)):
        wp, opt = route[i]
        loc = wp.transform.location
        if i > idx:
            traveled += prev.distance(loc)
        prev = loc
        end_loc = loc
        end_yaw = wp.transform.rotation.yaw
        if opt in (RoadOption.CHANGELANELEFT, RoadOption.CHANGELANERIGHT):
            saw_lane_change = opt
        if opt in (RoadOption.LEFT, RoadOption.RIGHT, RoadOption.STRAIGHT):
            saw_junction_option = True
        if first_turn is None and i > idx:
            dyaw_i = _wrap_yaw_delta_deg(end_yaw - start_yaw)
            _x_i, y_i = _local_xy(start_loc, start_yaw, end_loc)
            if dyaw_i <= -turn_yaw_deg or dyaw_i >= turn_yaw_deg:
                first_turn = (dyaw_i, y_i)
        if traveled >= lookahead_m:
            break

    if saw_lane_change is not None and traveled < lookahead_m * 0.5:
        return road_option_to_command(saw_lane_change)

    if first_turn is not None:
        dyaw, local_y = first_turn
    else:
        _local_x, local_y = _local_xy(start_loc, start_yaw, end_loc)
        dyaw = _wrap_yaw_delta_deg(end_yaw - start_yaw)

    # Yaw is the maneuver. A 2 m lateral offset over 28 m is a curve, not a turn,
    # and using it alone flipped left corners to TURN_RIGHT.
    left = dyaw <= -turn_yaw_deg
    right = dyaw >= turn_yaw_deg
    if left and local_y < turn_lateral_m:
        return COMMAND_TURN_LEFT
    if right and local_y > -turn_lateral_m:
        return COMMAND_TURN_RIGHT
    if saw_junction_option and abs(dyaw) <= straight_yaw_deg and abs(local_y) < turn_lateral_m:
        return COMMAND_GO_STRAIGHT
    return COMMAND_FOLLOW_LANE


def upcoming_command(
    vehicle_loc: carla.Location,
    route: Sequence[tuple[carla.Waypoint, RoadOption]],
    *,
    start_hint: int = 0,
    lookahead_m: float = LOOKAHEAD_METERS,
) -> tuple[str, int]:
    """
    Label the upcoming maneuver from route geometry near the vehicle.

    Returns (command_name, nearest_route_index).
    """
    if not route:
        return COMMAND_FOLLOW_LANE, 0

    idx = nearest_route_index(vehicle_loc, route, start_hint=start_hint)
    command = geometric_command_from_route(
        route, start_index=idx, lookahead_m=lookahead_m
    )
    return command, idx


def hold_turn_command(
    vehicle_loc: carla.Location,
    route: Sequence[tuple[carla.Waypoint, RoadOption]],
    *,
    start_hint: int = 0,
    sticky_command: str | None = None,
    sticky_start_index: int | None = None,
    lookahead_m: float = 28.0,
    hold_lookahead_m: float = 10.0,
    min_hold_m: float = 28.0,
    max_hold_m: float = 70.0,
    min_turn_completed_deg: float = 45.0,
    vehicle_yaw: float | None = None,
) -> tuple[str, int, str | None, int | None]:
    """
    Next-maneuver intent, not a "steer now" trigger.

    TURN_LEFT means the upcoming intersection is a left, held from when that
    corner enters lookahead until the exit lane is straight ahead. Approach
    frames keep the expert's small steer, so the image decides when to turn.
    GO_STRAIGHT is held the same way so it does not flicker for half a second.

    Returns (command, route_index, sticky_command, sticky_start_index).
    """
    intent_cmds = (COMMAND_TURN_LEFT, COMMAND_TURN_RIGHT, COMMAND_GO_STRAIGHT)
    raw, idx = upcoming_command(
        vehicle_loc, route, start_hint=start_hint, lookahead_m=lookahead_m
    )
    sticky = sticky_command
    start = sticky_start_index

    # A held left must not still be saying "left" once the path ahead is a right,
    # and the other way around. That was the wrong-side command at the next corner.
    if (
        sticky in (COMMAND_TURN_LEFT, COMMAND_TURN_RIGHT)
        and raw in (COMMAND_TURN_LEFT, COMMAND_TURN_RIGHT)
        and sticky != raw
    ):
        return raw, idx, raw, idx

    if raw in intent_cmds and sticky != raw:
        # Don't abandon a turn that has not finished just because a later
        # maneuver already sits inside the long lookahead.
        if sticky in (COMMAND_TURN_LEFT, COMMAND_TURN_RIGHT) and start is not None:
            if not _turn_intent_finished(
                route,
                start,
                idx,
                sticky,
                hold_lookahead_m=hold_lookahead_m,
                min_hold_m=min_hold_m,
                min_turn_completed_deg=min_turn_completed_deg,
                max_hold_m=max_hold_m,
                vehicle_yaw=vehicle_yaw,
            ):
                return sticky, idx, sticky, start
        return raw, idx, raw, idx

    if sticky in intent_cmds and start is not None:
        if sticky in (COMMAND_TURN_LEFT, COMMAND_TURN_RIGHT):
            if not _turn_intent_finished(
                route,
                start,
                idx,
                sticky,
                hold_lookahead_m=hold_lookahead_m,
                min_hold_m=min_hold_m,
                min_turn_completed_deg=min_turn_completed_deg,
                max_hold_m=max_hold_m,
                vehicle_yaw=vehicle_yaw,
            ):
                return sticky, idx, sticky, start
        else:
            traveled = _route_distance(route, start, idx)
            short = geometric_command_from_route(
                route, start_index=idx, lookahead_m=hold_lookahead_m
            )
            if traveled < min_hold_m or short == sticky:
                return sticky, idx, sticky, start
        return raw, idx, None, None

    if raw in intent_cmds:
        return raw, idx, raw, idx
    return raw, idx, None, None


def _route_distance(
    route: Sequence[tuple[carla.Waypoint, RoadOption]],
    start_index: int,
    end_index: int,
) -> float:
    if not route:
        return 0.0
    a = max(0, min(start_index, len(route) - 1))
    b = max(0, min(end_index, len(route) - 1))
    if b < a:
        a, b = b, a
    traveled = 0.0
    prev = route[a][0].transform.location
    for i in range(a + 1, b + 1):
        loc = route[i][0].transform.location
        traveled += prev.distance(loc)
        prev = loc
    return traveled


def _turn_intent_finished(
    route: Sequence[tuple[carla.Waypoint, RoadOption]],
    start_index: int,
    index: int,
    sticky: str,
    *,
    hold_lookahead_m: float,
    min_hold_m: float = 28.0,
    min_turn_completed_deg: float,
    max_hold_m: float = 70.0,
    vehicle_yaw: float | None = None,
) -> bool:
    """True only once the exit is straight, not at mid-corner."""
    if not route:
        return True
    start = max(0, min(start_index, len(route) - 1))
    idx = max(0, min(index, len(route) - 1))
    short = geometric_command_from_route(
        route,
        start_index=idx,
        lookahead_m=hold_lookahead_m,
        turn_lateral_m=1.2,
        turn_yaw_deg=12.0,
    )
    traveled = _route_distance(route, start, idx)
    start_yaw = route[start][0].transform.rotation.yaw
    now_yaw = route[idx][0].transform.rotation.yaw
    completed = abs(_wrap_yaw_delta_deg(now_yaw - start_yaw))
    # False engage: called a turn but the path never rotated. Don't wait forever.
    if traveled >= max_hold_m and completed < 20.0 and short != sticky:
        return True
    # Stay on TURN_* through the bend — don't clear until we've actually
    # traveled through the corner and rotated enough.
    if traveled < min_hold_m or completed < min_turn_completed_deg:
        return False
    ahead = _advance_along_route(route, idx, 22.0)
    ahead_yaw = route[ahead][0].transform.rotation.yaw
    if abs(_wrap_yaw_delta_deg(ahead_yaw - now_yaw)) > 6.0:
        return False
    # Hold until the vehicle itself is pointed down the exit.
    if vehicle_yaw is not None and abs(
        _wrap_yaw_delta_deg(vehicle_yaw - ahead_yaw)
    ) > 8.0:
        return False
    return short != sticky

def _advance_along_route(
    route: Sequence[tuple[carla.Waypoint, RoadOption]],
    start_index: int,
    distance_m: float,
) -> int:
    if not route:
        return 0
    i = max(0, min(start_index, len(route) - 1))
    traveled = 0.0
    prev = route[i][0].transform.location
    while i + 1 < len(route) and traveled < distance_m:
        i += 1
        loc = route[i][0].transform.location
        traveled += prev.distance(loc)
        prev = loc
    return i


def analyze_route_turns(
    route: Sequence[tuple[carla.Waypoint, RoadOption]],
    *,
    lookahead_m: float = 15.0,
    skip_after_turn_m: float = 22.0,
) -> dict[str, float | int]:
    """
    Walk a planned route and count geometric LEFT/RIGHT/STRAIGHT maneuvers.

    Better than raw RoadOption counts (those can be L/R-flipped vs the path).
    """
    left = right = straight = 0
    i = 0
    while i < len(route) - 1:
        cmd = geometric_command_from_route(
            route, start_index=i, lookahead_m=lookahead_m
        )
        if cmd == COMMAND_TURN_LEFT:
            left += 1
            i = _advance_along_route(route, i, skip_after_turn_m)
        elif cmd == COMMAND_TURN_RIGHT:
            right += 1
            i = _advance_along_route(route, i, skip_after_turn_m)
        elif cmd == COMMAND_GO_STRAIGHT:
            straight += 1
            i = _advance_along_route(route, i, skip_after_turn_m * 0.6)
        else:
            i = _advance_along_route(route, i, lookahead_m * 0.5)

    turns = left + right
    balance = 0.0
    if turns > 0:
        balance = 1.0 - abs(left - right) / turns
    length_m = 0.0
    for j in range(len(route) - 1):
        a = route[j][0].transform.location
        b = route[j + 1][0].transform.location
        length_m += a.distance(b)

    # Prefer long routes with several turns and left/right balance.
    # Hard-penalize one-sided turn salads (often become TM spin loops).
    score = (turns + 0.25 * straight) * (0.35 + 0.65 * balance) * min(
        1.0, length_m / 200.0
    )
    if turns >= 3 and min(left, right) == 0:
        score *= 0.12
    elif turns >= 2 and min(left, right) == 0:
        score *= 0.4
    return {
        "left": left,
        "right": right,
        "straight": straight,
        "turns": turns,
        "balance": balance,
        "length_m": length_m,
        "score": score,
    }


def pick_balanced_route(
    grp: GlobalRoutePlanner,
    spawn_points: Sequence[carla.Transform],
    *,
    start_index: int | None = None,
    pair_samples: int = 48,
    min_distance_m: float = 120.0,
    rng: random.Random | None = None,
    exclude_dests: Sequence[int] = (),
    top_k: int = 1,
) -> tuple[
    int,
    int,
    carla.Location,
    list[tuple[carla.Waypoint, RoadOption]],
    dict[str, float | int],
]:
    """
    Sample many spawn→spawn A* routes; pick a strong left/right-balanced course.

    With top_k > 1, randomly choose among the top-scoring candidates (more variety).
    exclude_dests skips recently used destinations.
    """
    rng = rng or random.Random()
    n = len(spawn_points)
    if n < 2:
        raise RuntimeError("Need at least 2 spawn points")
    banned = {int(i) for i in exclude_dests}

    origins = [start_index] if start_index is not None else list(range(n))
    if start_index is None and len(origins) > 16:
        origins = rng.sample(origins, 16)

    candidates: list[tuple[int, int]] = []
    for oi in origins:
        origin = spawn_points[oi].location
        far = [
            i
            for i, sp in enumerate(spawn_points)
            if i != oi
            and i not in banned
            and origin.distance(sp.location) >= min_distance_m
        ]
        if len(far) < 3:
            far = [i for i in range(n) if i != oi and i not in banned]
        if not far:
            far = [i for i in range(n) if i != oi]
        take = min(max(4, pair_samples // max(1, len(origins))), len(far))
        for di in rng.sample(far, take):
            candidates.append((oi, di))

    for oi in origins:
        fi = farthest_spawn_index(spawn_points, oi)
        if fi not in banned:
            candidates.append((oi, fi))

    seen: set[tuple[int, int]] = set()
    uniq: list[tuple[int, int]] = []
    for pair in candidates:
        if pair not in seen and pair[0] != pair[1]:
            seen.add(pair)
            uniq.append(pair)

    scored: list[
        tuple[
            float,
            int,
            int,
            carla.Location,
            list[tuple[carla.Waypoint, RoadOption]],
            dict[str, float | int],
        ]
    ] = []

    for oi, di in uniq:
        try:
            dest, route, _junc = plan_spawn_to_spawn(grp, spawn_points, oi, di)
        except Exception:
            continue
        stats = analyze_route_turns(route)
        scored.append((float(stats["score"]), oi, di, dest, route, stats))

    if not scored:
        oi = start_index if start_index is not None else 0
        di = farthest_spawn_index(spawn_points, oi)
        dest, route, _ = plan_spawn_to_spawn(grp, spawn_points, oi, di)
        stats = analyze_route_turns(route)
        return oi, di, dest, route, stats

    scored.sort(key=lambda row: row[0], reverse=True)
    k = max(1, min(top_k, len(scored)))
    # Softmax-ish: prefer better scores but still explore.
    weights = [max(row[0], 0.05) ** 1.5 for row in scored[:k]]
    pick = rng.choices(scored[:k], weights=weights, k=1)[0]
    _score, oi, di, dest, route, stats = pick
    return oi, di, dest, route, stats


def route_forward_alignment(
    route: Sequence[tuple[carla.Waypoint, RoadOption]],
    origin: carla.Location,
    yaw_deg: float,
    *,
    sample_m: float = 14.0,
) -> float:
    """
    How well the start of the route matches current heading.

    Returns cos(angle) in [-1, 1]: +1 = leaves straight ahead, -1 = immediate U-turn.
    Destination may be anywhere; this only inspects the first ~sample_m of path.
    """
    if len(route) < 2:
        return 0.0
    traveled = 0.0
    prev = route[0][0].transform.location
    target = route[min(1, len(route) - 1)][0].transform.location
    for i in range(1, len(route)):
        loc = route[i][0].transform.location
        traveled += prev.distance(loc)
        prev = loc
        target = loc
        if traveled >= sample_m:
            break
    dx = target.x - origin.x
    dy = target.y - origin.y
    norm = math.hypot(dx, dy)
    if norm < 1e-3:
        return 0.0
    fx = math.cos(math.radians(yaw_deg))
    fy = math.sin(math.radians(yaw_deg))
    return (fx * dx + fy * dy) / norm


def plan_origin_along_heading(
    world_map: carla.Map,
    location: carla.Location,
    yaw_deg: float,
    *,
    distance_m: float = 6.0,
) -> carla.Location:
    """
    Road point a few meters ahead of the vehicle.

    Planning from here makes GlobalRoutePlanner enter the graph facing forward
    instead of possibly linking backward from the nearest node.
    """
    yaw = math.radians(yaw_deg)
    ahead = carla.Location(
        location.x + distance_m * math.cos(yaw),
        location.y + distance_m * math.sin(yaw),
        location.z,
    )
    try:
        wp = world_map.get_waypoint(
            ahead, project_to_road=True, lane_type=carla.LaneType.Driving
        )
        if wp is not None:
            # If projected waypoint faces opposite, try next() in our direction.
            wp_yaw = wp.transform.rotation.yaw
            align = math.cos(math.radians(_wrap_yaw_delta_deg(wp_yaw - yaw_deg)))
            if align < 0.0:
                nxt = wp.next(2.0)
                if nxt:
                    wp = nxt[0]
            return wp.transform.location
    except Exception:
        pass
    return ahead


def plan_forward_continuation(
    grp: GlobalRoutePlanner,
    world_map: carla.Map,
    origin: carla.Location,
    yaw_deg: float,
    *,
    distance_m: float = 140.0,
) -> tuple[carla.Location, list[tuple[carla.Waypoint, RoadOption]]]:
    """
    Build a short route that only continues forward along the current road.

    Used when every spawn-to-spawn plan would U-turn from the current heading.
    """
    plan_from = plan_origin_along_heading(world_map, origin, yaw_deg)
    try:
        wp = world_map.get_waypoint(
            plan_from, project_to_road=True, lane_type=carla.LaneType.Driving
        )
    except Exception:
        wp = None
    if wp is None:
        dest = plan_from
        route = plan_route(grp, origin, dest)
        return dest, route

    # Prefer the next() candidate that best matches heading.
    traveled = 0.0
    prev_loc = wp.transform.location
    while traveled < distance_m:
        options = wp.next(2.5) or []
        if not options:
            break
        best_wp = options[0]
        best_align = -2.0
        for cand in options:
            cy = cand.transform.rotation.yaw
            align = math.cos(math.radians(_wrap_yaw_delta_deg(cy - yaw_deg)))
            if align > best_align:
                best_align = align
                best_wp = cand
        # If all options face backward, stop — don't walk into a U-turn.
        if best_align < 0.15:
            break
        wp = best_wp
        loc = wp.transform.location
        traveled += prev_loc.distance(loc)
        prev_loc = loc

    dest = wp.transform.location
    route = plan_route(grp, plan_from, dest)
    # If GRP still U-turns, synthesize a trivial forward segment list.
    align = route_forward_alignment(route, origin, yaw_deg)
    if align < 0.15 or len(route) < 3:
        # Fall back: dense next() chain as fake route with LANEFOLLOW.
        chain: list[tuple[carla.Waypoint, RoadOption]] = []
        wp0 = world_map.get_waypoint(
            plan_from, project_to_road=True, lane_type=carla.LaneType.Driving
        )
        if wp0 is not None:
            chain.append((wp0, RoadOption.LANEFOLLOW))
            cur = wp0
            walked = 0.0
            prev = wp0.transform.location
            while walked < distance_m:
                opts = cur.next(2.0) or []
                if not opts:
                    break
                pick = max(
                    opts,
                    key=lambda c: math.cos(
                        math.radians(
                            _wrap_yaw_delta_deg(c.transform.rotation.yaw - yaw_deg)
                        )
                    ),
                )
                if (
                    math.cos(
                        math.radians(
                            _wrap_yaw_delta_deg(pick.transform.rotation.yaw - yaw_deg)
                        )
                    )
                    < 0.15
                ):
                    break
                chain.append((pick, RoadOption.LANEFOLLOW))
                walked += prev.distance(pick.transform.location)
                prev = pick.transform.location
                cur = pick
            if len(chain) >= 3:
                return chain[-1][0].transform.location, chain
    return dest, route


def pick_next_destination(
    grp: GlobalRoutePlanner,
    origin: carla.Location,
    spawn_points: Sequence[carla.Transform],
    *,
    route_mode: str = "balanced",
    rng: random.Random | None = None,
    exclude_dests: Sequence[int] = (),
    top_k: int = 8,
    heading_yaw: float | None = None,
    world_map: carla.Map | None = None,
    min_forward_alignment: float = 0.25,
) -> tuple[carla.Location, list[tuple[carla.Waypoint, RoadOption]], float, dict]:
    """
    Choose any far destination, but prefer a path that *leaves* in heading_yaw.

    Destinations are not required to be in front of the car — only the first
    segment of the planned polyline should not reverse/U-turn.
    """
    rng = rng or random.Random()
    mode = (route_mode or "balanced").lower()
    banned = {int(i) for i in exclude_dests}
    n = len(spawn_points)
    # Reject anything that leaves backward relative to heading.
    reverse_cut = -0.05

    plan_from = origin
    if heading_yaw is not None and world_map is not None:
        plan_from = plan_origin_along_heading(world_map, origin, heading_yaw)

    def _score_route(
        route: list[tuple[carla.Waypoint, RoadOption]],
        base: float,
    ) -> tuple[float, float]:
        align = 1.0
        if heading_yaw is not None:
            align = route_forward_alignment(route, origin, heading_yaw)
            if align < reverse_cut:
                return -1.0, align
            if align < min_forward_alignment:
                base *= 0.15
            else:
                base = base * (0.45 + 0.55 * max(0.0, align))
        return base, align

    def _try_plan(dest_loc: carla.Location) -> list[tuple[carla.Waypoint, RoadOption]] | None:
        try:
            route = plan_route(grp, plan_from, dest_loc)
        except Exception:
            return None
        if len(route) < 5:
            return None
        return route

    def _forward_fallback() -> tuple[
        carla.Location, list[tuple[carla.Waypoint, RoadOption]], float, dict
    ]:
        """Never return a reverse departure — continue straight if needed."""
        # Exhaustive search over all spawns for any forward-leaving plan.
        candidates: list = []
        for di in range(n):
            if di in banned:
                continue
            route = _try_plan(spawn_points[di].location)
            if route is None:
                continue
            align = (
                route_forward_alignment(route, origin, heading_yaw)
                if heading_yaw is not None
                else 1.0
            )
            if align < min_forward_alignment:
                continue
            stats = analyze_route_turns(route)
            candidates.append(
                (align, float(stats["score"]), di, spawn_points[di].location, route, stats)
            )
        if candidates:
            candidates.sort(key=lambda row: (row[0], row[1]), reverse=True)
            align, score, di, dest, route, stats = candidates[0]
            return dest, route, score, {
                "mode": f"{mode}-forward-search",
                "dest": di,
                "forward_align": align,
                **stats,
            }

        if heading_yaw is not None and world_map is not None:
            dest, route = plan_forward_continuation(
                grp, world_map, origin, heading_yaw
            )
            stats = analyze_route_turns(route)
            align = route_forward_alignment(route, origin, heading_yaw)
            return dest, route, float(stats["score"]), {
                "mode": f"{mode}-forward-escape",
                "dest": -1,
                "forward_align": align,
                **stats,
            }

        # Last resort (no heading): old farthest spawn.
        di = max(range(n), key=lambda i: origin.distance(spawn_points[i].location))
        dest = spawn_points[di].location
        route = _try_plan(dest) or plan_route(grp, origin, dest)
        stats = analyze_route_turns(route)
        return dest, route, float(stats["score"]), {
            "mode": mode,
            "dest": di,
            "forward_align": 1.0,
            **stats,
        }

    # Candidate destinations: far in map distance (any direction).
    far = [
        i
        for i, sp in enumerate(spawn_points)
        if i not in banned and origin.distance(sp.location) >= 100.0
    ]
    if len(far) < 4:
        far = [
            i
            for i in range(n)
            if i not in banned and origin.distance(spawn_points[i].location) >= 40.0
        ]
    if not far:
        far = [i for i in range(n) if origin.distance(spawn_points[i].location) >= 20.0]
    if not far:
        far = list(range(n))

    if mode == "farthest":
        ranked = sorted(
            far,
            key=lambda i: origin.distance(spawn_points[i].location),
            reverse=True,
        )[:20]
        best = None
        for di in ranked:
            route = _try_plan(spawn_points[di].location)
            if route is None:
                continue
            stats = analyze_route_turns(route)
            score, align = _score_route(
                route, float(stats["score"]) + float(stats["length_m"]) / 200.0
            )
            if score < 0:
                continue
            if best is None or score > best[0]:
                best = (score, di, spawn_points[di].location, route, stats, align)
        if best is None:
            return _forward_fallback()
        _s, di, dest, route, stats, align = best
        return dest, route, _s, {
            "mode": mode,
            "dest": di,
            "forward_align": align,
            **stats,
        }

    if mode == "junction":
        scored: list = []
        sample = far if len(far) <= 24 else rng.sample(far, 24)
        for di in sample:
            route = _try_plan(spawn_points[di].location)
            if route is None:
                continue
            junc = route_junction_score(route)
            stats = analyze_route_turns(route)
            score, align = _score_route(route, junc)
            if score < 0:
                continue
            scored.append((score, di, spawn_points[di].location, route, stats, align))
        if not scored:
            return _forward_fallback()
        scored.sort(key=lambda row: row[0], reverse=True)
        k = min(top_k, len(scored))
        pick = rng.choices(
            scored[:k], weights=[max(r[0], 0.05) for r in scored[:k]], k=1
        )[0]
        score, di, dest, route, stats, align = pick
        return dest, route, score, {
            "mode": mode,
            "dest": di,
            "forward_align": align,
            **stats,
        }

    # balanced (default)
    sample = far if len(far) <= 40 else rng.sample(far, 40)
    farthest = sorted(
        range(n),
        key=lambda i: origin.distance(spawn_points[i].location),
        reverse=True,
    )[:5]
    for di in farthest:
        if di not in sample and di not in banned:
            sample.append(di)

    scored = []
    for di in sample:
        route = _try_plan(spawn_points[di].location)
        if route is None:
            continue
        stats = analyze_route_turns(route)
        score, align = _score_route(route, float(stats["score"]))
        if score < 0:
            continue
        scored.append((score, di, spawn_points[di].location, route, stats, align))

    if not scored:
        return _forward_fallback()

    scored.sort(key=lambda row: row[0], reverse=True)
    k = max(1, min(top_k, len(scored)))
    weights = [max(row[0], 0.05) ** 1.5 for row in scored[:k]]
    pick = rng.choices(scored[:k], weights=weights, k=1)[0]
    score, di, dest, route, stats, align = pick
    return dest, route, score, {
        "mode": "balanced",
        "dest": di,
        "forward_align": align,
        **stats,
    }


def needs_replan(
    vehicle_loc: carla.Location,
    route: Sequence[tuple[carla.Waypoint, RoadOption]],
    route_index: int,
    *,
    threshold_m: float = ROUTE_REPLAN_REMAINING_M,
) -> bool:
    return remaining_route_distance_m(vehicle_loc, route, start_index=route_index) < threshold_m


_OPTION_COLORS: dict[RoadOption, carla.Color] = {
    RoadOption.VOID: carla.Color(180, 180, 180),
    RoadOption.LEFT: carla.Color(0, 180, 255),
    RoadOption.RIGHT: carla.Color(255, 140, 0),
    RoadOption.STRAIGHT: carla.Color(255, 220, 0),
    RoadOption.LANEFOLLOW: carla.Color(40, 220, 90),
    RoadOption.CHANGELANELEFT: carla.Color(180, 100, 255),
    RoadOption.CHANGELANERIGHT: carla.Color(255, 80, 180),
}


def draw_route(
    world: carla.World,
    route: Sequence[tuple[carla.Waypoint, RoadOption]],
    *,
    life_time: float = 0.15,
    z: float = 0.4,
    thickness: float = 0.12,
    start_index: int = 0,
    hud: bool = True,
) -> None:
    """Draw the planned route as colored debug lines (by RoadOption).

    ``hud=True`` (default) uses draw_hud_* so overlays stay on the server
    spectator and do **not** leak into RGB camera / training images.
    """
    if len(route) < 2:
        return
    start = max(0, min(start_index, len(route) - 2))
    # Cap segments so a huge route cannot flood the HUD renderer.
    end = min(len(route) - 1, start + 80)
    debug = world.debug
    use_hud = hud and hasattr(debug, "draw_hud_line")
    draw_line = debug.draw_hud_line if use_hud else debug.draw_line
    # HUD thickness is pixels; world thickness is meters.
    line_thickness = 2.0 if use_hud else thickness
    try:
        for i in range(start, end):
            a = route[i][0].transform.location
            b = route[i + 1][0].transform.location
            opt = route[i][1]
            color = _OPTION_COLORS.get(
                RoadOption(int(opt)), carla.Color(40, 220, 90)
            )
            draw_line(
                carla.Location(a.x, a.y, a.z + z),
                carla.Location(b.x, b.y, b.z + z),
                line_thickness,
                color,
                life_time,
                False,  # persistent_lines — must be False with short life_time
            )
    except Exception:
        # Never let debug drawing tear down the sim / Python process.
        return


def draw_target(
    world: carla.World,
    location: carla.Location,
    *,
    life_time: float = 0.15,
    z: float = 0.6,
    size: float = 0.25,
    hud: bool = True,
) -> None:
    """Mark the current pursuit target (HUD by default — not in RGB sensors)."""
    debug = world.debug
    use_hud = hud and hasattr(debug, "draw_hud_point")
    draw_point = debug.draw_hud_point if use_hud else debug.draw_point
    point_size = 6.0 if use_hud else size
    try:
        draw_point(
            carla.Location(location.x, location.y, location.z + z),
            point_size,
            carla.Color(255, 40, 40),
            life_time,
            False,
        )
    except Exception:
        return


def draw_destination(
    world: carla.World,
    location: carla.Location,
    *,
    life_time: float = 0.15,
    z: float = 1.0,
    size: float = 0.35,
    hud: bool = True,
) -> None:
    """Yellow destination marker (HUD by default — not in RGB sensors)."""
    debug = world.debug
    use_hud = hud and hasattr(debug, "draw_hud_point")
    draw_point = debug.draw_hud_point if use_hud else debug.draw_point
    point_size = 8.0 if use_hud else size
    try:
        draw_point(
            carla.Location(location.x, location.y, location.z + z),
            point_size,
            carla.Color(255, 255, 0),
            life_time,
            False,
        )
    except Exception:
        return


def lookahead_target(
    vehicle_loc: carla.Location,
    route: Sequence[tuple[carla.Waypoint, RoadOption]],
    *,
    start_hint: int = 0,
    lookahead_m: float = LOOKAHEAD_METERS,
) -> tuple[carla.Location, int]:
    """
    Pick a route location ~lookahead_m ahead of the nearest waypoint.

    Returns (target_location, nearest_route_index).
    """
    if not route:
        return vehicle_loc, 0
    idx = nearest_route_index(vehicle_loc, route, start_hint=start_hint)
    traveled = 0.0
    prev = route[idx][0].transform.location
    target = route[min(idx + 1, len(route) - 1)][0].transform.location
    for i in range(idx, len(route)):
        loc = route[i][0].transform.location
        if i > idx:
            traveled += prev.distance(loc)
        prev = loc
        target = loc
        if traveled >= lookahead_m:
            break
    return target, idx
