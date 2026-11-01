"""Lane-keeping reward: progress, centering, collision, off-lane, stall, comfort.

Traffic runs add a hazard speed cap (red light / car / pedestrian ahead, from
sim truth) and a red-light penalty, Roach-style: the speed term rewards
matching the cap, so waiting at a red scores like cruising, not like idling.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class RewardWeights:
    progress: float = 1.0
    cte: float = 0.5
    heading: float = 0.25
    invasion: float = 1.0
    collision: float = 20.0
    # Leaving the lane (env CTE_FAIL_M) costs as much as a crash, so a missed
    # turn is not a cheap way to end the episode.
    off_lane: float = 20.0
    in_lane_cte_m: float = 1.0
    # Same price for stalling (env ends the episode), so crawling can't
    # push the off-lane penalty out past the discount horizon.
    stall: float = 20.0
    # Crossing a red light's stop line: same price as a crash; env ends the episode.
    red_light: float = 20.0
    steer_jerk: float = 0.8
    steer: float = 0.25
    steady: float = 0.25
    cte_deadzone_m: float = 0.2
    heading_deadzone_deg: float = 2.0
    center_steer_extra: float = 2.0
    # Ramps from -idle at a standstill to 0 at idle_kmh. A hard cutoff at
    # 3 km/h let the car crawl just above it for free.
    idle: float = 0.5
    idle_kmh: float = 10.0
    speed: float = 0.45
    cruise_kmh: float = 18.0
    too_fast_kmh: float = 28.0
    min_steady_kmh: float = 14.0
    turn_hold: float = 0.22
    turn_flip: float = 0.6
    turn_match: float = 0.25


DEFAULT_WEIGHTS = RewardWeights()


def _cruise_speed_reward(speed: float, weights: RewardWeights) -> float:
    """Peak at cruise_kmh, fade to 0 by too_fast_kmh so corners can slow down."""
    cruise = max(1.0, float(weights.cruise_kmh))
    cap = max(cruise + 1.0, float(weights.too_fast_kmh))
    if speed <= cruise:
        scale = speed / cruise
    elif speed <= cap:
        scale = max(0.0, 1.0 - (speed - cruise) / (cap - cruise))
    else:
        scale = -min(1.0, (speed - cap) / 10.0)
    return weights.speed * scale


def step_reward(
    *,
    progress_m: float,
    cte_m: float,
    heading_err_deg: float,
    n_invasions: int,
    collided: bool,
    d_steer: float,
    off_lane: bool = False,
    stalled: bool = False,
    ran_red: bool = False,
    speed_cap_kmh: float | None = None,
    steer: float = 0.0,
    speed_kmh: float = 0.0,
    turning: bool = False,
    weights: RewardWeights = DEFAULT_WEIGHTS,
) -> tuple[float, dict[str, float]]:
    """Scalar reward plus a breakdown for logs."""
    signed_cte = float(cte_m)
    heading_signed = float(heading_err_deg)
    cte = abs(signed_cte)
    heading = abs(heading_signed)
    steer_val = float(steer)
    d_steer_val = float(d_steer)
    steer_abs = abs(steer_val)
    speed = max(0.0, float(speed_kmh))
    in_lane = cte < weights.in_lane_cte_m
    # Progress and speed only pay while in lane; drifting off a missed turn
    # earns nothing (too-fast penalty still applies).
    r_progress = weights.progress * float(progress_m) if in_lane else 0.0
    r_cte = -weights.cte * max(0.0, cte - weights.cte_deadzone_m)
    r_heading = (
        -weights.heading * max(0.0, heading - weights.heading_deadzone_deg) / 45.0
    )
    r_inv = -weights.invasion * float(max(0, int(n_invasions)))
    r_col = -weights.collision if collided else 0.0
    r_off = -weights.off_lane if off_lane and not collided else 0.0
    r_stall = (
        -weights.stall if stalled and not collided and not off_lane else 0.0
    )
    r_red = (
        -weights.red_light if ran_red and not collided and not off_lane else 0.0
    )
    jerk_scale = 0.35 if turning else 1.0
    r_jerk = -weights.steer_jerk * jerk_scale * abs(d_steer_val)
    # Extra |steer| on straights only. Base |steer| is also cut on a mapped
    # turn so the car can hold a radius instead of pulse-and-coast.
    if turning:
        on_line = 0.0
        steer_scale = 0.25
    else:
        on_line = max(0.0, 1.0 - cte / 0.6)
        steer_scale = 1.0
    r_steer = (
        -weights.steer * steer_scale * steer_abs * (1.0 + weights.center_steer_extra * on_line)
    )
    r_steady = 0.0
    if (
        not turning
        and cte < 0.25
        and heading < 8.0
        and steer_abs < 0.1
        and speed >= weights.min_steady_kmh
    ):
        r_steady = weights.steady
    # A hazard ahead lowers the speed the car should be at; idling is only
    # penalised below that, so stopping for a red or a lead car is free.
    idle_ref = weights.idle_kmh
    if speed_cap_kmh is not None:
        idle_ref = min(idle_ref, float(speed_cap_kmh))
    r_idle = 0.0
    if idle_ref > 0.5:
        r_idle = -weights.idle * max(0.0, 1.0 - speed / idle_ref)
    if speed_cap_kmh is not None and speed_cap_kmh < weights.cruise_kmh:
        # Peak (+speed weight) at the cap; -speed weight once off by cruise/2.
        gap = abs(speed - float(speed_cap_kmh))
        r_speed = weights.speed * max(-1.0, 1.0 - 2.0 * gap / weights.cruise_kmh)
    else:
        r_speed = _cruise_speed_reward(speed, weights)
    if not in_lane:
        r_speed = min(0.0, r_speed)
    hazard = speed_cap_kmh is not None and speed_cap_kmh < weights.cruise_kmh
    if hazard and speed > float(speed_cap_kmh) + 3.0:
        # Closing on a hazard too fast earns no progress or steady bonus.
        r_progress = 0.0
        r_steady = 0.0
    r_hold = 0.0
    r_flip = 0.0
    r_match = 0.0
    if turning:
        prev_steer = steer_val - d_steer_val
        if prev_steer * steer_val < 0.0 and abs(prev_steer) > 0.05 and steer_abs > 0.05:
            r_flip = -weights.turn_flip
        target = math.radians(heading_signed) + math.atan2(-signed_cte, 8.0)
        target = max(-1.0, min(1.0, target))
        # Hold bonus only when the held wheel points the way the path needs.
        if steer_abs > 0.08 and abs(d_steer_val) < 0.04 and steer_val * target > 0.0:
            r_hold = weights.turn_hold
        r_match = -weights.turn_match * abs(steer_val - target)
    total = (
        r_progress
        + r_cte
        + r_heading
        + r_inv
        + r_col
        + r_off
        + r_stall
        + r_red
        + r_jerk
        + r_steer
        + r_steady
        + r_idle
        + r_speed
        + r_hold
        + r_flip
        + r_match
    )
    parts = {
        "r_progress": r_progress,
        "r_cte": r_cte,
        "r_heading": r_heading,
        "r_invasion": r_inv,
        "r_collision": r_col,
        "r_off_lane": r_off,
        "r_stall": r_stall,
        "r_red_light": r_red,
        "r_jerk": r_jerk,
        "r_steer": r_steer,
        "r_steady": r_steady,
        "r_idle": r_idle,
        "r_speed": r_speed,
        "r_hold": r_hold,
        "r_flip": r_flip,
        "r_match": r_match,
        "reward": total,
    }
    return total, parts
