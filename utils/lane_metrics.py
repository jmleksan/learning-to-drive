"""Eval-only lane position metrics (never model inputs)."""
from __future__ import annotations

from typing import Sequence

import carla

from utils.waypoint_follower import _crosstrack_m, _wrap_deg


def lane_metrics_at(
    vehicle_loc: carla.Location,
    vehicle_yaw_deg: float,
    route: Sequence[tuple[carla.Waypoint, object]] | list,
    route_index: int,
) -> dict[str, float]:
    """
    Signed crosstrack to the route centerline + normalized lane offset.

    cte_m: + = vehicle right of path (same sign convention as steer).
    lane_offset_frac: cte / (0.5 * lane_width); ~0 center, ~±1 at lane edge.
    heading_err_deg: path yaw − vehicle yaw, wrapped to [-180, 180].
    """
    if not route:
        return {"cte_m": 0.0, "lane_offset_frac": 0.0, "heading_err_deg": 0.0}

    idx = max(0, min(int(route_index), len(route) - 1))
    wp = route[idx][0]
    path_tf = wp.transform
    cte = float(_crosstrack_m(path_tf, vehicle_loc))
    half_w = max(0.5 * float(getattr(wp, "lane_width", 3.5) or 3.5), 0.5)
    frac = cte / half_w
    heading_err = _wrap_deg(path_tf.rotation.yaw - vehicle_yaw_deg)
    return {
        "cte_m": cte,
        "lane_offset_frac": frac,
        "heading_err_deg": heading_err,
    }


def format_lane_metric_fields(metrics: dict[str, float]) -> dict[str, str]:
    return {
        "cte_m": f"{metrics['cte_m']:.4f}",
        "lane_offset_frac": f"{metrics['lane_offset_frac']:.4f}",
        "heading_err_deg": f"{metrics['heading_err_deg']:.3f}",
    }
