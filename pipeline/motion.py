"""Local motion planning: Frenet (live) + A* / RRT* teaching demos."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

import carla
import numpy as np

from pipeline.control import ControlTarget
from utils.routing import lookahead_target, nearest_route_index


@dataclass
class PlannedPath:
    """Selected local plan in world frame."""

    target: carla.Location
    path_transform: carla.Transform
    heading_yaw_deg: float
    desired_speed_kmh: float
    cost: float
    # Optional polyline for debug (world xy)
    samples_xy: list[tuple[float, float]] | None = None
    route_index: int = 0


def _route_arclength(route: list) -> np.ndarray:
    n = len(route)
    s = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        a = route[i - 1][0].transform.location
        b = route[i][0].transform.location
        s[i] = s[i - 1] + a.distance(b)
    return s


def frenet_plan(
    vehicle: carla.Vehicle,
    route: list,
    *,
    route_index_hint: int = 0,
    desired_speed_kmh: float = 25.0,
    predictions: list | None = None,
    lateral_offsets_m: tuple[float, ...] = (-1.2, -0.6, 0.0, 0.6, 1.2),
    horizon_m: float = 18.0,
    sample_step_m: float = 2.0,
) -> PlannedPath:
    """Sample lateral offsets along the GRP centerline and pick lowest cost."""
    if not route:
        loc = vehicle.get_location()
        tf = vehicle.get_transform()
        return PlannedPath(loc, tf, tf.rotation.yaw, 0.0, 1e9, route_index=0)

    loc = vehicle.get_location()
    idx = nearest_route_index(loc, route, start_hint=route_index_hint)
    s_arr = _route_arclength(route)
    s0 = float(s_arr[idx])

    best: PlannedPath | None = None
    for d_offset in lateral_offsets_m:
        samples: list[tuple[float, float]] = []
        blocked = False
        comfort = d_offset * d_offset
        progress = 0.0
        clearance = 0.0
        s = s0
        end_i = idx
        while s < s0 + horizon_m and end_i < len(route) - 1:
            # Advance along arc.
            while end_i < len(route) - 1 and s_arr[end_i] < s:
                end_i += 1
            wp = route[end_i][0]
            tf = wp.transform
            yaw = math.radians(tf.rotation.yaw)
            # Left-handed CARLA: +y is right of heading when using (cos,sin) for forward.
            # Lateral +offset = toward vehicle right (same as CTE sign).
            ox = tf.location.x + d_offset * math.sin(yaw)
            oy = tf.location.y - d_offset * math.cos(yaw)
            samples.append((ox, oy))
            # Collision vs predicted agents at matching time ≈ distance/speed.
            if predictions:
                speed_m_s = max(desired_speed_kmh / 3.6, 1.0)
                t = (s - s0) / speed_m_s
                for pred in predictions:
                    traj = pred.trajectory_xy
                    ti = int(round(t / 0.1))
                    ti = max(0, min(len(traj) - 1, ti))
                    # Pred is ego-frame; convert sample to ego frame.
                    ego_tf = vehicle.get_transform()
                    eyaw = math.radians(ego_tf.rotation.yaw)
                    dx = ox - ego_tf.location.x
                    dy = oy - ego_tf.location.y
                    ex = math.cos(eyaw) * dx + math.sin(eyaw) * dy
                    ey = -math.sin(eyaw) * dx + math.cos(eyaw) * dy
                    px, py = float(traj[ti, 0]), float(traj[ti, 1])
                    dist = math.hypot(ex - px, ey - py)
                    half = 0.5 * (float(pred.width_m) + 2.0)
                    if dist < half + 0.8:
                        blocked = True
                        break
                    clearance += min(dist, 20.0)
                if blocked:
                    break
            s += sample_step_m
            progress += sample_step_m

        if blocked or not samples:
            continue
        # Lookahead target ~ mid/far sample.
        tx, ty = samples[min(len(samples) - 1, max(1, len(samples) // 2))]
        target = carla.Location(x=tx, y=ty, z=loc.z)
        path_wp = route[idx][0]
        heading_wp = route[min(idx + 1, len(route) - 1)][0]
        n_clear = max(1, len(samples) * max(1, len(predictions or [])))
        mean_clear = clearance / n_clear if predictions else 10.0
        cost = (
            1.2 * comfort
            - 0.05 * progress
            - 0.15 * mean_clear
            + 0.02 * abs(d_offset) * desired_speed_kmh
        )
        plan = PlannedPath(
            target=target,
            path_transform=path_wp.transform,
            heading_yaw_deg=heading_wp.transform.rotation.yaw,
            desired_speed_kmh=desired_speed_kmh,
            cost=cost,
            samples_xy=samples,
            route_index=idx,
        )
        if best is None or plan.cost < best.cost:
            best = plan

    if best is None:
        target, new_idx = lookahead_target(
            loc, route, start_hint=idx, lookahead_m=8.0
        )
        wp = route[idx][0]
        heading_wp = route[min(idx + 1, len(route) - 1)][0]
        return PlannedPath(
            target=target,
            path_transform=wp.transform,
            heading_yaw_deg=heading_wp.transform.rotation.yaw,
            desired_speed_kmh=max(0.0, desired_speed_kmh * 0.5),
            cost=1e6,
            route_index=new_idx,
        )
    return best


def plan_to_control_target(plan: PlannedPath) -> ControlTarget:
    return ControlTarget(
        target=plan.target,
        path_transform=plan.path_transform,
        desired_speed_kmh=plan.desired_speed_kmh,
        heading_yaw_deg=plan.heading_yaw_deg,
    )


# --- Teaching demos: A* and RRT* on a 2D occupancy grid -------------------


def astar_grid(
    occ: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
) -> list[tuple[int, int]] | None:
    """4-connected A* on a binary occupancy grid (True = blocked)."""
    h, w = occ.shape
    sy, sx = start
    gy, gx = goal
    if not (0 <= sx < w and 0 <= sy < h and 0 <= gx < w and 0 <= gy < h):
        return None
    if occ[sy, sx] or occ[gy, gx]:
        return None

    def heur(y: int, x: int) -> float:
        return abs(y - gy) + abs(x - gx)

    open_set: dict[tuple[int, int], float] = {(sy, sx): 0.0}
    came: dict[tuple[int, int], tuple[int, int]] = {}
    gscore = {(sy, sx): 0.0}
    closed: set[tuple[int, int]] = set()
    while open_set:
        current = min(open_set, key=lambda n: open_set[n])
        del open_set[current]
        if current == (gy, gx):
            path = [current]
            while path[-1] in came:
                path.append(came[path[-1]])
            path.reverse()
            return path
        closed.add(current)
        cy, cx = current
        for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            ny, nx = cy + dy, cx + dx
            if not (0 <= nx < w and 0 <= ny < h) or occ[ny, nx]:
                continue
            if (ny, nx) in closed:
                continue
            tent = gscore[current] + 1.0
            if tent < gscore.get((ny, nx), float("inf")):
                came[(ny, nx)] = current
                gscore[(ny, nx)] = tent
                open_set[(ny, nx)] = tent + heur(ny, nx)
    return None


def rrt_star_grid(
    occ: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    *,
    iterations: int = 800,
    step: float = 3.0,
    goal_bias: float = 0.1,
    seed: int = 0,
) -> list[tuple[int, int]] | None:
    """Lightweight RRT* on a grid (teaching demo; not the live driver)."""
    rng = random.Random(seed)
    h, w = occ.shape
    sy, sx = start
    gy, gx = goal

    def free(y: float, x: float) -> bool:
        iy, ix = int(round(y)), int(round(x))
        if not (0 <= ix < w and 0 <= iy < h):
            return False
        return not bool(occ[iy, ix])

    if not free(sy, sx) or not free(gy, gx):
        return None

    nodes = [(float(sy), float(sx))]
    parent = [-1]
    cost = [0.0]

    def nearest(y: float, x: float) -> int:
        best_i, best_d = 0, float("inf")
        for i, (ny, nx) in enumerate(nodes):
            d = (ny - y) ** 2 + (nx - x) ** 2
            if d < best_d:
                best_d, best_i = d, i
        return best_i

    def steer(y0: float, x0: float, y1: float, x1: float) -> tuple[float, float]:
        dy, dx = y1 - y0, x1 - x0
        dist = math.hypot(dy, dx)
        if dist <= step:
            return y1, x1
        return y0 + step * dy / dist, x0 + step * dx / dist

    def collision_free(y0: float, x0: float, y1: float, x1: float) -> bool:
        dist = math.hypot(y1 - y0, x1 - x0)
        n = max(1, int(dist))
        for i in range(n + 1):
            t = i / n
            if not free(y0 + t * (y1 - y0), x0 + t * (x1 - x0)):
                return False
        return True

    for _ in range(iterations):
        if rng.random() < goal_bias:
            ry, rx = float(gy), float(gx)
        else:
            ry, rx = rng.uniform(0, h - 1), rng.uniform(0, w - 1)
        ni = nearest(ry, rx)
        ny, nx = steer(nodes[ni][0], nodes[ni][1], ry, rx)
        if not collision_free(nodes[ni][0], nodes[ni][1], ny, nx):
            continue
        # Near set for rewiring.
        near_r = step * 2.5
        near_ids = [
            i
            for i, (yy, xx) in enumerate(nodes)
            if math.hypot(yy - ny, xx - nx) <= near_r
        ]
        best_parent = ni
        best_cost = cost[ni] + math.hypot(nodes[ni][0] - ny, nodes[ni][1] - nx)
        for i in near_ids:
            c = cost[i] + math.hypot(nodes[i][0] - ny, nodes[i][1] - nx)
            if c < best_cost and collision_free(nodes[i][0], nodes[i][1], ny, nx):
                best_parent, best_cost = i, c
        nodes.append((ny, nx))
        parent.append(best_parent)
        cost.append(best_cost)
        new_i = len(nodes) - 1
        for i in near_ids:
            c = best_cost + math.hypot(nodes[i][0] - ny, nodes[i][1] - nx)
            if c < cost[i] and collision_free(ny, nx, nodes[i][0], nodes[i][1]):
                parent[i] = new_i
                cost[i] = c
        if math.hypot(ny - gy, nx - gx) < step and collision_free(ny, nx, gy, gx):
            nodes.append((float(gy), float(gx)))
            parent.append(new_i)
            cost.append(best_cost + math.hypot(ny - gy, nx - gx))
            break

    # Extract path if goal reached.
    goal_i = None
    for i, (yy, xx) in enumerate(nodes):
        if math.hypot(yy - gy, xx - gx) < 1.0:
            goal_i = i
            break
    if goal_i is None:
        return None
    path: list[tuple[int, int]] = []
    i = goal_i
    while i >= 0:
        path.append((int(round(nodes[i][0])), int(round(nodes[i][1]))))
        i = parent[i]
    path.reverse()
    return path


def demo_occupancy(h: int = 40, w: int = 40) -> np.ndarray:
    """Synthetic blocked corridor for offline A*/RRT* demos."""
    occ = np.zeros((h, w), dtype=bool)
    occ[5:35, 18:22] = True
    occ[20:24, 22:32] = True
    occ[10:14, 8:18] = True
    return occ
