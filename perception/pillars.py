"""Vote lidar points into a bird's-eye pillar map.

Each cell holds geometry stats only — no semantic tags. A network reads
this map and predicts occupied vs free. That replaces the hand-tuned
hit counts in ``bev_from_lidar``.
"""
from __future__ import annotations

import numpy as np

from perception.bev import CELL_M, FORWARD_M, LATERAL_M, MIN_HEIGHT_M
from perception.lidar import LIDAR_EGO_RADIUS_M, LIDAR_X, LIDAR_Y

# log1p(count), mean_z, max_z, min_z, mean dx-to-center, mean dy-to-center
PILLAR_CHANNELS = 6

__all__ = [
    "PILLAR_CHANNELS",
    "grid_shape",
    "pillar_features",
]


def grid_shape(
    *,
    forward_m: float = FORWARD_M,
    lateral_m: float = LATERAL_M,
    cell_m: float = CELL_M,
) -> tuple[int, int]:
    return int(round(forward_m / cell_m)), int(round(2.0 * lateral_m / cell_m))


def pillar_features(
    points: np.ndarray,
    *,
    forward_m: float = FORWARD_M,
    lateral_m: float = LATERAL_M,
    cell_m: float = CELL_M,
    min_height_m: float = MIN_HEIGHT_M,
) -> np.ndarray:
    """Nx4 (x, y, z, tag) → float32 [F, H, W] pillar features.

    Tag is ignored. Ego-body and below-road hits are dropped. Empty pillars
    stay zero.
    """
    n_forward, n_lateral = grid_shape(
        forward_m=forward_m, lateral_m=lateral_m, cell_m=cell_m
    )
    out = np.zeros((PILLAR_CHANNELS, n_forward, n_lateral), dtype=np.float32)
    if len(points) == 0:
        return out

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    dist = np.hypot(x - LIDAR_X, y - LIDAR_Y)
    valid = (
        (dist > LIDAR_EGO_RADIUS_M)
        & (z >= min_height_m)
        & (x >= 0.0)
        & (x < forward_m)
        & (y >= -lateral_m)
        & (y < lateral_m)
    )
    if not np.any(valid):
        return out

    xs = x[valid]
    ys = y[valid]
    zs = z[valid]
    col = np.floor((ys + lateral_m) / cell_m).astype(np.int32)
    row = np.floor((forward_m - xs) / cell_m).astype(np.int32)
    inside = (col >= 0) & (col < n_lateral) & (row >= 0) & (row < n_forward)
    if not np.any(inside):
        return out

    xs = xs[inside]
    ys = ys[inside]
    zs = zs[inside]
    row = row[inside]
    col = col[inside]
    flat = row * n_lateral + col
    n_cells = n_forward * n_lateral

    count = np.bincount(flat, minlength=n_cells).astype(np.float32)
    sum_z = np.bincount(flat, weights=zs, minlength=n_cells).astype(np.float32)
    max_z = np.full(n_cells, -np.inf, dtype=np.float32)
    min_z = np.full(n_cells, np.inf, dtype=np.float32)
    np.maximum.at(max_z, flat, zs)
    np.minimum.at(min_z, flat, zs)

    cx = forward_m - (row.astype(np.float32) + 0.5) * cell_m
    cy = -lateral_m + (col.astype(np.float32) + 0.5) * cell_m
    sum_dx = np.bincount(flat, weights=xs - cx, minlength=n_cells).astype(np.float32)
    sum_dy = np.bincount(flat, weights=ys - cy, minlength=n_cells).astype(np.float32)

    occupied = count > 0
    mean_z = np.zeros(n_cells, dtype=np.float32)
    mean_dx = np.zeros(n_cells, dtype=np.float32)
    mean_dy = np.zeros(n_cells, dtype=np.float32)
    mean_z[occupied] = sum_z[occupied] / count[occupied]
    mean_dx[occupied] = sum_dx[occupied] / count[occupied]
    mean_dy[occupied] = sum_dy[occupied] / count[occupied]
    max_z[~occupied] = 0.0
    min_z[~occupied] = 0.0

    out[0] = np.log1p(count).reshape(n_forward, n_lateral)
    out[1] = mean_z.reshape(n_forward, n_lateral)
    out[2] = max_z.reshape(n_forward, n_lateral)
    out[3] = min_z.reshape(n_forward, n_lateral)
    out[4] = mean_dx.reshape(n_forward, n_lateral)
    out[5] = mean_dy.reshape(n_forward, n_lateral)
    return out
