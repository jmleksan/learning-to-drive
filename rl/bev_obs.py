"""Ground-truth bird's-eye observation for RL: drivable road + planned route.

Roach-style privileged input (Zhang et al. 2021): the policy still learns only
from reward, but sees the road and the path ahead drawn from the CARLA map
instead of inferring them from pixels. Ego-centric, ego facing up.

Channel 0: drivable road (all Driving lanes, junctions included).
Channel 1: planned route from just behind the car to ~50 m ahead.
With objects (traffic runs), three more channels from true actor boxes:
Channel 2: other vehicles.
Channel 3: pedestrians.
Channel 4: stop lines of red / yellow traffic lights.

The road is pre-rendered once into a world-aligned raster and sampled per step
(nearest cell); the route is painted per step from the route points in view.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

BEV_RES_M = 0.5
BEV_FORWARD_M = 36.0
BEV_BACK_M = 4.0
BEV_LATERAL_M = 16.0
BEV_H = int(round((BEV_FORWARD_M + BEV_BACK_M) / BEV_RES_M))
BEV_W = int(round(2 * BEV_LATERAL_M / BEV_RES_M))
BEV_CHANNELS = 2
BEV_OBJECT_CHANNELS = 5
CH_ROAD, CH_ROUTE, CH_VEHICLES, CH_WALKERS, CH_STOP = range(BEV_OBJECT_CHANNELS)
# Pedestrians are ~0.3 m boxes; paint at least this half-size so they show.
MIN_WALKER_HALF_M = 0.5
ROUTE_HALF_WIDTH_M = 1.0
ROUTE_BEHIND_M = 4.0
ROUTE_AHEAD_M = 55.0
_MAP_MARGIN_M = 30.0
_ROAD_SPACING_M = 1.0


def _disc_offsets(radius_cells: float) -> tuple[np.ndarray, np.ndarray]:
    r = int(math.ceil(radius_cells))
    dr, dc = np.mgrid[-r : r + 1, -r : r + 1]
    keep = dr**2 + dc**2 <= radius_cells**2
    return dr[keep], dc[keep]


@dataclass
class BevObjects:
    """World-frame boxes (x, y, yaw_deg, half_length, half_width) to paint."""

    vehicles: list[tuple[float, float, float, float, float]] = field(default_factory=list)
    walkers: list[tuple[float, float, float, float, float]] = field(default_factory=list)
    stops: list[tuple[float, float, float, float, float]] = field(default_factory=list)


def _paint(grid: np.ndarray, rows: np.ndarray, cols: np.ndarray, dr, dc) -> None:
    """Set grid[rows+dr, cols+dc] = 255 for every point/offset pair in bounds."""
    rr = (rows[:, None] + dr[None, :]).ravel()
    cc = (cols[:, None] + dc[None, :]).ravel()
    ok = (rr >= 0) & (rr < grid.shape[0]) & (cc >= 0) & (cc < grid.shape[1])
    grid[rr[ok], cc[ok]] = 255


class BevRasterizer:
    """Road raster for one map plus a per-episode route; render() is per step."""

    def __init__(self, road_xy: np.ndarray, road_half_width: np.ndarray) -> None:
        road_xy = np.asarray(road_xy, dtype=np.float64)
        half_w = np.asarray(road_half_width, dtype=np.float64)
        self.x0 = float(road_xy[:, 0].min()) - _MAP_MARGIN_M
        self.y0 = float(road_xy[:, 1].min()) - _MAP_MARGIN_M
        nx = int(math.ceil((road_xy[:, 0].max() + _MAP_MARGIN_M - self.x0) / BEV_RES_M))
        ny = int(math.ceil((road_xy[:, 1].max() + _MAP_MARGIN_M - self.y0) / BEV_RES_M))
        self.road = np.zeros((nx, ny), dtype=np.uint8)
        ix = ((road_xy[:, 0] - self.x0) / BEV_RES_M).astype(np.int64)
        iy = ((road_xy[:, 1] - self.y0) / BEV_RES_M).astype(np.int64)
        # Group by width so each group shares one disc stencil.
        for w in np.unique(np.round(half_w / BEV_RES_M, 1)):
            sel = np.round(half_w / BEV_RES_M, 1) == w
            dr, dc = _disc_offsets(float(w))
            _paint(self.road, ix[sel], iy[sel], dr, dc)

        # Ego-frame offsets of every cell centre: +fx forward, +fy right.
        rows, cols = np.mgrid[0:BEV_H, 0:BEV_W]
        self._fx = BEV_FORWARD_M - (rows + 0.5) * BEV_RES_M
        self._fy = -BEV_LATERAL_M + (cols + 0.5) * BEV_RES_M
        self._route_dr, self._route_dc = _disc_offsets(ROUTE_HALF_WIDTH_M / BEV_RES_M)
        self._route_xy = np.zeros((0, 2))
        self._route_s = np.zeros(0)
        self._route_wp_s = np.zeros(0)

    @classmethod
    def from_map(cls, world_map) -> "BevRasterizer":
        wps = world_map.generate_waypoints(_ROAD_SPACING_M)
        xy = np.array([(w.transform.location.x, w.transform.location.y) for w in wps])
        half_w = np.array([0.5 * float(w.lane_width or 3.5) for w in wps])
        return cls(xy, half_w)

    def set_route(self, route_xy) -> None:
        """Route polyline in world XY, resampled to half-cell spacing."""
        pts = np.asarray(route_xy, dtype=np.float64)[:, :2]
        if len(pts) < 2:
            self._route_xy, self._route_s = pts, np.zeros(len(pts))
            self._route_wp_s = np.zeros(len(pts))
            return
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(seg)])
        fine = np.arange(0.0, s[-1], BEV_RES_M * 0.5)
        self._route_xy = np.stack(
            [np.interp(fine, s, pts[:, 0]), np.interp(fine, s, pts[:, 1])], axis=1
        )
        self._route_s = fine
        self._route_wp_s = s

    def render(
        self,
        x: float,
        y: float,
        yaw_deg: float,
        route_index: int = 0,
        objects: BevObjects | None = None,
    ) -> np.ndarray:
        """(C, BEV_H, BEV_W) uint8, ego at bottom centre facing up.

        C is BEV_CHANNELS, or BEV_OBJECT_CHANNELS when ``objects`` is given.
        """
        c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
        # CARLA: forward = (cos, sin), right = (-sin, cos).
        wx = x + self._fx * c - self._fy * s
        wy = y + self._fx * s + self._fy * c
        ix = ((wx - self.x0) / BEV_RES_M).astype(np.int64)
        iy = ((wy - self.y0) / BEV_RES_M).astype(np.int64)
        ok = (ix >= 0) & (ix < self.road.shape[0]) & (iy >= 0) & (iy < self.road.shape[1])
        n_ch = BEV_CHANNELS if objects is None else BEV_OBJECT_CHANNELS
        out = np.zeros((n_ch, BEV_H, BEV_W), dtype=np.uint8)
        out[CH_ROAD][ok] = self.road[ix[ok], iy[ok]]

        if len(self._route_xy):
            idx = min(max(0, int(route_index)), len(self._route_wp_s) - 1)
            here = self._route_wp_s[idx]
            sel = (self._route_s >= here - ROUTE_BEHIND_M) & (
                self._route_s <= here + ROUTE_AHEAD_M
            )
            dx = self._route_xy[sel, 0] - x
            dy = self._route_xy[sel, 1] - y
            fx = dx * c + dy * s
            fy = -dx * s + dy * c
            rows = np.floor((BEV_FORWARD_M - fx) / BEV_RES_M).astype(np.int64)
            cols = np.floor((fy + BEV_LATERAL_M) / BEV_RES_M).astype(np.int64)
            _paint(out[CH_ROUTE], rows, cols, self._route_dr, self._route_dc)
        if objects is not None:
            for ch, boxes in (
                (CH_VEHICLES, objects.vehicles),
                (CH_WALKERS, objects.walkers),
                (CH_STOP, objects.stops),
            ):
                for box in boxes:
                    _paint_box(out[ch], x, y, c, s, box)
        return out


def _paint_box(grid: np.ndarray, x: float, y: float, c: float, s: float, box) -> None:
    """Fill one world-frame oriented box into an ego-frame grid."""
    bx, by, byaw, hl, hw = box
    step = BEV_RES_M * 0.5
    lx, ly = np.meshgrid(np.arange(-hl, hl + 1e-6, step), np.arange(-hw, hw + 1e-6, step))
    cb, sb = math.cos(math.radians(byaw)), math.sin(math.radians(byaw))
    wx = bx + lx * cb - ly * sb
    wy = by + lx * sb + ly * cb
    dx, dy = wx - x, wy - y
    fx = dx * c + dy * s
    fy = -dx * s + dy * c
    rows = np.floor((BEV_FORWARD_M - fx) / BEV_RES_M).astype(np.int64).ravel()
    cols = np.floor((fy + BEV_LATERAL_M) / BEV_RES_M).astype(np.int64).ravel()
    ok = (rows >= 0) & (rows < grid.shape[0]) & (cols >= 0) & (cols < grid.shape[1])
    grid[rows[ok], cols[ok]] = 255


def bev_to_rgb(bev: np.ndarray, scale: int = 4) -> np.ndarray:
    """Grey road, green route, red stop lines, blue cars, yellow people, white ego."""
    h, w = bev.shape[1:]
    rgb = np.full((h, w, 3), 30, dtype=np.uint8)
    rgb[bev[CH_ROAD] > 0] = (110, 110, 110)
    rgb[bev[CH_ROUTE] > 0] = (40, 200, 70)
    if bev.shape[0] >= BEV_OBJECT_CHANNELS:
        rgb[bev[CH_STOP] > 0] = (230, 40, 40)
        rgb[bev[CH_VEHICLES] > 0] = (70, 130, 255)
        rgb[bev[CH_WALKERS] > 0] = (255, 210, 0)
    ego_r = int(BEV_FORWARD_M / BEV_RES_M)
    ego_c = w // 2
    rgb[max(0, ego_r - 4) : ego_r + 4, ego_c - 2 : ego_c + 2] = (240, 240, 240)
    return np.repeat(np.repeat(rgb, scale, axis=0), scale, axis=1)


# Perceived road: SegNet road / road-line, projected by perception.bev.
PERCEIVED_CAMERAS = ("center", "left", "right")
PERCEIVED_ROAD_IDS = (1, 24)


class PerceivedRoad:
    """Road channel from the car's own cameras instead of the map.

    Runs SegNet on center/left/right RGB in one batch and projects the labels
    onto the same grid as BevRasterizer with perception.bev.fill_ground_multi.
    Cells the cameras can't see (behind, beside the bumper) stay empty.
    """

    def __init__(self, seg_path, device: str = "auto") -> None:
        import torch

        from segmentation.model import load_seg_model

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model = load_seg_model(seg_path, device=device)
        self._torch = torch

    def road(self, rgb_by_camera: dict[str, np.ndarray]) -> np.ndarray:
        """(BEV_H, BEV_W) uint8 road mask (0/255) from {camera name: HxWx3 RGB}."""
        from perception.bev import fill_ground_multi

        torch = self._torch
        names = [n for n in PERCEIVED_CAMERAS if n in rgb_by_camera]
        batch = np.stack([rgb_by_camera[n] for n in names]).astype(np.float32) / 255.0
        tensor = torch.from_numpy(batch).permute(0, 3, 1, 2).to(self.device)
        with torch.no_grad():
            labels = self.model(tensor).argmax(dim=1).cpu().numpy().astype(np.int16)
        grid = np.full((BEV_H, BEV_W), -1, dtype=np.int16)
        # Center first so it wins ties, as in perception.bev.
        fill_ground_multi(
            grid,
            list(zip(labels, names)),
            forward_m=BEV_FORWARD_M,
            lateral_m=BEV_LATERAL_M,
            cell_m=BEV_RES_M,
        )
        return np.where(np.isin(grid, PERCEIVED_ROAD_IDS), 255, 0).astype(np.uint8)


def road_agreement(perceived: np.ndarray, truth: np.ndarray) -> tuple[float, float]:
    """(precision, recall) of perceived road vs map road, ahead of the car only."""
    ahead = np.zeros_like(truth, dtype=bool)
    ahead[: int(BEV_FORWARD_M / BEV_RES_M)] = True
    p = (perceived > 0) & ahead
    t = (truth > 0) & ahead
    hit = float((p & t).sum())
    precision = hit / max(1.0, float(p.sum()))
    recall = hit / max(1.0, float(t.sum()))
    return precision, recall
