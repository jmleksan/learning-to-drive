"""Bird's-eye grid: camera road + lidar objects.

The road and sidewalk come from the center semantic camera. Cars, people,
and poles come from the roof lidar. ``python -m perception.bev`` writes a
strip per frame: photo, tags, the CARLA answer key (true tags + depth), the
lidar grid, and a match picture. Green cells agree, red cells miss, blue
cells are extras.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from perception.depth import FOCAL_PX, IMAGE_HEIGHT, IMAGE_WIDTH, MAX_DEPTH_M
from segmentation.labels import CLASS_NAMES, PALETTE, colorize

# Camera mounts, vehicle frame: x forward, y right, z up, yaw deg (CARLA).
# Must stay in sync with utils.vehicles.CAMERA_SPECS.
CAMERA_X = 1.5
CAMERA_Y = 0.0
CAMERA_Z = 2.4
CAM_EXTRINSICS: dict[str, tuple[float, float, float, float]] = {
    "center": (1.5, 0.0, 2.4, 0.0),
    "left": (1.5, -0.40, 2.4, -55.0),
    "right": (1.5, 0.40, 2.4, 55.0),
    "left_wide": (1.5, -1.00, 2.4, 0.0),
    "right_wide": (1.5, 1.00, 2.4, 0.0),
}
SKY = 11

FORWARD_M = 40.0
LATERAL_M = 12.0
CELL_M = 0.25
# Road, sidewalk, terrain, road line, ground. These sit on z = 0.
GROUND_IDS = (1, 2, 10, 24, 25)
# Side cams: road + sidewalk + road line. Terrain/ground alone were the
# main wall→BEV ghosts; sidewalk is needed or the right wedge looks cut off
# (Town01 curb is often tagged sidewalk). Keep geometric clamps below.
SIDE_GROUND_IDS = (1, 2, 24)
# Points below this height are the road surface and are dropped.
MIN_HEIGHT_M = 0.30
# The classes the stereo miss showed up on: cars and people.
VEHICLE_IDS = (12, 13, 14, 15, 16, 18, 19)
# Classes that become one boxed object on the map.
OBJECT_IDS = VEHICLE_IDS + (6, 7, 8)  # pole, traffic light, traffic sign
# What a driving policy should see — drop poles/signs (big false-positive source).
DRIVE_OBJECT_IDS = VEHICLE_IDS
# The center camera sees the ego hood in the bottom rows, tagged as a car.
EGO_HOOD_ROWS = 18
# A cell survives only when this many hits land in it. One stray pixel does not.
STEREO_MIN_HITS = 4
LIDAR_MIN_HITS = 2
# Blobs smaller than this are dust, not an object. 4 cells is about 0.5 m.
MIN_BLOB_CELLS = 4
# Stricter cleanup for the learned occupancy path.
OCC_MIN_BLOB_CELLS = 8
OCC_THRESHOLD = 0.5
# Drop detections glued to the ego body (common occupancy false positives).
DRIVE_MIN_RANGE_M = 4.0

__all__ = [
    "BevObject",
    "CAM_EXTRINSICS",
    "CELL_M",
    "FORWARD_M",
    "GROUND_IDS",
    "LATERAL_M",
    "bev_accuracy",
    "bev_from_labels",
    "colorize_bev",
    "fill_ground_multi",
    "objects_from_grid",
    "paint_radar_range",
    "pixels_to_vehicle",
]


@dataclass(frozen=True)
class BevObject:
    """One connected blob on the bird's-eye grid."""

    class_id: int
    name: str
    x_m: float  # forward of the car
    y_m: float  # right of the car
    length_m: float  # extent along forward
    width_m: float  # extent along left/right
    cells: int


def pixels_to_vehicle(depth: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """HxW axial depth in meters → vehicle-frame x, y, z."""
    height, width = depth.shape
    rows, cols = np.mgrid[0:height, 0:width]
    forward = depth
    right = (cols - (width / 2.0)) / FOCAL_PX * depth
    up = -(rows - (height / 2.0)) / FOCAL_PX * depth
    return CAMERA_X + forward, CAMERA_Y + right, CAMERA_Z + up


def _grid_shape(forward_m: float, lateral_m: float, cell_m: float) -> tuple[int, int]:
    return int(round(forward_m / cell_m)), int(round(2.0 * lateral_m / cell_m))


def _paint_votes(
    grid: np.ndarray,
    row: np.ndarray,
    col: np.ndarray,
    tags: np.ndarray,
    *,
    min_hits: int,
) -> None:
    """Keep a cell when one class has enough hits there. The majority class wins."""
    buckets: dict[tuple[int, int], dict[int, int]] = {}
    for r, c, tag in zip(row.tolist(), col.tolist(), tags.tolist()):
        cell = buckets.setdefault((int(r), int(c)), {})
        cell[int(tag)] = cell.get(int(tag), 0) + 1
    for (r, c), counts in buckets.items():
        tag, hits = max(counts.items(), key=lambda item: item[1])
        if hits >= min_hits:
            grid[r, c] = tag


def _drop_small(grid: np.ndarray, *, min_cells: int) -> None:
    """Erase connected non-ground blobs smaller than min_cells. Road stays."""
    seen = np.zeros(grid.shape, dtype=bool)
    height, width = grid.shape
    for row in range(height):
        for col in range(width):
            tag = int(grid[row, col])
            if tag < 0 or seen[row, col] or tag in GROUND_IDS:
                continue
            stack = [(row, col)]
            seen[row, col] = True
            cells: list[tuple[int, int]] = []
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if (
                        0 <= ny < height
                        and 0 <= nx < width
                        and not seen[ny, nx]
                        and int(grid[ny, nx]) == tag
                    ):
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            if len(cells) < min_cells:
                for y, x in cells:
                    grid[y, x] = -1


def _fill_ground(
    grid: np.ndarray,
    labels: np.ndarray,
    *,
    forward_m: float,
    lateral_m: float,
    cell_m: float,
    camera_x: float = CAMERA_X,
    camera_y: float = CAMERA_Y,
    camera_z: float = CAMERA_Z,
    camera_yaw_deg: float = 0.0,
    ground_ids: tuple[int, ...] = GROUND_IDS,
    min_y_m: float | None = None,
    max_y_m: float | None = None,
    max_forward_m: float | None = None,
    min_image_v_frac: float = 0.0,
) -> None:
    """Paint road-like classes by shooting each cell at the ground plane.

    The segmenter already named the road. This puts that surface on the grid
    in meters so turns and width are readable. Stereo and lidar do not do this.
    ``camera_yaw_deg`` matches CARLA (positive yaw looks right).

    Side-cam knobs (``min_y_m`` / ``max_y_m`` / ``min_image_v_frac``) stop
    yawed views from painting building facades onto the ground as road.
    """
    height, width = labels.shape
    rows, cols = np.mgrid[0 : grid.shape[0], 0 : grid.shape[1]]
    x = forward_m - (rows + 0.5) * cell_m
    y = -lateral_m + (cols + 0.5) * cell_m
    dx = x - camera_x
    dy = y - camera_y
    dz = 0.0 - camera_z
    yaw = np.deg2rad(camera_yaw_deg)
    c = np.cos(yaw)
    s = np.sin(yaw)
    # Camera forward / right in the vehicle XY plane (CARLA yaw convention).
    cam_forward = c * dx + s * dy
    cam_right = -s * dx + c * dy
    cam_up = dz
    visible = cam_forward > 0.4
    if max_forward_m is not None:
        visible &= cam_forward < float(max_forward_m)
    if min_y_m is not None:
        visible &= y >= float(min_y_m)
    if max_y_m is not None:
        visible &= y <= float(max_y_m)
    denom = np.maximum(cam_forward, 1e-3)
    u = np.rint(width / 2.0 + FOCAL_PX * cam_right / denom).astype(np.int32)
    v = np.rint(height / 2.0 - FOCAL_PX * cam_up / denom).astype(np.int32)
    v_min = int(round(float(min_image_v_frac) * height))
    inside = (
        visible
        & (u >= 0)
        & (u < width)
        & (v >= v_min)
        & (v < height)
    )
    sampled = np.zeros(grid.shape, dtype=np.int16)
    sampled[inside] = labels[v[inside], u[inside]]
    ground = inside & np.isin(sampled, ground_ids)
    # Late multi-cam merge: keep an existing ground label if already painted.
    empty = grid < 0
    grid[ground & empty] = sampled[ground & empty]


def fill_ground_multi(
    grid: np.ndarray,
    views: list[tuple[np.ndarray, str]],
    *,
    forward_m: float,
    lateral_m: float,
    cell_m: float,
) -> None:
    """Project several SegNet maps into one BEV (OR merge on empty cells).

    ``views`` is a list of (labels HxW, camera_name). Camera names must exist
    in ``CAM_EXTRINSICS``. Center is usually first so it wins ties. Side cams
    only fill their lateral half, road/road-line only, lower image rows.
    """
    kwargs = {
        "forward_m": forward_m,
        "lateral_m": lateral_m,
        "cell_m": cell_m,
    }
    for labels, cam_name in views:
        if cam_name not in CAM_EXTRINSICS:
            raise KeyError(f"Unknown camera {cam_name!r}")
        cx, cy, cz, yaw = CAM_EXTRINSICS[cam_name]
        side_kw: dict = {}
        if cam_name == "left":
            side_kw = {
                "ground_ids": SIDE_GROUND_IDS,
                "max_y_m": -0.5,
                "max_forward_m": 30.0,
                "min_image_v_frac": 0.50,
            }
        elif cam_name == "right":
            side_kw = {
                "ground_ids": SIDE_GROUND_IDS,
                "min_y_m": 0.5,
                "max_forward_m": 30.0,
                "min_image_v_frac": 0.50,
            }
        _fill_ground(
            grid,
            labels,
            camera_x=cx,
            camera_y=cy,
            camera_z=cz,
            camera_yaw_deg=yaw,
            **kwargs,
            **side_kw,
        )


def paint_radar_range(
    panel: Image.Image,
    range_m: float | None,
    *,
    forward_m: float,
    lateral_m: float,
    cell_m: float,
    grid_shape: tuple[int, int],
) -> None:
    """Late-fusion cue: mark the nearest forward radar hit on the BEV panel."""
    if range_m is None or not np.isfinite(range_m) or range_m <= 0.5:
        return
    if range_m > forward_m:
        return
    scale = panel.height / grid_shape[0]
    row = (forward_m - float(range_m)) / cell_m
    col = (0.0 + lateral_m) / cell_m
    cx = int(col * scale)
    cy = int(row * scale)
    draw = ImageDraw.Draw(panel)
    r = max(3, int(2 * scale))
    draw.ellipse(
        (cx - r, cy - r, cx + r, cy + r),
        outline=(0, 220, 255),
        width=2,
    )
    draw.line((cx, cy, cx, panel.height - 1), fill=(0, 180, 220), width=1)


def _splat_objects(
    grid: np.ndarray,
    labels: np.ndarray,
    depth: np.ndarray,
    *,
    forward_m: float,
    lateral_m: float,
    cell_m: float,
) -> None:
    """Place pixels whose label is not road. Closest point wins.

    Stereo height is a guess, so the segmenter decides what is ground.
    """
    n_forward, n_lateral = grid.shape
    height = labels.shape[0]
    usable = labels.copy()
    usable[height - EGO_HOOD_ROWS :, :] = 0
    x, y, _z = pixels_to_vehicle(depth.astype(np.float32, copy=False))
    ground = np.isin(usable, GROUND_IDS)
    valid = (
        (depth > 0.5)
        & (depth < MAX_DEPTH_M)
        & (usable > 0)
        & (usable != SKY)
        & ~ground
    )
    if not np.any(valid):
        return
    xs = x[valid]
    ys = y[valid]
    tags = usable[valid].astype(np.int16, copy=False)
    col = np.floor((ys + lateral_m) / cell_m).astype(np.int32)
    row = np.floor((forward_m - xs) / cell_m).astype(np.int32)
    inside = (xs >= 0.0) & (xs < forward_m) & (col >= 0) & (col < n_lateral)
    inside &= (row >= 0) & (row < n_forward)
    if not np.any(inside):
        return
    _paint_votes(grid, row[inside], col[inside], tags[inside], min_hits=STEREO_MIN_HITS)
    _drop_small(grid, min_cells=MIN_BLOB_CELLS)


def bev_from_labels(
    labels: np.ndarray,
    depth: np.ndarray,
    *,
    forward_m: float = FORWARD_M,
    lateral_m: float = LATERAL_M,
    cell_m: float = CELL_M,
) -> np.ndarray:
    """Project labels into a top-down grid: road from the camera, objects on top.

    Row 0 is the far edge. Column 0 is to the car's left. Empty cells are -1.
    Road and sidewalk are filled from the segmenter on the ground plane.
    Everything else is placed with measured depth and needs several hits.
    """
    if labels.shape != depth.shape:
        raise ValueError(
            f"labels {labels.shape} and depth {depth.shape} do not match"
        )
    n_forward, n_lateral = _grid_shape(forward_m, lateral_m, cell_m)
    grid = np.full((n_forward, n_lateral), -1, dtype=np.int16)
    kwargs = {"forward_m": forward_m, "lateral_m": lateral_m, "cell_m": cell_m}
    _fill_ground(grid, labels, **kwargs)
    _splat_objects(grid, labels, depth, **kwargs)
    return grid


def bev_from_lidar(
    points: np.ndarray,
    *,
    labels: np.ndarray | None = None,
    forward_m: float = FORWARD_M,
    lateral_m: float = LATERAL_M,
    cell_m: float = CELL_M,
) -> np.ndarray:
    """Lidar objects on a camera road. A cell needs several hits.

    ``points`` is Nx4 (x, y, z, tag) in the vehicle frame, z up. Hits on the
    ego body and anything below road level are dropped. If ``labels`` is set,
    the road and sidewalk come from the camera, not from the lidar rings.
    """
    from perception.lidar import LIDAR_EGO_RADIUS_M, LIDAR_X, LIDAR_Y

    n_forward, n_lateral = _grid_shape(forward_m, lateral_m, cell_m)
    grid = np.full((n_forward, n_lateral), -1, dtype=np.int16)
    kwargs = {"forward_m": forward_m, "lateral_m": lateral_m, "cell_m": cell_m}
    if labels is not None:
        _fill_ground(grid, labels, **kwargs)
    if len(points) == 0:
        return grid
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    tag = points[:, 3].astype(np.int32)
    dist = np.hypot(x - LIDAR_X, y - LIDAR_Y)
    valid = (
        (dist > LIDAR_EGO_RADIUS_M)
        & (tag > 0)
        & (tag != SKY)
        & (tag < len(PALETTE))
        & (z >= MIN_HEIGHT_M)
    )
    if not np.any(valid):
        return grid
    xs = x[valid]
    ys = y[valid]
    tags = tag[valid].astype(np.int16)
    col = np.floor((ys + lateral_m) / cell_m).astype(np.int32)
    row = np.floor((forward_m - xs) / cell_m).astype(np.int32)
    inside = (xs >= 0.0) & (xs < forward_m) & (col >= 0) & (col < n_lateral)
    inside &= (row >= 0) & (row < n_forward)
    if not np.any(inside):
        return grid
    _paint_votes(grid, row[inside], col[inside], tags[inside], min_hits=LIDAR_MIN_HITS)
    _drop_small(grid, min_cells=MIN_BLOB_CELLS)
    return grid


def colorize_bev(grid: np.ndarray) -> np.ndarray:
    """Grid of tag ids (-1 empty) → RGB. Empty cells are dark gray."""
    rgb = np.full(grid.shape + (3,), 24, dtype=np.uint8)
    filled = grid >= 0
    rgb[filled] = PALETTE[grid[filled]]
    return rgb


def bev_accuracy(
    actual: np.ndarray, predicted: np.ndarray
) -> tuple[float, int, float, int, float, int]:
    """Cell match on the whole grid, on objects, then on vehicles and people."""
    occupied = actual >= 0
    objects = occupied & ~np.isin(actual, GROUND_IDS)
    vehicles = occupied & np.isin(actual, VEHICLE_IDS)

    def _rate(mask: np.ndarray) -> tuple[float, int]:
        count = int(mask.sum())
        if count == 0:
            return 0.0, 0
        hit = int(((actual == predicted) & mask).sum())
        return hit / count, count

    all_rate, all_count = _rate(occupied)
    obj_rate, obj_count = _rate(objects)
    veh_rate, veh_count = _rate(vehicles)
    return all_rate, all_count, obj_rate, obj_count, veh_rate, veh_count


def _match_image(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    rgb = np.full(actual.shape + (3,), 24, dtype=np.uint8)
    agree = (actual >= 0) & (actual == predicted)
    miss = (actual >= 0) & (actual != predicted)
    extra = (actual < 0) & (predicted >= 0)
    rgb[agree] = (40, 160, 60)
    rgb[miss] = (210, 40, 40)
    rgb[extra] = (40, 90, 210)
    return rgb


def _load_rgb(path: Path) -> np.ndarray:
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    if rgb.shape[0] != IMAGE_HEIGHT or rgb.shape[1] != IMAGE_WIDTH:
        raise ValueError(
            f"{path} is {rgb.shape[1]}x{rgb.shape[0]}, "
            f"expected {IMAGE_WIDTH}x{IMAGE_HEIGHT}"
        )
    return rgb


def _panel(rgb: np.ndarray, height: int) -> Image.Image:
    image = Image.fromarray(rgb)
    scale = height / image.height
    width = max(1, int(round(image.width * scale)))
    return image.resize((width, height), Image.Resampling.NEAREST)


_SHORT_NAME = {
    "Building": "bldg",
    "Vegetation": "tree",
    "TrafficLight": "light",
    "TrafficSign": "sign",
    "Pedestrian": "person",
    "GuardRail": "rail",
    "Motorcycle": "moto",
}


def keep_drive_objects(grid: np.ndarray) -> np.ndarray:
    """Clear non-vehicle object cells; keep road and empty. Copy returned."""
    out = grid.copy()
    clutter = (
        (out >= 0)
        & ~np.isin(out, GROUND_IDS)
        & ~np.isin(out, DRIVE_OBJECT_IDS)
    )
    out[clutter] = -1
    return out


def objects_from_grid(
    grid: np.ndarray,
    *,
    forward_m: float = FORWARD_M,
    lateral_m: float = LATERAL_M,
    cell_m: float = CELL_M,
    min_cells: int = MIN_BLOB_CELLS,
    class_ids: tuple[int, ...] = OBJECT_IDS,
) -> list[BevObject]:
    """Turn touching same-class cells into one object with a center and size.

    Flood-fill each connected blob. The center is the mean cell, converted to
    meters. The size is the axis-aligned box around those cells.
    """
    wanted = set(class_ids)
    seen = np.zeros(grid.shape, dtype=bool)
    height, width = grid.shape
    found: list[BevObject] = []
    for row in range(height):
        for col in range(width):
            tag = int(grid[row, col])
            if tag not in wanted or seen[row, col]:
                continue
            stack = [(row, col)]
            seen[row, col] = True
            cells: list[tuple[int, int]] = []
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if (
                        0 <= ny < height
                        and 0 <= nx < width
                        and not seen[ny, nx]
                        and int(grid[ny, nx]) == tag
                    ):
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            if len(cells) < min_cells:
                continue
            rows = [y for y, _x in cells]
            cols = [x for _y, x in cells]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            # Cell centers → vehicle frame meters.
            x_m = forward_m - (sum(rows) / len(rows) + 0.5) * cell_m
            y_m = -lateral_m + (sum(cols) / len(cols) + 0.5) * cell_m
            length_m = (r1 - r0 + 1) * cell_m
            width_m = (c1 - c0 + 1) * cell_m
            name = CLASS_NAMES[tag] if tag < len(CLASS_NAMES) else str(tag)
            found.append(
                BevObject(
                    class_id=tag,
                    name=_SHORT_NAME.get(name, name.lower()),
                    x_m=float(x_m),
                    y_m=float(y_m),
                    length_m=float(length_m),
                    width_m=float(width_m),
                    cells=len(cells),
                )
            )
    found.sort(key=lambda obj: obj.x_m)
    return found


def _object_label(obj: BevObject) -> str:
    """Class + range; tracked objects also show id and speed."""
    track_id = getattr(obj, "track_id", None)
    if track_id is None:
        return f"{obj.name} {obj.x_m:.0f}m"
    speed = (getattr(obj, "vx_mps", 0.0) ** 2 + getattr(obj, "vy_mps", 0.0) ** 2) ** 0.5
    return f"{obj.name}#{track_id} {obj.x_m:.0f}m {speed:.1f}m/s"


def _draw_objects(
    panel: Image.Image,
    objects: list,
    *,
    forward_m: float,
    lateral_m: float,
    cell_m: float,
    grid_shape: tuple[int, int],
) -> None:
    """Draw one box per object. The label is class and distance ahead."""
    scale = panel.height / grid_shape[0]
    draw = ImageDraw.Draw(panel)
    font = ImageFont.load_default(size=16)
    for obj in objects:
        # Box corners in cell space, then scale to the panel.
        r0 = (forward_m - (obj.x_m + 0.5 * obj.length_m)) / cell_m
        r1 = (forward_m - (obj.x_m - 0.5 * obj.length_m)) / cell_m
        c0 = (obj.y_m - 0.5 * obj.width_m + lateral_m) / cell_m
        c1 = (obj.y_m + 0.5 * obj.width_m + lateral_m) / cell_m
        x0 = int(min(c0, c1) * scale)
        y0 = int(min(r0, r1) * scale)
        x1 = int(max(c0, c1) * scale)
        y1 = int(max(r0, r1) * scale)
        draw.rectangle((x0, y0, x1, y1), outline=(255, 255, 255), width=2)
        label = _object_label(obj)
        draw.text((x0 + 1, max(0, y0 - 16) + 1), label, fill=(0, 0, 0), font=font)
        draw.text((x0, max(0, y0 - 16)), label, fill=(255, 255, 255), font=font)


def _mark_ego(grid_rgb: np.ndarray, *, forward_m: float, lateral_m: float, cell_m: float) -> None:
    n_forward, n_lateral, _ = grid_rgb.shape
    row = int(np.floor((forward_m - 0.0) / cell_m)) - 1
    col = int(np.floor((0.0 + lateral_m) / cell_m))
    row = min(max(row, 0), n_forward - 1)
    col = min(max(col, 1), n_lateral - 2)
    grid_rgb[row - 2 : row + 1, col - 1 : col + 2] = (255, 255, 255)


def _predict_seg_labels(seg_model, center_rgb: np.ndarray, device: str) -> np.ndarray:
    """HxWx3 float RGB → HxW tag ids from SegNet."""
    import torch

    tensor = (
        torch.from_numpy(center_rgb)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device)
    )
    seg_model.eval()
    with torch.no_grad():
        logits = seg_model(tensor)
    return logits.argmax(dim=1)[0].detach().cpu().numpy().astype(np.int16)


def _load_radar_m(run_dir: Path, frame: int) -> float | None:
    """Optional late cue from controls.csv ``radar_m`` (center row)."""
    import csv

    path = run_dir / "controls.csv"
    if not path.exists():
        return None
    key = str(frame)
    with path.open() as fh:
        for row in csv.DictReader(fh):
            if row.get("frame") == key and row.get("camera", "center") == "center":
                text = (row.get("radar_m") or "").strip()
                if not text:
                    return None
                try:
                    return float(text)
                except ValueError:
                    return None
    return None


def _seg_views_for_frame(
    run_dir: Path,
    frame: int,
    seg_model,
    device: str,
    *,
    multicam: bool,
) -> tuple[np.ndarray, list[tuple[np.ndarray, str]] | None]:
    """Center labels always; optional left/right views when images exist."""
    name = f"{frame:06d}"
    center = _load_rgb(run_dir / "images" / "center" / f"{name}.png")
    pred_center = _predict_seg_labels(seg_model, center, device)
    if not multicam:
        return pred_center, None
    views: list[tuple[np.ndarray, str]] = [(pred_center, "center")]
    for cam in ("left", "right"):
        path = run_dir / "images" / cam / f"{name}.png"
        if not path.exists():
            continue
        rgb = _load_rgb(path)
        views.append((_predict_seg_labels(seg_model, rgb, device), cam))
    if len(views) == 1:
        return pred_center, None
    return pred_center, views


def render_strip(
    run_dir: Path,
    frame: int,
    *,
    forward_m: float = FORWARD_M,
    lateral_m: float = LATERAL_M,
    cell_m: float = CELL_M,
    mode: str = "full",
    tracker=None,
    occupancy_model=None,
    occupancy_device: str = "cpu",
    seg_model=None,
    multicam: bool = False,
    show_radar: bool = False,
) -> tuple[Image.Image, dict]:
    """Build one comparison strip.

    ``watch`` — photo | lidar/occ | match-to-CARLA.
    ``full`` — five panels including CARLA tags and answer key.
    ``live`` — photo | SegNet tags | OccupancyNet BEV. No CARLA GT.
    ``compare`` — photo | CARLA BEV | live BEV | match (green=agree).
    """
    from perception.lidar import load_lidar_points

    name = f"{frame:06d}"
    center = _load_rgb(run_dir / "images" / "center" / f"{name}.png")
    lidar_path = run_dir / "lidar" / f"{name}.npy"
    if not lidar_path.exists():
        raise FileNotFoundError(f"No lidar frame at {lidar_path}")

    kwargs = {"forward_m": forward_m, "lateral_m": lateral_m, "cell_m": cell_m}
    points = load_lidar_points(lidar_path)
    radar_m = _load_radar_m(run_dir, frame) if show_radar else None

    if mode in ("live", "compare"):
        if seg_model is None or occupancy_model is None:
            raise ValueError(f"{mode} mode needs --seg and --occupancy models")
        from perception.occupancy import bev_from_occupancy

        pred_labels, label_views = _seg_views_for_frame(
            run_dir,
            frame,
            seg_model,
            occupancy_device,
            multicam=multicam,
        )
        live_grid = bev_from_occupancy(
            points,
            occupancy_model,
            labels=None if label_views else pred_labels,
            label_views=label_views,
            device=occupancy_device,
            occ_threshold=OCC_THRESHOLD,
            min_blob_cells=OCC_MIN_BLOB_CELLS,
            **kwargs,
        )
        # Driving output: vehicles/people only — drop pole/sign spam.
        live_grid = keep_drive_objects(live_grid)
        live_rgb = colorize_bev(live_grid)
        _mark_ego(live_rgb, **kwargs)
        height = live_rgb.shape[0] * 4
        live_objects = [
            obj
            for obj in objects_from_grid(
                live_grid, class_ids=DRIVE_OBJECT_IDS, **kwargs
            )
            if obj.x_m >= DRIVE_MIN_RANGE_M
        ]
        tracked = tracker.update(live_objects) if tracker is not None else None
        draw_objs = tracked if tracked is not None else live_objects
        live_panel = _panel(live_rgb, height)
        draw_kw = {**kwargs, "grid_shape": live_grid.shape}
        _draw_objects(live_panel, draw_objs, **draw_kw)
        if show_radar:
            paint_radar_range(live_panel, radar_m, **draw_kw)
        photo = _panel((center * 255.0).astype(np.uint8), height)
        cam_note = (
            f"seg×{len(label_views)}" if label_views else "seg (trained)"
        )

        if mode == "live":
            tags = _panel(colorize(pred_labels), height)
            panels = [photo, tags, live_panel]
            titles = [
                f"photo  {frame:06d}",
                cam_note,
                f"bev ({len(draw_objs)} obj)",
            ]
            result = {
                "actual_objects": [],
                "lidar_objects": live_objects,
                "tracked_objects": tracked,
                "pred_labels": pred_labels,
            }
        else:
            true_labels = np.asarray(
                Image.open(run_dir / "seg" / "center" / f"{name}.png")
            )
            depth_path = run_dir / "depth" / "center" / f"{name}.npy"
            if not depth_path.exists():
                raise FileNotFoundError(
                    f"compare mode needs depth GT at {depth_path}"
                )
            true_depth = np.load(depth_path).astype(np.float32)
            actual = keep_drive_objects(
                bev_from_labels(true_labels, true_depth, **kwargs)
            )
            (
                accuracy,
                occupied,
                obj_rate,
                obj_count,
                veh_rate,
                veh_count,
            ) = bev_accuracy(actual, live_grid)
            actual_rgb = colorize_bev(actual)
            _mark_ego(actual_rgb, **kwargs)
            actual_objects = [
                obj
                for obj in objects_from_grid(
                    actual, class_ids=DRIVE_OBJECT_IDS, **kwargs
                )
                if obj.x_m >= DRIVE_MIN_RANGE_M
            ]
            actual_panel = _panel(actual_rgb, height)
            _draw_objects(actual_panel, actual_objects, **draw_kw)
            match = _panel(_match_image(actual, live_grid), height)
            panels = [photo, actual_panel, live_panel, match]
            titles = [
                f"photo  {frame:06d}",
                f"truth ({len(actual_objects)} obj)",
                f"live ({len(draw_objs)} obj)",
                f"match {100 * accuracy:.0f}%  obj {100 * obj_rate:.0f}%",
            ]
            result = {
                "accuracy": accuracy,
                "occupied": occupied,
                "lidar_obj": obj_rate,
                "obj_count": obj_count,
                "lidar_veh": veh_rate,
                "veh_count": veh_count,
                "actual_objects": actual_objects,
                "lidar_objects": live_objects,
                "tracked_objects": tracked,
                "pred_labels": pred_labels,
            }

        strip = Image.new(
            "RGB", (sum(panel.width for panel in panels), height + 28), (16, 16, 16)
        )
        draw = ImageDraw.Draw(strip)
        x = 0
        for panel, title in zip(panels, titles):
            strip.paste(panel, (x, 28))
            draw.text((x + 6, 6), title, fill=(230, 230, 230))
            x += panel.width
        return strip, result

    true_labels = np.asarray(Image.open(run_dir / "seg" / "center" / f"{name}.png"))
    depth_path = run_dir / "depth" / "center" / f"{name}.npy"
    if depth_path.exists():
        true_depth = np.load(depth_path).astype(np.float32)
        actual = bev_from_labels(true_labels, true_depth, **kwargs)
    else:
        actual = bev_from_lidar(points, labels=true_labels, **kwargs)

    if occupancy_model is not None:
        from perception.occupancy import bev_from_occupancy

        lidar_grid = bev_from_occupancy(
            points,
            occupancy_model,
            labels=true_labels,
            device=occupancy_device,
            **kwargs,
        )
        pred_name = "occ"
    else:
        lidar_grid = bev_from_lidar(points, labels=true_labels, **kwargs)
        pred_name = "lidar"
    (
        accuracy,
        occupied,
        lidar_obj,
        obj_count,
        lidar_veh,
        veh_count,
    ) = bev_accuracy(actual, lidar_grid)

    actual_rgb = colorize_bev(actual)
    lidar_rgb = colorize_bev(lidar_grid)
    _mark_ego(actual_rgb, **kwargs)
    _mark_ego(lidar_rgb, **kwargs)
    height = actual_rgb.shape[0] * 4
    actual_objects = objects_from_grid(actual, **kwargs)
    lidar_objects = objects_from_grid(lidar_grid, **kwargs)
    tracked = tracker.update(lidar_objects) if tracker is not None else None
    draw_lidar = tracked if tracked is not None else lidar_objects
    actual_panel = _panel(actual_rgb, height)
    lidar_panel = _panel(lidar_rgb, height)
    draw_kw = {**kwargs, "grid_shape": actual.shape}
    _draw_objects(actual_panel, actual_objects, **draw_kw)
    _draw_objects(lidar_panel, draw_lidar, **draw_kw)
    photo = _panel((center * 255.0).astype(np.uint8), height)
    match = _panel(_match_image(actual, lidar_grid), height)
    n_lidar = len(draw_lidar)
    if mode == "watch":
        panels = [photo, lidar_panel, match]
        titles = [
            f"photo  {frame:06d}",
            f"{pred_name} ({n_lidar} obj)",
            f"match {100 * accuracy:.0f}%  obj {100 * lidar_obj:.0f}%",
        ]
    else:
        panels = [
            photo,
            _panel(colorize(true_labels), height),
            actual_panel,
            lidar_panel,
            match,
        ]
        titles = [
            "photo",
            "tags",
            f"actual ({len(actual_objects)} obj)",
            f"{pred_name} ({n_lidar} obj)",
            f"match {100 * accuracy:.0f}%  obj {100 * lidar_obj:.0f}%",
        ]
    strip = Image.new(
        "RGB", (sum(panel.width for panel in panels), height + 28), (16, 16, 16)
    )
    draw = ImageDraw.Draw(strip)
    x = 0
    for panel, title in zip(panels, titles):
        strip.paste(panel, (x, 28))
        draw.text((x + 6, 6), title, fill=(230, 230, 230))
        x += panel.width
    return strip, {
        "accuracy": accuracy,
        "occupied": occupied,
        "lidar_obj": lidar_obj,
        "obj_count": obj_count,
        "lidar_veh": lidar_veh,
        "veh_count": veh_count,
        "actual_objects": actual_objects,
        "lidar_objects": lidar_objects,
        "tracked_objects": tracked,
    }


def write_video(
    run_dir: Path,
    out_path: Path,
    *,
    fps: float | None = None,
    stride: int = 1,
    max_frames: int | None = None,
    mode: str = "watch",
    occupancy_model=None,
    occupancy_device: str = "cpu",
    seg_model=None,
    multicam: bool = False,
    show_radar: bool = False,
) -> Path:
    """Write a BEV strip video for the run."""
    import imageio.v2 as imageio

    from perception.track import ObjectTracker

    meta_path = run_dir / "meta.json"
    if fps is None and meta_path.exists():
        import json

        fps = float(json.loads(meta_path.read_text()).get("fps", 10.0))
    if fps is None:
        fps = 10.0

    names = sorted((run_dir / "lidar").glob("*.npy"))
    if not names:
        raise FileNotFoundError(f"No lidar frames in {run_dir}")
    stride = max(1, int(stride))
    frames = [int(p.stem) for p in names[::stride]]
    if max_frames is not None:
        frames = frames[: max(0, int(max_frames))]

    tracker = ObjectTracker(dt_s=stride / fps, min_confirm=3)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        out_path,
        fps=fps / stride,
        codec="libx264",
        quality=8,
        pixelformat="yuv420p",
        macro_block_size=1,
    )
    try:
        for i, frame in enumerate(frames):
            strip, _ = render_strip(
                run_dir,
                frame,
                mode=mode,
                tracker=tracker,
                occupancy_model=occupancy_model,
                occupancy_device=occupancy_device,
                seg_model=seg_model,
                multicam=multicam,
                show_radar=show_radar,
            )
            # H.264 wants even dimensions.
            width, height = strip.size
            if width % 2 or height % 2:
                strip = strip.resize(
                    (width + width % 2, height + height % 2),
                    Image.Resampling.NEAREST,
                )
            writer.append_data(np.asarray(strip))
            if (i + 1) % 50 == 0 or i + 1 == len(frames):
                print(f"video {i + 1}/{len(frames)}  frame {frame:06d}")
    finally:
        writer.close()
    print(f"Wrote {out_path}  ({len(frames)} frames @ {fps / stride:.1f} fps)")
    return out_path


def _parse_frames(value: str, run_dir: Path) -> list[int]:
    if value.strip():
        return [int(part) for part in value.split(",") if part.strip()]
    names = sorted((run_dir / "lidar").glob("*.npy"))
    if not names:
        raise FileNotFoundError(f"No lidar frames in {run_dir}")
    picks = [
        0,
        len(names) // 5,
        2 * len(names) // 5,
        3 * len(names) // 5,
        4 * len(names) // 5,
    ]
    return [int(names[i].stem) for i in picks]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="One run directory")
    parser.add_argument(
        "--frames",
        type=str,
        default="",
        help="Comma-separated frame numbers. Default: five spread through the run.",
    )
    parser.add_argument("--out", type=Path, default=Path("models/bev_preview"))
    parser.add_argument(
        "--video",
        action="store_true",
        help="Write an mp4 of photo | lidar | match for the whole run",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Keep every Nth frame in the video (default: 1)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Cap how many video frames to write (for a quick preview)",
    )
    parser.add_argument(
        "--mode",
        choices=["watch", "full", "live", "compare"],
        default="watch",
        help="watch=photo|lidar|match. full=five panels. "
        "live=photo|SegNet|OccupancyNet (no CARLA GT). "
        "compare=photo|truth BEV|live BEV|match.",
    )
    parser.add_argument(
        "--occupancy",
        type=Path,
        default=None,
        help="OccupancyNet checkpoint; replaces hand-voted lidar objects",
    )
    parser.add_argument(
        "--seg",
        type=Path,
        default=None,
        help="SegNet checkpoint (required for --mode live)",
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu", "auto"],
        default="auto",
    )
    parser.add_argument(
        "--multicam",
        action="store_true",
        help=(
            "Late-fuse left/center/right SegNet into BEV road "
            "(needs images/left and images/right in the run)"
        ),
    )
    parser.add_argument(
        "--radar",
        action="store_true",
        help="Overlay controls.csv radar_m as a cyan range cue on the live BEV",
    )
    args = parser.parse_args()

    from utils.device import resolve_device

    occupancy_model = None
    seg_model = None
    occupancy_device = str(resolve_device(args.device))
    if args.mode in ("live", "compare"):
        if args.seg is None:
            args.seg = Path("models/seg.pt")
        if args.occupancy is None:
            args.occupancy = Path("models/occupancy.pt")
    if args.occupancy is not None:
        from perception.occupancy import load_occupancy_model

        occupancy_model = load_occupancy_model(args.occupancy, device=occupancy_device)
        print(f"Occupancy model: {args.occupancy} on {occupancy_device}")
    if args.seg is not None:
        from segmentation.model import load_seg_model

        seg_model = load_seg_model(args.seg, device=occupancy_device)
        print(f"Seg model: {args.seg} on {occupancy_device}")

    if args.video:
        out = args.out
        if out.is_dir() or out.suffix == "":
            if args.mode == "live":
                suffix = "_live.mp4"
            elif args.mode == "compare":
                suffix = "_compare.mp4"
            elif occupancy_model is not None:
                suffix = "_occ.mp4"
            else:
                suffix = "_bev.mp4"
            out = Path(out) / f"{args.data.name}{suffix}"
        write_video(
            args.data,
            out,
            stride=max(1, args.stride),
            max_frames=args.max_frames,
            mode=args.mode,
            occupancy_model=occupancy_model,
            occupancy_device=occupancy_device,
            seg_model=seg_model,
            multicam=args.multicam,
            show_radar=args.radar,
        )
        return

    frames = _parse_frames(args.frames, args.data)
    print(
        f"Grid: {FORWARD_M:.0f} m forward, ±{LATERAL_M:.0f} m, {CELL_M:.2f} m cells. "
        "White mark is the car."
    )
    for frame in frames:
        out_path = args.out / f"bev_{frame:06d}.png"
        strip, result = render_strip(
            args.data,
            frame,
            mode=args.mode if args.mode in ("live", "compare") else "full",
            occupancy_model=occupancy_model,
            occupancy_device=occupancy_device,
            seg_model=seg_model,
            multicam=args.multicam,
            show_radar=args.radar,
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        strip.save(out_path)
        print(
            f"frame {frame:06d}  cells={result['occupied']}  "
            f"match={100 * result['accuracy']:.1f}%  "
            f"lidar_obj={100 * result['lidar_obj']:.1f}%  "
            f"lidar_vehicles={100 * result['lidar_veh']:.1f}% "
            f"({result['veh_count']})  {out_path}"
        )
        for source in ("actual", "lidar"):
            objs: list[BevObject] = result[f"{source}_objects"]
            parts = [
                f"{obj.name}@{obj.x_m:.0f}m"
                for obj in objs
                if obj.class_id in VEHICLE_IDS
            ]
            if parts:
                print(f"  {source}: {', '.join(parts)}")


if __name__ == "__main__":
    main()
