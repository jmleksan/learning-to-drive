"""Labels and dataset for TransFuser-lite.

Reads runs recorded with
``expert.record --cameras center,left,right --semantic --lidar --gnss``.

Per frame:
  inputs  RGB strip (left|center|right, 3×120×480), lidar BEV (4×128×128: two
          height bins plus one goal channel per target), speed (m/s), and
          two target points (ego x fwd / y right, m)
  labels  4 future waypoints at 0.5 s (ego frame, m)
          aux: image semantics (left|center|right), BEV semantics from lidar
          tags, red light affecting ego, expert braking

Waypoints and the target point come from the logged ground-truth pose
(``localization.csv``). The target points are TARGET_M and TARGET_FAR_M ahead
along the path the expert actually drove, which is what ``lookahead_target``
gives on the planned route at drive time.

The goal channels are not sensor data. They are drawn from the target points,
as TransFuser rasterizes its goal into the lidar BEV. The far target gives
earlier notice of a turn.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from perception.lidar import LIDAR_EGO_RADIUS_M, load_lidar_points

CAMERAS = ("left", "center", "right")

NUM_WAYPOINTS = 4
WAYPOINT_DT_S = 0.5
TARGET_M = 25.0
TARGET_FAR_M = 50.0
NUM_TARGETS = 2
# A pose jump bigger than this between two ticks is a respawn, not driving.
MAX_STEP_M = 3.0
# Long stops (traffic deadlocks) are thousands of identical frames. Keep the
# start of each stop (a normal light/queue wait) and the end (pulling away).
STOP_SPEED_MPS = 0.1
STOP_KEEP_HEAD_S = 10.0
STOP_KEEP_TAIL_S = 4.0

# Lidar BEV grid: 32 m ahead, ±16 m sideways, 0.25 m cells. Row 0 is far.
BEV_FORWARD_M = 32.0
BEV_LATERAL_M = 16.0
BEV_CELL_M = 0.25
BEV_SIZE = int(round(BEV_FORWARD_M / BEV_CELL_M))  # 128 (square grid)
GROUND_SPLIT_M = 0.30
MAX_HEIGHT_M = 4.0
HIST_MAX_HITS = 5.0
# Aux BEV labels at half resolution (0.5 m cells, 64×64).
BEV_LABEL_STRIDE = 2
LIDAR_CHANNELS = 2 + NUM_TARGETS
GOAL_RADIUS_CELLS = 3
# Near target this far sideways: a turn is coming, sample the frame more often.
JUNCTION_LATERAL_M = 4.0

IGNORE = 255

# Image semantics, reduced from CARLA tags (segmentation/labels.py).
IMG_CLASSES = ("other", "road", "roadline", "sidewalk", "vehicle", "pedestrian", "light_sign")
_IMG_LUT = np.zeros(256, dtype=np.uint8)
_IMG_LUT[1] = 1
_IMG_LUT[24] = 2
_IMG_LUT[2] = 3
_IMG_LUT[[14, 15, 16, 18, 19]] = 4
_IMG_LUT[[12, 13]] = 5
_IMG_LUT[[7, 8]] = 6

# BEV semantics from lidar tags. Cells with no hit are IGNORE. Ids are in
# priority order: when tags share a cell the higher id wins (a car on the
# road is a car).
BEV_CLASSES = ("road", "other", "vehicle", "pedestrian")
_BEV_LUT = np.full(256, 1, dtype=np.uint8)
_BEV_LUT[[1, 24, 25]] = 0
_BEV_LUT[[14, 15, 16, 18, 19]] = 2
_BEV_LUT[[12, 13]] = 3

IMAGE_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGE_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

__all__ = [
    "BEV_CLASSES",
    "IMG_CLASSES",
    "LIDAR_CHANNELS",
    "NUM_TARGETS",
    "NUM_WAYPOINTS",
    "TARGET_FAR_M",
    "TARGET_M",
    "TransfuserDataset",
    "bev_labels_from_lidar",
    "goal_channel",
    "image_strip",
    "lidar_histogram",
    "lidar_input",
]


def _bev_cells(points: np.ndarray, stride: int = 1):
    """Points → (row, col, mask) on the BEV grid, ego body dropped."""
    cell = BEV_CELL_M * stride
    size = BEV_SIZE // stride
    x, y = points[:, 0], points[:, 1]
    keep = (
        (np.hypot(x, y) > LIDAR_EGO_RADIUS_M)
        & (x >= 0.0)
        & (x < BEV_FORWARD_M)
        & (y >= -BEV_LATERAL_M)
        & (y < BEV_LATERAL_M)
        & (points[:, 2] < MAX_HEIGHT_M)
    )
    row = np.floor((BEV_FORWARD_M - x) / cell).astype(np.int64)
    col = np.floor((y + BEV_LATERAL_M) / cell).astype(np.int64)
    keep &= (row >= 0) & (row < size) & (col >= 0) & (col < size)
    return row, col, keep, size


def lidar_histogram(points: np.ndarray) -> np.ndarray:
    """Nx4 lidar (tags ignored) → float32 [2, 128, 128]: ground hits, above-ground hits."""
    out = np.zeros((2, BEV_SIZE, BEV_SIZE), dtype=np.float32)
    if len(points) == 0:
        return out
    row, col, keep, size = _bev_cells(points)
    above = (points[:, 2] >= GROUND_SPLIT_M).astype(np.int64)
    flat = (above * size + row) * size + col
    counts = np.bincount(flat[keep], minlength=2 * size * size)
    out[:] = counts.reshape(2, size, size)
    return np.minimum(out, HIST_MAX_HITS) / HIST_MAX_HITS


def goal_channel(target: np.ndarray) -> np.ndarray:
    """Ego (x fwd, y right) target → float32 [128, 128] with a blob at it.

    A target outside the grid is clamped to the nearest edge, which still
    shows its direction.
    """
    out = np.zeros((BEV_SIZE, BEV_SIZE), dtype=np.float32)
    row = int(np.clip((BEV_FORWARD_M - target[0]) / BEV_CELL_M, 0, BEV_SIZE - 1))
    col = int(np.clip((target[1] + BEV_LATERAL_M) / BEV_CELL_M, 0, BEV_SIZE - 1))
    r = GOAL_RADIUS_CELLS
    out[max(0, row - r) : row + r + 1, max(0, col - r) : col + r + 1] = 1.0
    return out


def lidar_input(points: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Nx4 lidar + [near x, near y, far x, far y] → float32 [4, 128, 128]."""
    return np.concatenate(
        [lidar_histogram(points), goal_channel(targets[:2])[None], goal_channel(targets[2:])[None]]
    )


def bev_labels_from_lidar(points: np.ndarray) -> np.ndarray:
    """Nx4 lidar with tags → uint8 [64, 64] BEV class per cell (IGNORE = no hit)."""
    size = BEV_SIZE // BEV_LABEL_STRIDE
    if len(points) == 0:
        return np.full((size, size), IGNORE, dtype=np.uint8)
    row, col, keep, size = _bev_cells(points, BEV_LABEL_STRIDE)
    cls = _BEV_LUT[points[:, 3].astype(np.int64).clip(0, 255)]
    best = np.full(size * size, -1, dtype=np.int16)
    np.maximum.at(best, row[keep] * size + col[keep], cls[keep].astype(np.int16))
    return np.where(best >= 0, best, IGNORE).astype(np.uint8).reshape(size, size)


def image_strip(images: list[np.ndarray]) -> np.ndarray:
    """HxWx3 uint8 per camera (left, center, right) → normalized float32 [3, H, 3W]."""
    strip = np.concatenate(images, axis=1).astype(np.float32) / 255.0
    strip = (strip - IMAGE_MEAN) / IMAGE_STD
    return np.ascontiguousarray(strip.transpose(2, 0, 1))


def to_ego(
    e: np.ndarray, n: np.ndarray, e0: float, n0: float, yaw_enu: float
) -> np.ndarray:
    """ENU points → ego frame (x forward, y right). ENU yaw: 0 = north, + east.

    On Town01 this repo's ENU is e = +x, n = +y of CARLA's left-handed world, a
    mirror of true east/north. So ego right is (-cos yaw, +sin yaw) in (e, n).
    """
    de = e - e0
    dn = n - n0
    s, c = np.sin(yaw_enu), np.cos(yaw_enu)
    return np.stack([de * s + dn * c, dn * s - de * c], axis=-1)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _stop_mask(stopped: np.ndarray, fps: float) -> np.ndarray:
    """True for frames to keep: all moving frames, plus the head and tail of each stop."""
    keep = np.ones(len(stopped), dtype=bool)
    head = int(STOP_KEEP_HEAD_S * fps)
    tail = int(STOP_KEEP_TAIL_S * fps)
    edges = np.flatnonzero(np.diff(np.concatenate([[0], stopped.astype(np.int8), [0]])))
    for start, end in zip(edges[::2], edges[1::2]):
        if end - start > head + tail:
            keep[start + head : end - tail] = False
    return keep


class _Run:
    """One recorded run with per-frame labels precomputed from the pose log."""

    def __init__(
        self, run_dir: Path, *, fps: float, target_m: float, target_far_m: float, stride: int
    ) -> None:
        self.dir = Path(run_dir)
        for sub in ("lidar", *(f"images/{c}" for c in CAMERAS)):
            if not (self.dir / sub).is_dir():
                raise FileNotFoundError(
                    f"{self.dir} has no {sub}/. Record with "
                    "--cameras center,left,right --semantic --lidar --gnss"
                )
        self.has_seg = all((self.dir / "seg" / c).is_dir() for c in CAMERAS)

        loc_rows = _read_csv(self.dir / "localization.csv")
        frames = np.array([int(r["frame"]) for r in loc_rows])
        e = np.array([float(r["gt_e_m"]) for r in loc_rows])
        n = np.array([float(r["gt_n_m"]) for r in loc_rows])
        yaw = np.array([float(r["gt_yaw_rad"]) for r in loc_rows])
        speed = np.array([float(r["gt_speed_mps"]) for r in loc_rows])

        ctrl = {
            int(r["frame"]): r
            for r in _read_csv(self.dir / "controls.csv")
            if r.get("camera", "center") == "center"
        }

        step = np.hypot(np.diff(e), np.diff(n))
        # Path length from frame 0; a respawn jump breaks the path.
        arc = np.concatenate([[0.0], np.cumsum(np.minimum(step, MAX_STEP_M))])
        jump = np.concatenate([[False], step > MAX_STEP_M])
        jumps_before = np.cumsum(jump)

        horizon = int(round(WAYPOINT_DT_S * fps))
        offsets = horizon * np.arange(1, NUM_WAYPOINTS + 1)
        keep = _stop_mask(speed < STOP_SPEED_MPS, fps)

        self.samples: list[dict] = []
        for i in range(0, len(frames), stride):
            if not keep[i]:
                continue
            j_wp = i + offsets
            if j_wp[-1] >= len(frames):
                break
            j_tp = int(np.searchsorted(arc, arc[i] + target_m))
            j_far = int(np.searchsorted(arc, arc[i] + target_far_m))
            if j_far >= len(frames):
                break
            last = max(j_wp[-1], j_far)
            if jumps_before[last] != jumps_before[i]:
                continue
            if np.any(np.diff(frames[i : last + 1]) != 1):
                continue
            row = ctrl.get(int(frames[i]))
            if row is None:
                continue
            frame = int(frames[i])
            if not (self.dir / "lidar" / f"{frame:06d}.npy").exists():
                continue
            wps = to_ego(e[j_wp], n[j_wp], e[i], n[i], yaw[i])
            tp = to_ego(e[[j_tp, j_far]], n[[j_tp, j_far]], e[i], n[i], yaw[i]).ravel()
            red = row.get("at_traffic_light") == "1" and row.get("light_state") in (
                "Red",
                "Yellow",
            )
            self.samples.append(
                {
                    "frame": frame,
                    "waypoints": wps.astype(np.float32),
                    "target": tp.astype(np.float32),
                    "speed": np.float32(speed[i]),
                    "red_light": np.float32(red),
                    "brake": np.float32(float(row.get("brake") or 0.0) > 0.1),
                    "command": row.get("command", ""),
                }
            )


class TransfuserDataset(Dataset):
    def __init__(
        self,
        runs: list[Path],
        *,
        fps: float = 20.0,
        target_m: float = TARGET_M,
        target_far_m: float = TARGET_FAR_M,
        stride: int = 2,
    ) -> None:
        self.runs = [
            _Run(r, fps=fps, target_m=target_m, target_far_m=target_far_m, stride=stride)
            for r in runs
        ]
        self.index = [
            (ri, si) for ri, run in enumerate(self.runs) for si in range(len(run.samples))
        ]
        self.has_seg = all(run.has_seg for run in self.runs)
        if not self.index:
            raise ValueError(f"No usable frames in {[str(r) for r in runs]}")

    def __len__(self) -> int:
        return len(self.index)

    def junction_weights(self, weight: float) -> list[float]:
        """Sampling weight per frame: `weight` where the near target is off to the side."""
        out = []
        for ri, si in self.index:
            side = abs(self.runs[ri].samples[si]["target"][1])
            out.append(weight if side > JUNCTION_LATERAL_M else 1.0)
        return out

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ri, si = self.index[idx]
        run = self.runs[ri]
        s = run.samples[si]
        name = f"{s['frame']:06d}"

        rgb = [np.asarray(Image.open(run.dir / "images" / c / f"{name}.png").convert("RGB")) for c in CAMERAS]
        points = load_lidar_points(run.dir / "lidar" / f"{name}.npy")

        item = {
            "image": torch.from_numpy(image_strip(rgb)),
            "lidar": torch.from_numpy(lidar_input(points, s["target"])),
            "speed": torch.tensor([s["speed"]]),
            "target": torch.from_numpy(s["target"]),
            "waypoints": torch.from_numpy(s["waypoints"]),
            "affordance": torch.tensor([s["red_light"], s["brake"]]),
            "bev_seg": torch.from_numpy(bev_labels_from_lidar(points).astype(np.int64)),
        }
        if run.has_seg:
            tags = [np.asarray(Image.open(run.dir / "seg" / c / f"{name}.png")) for c in CAMERAS]
            # Label at the decoder's 1/4 resolution.
            seg = _IMG_LUT[np.concatenate(tags, axis=1)[2::4, 2::4]]
            item["img_seg"] = torch.from_numpy(seg.astype(np.int64))
        return item
