"""Learned bird's-eye occupancy from lidar pillars.

Hand voting needs hit thresholds and blob cleanup. This network reads
geometry pillars and predicts free vs occupied, then a class for each
occupied cell. Road still comes from the camera segmenter.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

from perception.bev import (
    CELL_M,
    FORWARD_M,
    LATERAL_M,
    MIN_BLOB_CELLS,
    OBJECT_IDS,
    bev_from_labels,
    _drop_small,
    _fill_ground,
)
from perception.lidar import load_lidar_points
from perception.pillars import PILLAR_CHANNELS, grid_shape, pillar_features

# Compact class list for the head: free + object tags we care about.
OCC_CLASS_IDS: tuple[int, ...] = (0,) + tuple(OBJECT_IDS)  # 0 = free
OCC_ID_TO_INDEX = {cid: i for i, cid in enumerate(OCC_CLASS_IDS)}
NUM_OCC_CLASSES = len(OCC_CLASS_IDS)

__all__ = [
    "NUM_OCC_CLASSES",
    "OCC_CLASS_IDS",
    "OccupancyDataset",
    "OccupancyNet",
    "bev_from_occupancy",
    "load_occupancy_model",
    "target_from_grid",
]


def target_from_grid(grid: np.ndarray) -> np.ndarray:
    """BEV tag grid (-1 empty) → class index map. Ground and empty → free (0)."""
    target = np.zeros(grid.shape, dtype=np.int64)
    for tag, index in OCC_ID_TO_INDEX.items():
        if tag == 0:
            continue
        target[grid == tag] = index
    return target


def _is_occupancy_run(path: Path) -> bool:
    return (
        (path / "lidar").is_dir()
        and (path / "seg" / "center").is_dir()
        and (path / "depth" / "center").is_dir()
    )


def resolve_occupancy_runs(paths: list[Path]) -> list[Path]:
    runs: list[Path] = []
    for path in paths:
        path = Path(path)
        if _is_occupancy_run(path):
            runs.append(path)
            continue
        if not path.is_dir():
            raise FileNotFoundError(f"Not a directory: {path}")
        children = sorted(p for p in path.iterdir() if p.is_dir() and _is_occupancy_run(p))
        if not children:
            raise FileNotFoundError(
                f"No occupancy dataset (lidar, seg/center, depth/center) at {path}"
            )
        runs.extend(children)
    if not runs:
        raise FileNotFoundError("No occupancy datasets found.")
    return runs


class OccupancyDataset(Dataset):
    """Lidar pillars supervised by camera+depth BEV object cells."""

    def __init__(
        self,
        run_dirs: Path | str | list[Path | str],
        *,
        stride: int = 1,
        forward_m: float = FORWARD_M,
        lateral_m: float = LATERAL_M,
        cell_m: float = CELL_M,
    ) -> None:
        if isinstance(run_dirs, (str, Path)):
            run_dirs = resolve_occupancy_runs([Path(run_dirs)])
        else:
            run_dirs = resolve_occupancy_runs([Path(p) for p in run_dirs])
        self.run_dirs = run_dirs
        self.forward_m = forward_m
        self.lateral_m = lateral_m
        self.cell_m = cell_m
        self.samples: list[tuple[Path, Path, Path]] = []
        stride = max(1, int(stride))
        for run_dir in run_dirs:
            lidar_dir = run_dir / "lidar"
            seg_dir = run_dir / "seg" / "center"
            depth_dir = run_dir / "depth" / "center"
            names = sorted(lidar_dir.glob("*.npy"))[::stride]
            for lidar_path in names:
                name = lidar_path.stem
                seg_path = seg_dir / f"{name}.png"
                depth_path = depth_dir / f"{name}.npy"
                if seg_path.exists() and depth_path.exists():
                    self.samples.append((lidar_path, seg_path, depth_path))
        if not self.samples:
            raise FileNotFoundError("No paired lidar / seg / depth frames.")

    def __len__(self) -> int:
        return len(self.samples)

    def get(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        lidar_path, seg_path, depth_path = self.samples[index]
        points = load_lidar_points(lidar_path)
        labels = np.asarray(Image.open(seg_path))
        depth = np.load(depth_path).astype(np.float32)
        kwargs = {
            "forward_m": self.forward_m,
            "lateral_m": self.lateral_m,
            "cell_m": self.cell_m,
        }
        pillars = pillar_features(points, **kwargs)
        gt = bev_from_labels(labels, depth, **kwargs)
        target = target_from_grid(gt)
        return (
            torch.from_numpy(pillars).contiguous(),
            torch.from_numpy(target).contiguous(),
        )

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.get(index)


class _ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class OccupancyNet(nn.Module):
    """Small U-Net over the pillar map → per-cell class logits."""

    def __init__(
        self,
        *,
        in_channels: int = PILLAR_CHANNELS,
        num_classes: int = NUM_OCC_CLASSES,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.enc1 = _ConvBlock(in_channels, 32)
        self.enc2 = _ConvBlock(32, 64)
        self.enc3 = _ConvBlock(64, 128)
        self.pool = nn.MaxPool2d(2)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = _ConvBlock(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = _ConvBlock(64, 32)
        self.head = nn.Conv2d(32, num_classes, 1)

    def forward(self, pillars: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(pillars)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        d2 = self.up2(e3)
        if d2.shape[-2:] != e2.shape[-2:]:
            d2 = F.interpolate(d2, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        if d1.shape[-2:] != e1.shape[-2:]:
            d1 = F.interpolate(d1, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.head(d1)


def logits_to_grid(
    logits: torch.Tensor | np.ndarray,
    *,
    occ_threshold: float = 0.5,
) -> np.ndarray:
    """Class logits [C,H,W] or [1,C,H,W] → tag grid (-1 free).

    A cell is occupied when free-class probability is below ``1 - occ_threshold``
    (i.e. some object mass ≥ threshold). Class is the best non-free logit.
    """
    if isinstance(logits, torch.Tensor):
        logits = logits.detach().float().cpu()
        if logits.ndim == 4:
            logits = logits[0]
        probs = torch.softmax(logits, dim=0).numpy()
        logits_np = logits.numpy()
    else:
        logits_np = np.asarray(logits, dtype=np.float32)
        if logits_np.ndim == 4:
            logits_np = logits_np[0]
        exp = np.exp(logits_np - logits_np.max(axis=0, keepdims=True))
        probs = exp / np.clip(exp.sum(axis=0, keepdims=True), 1e-6, None)

    occupied_prob = 1.0 - probs[0]
    # Best object class among non-free channels.
    obj_logits = logits_np[1:]
    pred_obj = obj_logits.argmax(axis=0) + 1
    grid = np.full(occupied_prob.shape, -1, dtype=np.int16)
    mask = occupied_prob >= occ_threshold
    for index, tag in enumerate(OCC_CLASS_IDS):
        if index == 0:
            continue
        grid[mask & (pred_obj == index)] = tag
    return grid


def bev_from_occupancy(
    points: np.ndarray,
    model: OccupancyNet,
    *,
    labels: np.ndarray | None = None,
    label_views: list[tuple[np.ndarray, str]] | None = None,
    device: torch.device | str = "cpu",
    forward_m: float = FORWARD_M,
    lateral_m: float = LATERAL_M,
    cell_m: float = CELL_M,
    occ_threshold: float = 0.5,
    min_blob_cells: int = MIN_BLOB_CELLS,
) -> np.ndarray:
    """Pillars → network → object cells; optional camera road underneath.

    Pass ``labels`` for center-only road, or ``label_views`` for multi-cam
    late fusion: ``[(labels, "center"), (labels_l, "left"), ...]``.
    """
    from perception.bev import fill_ground_multi

    kwargs = {"forward_m": forward_m, "lateral_m": lateral_m, "cell_m": cell_m}
    n_forward, n_lateral = grid_shape(**kwargs)
    grid = np.full((n_forward, n_lateral), -1, dtype=np.int16)
    if label_views:
        fill_ground_multi(grid, label_views, **kwargs)
    elif labels is not None:
        _fill_ground(grid, labels, **kwargs)

    pillars = pillar_features(points, **kwargs)
    tensor = torch.from_numpy(pillars).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        logits = model(tensor)
    objects = logits_to_grid(logits, occ_threshold=occ_threshold)
    # Objects sit on the road — overwrite ground the same way hand lidar does.
    object_mask = objects >= 0
    grid[object_mask] = objects[object_mask]
    _drop_small(grid, min_cells=min_blob_cells)
    return grid


def load_occupancy_model(
    path: Path | str, device: torch.device | str = "cpu"
) -> OccupancyNet:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = OccupancyNet(
        in_channels=int(ckpt.get("in_channels", PILLAR_CHANNELS)),
        num_classes=int(ckpt.get("num_classes", NUM_OCC_CLASSES)),
    )
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model


def occupancy_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    free_weight: float = 0.02,
) -> torch.Tensor:
    """Cross-entropy + binary occupancy. Free cells dominate (~99.9%).

    Class CE down-weights free. Binary BCE pushes occupied probability up
    with a large positive weight so the net does not collapse to empty.
    """
    weights = torch.ones(NUM_OCC_CLASSES, device=logits.device, dtype=logits.dtype)
    weights[0] = free_weight
    weights[1:] = 5.0
    ce = F.cross_entropy(logits, target, weight=weights)

    # Occupied = any non-free class. Compare free logit vs logsumexp(objects).
    free_logit = logits[:, 0]
    occ_logit = logits[:, 1:].logsumexp(dim=1)
    binary_logits = occ_logit - free_logit
    binary_target = (target > 0).float()
    bce = F.binary_cross_entropy_with_logits(
        binary_logits, binary_target, pos_weight=torch.tensor(40.0, device=logits.device)
    )
    return ce + 0.5 * bce
