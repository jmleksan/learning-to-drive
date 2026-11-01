"""Center + right_wide photos → meters, supervised by the center depth camera."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import Dataset

from perception.depth import (
    BASELINE_M,
    FOCAL_PX,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    MAX_DISP,
    STEREO_CAMERA,
    load_depth_meters,
)

__all__ = [
    "IMAGE_HEIGHT",
    "IMAGE_WIDTH",
    "StereoDataset",
    "StereoNet",
    "load_stereo_model",
]


def _is_stereo_run(path: Path) -> bool:
    return (
        (path / "images" / "center").is_dir()
        and (path / "images" / STEREO_CAMERA).is_dir()
        and (path / "depth" / "center").is_dir()
    )


def resolve_stereo_runs(paths: list[Path]) -> list[Path]:
    runs: list[Path] = []
    for path in paths:
        path = Path(path)
        if _is_stereo_run(path):
            runs.append(path)
            continue
        if not path.is_dir():
            raise FileNotFoundError(f"Not a directory: {path}")
        children = sorted(p for p in path.iterdir() if p.is_dir() and _is_stereo_run(p))
        if not children:
            raise FileNotFoundError(
                f"No stereo dataset (images/center, images/{STEREO_CAMERA}, depth/center) at {path}"
            )
        runs.extend(children)
    if not runs:
        raise FileNotFoundError("No stereo datasets found.")
    return runs


def _load_rgb(path: Path) -> np.ndarray:
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    if rgb.shape[0] != IMAGE_HEIGHT or rgb.shape[1] != IMAGE_WIDTH:
        raise ValueError(
            f"{path} is {rgb.shape[1]}x{rgb.shape[0]}, "
            f"expected {IMAGE_WIDTH}x{IMAGE_HEIGHT}"
        )
    return rgb


class StereoDataset(Dataset):
    """Center and right_wide RGB paired with center depth in meters."""

    def __init__(self, run_dirs: Path | str | list[Path | str]) -> None:
        if isinstance(run_dirs, (str, Path)):
            run_dirs = resolve_stereo_runs([Path(run_dirs)])
        else:
            run_dirs = resolve_stereo_runs([Path(p) for p in run_dirs])
        self.run_dirs = run_dirs
        self.samples: list[tuple[Path, Path, Path]] = []
        for run_dir in run_dirs:
            center_dir = run_dir / "images" / "center"
            right_dir = run_dir / "images" / STEREO_CAMERA
            depth_dir = run_dir / "depth" / "center"
            for depth_path in sorted(depth_dir.glob("*.npy")):
                name = depth_path.with_suffix(".png").name
                center_path = center_dir / name
                right_path = right_dir / name
                if center_path.exists() and right_path.exists():
                    self.samples.append((center_path, right_path, depth_path))
        if not self.samples:
            raise FileNotFoundError(
                "Stereo runs have no paired center, right_wide, and depth frames."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def get(self, index: int, *, augment: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        center_path, right_path, depth_path = self.samples[index]
        center = _load_rgb(center_path)
        right = _load_rgb(right_path)
        depth = load_depth_meters(depth_path)
        if augment:
            scale = float(np.random.uniform(0.7, 1.3))
            center = np.clip(center * scale, 0.0, 1.0)
            right = np.clip(right * scale, 0.0, 1.0)
        center_t = torch.from_numpy(center).permute(2, 0, 1).contiguous()
        right_t = torch.from_numpy(right).permute(2, 0, 1).contiguous()
        depth_t = torch.from_numpy(depth).unsqueeze(0).contiguous()
        return center_t, right_t, depth_t

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.get(index, augment=False)


class _Features(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.net(image)


class StereoNet(nn.Module):
    """Match the center photo to the right-wide photo and convert the shift to meters.

    A point at center pixel x appears at x - d in the right-wide photo, because
    that camera sits 1.00 m to the right. Depth is focal_px * baseline / disparity.
    """

    def __init__(self, max_disp: int = MAX_DISP) -> None:
        super().__init__()
        self.max_disp = max_disp
        self.focal_px = FOCAL_PX
        self.baseline_m = BASELINE_M
        self.camera_name = STEREO_CAMERA
        self.features = _Features()
        self.head = nn.Sequential(
            nn.Conv2d(max_disp, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, max_disp, 3, padding=1),
        )

    def forward(self, center: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        feat_c = self.features(center)
        feat_r = self.features(right)
        costs = []
        for disparity in range(self.max_disp):
            if disparity == 0:
                shifted = feat_r
            else:
                shifted = F.pad(feat_r, (disparity, 0))[:, :, :, :-disparity]
            costs.append((feat_c * shifted).mean(dim=1))
        cost = torch.stack(costs, dim=1)
        logits = self.head(cost)
        probability = torch.softmax(logits, dim=1)
        values = torch.arange(
            self.max_disp, device=center.device, dtype=center.dtype
        ).view(1, -1, 1, 1)
        disparity = (probability * values).sum(dim=1, keepdim=True)
        return (self.focal_px * self.baseline_m) / disparity.clamp(min=0.25)


def load_stereo_model(path: Path | str, device: torch.device | str = "cpu") -> StereoNet:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = StereoNet(max_disp=int(ckpt.get("max_disp", MAX_DISP)))
    model.baseline_m = float(ckpt.get("baseline_m", model.baseline_m))
    model.camera_name = str(ckpt.get("stereo_camera", "right"))
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model
