"""RGB → CARLA semantic-tag model and paired dataset."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset

from segmentation.labels import CLASS_NAMES, NUM_CLASSES, sanitize_labels

# Must match the center camera used while recording.
IMAGE_WIDTH = 160
IMAGE_HEIGHT = 120

__all__ = [
    "CLASS_NAMES",
    "IMAGE_HEIGHT",
    "IMAGE_WIDTH",
    "NUM_CLASSES",
    "SegDataset",
    "SegNet",
    "SegSubset",
    "class_weights_from_counts",
    "label_histogram",
    "load_seg_model",
]


def _is_seg_run(path: Path) -> bool:
    return (path / "seg" / "center").is_dir() and (
        path / "images" / "center"
    ).is_dir()


def resolve_seg_runs(paths: list[Path]) -> list[Path]:
    """Expand a data/ folder into runs that have center RGB and tag maps."""
    runs: list[Path] = []
    for path in paths:
        path = Path(path)
        if _is_seg_run(path):
            runs.append(path)
            continue
        if not path.is_dir():
            raise FileNotFoundError(f"Not a directory: {path}")
        children = sorted(p for p in path.iterdir() if p.is_dir() and _is_seg_run(p))
        if not children:
            raise FileNotFoundError(
                f"No semantic dataset (images/center + seg/center) at {path}"
            )
        runs.extend(children)
    if not runs:
        raise FileNotFoundError("No semantic datasets found.")
    return runs


def _shift_x(array: np.ndarray, dx: int) -> np.ndarray:
    if dx == 0:
        return array
    out = np.empty_like(array)
    if array.ndim == 2:
        if dx > 0:
            out[:, dx:] = array[:, :-dx]
            out[:, :dx] = array[:, :1]
        else:
            out[:, :dx] = array[:, -dx:]
            out[:, dx:] = array[:, -1:]
        return out
    if dx > 0:
        out[:, dx:] = array[:, :-dx]
        out[:, :dx] = array[:, :1]
    else:
        out[:, :dx] = array[:, -dx:]
        out[:, dx:] = array[:, -1:]
    return out


class SegDataset(Dataset):
    """RGB paired with CARLA tag ids under seg/<camera>.

    Includes every camera that has both ``images/<cam>`` and ``seg/<cam>``
    (center, and left/right when recorded with --semantic for multi-cam).
    """

    def __init__(
        self,
        run_dirs: Path | str | list[Path | str],
        *,
        cameras: tuple[str, ...] | None = None,
    ) -> None:
        if isinstance(run_dirs, (str, Path)):
            run_dirs = resolve_seg_runs([Path(run_dirs)])
        else:
            run_dirs = resolve_seg_runs([Path(p) for p in run_dirs])
        self.run_dirs = run_dirs
        self.samples: list[tuple[Path, Path]] = []
        wanted = None if cameras is None else {c.lower() for c in cameras}
        for run_dir in run_dirs:
            seg_root = run_dir / "seg"
            if not seg_root.is_dir():
                continue
            for seg_dir in sorted(p for p in seg_root.iterdir() if p.is_dir()):
                cam = seg_dir.name
                if wanted is not None and cam not in wanted:
                    continue
                rgb_dir = run_dir / "images" / cam
                if not rgb_dir.is_dir():
                    continue
                for seg_path in sorted(seg_dir.glob("*.png")):
                    rgb_path = rgb_dir / seg_path.name
                    if rgb_path.exists():
                        self.samples.append((rgb_path, seg_path))
        if not self.samples:
            raise FileNotFoundError(
                "Semantic runs have no paired images/<cam> and seg/<cam> frames."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def get(
        self, index: int, *, augment: bool = False, max_shift_px: int = 8
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rgb_path, seg_path = self.samples[index]
        rgb = np.asarray(Image.open(rgb_path).convert("RGB"), dtype=np.float32) / 255.0
        labels = sanitize_labels(np.asarray(Image.open(seg_path)))
        if rgb.shape[0] != IMAGE_HEIGHT or rgb.shape[1] != IMAGE_WIDTH:
            raise ValueError(
                f"{rgb_path} is {rgb.shape[1]}x{rgb.shape[0]}, "
                f"expected {IMAGE_WIDTH}x{IMAGE_HEIGHT}"
            )
        if labels.shape != (IMAGE_HEIGHT, IMAGE_WIDTH):
            raise ValueError(
                f"{seg_path} is {labels.shape[1]}x{labels.shape[0]}, "
                f"expected {IMAGE_WIDTH}x{IMAGE_HEIGHT}"
            )
        if augment:
            rgb = np.clip(rgb * float(np.random.uniform(0.7, 1.3)), 0.0, 1.0)
            if np.random.random() < 0.5:
                rgb = rgb[:, ::-1]
                labels = labels[:, ::-1]
            if max_shift_px > 0:
                dx = int(np.random.randint(-max_shift_px, max_shift_px + 1))
                rgb = _shift_x(rgb, dx)
                labels = _shift_x(labels, dx)
            rgb = np.ascontiguousarray(rgb)
            labels = np.ascontiguousarray(labels)
        image = torch.from_numpy(rgb).permute(2, 0, 1)
        target = torch.from_numpy(labels.astype(np.int64, copy=False))
        return image, target

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.get(index, augment=False)


class SegSubset(Dataset):
    def __init__(
        self,
        dataset: SegDataset,
        indices: list[int],
        *,
        augment: bool = False,
    ) -> None:
        self.dataset = dataset
        self.indices = list(indices)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.dataset.get(self.indices[index], augment=self.augment)


def label_histogram(paths: list[Path]) -> np.ndarray:
    counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    for path in paths:
        labels = sanitize_labels(np.asarray(Image.open(path)))
        counts += np.bincount(labels.ravel(), minlength=NUM_CLASSES)
    return counts


def class_weights_from_counts(counts: np.ndarray) -> np.ndarray:
    """ENet-style inverse-log frequency. Absent tags stay at weight 0."""
    counts = np.asarray(counts, dtype=np.float64)
    total = float(counts.sum())
    weights = np.zeros(NUM_CLASSES, dtype=np.float64)
    if total <= 0:
        return np.ones(NUM_CLASSES, dtype=np.float32)
    present = counts > 0
    freq = counts / total
    weights[present] = 1.0 / np.log(1.02 + freq[present])
    weights[present] *= float(present.sum()) / float(weights[present].sum())
    return weights.astype(np.float32)


class _Block(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SegNet(nn.Module):
    """Small U-Net. 160x120 is divisible by 8, so three pools round-trip."""

    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.enc1 = _Block(3, 32)
        self.enc2 = _Block(32, 64)
        self.enc3 = _Block(64, 128)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = _Block(128, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = _Block(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = _Block(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = _Block(64, 32)
        self.head = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        bottleneck = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(bottleneck), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)


def load_seg_model(path: Path | str, device: torch.device | str = "cpu") -> SegNet:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = SegNet(num_classes=int(ckpt.get("num_classes", NUM_CLASSES)))
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model
