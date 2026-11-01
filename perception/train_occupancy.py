"""
Train bird's-eye occupancy from lidar pillars.

Labels come from the center semantic camera + depth projected to BEV.
Inputs are geometry-only pillars (no tags).

  python -m perception.train_occupancy --data data/run_20261016_200616 --epochs 8
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from perception.bev import colorize_bev
from perception.occupancy import (
    NUM_OCC_CLASSES,
    OccupancyDataset,
    OccupancyNet,
    logits_to_grid,
    occupancy_loss,
)
from perception.pillars import PILLAR_CHANNELS
from utils.device import describe_device, resolve_device


class _Subset(torch.utils.data.Dataset):
    def __init__(self, base: OccupancyDataset, indices: list[int]) -> None:
        self.base = base
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        return self.base.get(self.indices[index])


def _cell_stats(pred: torch.Tensor, target: torch.Tensor) -> tuple[float, float, float]:
    """Return (accuracy, occupied-IoU, free accuracy) on argmax classes."""
    pred_idx = pred.argmax(dim=1)
    correct = (pred_idx == target).float().mean().item()
    occ_t = target > 0
    occ_p = pred_idx > 0
    inter = (occ_t & occ_p).sum().item()
    union = (occ_t | occ_p).sum().item()
    iou = inter / union if union else 0.0
    free = target == 0
    free_acc = (pred_idx[free] == 0).float().mean().item() if free.any() else 0.0
    return float(correct), float(iou), float(free_acc)


def _save_previews(
    model: OccupancyNet,
    dataset: OccupancyDataset,
    indices: list[int],
    device: torch.device,
    out_dir: Path,
    *,
    n: int = 8,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    with torch.no_grad():
        for i, index in enumerate(indices[:n]):
            pillars, target = dataset.get(index)
            logits = model(pillars.unsqueeze(0).to(device))[0]
            pred_grid = logits_to_grid(logits)
            gt_grid = np.full(target.shape, -1, dtype=np.int16)
            from perception.occupancy import OCC_CLASS_IDS

            for class_index, tag in enumerate(OCC_CLASS_IDS):
                if class_index == 0:
                    continue
                gt_grid[target.numpy() == class_index] = tag
            strip = np.concatenate(
                [colorize_bev(gt_grid), colorize_bev(pred_grid)],
                axis=1,
            )
            Image.fromarray(strip).save(out_dir / f"preview_{i:02d}.png")


def train_occupancy(
    data: Path | list[Path],
    *,
    epochs: int = 8,
    batch_size: int = 8,
    lr: float = 1e-3,
    stride: int = 5,
    out: Path = Path("models/occupancy.pt"),
    device_name: str = "auto",
    preview_dir: Path | None = None,
) -> Path:
    device = resolve_device(device_name)
    print(f"Device: {describe_device(device)}")
    dataset = OccupancyDataset(data, stride=stride)
    print(
        f"Occupancy frames: {len(dataset)} from {len(dataset.run_dirs)} run(s) "
        f"(stride={stride})"
    )
    val_size = max(1, int(round(0.1 * len(dataset))))
    train_size = len(dataset) - val_size
    train_base, val_base = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    train_set = _Subset(dataset, list(train_base.indices))
    val_set = _Subset(dataset, list(val_base.indices))
    use_cuda = device.type == "cuda"
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=use_cuda,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=use_cuda,
    )
    model = OccupancyNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    out.parent.mkdir(parents=True, exist_ok=True)
    best_iou = -1.0
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_bar = tqdm(
            train_loader,
            desc=f"epoch {epoch:02d}/{epochs} train",
            leave=False,
            unit="batch",
        )
        for pillars, target in train_bar:
            pillars = pillars.to(device, non_blocking=use_cuda)
            target = target.to(device, non_blocking=use_cuda)
            logits = model(pillars)
            loss = occupancy_loss(logits, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item()) * pillars.size(0)
            train_bar.set_postfix(loss=f"{loss.item():.4f}")
        train_loss /= max(1, train_size)

        model.eval()
        val_loss = 0.0
        acc_sum = 0.0
        iou_sum = 0.0
        free_sum = 0.0
        batches = 0
        with torch.no_grad():
            val_bar = tqdm(
                val_loader,
                desc=f"epoch {epoch:02d}/{epochs} val",
                leave=False,
                unit="batch",
            )
            for pillars, target in val_bar:
                pillars = pillars.to(device, non_blocking=use_cuda)
                target = target.to(device, non_blocking=use_cuda)
                logits = model(pillars)
                batch_loss = occupancy_loss(logits, target)
                val_loss += float(batch_loss.item()) * pillars.size(0)
                acc, iou, free_acc = _cell_stats(logits, target)
                acc_sum += acc
                iou_sum += iou
                free_sum += free_acc
                batches += 1
                val_bar.set_postfix(loss=f"{batch_loss.item():.4f}", iou=f"{iou:.3f}")
        val_loss /= max(1, val_size)
        mean_acc = acc_sum / max(1, batches)
        mean_iou = iou_sum / max(1, batches)
        mean_free = free_sum / max(1, batches)
        print(
            f"epoch {epoch:02d}  train={train_loss:.4f}  val={val_loss:.4f}  "
            f"acc={100 * mean_acc:.1f}%  occ_iou={100 * mean_iou:.1f}%  "
            f"free={100 * mean_free:.1f}%"
        )
        if mean_iou > best_iou:
            best_iou = mean_iou
            best_epoch = epoch
            torch.save(
                {
                    "model": model.state_dict(),
                    "in_channels": PILLAR_CHANNELS,
                    "num_classes": NUM_OCC_CLASSES,
                    "forward_m": dataset.forward_m,
                    "lateral_m": dataset.lateral_m,
                    "cell_m": dataset.cell_m,
                    "epoch": epoch,
                    "val_occ_iou": mean_iou,
                    "val_acc": mean_acc,
                },
                out,
            )

    print(f"Saved {out}  best epoch {best_epoch:02d}  occ_iou={100 * best_iou:.1f}%")
    if preview_dir is None:
        preview_dir = Path(str(out.with_suffix("")) + "_preview")
    ckpt = torch.load(out, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    _save_previews(model, dataset, list(val_base.indices), device, preview_dir)
    print(f"Previews (truth | pred): {preview_dir}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        nargs="+",
        required=True,
        help="Run directory or folder of runs with lidar, seg, depth",
    )
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--stride",
        type=int,
        default=5,
        help="Keep every Nth frame (default 5; CPU-friendly)",
    )
    parser.add_argument("--out", type=Path, default=Path("models/occupancy.pt"))
    parser.add_argument("--device", choices=["cuda", "cpu", "auto"], default="auto")
    parser.add_argument("--preview-dir", type=Path, default=None)
    args = parser.parse_args()
    train_occupancy(
        args.data if len(args.data) > 1 else args.data[0],
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        stride=args.stride,
        out=args.out,
        device_name=args.device,
        preview_dir=args.preview_dir,
    )


if __name__ == "__main__":
    main()
