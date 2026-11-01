"""
Train stereo depth from the center and right-wide cameras.

The depth camera is recorded as the label. It is not a model input.
right_wide is 1.00 m to the right of center, on the outer corner of the roof.

  python -m expert.record --depth --semantic --cameras center,right_wide --seconds 300

  python -m perception.train --data data/ --epochs 20
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from perception.depth import MAX_DEPTH_M
from perception.model import IMAGE_HEIGHT, IMAGE_WIDTH, StereoDataset, StereoNet
from utils.device import describe_device, resolve_device


def _masked_log_l1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    valid = (target > 0.5) & (target < MAX_DEPTH_M)
    if int(valid.sum().item()) == 0:
        return pred.sum() * 0.0
    log_pred = torch.log(pred.clamp(min=0.5))
    log_target = torch.log(target.clamp(min=0.5))
    return (log_pred - log_target).abs()[valid].mean()


def _depth_stats(pred: torch.Tensor, target: torch.Tensor) -> tuple[float, float, int]:
    valid = (target > 0.5) & (target < MAX_DEPTH_M)
    count = int(valid.sum().item())
    if count == 0:
        return 0.0, 0.0, 0
    abs_err = (pred - target).abs()[valid]
    ratio = torch.maximum(pred[valid] / target[valid], target[valid] / pred[valid])
    within = float((ratio < 1.25).float().mean().item())
    return float(abs_err.mean().item()), within, count


def _colorize_depth(meters: np.ndarray, max_m: float = 40.0) -> np.ndarray:
    x = np.clip(meters / max_m, 0.0, 1.0)
    red = np.clip(1.5 - 2.0 * x, 0.0, 1.0)
    green = np.clip(1.5 - np.abs(2.0 * x - 1.0), 0.0, 1.0)
    blue = np.clip(2.0 * x - 0.5, 0.0, 1.0)
    rgb = np.stack([red, green, blue], axis=-1)
    rgb[meters >= max_m] = 0.0
    return (rgb * 255.0).astype(np.uint8)


def _save_previews(
    model: StereoNet,
    dataset: StereoDataset,
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
            center, right, target = dataset.get(index, augment=False)
            pred = model(
                center.unsqueeze(0).to(device),
                right.unsqueeze(0).to(device),
            )[0, 0].cpu().numpy()
            rgb = (center.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
            gt = _colorize_depth(target[0].numpy())
            pred_rgb = _colorize_depth(pred)
            strip = np.concatenate([rgb, gt, pred_rgb], axis=1)
            Image.fromarray(strip).save(out_dir / f"preview_{i:02d}.png")


class _Subset(torch.utils.data.Dataset):
    def __init__(self, base: StereoDataset, indices: list[int], *, augment: bool) -> None:
        self.base = base
        self.indices = indices
        self.augment = augment

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        return self.base.get(self.indices[index], augment=self.augment)


def train_stereo(
    data: Path | list[Path],
    *,
    epochs: int = 20,
    batch_size: int = 16,
    lr: float = 1e-3,
    out: Path = Path("models/stereo.pt"),
    device_name: str = "cuda",
    preview_dir: Path | None = None,
) -> Path:
    device = resolve_device(device_name)
    print(f"Device: {describe_device(device)}")
    dataset = StereoDataset(data)
    print(f"Stereo frames: {len(dataset)} from {len(dataset.run_dirs)} run(s)")
    val_size = max(1, int(round(0.1 * len(dataset))))
    train_size = len(dataset) - val_size
    train_base, val_base = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    train_set = _Subset(dataset, list(train_base.indices), augment=True)
    val_set = _Subset(dataset, list(val_base.indices), augment=False)
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
    model = StereoNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    out.parent.mkdir(parents=True, exist_ok=True)
    best_abs = float("inf")
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
        for center, right, target in train_bar:
            center = center.to(device, non_blocking=use_cuda)
            right = right.to(device, non_blocking=use_cuda)
            target = target.to(device, non_blocking=use_cuda)
            pred = model(center, right)
            loss = _masked_log_l1(pred, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item()) * center.size(0)
            train_bar.set_postfix(loss=f"{loss.item():.4f}")
        train_loss /= train_size

        model.eval()
        val_loss = 0.0
        abs_sum = 0.0
        within_sum = 0.0
        pixels = 0
        with torch.no_grad():
            val_bar = tqdm(
                val_loader,
                desc=f"epoch {epoch:02d}/{epochs} val",
                leave=False,
                unit="batch",
            )
            for center, right, target in val_bar:
                center = center.to(device, non_blocking=use_cuda)
                right = right.to(device, non_blocking=use_cuda)
                target = target.to(device, non_blocking=use_cuda)
                pred = model(center, right)
                batch_loss = _masked_log_l1(pred, target)
                val_loss += float(batch_loss.item()) * center.size(0)
                batch_abs, batch_within, count = _depth_stats(pred, target)
                abs_sum += batch_abs * count
                within_sum += batch_within * count
                pixels += count
                val_bar.set_postfix(loss=f"{batch_loss.item():.4f}")
        val_loss /= val_size
        mean_abs = abs_sum / pixels if pixels else 0.0
        within = within_sum / pixels if pixels else 0.0
        print(
            f"epoch {epoch:02d}  train={train_loss:.4f}  val={val_loss:.4f}  "
            f"abs={mean_abs:.2f}m  within_1.25={100 * within:.1f}%"
        )
        if mean_abs < best_abs and pixels:
            best_abs = mean_abs
            best_epoch = epoch
            torch.save(
                {
                    "model": model.state_dict(),
                    "max_disp": model.max_disp,
                    "focal_px": model.focal_px,
                    "baseline_m": model.baseline_m,
                    "stereo_camera": model.camera_name,
                    "image_width": IMAGE_WIDTH,
                    "image_height": IMAGE_HEIGHT,
                    "epoch": epoch,
                    "val_abs_m": mean_abs,
                    "val_within_1.25": within,
                },
                out,
            )

    print(
        f"Saved {out}  best epoch {best_epoch:02d}  "
        f"abs={best_abs:.2f}m"
    )
    if preview_dir is None:
        preview_dir = Path(str(out.with_suffix("")) + "_preview")
    ckpt = torch.load(out, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    _save_previews(model, dataset, list(val_base.indices), device, preview_dir)
    print(f"Previews (rgb | truth | pred): {preview_dir}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        nargs="+",
        required=True,
        help="Run directory or a folder of runs with center, right_wide, and depth",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out", type=Path, default=Path("models/stereo.pt"))
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu", "auto"],
        default="cuda",
    )
    parser.add_argument("--preview-dir", type=Path, default=None)
    args = parser.parse_args()
    train_stereo(
        args.data if len(args.data) > 1 else args.data[0],
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        out=args.out,
        device_name=args.device,
        preview_dir=args.preview_dir,
    )


if __name__ == "__main__":
    main()
