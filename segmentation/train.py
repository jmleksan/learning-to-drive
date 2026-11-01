"""
Train a semantic segmentation net on CARLA tag maps.

Record first (RGB + matching semantic cameras):

  python -m expert.record --semantic --cameras center --seconds 60

For multi-cam BEV (yawed left/right), record tags on those mounts too:

  python -m expert.record --semantic --lidar --cameras center,left,right --seconds 180

Then mix old center runs with the new multi-cam run:

  python -m segmentation.train --data data/run_old data/run_new --epochs 20
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from segmentation.labels import colorize
from segmentation.model import (
    CLASS_NAMES,
    NUM_CLASSES,
    SegDataset,
    SegNet,
    SegSubset,
    class_weights_from_counts,
    label_histogram,
)
from utils.device import describe_device, resolve_device


def _confusion(
    pred: torch.Tensor, target: torch.Tensor, num_classes: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-class intersection and union counts."""
    inter = torch.zeros(num_classes, device=pred.device)
    union = torch.zeros(num_classes, device=pred.device)
    for cls in range(num_classes):
        p = pred == cls
        t = target == cls
        inter[cls] = (p & t).sum()
        union[cls] = (p | t).sum()
    return inter, union


def _save_previews(
    model: SegNet,
    dataset: SegDataset,
    indices: list[int],
    device: torch.device,
    out_dir: Path,
    *,
    n: int = 8,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    chosen = indices[:n]
    with torch.no_grad():
        for i, index in enumerate(chosen):
            image, target = dataset.get(index, augment=False)
            logits = model(image.unsqueeze(0).to(device))
            pred = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)
            rgb = (image.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
            gt = colorize(target.numpy().astype(np.uint8))
            pred_rgb = colorize(pred)
            strip = np.concatenate([rgb, gt, pred_rgb], axis=1)
            Image.fromarray(strip).save(out_dir / f"preview_{i:02d}.png")


def train_seg(
    data: Path | list[Path],
    *,
    epochs: int = 20,
    batch_size: int = 16,
    lr: float = 1e-3,
    out: Path = Path("models/seg.pt"),
    device_name: str = "cuda",
    preview_dir: Path | None = None,
) -> Path:
    device = resolve_device(device_name)
    use_cuda = device.type == "cuda"
    paths = [data] if isinstance(data, Path) else data
    dataset = SegDataset(paths)
    cam_counts: dict[str, int] = {}
    for _rgb, seg in dataset.samples:
        cam = seg.parent.name
        cam_counts[cam] = cam_counts.get(cam, 0) + 1
    print(
        f"Paired frames: {len(dataset)} across {len(dataset.run_dirs)} run(s)  "
        f"cams={cam_counts}"
    )
    counts = label_histogram([seg for _rgb, seg in dataset.samples])
    weights = class_weights_from_counts(counts)
    total_px = int(counts.sum()) or 1
    present = [
        f"{CLASS_NAMES[i]}={100.0 * counts[i] / total_px:.1f}%"
        for i in range(NUM_CLASSES)
        if counts[i] > 0 and 100.0 * counts[i] / total_px >= 0.05
    ]
    print("Tag mix: " + ", ".join(present))

    val_size = max(1, int(0.1 * len(dataset)))
    train_size = len(dataset) - val_size
    train_base, val_base = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    train_ds = SegSubset(dataset, list(train_base.indices), augment=True)
    val_ds = SegSubset(dataset, list(val_base.indices), augment=False)
    loader_kwargs: dict = {"batch_size": batch_size}
    if use_cuda:
        loader_kwargs["pin_memory"] = True
        loader_kwargs["num_workers"] = 2
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, **loader_kwargs)

    model = SegNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss(
        weight=torch.from_numpy(weights).to(device)
    )
    print(
        f"Samples: {len(dataset)} | device: {describe_device(device)} | "
        f"classes: {NUM_CLASSES}"
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    best_miou = -1.0
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
        for images, target in train_bar:
            images = images.to(device, non_blocking=use_cuda)
            target = target.to(device, non_blocking=use_cuda)
            logits = model(images)
            loss = loss_fn(logits, target)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item() * images.size(0)
            train_bar.set_postfix(loss=f"{loss.item():.4f}")
        train_loss /= train_size

        model.eval()
        val_loss = 0.0
        correct = 0
        pixels = 0
        inter = torch.zeros(NUM_CLASSES, device=device)
        union = torch.zeros(NUM_CLASSES, device=device)
        val_bar = tqdm(
            val_loader,
            desc=f"epoch {epoch:02d}/{epochs} val",
            leave=False,
            unit="batch",
        )
        with torch.no_grad():
            for images, target in val_bar:
                images = images.to(device, non_blocking=use_cuda)
                target = target.to(device, non_blocking=use_cuda)
                logits = model(images)
                batch_loss = loss_fn(logits, target).item()
                val_loss += batch_loss * images.size(0)
                pred = logits.argmax(1)
                correct += int((pred == target).sum().item())
                pixels += int(target.numel())
                batch_inter, batch_union = _confusion(pred, target, NUM_CLASSES)
                inter += batch_inter
                union += batch_union
                val_bar.set_postfix(loss=f"{batch_loss:.4f}")
        val_loss /= val_size
        present_cls = union > 0
        iou = torch.zeros_like(inter)
        iou[present_cls] = inter[present_cls] / union[present_cls]
        miou = float(iou[present_cls].mean().item()) if present_cls.any() else 0.0
        pix_acc = correct / pixels if pixels else 0.0
        print(
            f"epoch {epoch:02d}  train={train_loss:.4f}  val={val_loss:.4f}  "
            f"pix={100 * pix_acc:.1f}%  mIoU={100 * miou:.1f}%"
        )
        if miou > best_miou:
            best_miou = miou
            best_epoch = epoch
            torch.save(
                {
                    "model": model.state_dict(),
                    "num_classes": NUM_CLASSES,
                    "class_names": list(CLASS_NAMES),
                    "epoch": epoch,
                    "val_miou": miou,
                    "val_pix_acc": pix_acc,
                },
                out,
            )

    print(
        f"Saved {out}  best epoch {best_epoch:02d}  "
        f"mIoU={100 * best_miou:.1f}%"
    )
    if preview_dir is None:
        preview_dir = Path(str(out.with_suffix("")) + "_preview")
    # Reload the best weights so the strip matches the checkpoint.
    ckpt = torch.load(out, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    _save_previews(
        model, dataset, list(val_base.indices), device, preview_dir
    )
    print(f"Previews (rgb | truth | pred): {preview_dir}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        nargs="+",
        required=True,
        help="Run directory or a folder of runs that contain seg/center",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out", type=Path, default=Path("models/seg.pt"))
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu", "auto"],
        default="cuda",
    )
    parser.add_argument(
        "--preview-dir",
        type=Path,
        default=None,
        help="Where to write rgb | truth | pred strips (default: next to --out)",
    )
    args = parser.parse_args()
    train_seg(
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
