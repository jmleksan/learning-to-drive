"""
Train TransFuser-lite on recorded runs.

Hold out whole runs for validation (frames inside one run are near-duplicates).

Examples:
  python -m transfuser.train --data data/run_A data/run_B --val data/run_C
  python -m transfuser.train --data data/run_* --val data/run_C --modality lidar --out models/tf_lidar.pt
  python -m transfuser.train --data data/run_* --val data/run_C --no-aux --out models/tf_noaux.pt
  python -m transfuser.train --data data/run_* --val data/run_C --no-pretrained --out models/tf_scratch.pt

The GPU can die with "CUDA error: unknown error" on a long run. --auto-resume
continues from <out>_last.pt, saved every --save-every batches, so a shell loop
restarts it until it finishes:
  until python -m transfuser.train --data data/run_* --val data/run_C --auto-resume; do sleep 30; done
"""
from __future__ import annotations

import argparse
import json
import math
import time
from itertools import islice
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from transfuser.data import IGNORE, TransfuserDataset
from transfuser.model import BACKBONES, MODALITIES, TransFuserLite, save_checkpoint
from utils.device import describe_device, resolve_device

# Loss weights. Waypoint L1 is in meters.
W_WAYPOINT = 1.0
W_IMG_SEG = 0.5
W_BEV_SEG = 0.5
W_AFFORDANCE = 0.2


def to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def run_model(model: TransFuserLite, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return model(
        batch["image"] if model.use_camera else None,
        batch["lidar"] if model.use_lidar else None,
        batch["speed"],
        batch["target"],
    )


def compute_losses(
    out: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    losses = {"waypoint": F.l1_loss(out["waypoints"].float(), batch["waypoints"])}
    if "img_seg" in out and "img_seg" in batch:
        losses["img_seg"] = F.cross_entropy(out["img_seg"].float(), batch["img_seg"])
    if "bev_seg" in out:
        losses["bev_seg"] = F.cross_entropy(
            out["bev_seg"].float(), batch["bev_seg"], ignore_index=IGNORE
        )
    if "affordance" in out:
        losses["affordance"] = F.binary_cross_entropy_with_logits(
            out["affordance"].float(), batch["affordance"]
        )
    weights = {
        "waypoint": W_WAYPOINT,
        "img_seg": W_IMG_SEG,
        "bev_seg": W_BEV_SEG,
        "affordance": W_AFFORDANCE,
    }
    losses["total"] = sum(weights[k] * v for k, v in losses.items())
    return losses


@torch.no_grad()
def validate(model: TransFuserLite, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    sums: dict[str, float] = {}
    n = 0
    for batch in loader:
        batch = to_device(batch, device)
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            out = run_model(model, batch)
        losses = compute_losses(out, batch)
        err = torch.linalg.norm(out["waypoints"].float() - batch["waypoints"], dim=-1)
        bs = batch["waypoints"].shape[0]
        for k, v in losses.items():
            sums[k] = sums.get(k, 0.0) + float(v) * bs
        sums["ade_m"] = sums.get("ade_m", 0.0) + float(err.mean(dim=1).sum())
        sums["fde_m"] = sums.get("fde_m", 0.0) + float(err[:, -1].sum())
        n += bs
    return {k: v / max(1, n) for k, v in sums.items()}


def train(
    *,
    data: list[Path],
    val: list[Path],
    out: Path,
    modality: str = "both",
    aux: bool = True,
    backbone: str = "resnet18",
    pretrained: bool = True,
    epochs: int = 30,
    batch_size: int = 32,
    lr: float | None = None,
    stride: int = 2,
    workers: int = 6,
    device_name: str = "auto",
    auto_resume: bool = False,
    save_every: int = 250,
    junction_weight: float = 2.0,
) -> Path:
    device = resolve_device(device_name)
    if lr is None:
        # Lower for resnet18 so early updates don't wreck the ImageNet features.
        lr = 1e-4 if backbone == "resnet18" else 3e-4
    val_set = {p.resolve() for p in val}
    train_runs = [p for p in data if p.resolve() not in val_set]
    if not train_runs:
        raise SystemExit("Every --data run is also in --val; nothing left to train on.")
    train_ds = TransfuserDataset(train_runs, stride=stride)
    val_ds = TransfuserDataset(val, stride=stride)
    print(
        f"device={describe_device(device)}  modality={modality}  aux={aux}  "
        f"backbone={backbone}{' (ImageNet camera)' if pretrained and backbone == 'resnet18' else ''}  lr={lr:g}\n"
        f"train: {len(train_ds)} frames from {len(train_runs)} runs   "
        f"val: {len(val_ds)} frames from {len(val)} runs"
    )
    if aux and not train_ds.has_seg:
        print("No seg/ in every run: image-semantics aux loss is off.")

    loader_kw = dict(
        batch_size=batch_size,
        num_workers=workers,
        # Off: CUDA aborts on this GPU under WSL surfaced in the pinned allocator.
        pin_memory=False,
        persistent_workers=workers > 0,
    )
    weights = train_ds.junction_weights(junction_weight)
    print(f"junction frames: {sum(w > 1 for w in weights)}/{len(weights)}, sampled ×{junction_weight:g}")
    sampler = WeightedRandomSampler(weights, len(weights))
    train_loader = DataLoader(train_ds, sampler=sampler, drop_last=True, **loader_kw)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kw)

    model = TransFuserLite(
        modality=modality, aux=aux, backbone=backbone, pretrained=pretrained
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params / 1e6:.1f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps = epochs * len(train_loader)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: 0.5 * (1.0 + math.cos(math.pi * min(1.0, s / max(1, steps))))
    )
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    last_path = out.with_name(out.stem + "_last.pt")
    history = []
    best = float("inf")
    start_epoch = 1
    start_batch = 0
    if auto_resume and last_path.exists():
        ckpt = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        opt.load_state_dict(ckpt["optimizer"])
        sched.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        history = ckpt["history"]
        best = ckpt["best"]
        start_epoch = ckpt["epoch"] + 1
        start_batch = ckpt["batch"]
        print(f"Resuming {last_path} at epoch {start_epoch}/{epochs}, batch {start_batch}")

    def save_last(done_epochs: int, batch_index: int, **meta) -> None:
        save_checkpoint(
            last_path,
            model,
            optimizer=opt.state_dict(),
            scheduler=sched.state_dict(),
            scaler=scaler.state_dict(),
            history=history,
            best=best,
            batch=batch_index,
            epoch=done_epochs,
            **meta,
        )

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        t0 = time.time()
        sums: dict[str, float] = {}
        n_train = len(train_loader)
        # A resumed epoch reshuffles and runs only the batches it still owes.
        batches = enumerate(islice(train_loader, n_train - start_batch), start_batch + 1)
        n_seen = 0
        for batch_index, batch in tqdm(
            batches, desc=f"epoch {epoch}/{epochs}", leave=False, total=n_train, initial=start_batch
        ):
            batch = to_device(batch, device)
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                out_ = run_model(model, batch)
            losses = compute_losses(out_, batch)
            opt.zero_grad(set_to_none=True)
            scaler.scale(losses["total"]).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            for k, v in losses.items():
                sums[k] = sums.get(k, 0.0) + float(v.detach())
            n_seen += 1
            if save_every and batch_index % save_every == 0 and batch_index < n_train:
                save_last(epoch - 1, batch_index)
        start_batch = 0
        train_stats = {k: v / max(1, n_seen) for k, v in sums.items()}
        val_stats = validate(model, val_loader, device)
        history.append({"epoch": epoch, "train": train_stats, "val": val_stats})
        print(
            f"epoch {epoch:3d}  {time.time() - t0:5.0f}s  "
            f"train wp={train_stats['waypoint']:.3f}  "
            f"val wp={val_stats['waypoint']:.3f} ade={val_stats['ade_m']:.2f}m "
            f"fde={val_stats['fde_m']:.2f}m  "
            + "  ".join(
                f"{k}={val_stats[k]:.3f}" for k in ("img_seg", "bev_seg", "affordance") if k in val_stats
            )
        )
        meta = dict(
            val=val_stats,
            train_runs=[str(p) for p in train_runs],
            val_runs=[str(p) for p in val],
        )
        # Select on waypoint error only, so aux / no-aux checkpoints compare fairly.
        if val_stats["waypoint"] < best:
            best = val_stats["waypoint"]
            save_checkpoint(out, model, epoch=epoch, **meta)
        save_last(epoch, 0, **meta)
        out.with_suffix(".json").write_text(json.dumps(history, indent=2))
    print(f"best val waypoint L1 {best:.3f} m → {out}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, nargs="+", required=True)
    parser.add_argument("--val", type=Path, nargs="+", required=True, help="Held-out runs (dropped from --data)")
    parser.add_argument("--out", type=Path, default=Path("models/transfuser.pt"))
    parser.add_argument("--modality", choices=MODALITIES, default="both")
    parser.add_argument("--no-aux", action="store_true", help="Waypoint loss only")
    parser.add_argument("--backbone", choices=BACKBONES, default="resnet18")
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="resnet18 camera branch from random init instead of ImageNet",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=None, help="Default 1e-4 (resnet18) / 3e-4 (small)")
    parser.add_argument("--stride", type=int, default=2, help="Use every Nth frame (20 fps → 10 Hz)")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--auto-resume", action="store_true", help="Continue from <out>_last.pt if it exists")
    parser.add_argument("--save-every", type=int, default=250, help="Batches between mid-epoch resume saves (0 = off)")
    parser.add_argument("--junction-weight", type=float, default=2.0, help="Sampling weight of frames near a turn")
    args = parser.parse_args()
    train(
        data=args.data,
        val=args.val,
        out=args.out,
        modality=args.modality,
        aux=not args.no_aux,
        backbone=args.backbone,
        pretrained=not args.no_pretrained,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        stride=args.stride,
        workers=args.workers,
        device_name=args.device,
        auto_resume=args.auto_resume,
        save_every=args.save_every,
        junction_weight=args.junction_weight,
    )


if __name__ == "__main__":
    main()
