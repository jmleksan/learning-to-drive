"""
Train a behavioral cloning model from recorded expert runs.

Examples:
  python -m expert.train --data data/run_20260101_120000 --epochs 10
  python -m expert.train --data data/ --oversample-turns
  python -m expert.train --data data/run_a data/run_b --epochs 15
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split
from tqdm import tqdm

from expert.model import (
    STACK_FRAMES,
    AugmentedSubset,
    DrivingDataset,
    PilotNet,
    save_checkpoint,
)
from utils.device import describe_device, resolve_device


def train_model(
    data: Path | list[Path],
    *,
    epochs: int = 10,
    batch_size: int = 64,
    lr: float = 1e-3,
    out: Path = Path("models/bc.pt"),
    device_name: str = "cuda",
    center_only: bool = False,
    oversample_turns: bool = False,
    turn_boost: float = 5.0,
    turn_threshold: float = 0.15,
    oversample_lights: bool = True,
    light_boost: float = 2.0,
    light_go_boost: float = 1.8,
    light_wait_weight: float = 0.35,
    perspective_aug: bool = True,
    max_shift_px: int = 16,
    max_steer_nudge: float = 0.10,
    command_dropout_p: float = 0.0,
    stack_frames: int = STACK_FRAMES,
    oversample_dagger: bool | None = None,
    dagger_boost: float = 4.0,
    downweight_traffic_follow: bool | None = None,
    traffic_follow_weight: float = 0.25,
    use_radar: bool = False,
) -> Path:
    device = resolve_device(device_name)
    use_cuda = device.type == "cuda"
    paths = [data] if isinstance(data, Path) else data
    dataset = DrivingDataset(
        paths, center_only=center_only, stack_frames=stack_frames
    )
    turns, total = dataset.turn_stats(steer_threshold=turn_threshold)
    print(
        f"Turn frames (|steer|>={turn_threshold}): {turns}/{total} "
        f"({100 * turns / total:.1f}%) across {len(dataset.run_dirs)} run(s)"
    )
    dagger_n, dagger_total = dataset.dagger_stats()
    if dagger_n:
        print(
            f"DAgger correction frames: {dagger_n}/{dagger_total} "
            f"({100 * dagger_n / dagger_total:.1f}%)"
        )
    # Auto-on when dagger_* samples exist unless explicitly disabled.
    if oversample_dagger is None:
        oversample_dagger = dagger_n > 0
    traffic_n, traffic_total = dataset.traffic_follow_stats()
    if traffic_n:
        print(
            f"Traffic-run frames: {traffic_n}/{traffic_total} "
            f"({100 * traffic_n / traffic_total:.1f}%) "
            f"from {len(dataset.traffic_runs)} traffic run(s)"
        )
    # Auto-on when mixing traffic + empty runs unless explicitly disabled.
    if downweight_traffic_follow is None:
        downweight_traffic_follow = (
            traffic_n > 0 and traffic_n < traffic_total
        )
    if use_radar and dataset.missing_radar_traffic_runs:
        print(
            "WARNING: these traffic runs have no radar_m column, so a car "
            "ahead looks like a clear lane: "
            + ", ".join(dataset.missing_radar_traffic_runs)
        )
        print("Re-record them before training a radar brake head.")
    if center_only:
        print("Using center camera only")
    else:
        print("Using all cameras in controls.csv (center+left+right when present)")
    cmd_counts = dataset.command_stats()
    cmd_total = sum(cmd_counts.values()) or 1
    mix = ", ".join(
        f"{name}={100 * n / cmd_total:.0f}%"
        for name, n in cmd_counts.items()
        if n > 0
    )
    print(f"Command mix (one-hot input): {mix}")
    light_counts = dataset.light_stats()
    light_total = sum(light_counts.values()) or 1
    light_mix = ", ".join(
        f"{name}={100 * n / light_total:.0f}%"
        for name, n in light_counts.items()
        if n > 0
    )
    print(f"Light mix (one-hot input): {light_mix}")
    if perspective_aug:
        print(
            f"Perspective aug ON (train only): ±{max_shift_px}px shift, "
            f"steer nudge up to ±{max_steer_nudge:.2f}"
        )
    if command_dropout_p > 0:
        print(
            f"Command dropout ON (train only): p={command_dropout_p:.2f} "
            f"on |steer|>=0.12 → FOLLOW_LANE (image must recover/finish turn)"
        )

    val_size = max(1, int(0.1 * len(dataset)))
    train_size = len(dataset) - val_size
    train_base, val_base = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    train_ds = AugmentedSubset(
        dataset,
        list(train_base.indices),
        augment=perspective_aug or command_dropout_p > 0,
        max_shift_px=max_shift_px if perspective_aug else 0,
        max_steer_nudge=max_steer_nudge,
        command_dropout_p=command_dropout_p,
    )
    val_ds = AugmentedSubset(
        dataset,
        list(val_base.indices),
        augment=False,
    )

    loader_kwargs: dict = {"batch_size": batch_size}
    if use_cuda:
        loader_kwargs["pin_memory"] = True
        loader_kwargs["num_workers"] = 2

    if (
        oversample_turns
        or oversample_lights
        or oversample_dagger
        or downweight_traffic_follow
    ):
        weights = torch.ones(len(dataset), dtype=torch.float64)
        if oversample_turns:
            weights = weights * dataset.turn_sample_weights(
                steer_threshold=turn_threshold,
                boost=turn_boost,
            )
            print(
                f"Oversampling turns (boost={turn_boost}x for |steer|>={turn_threshold})"
            )
        if oversample_lights:
            weights = weights * dataset.light_stop_sample_weights(
                red_approach_boost=light_boost,
                red_wait_weight=light_wait_weight,
                go_boost=light_go_boost,
            )
            print(
                f"Oversampling lights: red/yellow approach×{light_boost:.1f}, "
                f"red wait×{light_wait_weight:.2f}, "
                f"green/Off go×{light_go_boost:.1f}"
            )
        if oversample_dagger:
            weights = weights * dataset.dagger_sample_weights(boost=dagger_boost)
            print(
                f"Oversampling DAgger corrections "
                f"(boost={dagger_boost:.1f}x on dagger_* frames)"
            )
        if downweight_traffic_follow:
            weights = weights * dataset.traffic_follow_sample_weights(
                weight=traffic_follow_weight
            )
            print(
                f"Downweighting traffic FOLLOW/GO cruise "
                f"(×{traffic_follow_weight:.2f}; keep brake/turns)"
            )
        train_indices = train_ds.indices
        sampler = WeightedRandomSampler(
            weights[train_indices],
            num_samples=len(train_indices),
            replacement=True,
        )
        train_loader = DataLoader(train_ds, sampler=sampler, **loader_kwargs)
    else:
        train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)

    val_loader = DataLoader(val_ds, **loader_kwargs)

    model = PilotNet(
        use_command=True,
        branched=True,
        stack_frames=dataset.stack_frames,
        use_light=True,
        use_radar=use_radar,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    radar_bit = "+radar(range)" if model.use_radar else ""
    print(
        f"Samples: {len(dataset)} | device: {describe_device(device)} | "
        f"inputs: image+speed+command({model.n_commands}-way branched CIL)"
        f"+light({model.n_lights}-way){radar_bit}"
    )
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_bar = tqdm(
            train_loader,
            desc=f"epoch {epoch:02d}/{epochs} train",
            leave=False,
            unit="batch",
        )
        for images, speed, command, light, radar, controls in train_bar:
            images = images.to(device, non_blocking=use_cuda)
            speed = speed.to(device, non_blocking=use_cuda)
            command = command.to(device, non_blocking=use_cuda)
            light = light.to(device, non_blocking=use_cuda)
            radar = radar.to(device, non_blocking=use_cuda)
            controls = controls.to(device, non_blocking=use_cuda)
            pred = model(images, speed, command, light, radar)
            loss = loss_fn(pred, controls)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item() * images.size(0)
            train_bar.set_postfix(loss=f"{loss.item():.4f}")
        train_loss /= train_size

        model.eval()
        val_loss = 0.0
        val_bar = tqdm(
            val_loader,
            desc=f"epoch {epoch:02d}/{epochs} val",
            leave=False,
            unit="batch",
        )
        with torch.no_grad():
            for images, speed, command, light, radar, controls in val_bar:
                images = images.to(device, non_blocking=use_cuda)
                speed = speed.to(device, non_blocking=use_cuda)
                command = command.to(device, non_blocking=use_cuda)
                light = light.to(device, non_blocking=use_cuda)
                radar = radar.to(device, non_blocking=use_cuda)
                controls = controls.to(device, non_blocking=use_cuda)
                batch_loss = loss_fn(
                    model(images, speed, command, light, radar), controls
                ).item()
                val_loss += batch_loss * images.size(0)
                val_bar.set_postfix(loss=f"{batch_loss:.4f}")
        val_loss /= val_size
        print(f"epoch {epoch:02d}  train={train_loss:.5f}  val={val_loss:.5f}")

    save_checkpoint(model, out)
    print(f"Saved {out}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        nargs="+",
        required=True,
        help="Run folder(s) or parent data/ directory",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out", type=Path, default=Path("models/bc.pt"))
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu", "auto"],
        default="cuda",
        help="Training device (default: cuda)",
    )
    parser.add_argument(
        "--oversample-turns",
        action="store_true",
        help="Sample turning frames more often during training",
    )
    parser.add_argument("--turn-boost", type=float, default=5.0)
    parser.add_argument("--turn-threshold", type=float, default=0.15)
    parser.add_argument(
        "--no-oversample-lights",
        action="store_true",
        help="Disable red/yellow light frame oversampling (on by default)",
    )
    parser.add_argument("--light-boost", type=float, default=2.0)
    parser.add_argument(
        "--light-go-boost",
        type=float,
        default=1.8,
        help="Mild boost for green/Off frames that are actually moving (default 1.8)",
    )
    parser.add_argument(
        "--light-wait-weight",
        type=float,
        default=0.35,
        help="Downweight red/yellow frames while stopped (default 0.35)",
    )
    parser.add_argument(
        "--center-only",
        action="store_true",
        help="Train on center camera only (default: use left/right recovery frames too)",
    )
    parser.add_argument(
        "--no-perspective-aug",
        action="store_true",
        help="Disable synthetic left/right shift recovery aug (on by default)",
    )
    parser.add_argument("--max-shift-px", type=int, default=16)
    parser.add_argument("--max-steer-nudge", type=float, default=0.10)
    parser.add_argument(
        "--command-dropout",
        type=float,
        default=0.0,
        help=(
            "Train-time prob of replacing command with FOLLOW_LANE on "
            "|steer|>=0.12 frames (0=off, default). Try 0.1–0.15 only if "
            "lane-keep is solid and mid-turn recovery is still weak."
        ),
    )
    parser.add_argument(
        "--stack-frames",
        type=int,
        default=STACK_FRAMES,
        help="RGB frames to stack as input channels (default: 3 → 9-ch)",
    )
    parser.add_argument(
        "--no-oversample-dagger",
        action="store_true",
        help="Disable auto DAgger oversampling when dagger_* runs are present",
    )
    parser.add_argument(
        "--dagger-boost",
        type=float,
        default=4.0,
        help="Sample weight for dagger_* frames (default 4.0; auto-on if present)",
    )
    parser.add_argument(
        "--downweight-traffic-follow",
        action="store_true",
        default=None,
        help=(
            "Downweight FOLLOW/GO cruise frames from traffic runs "
            "(auto-on when mixing traffic + empty runs)"
        ),
    )
    parser.add_argument(
        "--no-downweight-traffic-follow",
        action="store_true",
        help="Disable traffic FOLLOW downweight even when mixing runs",
    )
    parser.add_argument(
        "--traffic-follow-weight",
        type=float,
        default=0.25,
        help="Sample weight for traffic FOLLOW/GO cruise (default 0.25)",
    )
    parser.add_argument(
        "--radar",
        action="store_true",
        help=(
            "Train throttle/brake on the logged radar range. "
            "Steer still ignores it. Needs a new recording that has radar_m."
        ),
    )
    args = parser.parse_args()
    if args.no_downweight_traffic_follow:
        downweight_traffic_follow: bool | None = False
    elif args.downweight_traffic_follow:
        downweight_traffic_follow = True
    else:
        downweight_traffic_follow = None
    train_model(
        args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        out=args.out,
        device_name=args.device,
        center_only=args.center_only,
        oversample_turns=args.oversample_turns,
        turn_boost=args.turn_boost,
        turn_threshold=args.turn_threshold,
        oversample_lights=not args.no_oversample_lights,
        light_boost=args.light_boost,
        light_go_boost=args.light_go_boost,
        light_wait_weight=args.light_wait_weight,
        perspective_aug=not args.no_perspective_aug,
        max_shift_px=args.max_shift_px,
        max_steer_nudge=args.max_steer_nudge,
        command_dropout_p=args.command_dropout,
        stack_frames=args.stack_frames,
        oversample_dagger=False if args.no_oversample_dagger else None,
        dagger_boost=args.dagger_boost,
        downweight_traffic_follow=downweight_traffic_follow,
        traffic_follow_weight=args.traffic_follow_weight,
        use_radar=args.radar,
    )


if __name__ == "__main__":
    main()
