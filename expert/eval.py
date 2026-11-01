"""
Phase 2 diagnostics: figure out if failures are arch/training vs data.

Examples:
  python -m expert.eval --data data/run_20260831_201524
  python -m expert.eval --data data/run_20260831_201524 --model models/bc.pt
"""
from __future__ import annotations

import argparse
import csv
import statistics as stats
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, random_split

from expert.model import DrivingDataset, PilotNet, load_checkpoint
from utils.device import describe_device, resolve_device


def dataset_stats(data: Path) -> None:
    csv_path = data / "controls_balanced.csv"
    if not csv_path.exists():
        csv_path = data / "controls.csv"
    rows = list(csv.DictReader(csv_path.open()))
    if rows and "camera" in rows[0]:
        rows = [r for r in rows if r.get("camera") == "center"] or rows
    steers = [float(r["steer"]) for r in rows]
    throttles = [float(r["throttle"]) for r in rows]
    brakes = [float(r["brake"]) for r in rows]
    abs_steers = [abs(s) for s in steers]

    print("=== Dataset stats ===")
    print(f"  frames: {len(rows)}  ({csv_path.name})")
    print(f"  steer:  mean={stats.mean(steers):+.3f}  std={stats.pstdev(steers):.3f}")
    print(
        f"  throttle: mean={stats.mean(throttles):.3f}  "
        f"brake: mean={stats.mean(brakes):.3f}"
    )
    sharp = sum(1 for s in abs_steers if s > 0.3)
    print(f"  sharp turns (|steer|>0.3): {sharp} ({100 * sharp / len(rows):.1f}%)")
    if rows and "command" in rows[0]:
        from collections import Counter

        counts = Counter(r["command"] for r in rows)
        total = sum(counts.values()) or 1
        mix = ", ".join(f"{k}={100 * v / total:.0f}%" for k, v in sorted(counts.items()))
        print(f"  commands: {mix}")


def overfit_test(
    data: Path, *, n_frames: int = 32, epochs: int = 100, device_name: str = "cuda"
) -> float:
    device = resolve_device(device_name)
    dataset = DrivingDataset(data)
    n = min(n_frames, len(dataset))
    subset = Subset(dataset, range(n))
    loader = DataLoader(subset, batch_size=min(32, n), shuffle=True)

    model = PilotNet(
        use_command=True,
        stack_frames=dataset.stack_frames,
        use_light=True,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    print(f"\n=== Test 1: Overfit {n} frames ({epochs} epochs) ===")
    print(f"  device: {describe_device(device)}")
    final_loss = float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for images, speed, command, light, radar, controls in loader:
            images = images.to(device)
            speed = speed.to(device)
            command = command.to(device)
            light = light.to(device)
            radar = radar.to(device)
            controls = controls.to(device)
            pred = model(images, speed, command, light, radar)
            loss = loss_fn(pred, controls)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * images.size(0)
        final_loss = epoch_loss / n
        if epoch in {1, 10, 50, 100} or epoch == epochs:
            print(f"  epoch {epoch:03d}  loss={final_loss:.6f}")

    if final_loss < 0.005:
        print("  PASS — model can memorize a tiny batch (arch/training OK)")
    else:
        print("  FAIL — could not overfit 32 frames (check training/arch/labels)")
    return final_loss


def load_model(model_path: Path, device: torch.device) -> PilotNet:
    return load_checkpoint(model_path, device)


def offline_eval(
    data: Path,
    model: PilotNet,
    device: torch.device,
    *,
    val_fraction: float = 0.1,
) -> tuple[float, list[torch.Tensor]]:
    dataset = DrivingDataset(data)
    val_size = max(1, int(val_fraction * len(dataset)))
    train_size = len(dataset) - val_size
    _, val_ds = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    loader = DataLoader(val_ds, batch_size=64)

    loss_fn = nn.MSELoss(reduction="none")
    total_loss = 0.0
    preds: list[torch.Tensor] = []

    print(f"\n=== Test 2: Offline eval ({val_size} held-out frames) ===")
    with torch.no_grad():
        for images, speed, command, light, radar, controls in loader:
            images = images.to(device)
            speed = speed.to(device)
            command = command.to(device)
            light = light.to(device)
            radar = radar.to(device)
            controls = controls.to(device)
            if model.use_command:
                pred = model(
                    images,
                    speed,
                    command,
                    light if model.use_light else None,
                    radar if model.use_radar else None,
                )
            else:
                pred = model(
                    images,
                    speed,
                    None,
                    light if model.use_light else None,
                    radar if model.use_radar else None,
                )
            total_loss += loss_fn(pred, controls).mean().item() * images.size(0)
            preds.append(pred.cpu())

    mse = total_loss / val_size
    if mse < 0.02:
        print(f"  PASS — val MSE={mse:.5f} (matches expert labels on saved images)")
    elif mse < 0.05:
        print(f"  OK   — val MSE={mse:.5f} (decent but not tight)")
    else:
        print(f"  WEAK — val MSE={mse:.5f} (model not fitting the data well)")
    return mse, preds


def prediction_spread(preds: list[torch.Tensor]) -> tuple[float, float, float]:
    all_preds = torch.cat(preds, dim=0)
    steer_std = all_preds[:, 0].std().item()
    throttle_std = all_preds[:, 1].std().item()
    brake_std = all_preds[:, 2].std().item()

    print("\n=== Test 3: Prediction spread (held-out set) ===")
    print(f"  steer std={steer_std:.4f}  throttle std={throttle_std:.4f}  brake std={brake_std:.4f}")

    collapsed = steer_std < 0.02 and throttle_std < 0.02
    if collapsed:
        print("  FAIL — predictions barely change (mean-collapse)")
    else:
        print("  PASS — predictions vary across frames")
    return steer_std, throttle_std, brake_std


def print_verdict(
    overfit_loss: float | None,
    offline_mse: float | None,
    steer_std: float | None,
) -> None:
    print("\n=== Verdict ===")
    if overfit_loss is not None and overfit_loss >= 0.005:
        print("  Training/arch bug — fix overfit before collecting more data.")
        return

    if steer_std is not None and steer_std < 0.02:
        print("  Mean collapse — model outputs nearly constant controls.")
        print("  Try: more moving frames, check labels, retrain without throttle nudge.")
        return

    if offline_mse is not None and offline_mse < 0.03:
        print("  Arch is fine on recorded data.")
        print("  If drive still crashes → data / distribution shift:")
        print("    - record more turns and recovery states")
        print("    - multiple spawns/routes")
        print("    - re-record with speed_kmh (python -m expert.record)")
        return

    if offline_mse is not None:
        print("  Model underfits your dataset — train longer or record cleaner data.")
        return

    print("  Overfit passed. Train a model, then re-run with --model for full diagnosis.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--overfit-epochs", type=int, default=100)
    parser.add_argument("--skip-overfit", action="store_true")
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu", "auto"],
        default="cuda",
    )
    args = parser.parse_args()

    device = resolve_device(args.device)
    print(f"Device: {describe_device(device)}\n")

    dataset_stats(args.data)

    overfit_loss = None
    if not args.skip_overfit:
        overfit_loss = overfit_test(
            args.data, epochs=args.overfit_epochs, device_name=args.device
        )

    offline_mse = None
    steer_std = None
    if args.model is not None:
        if not args.model.exists():
            raise FileNotFoundError(f"Model not found: {args.model}")
        model = load_model(args.model, device)
        offline_mse, preds = offline_eval(args.data, model, device)
        steer_std, _, _ = prediction_spread(preds)
    else:
        print("\n(Skip tests 2–3: pass --model models/bc.pt after training)")

    print_verdict(overfit_loss, offline_mse, steer_std)


if __name__ == "__main__":
    main()
