"""
Record + train (+ optional matched drive) for lane-keeping data collection.

Defaults favor Town01, no lane changes, obey traffic lights, turn oversampling.

Examples:
  python -m expert.fit --spawn-index 25 --drive
  python -m expert.fit --map Town02 --seconds 240 --spawn-index 10 --drive
"""
from __future__ import annotations

import argparse
from pathlib import Path

from expert.drive import drive_run
from expert.record import record_run
from expert.train import train_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=180.0, help="Expert recording length")
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--model", type=Path, default=Path("models/bc.pt"))
    parser.add_argument("--tm-port", type=int, default=8000)
    parser.add_argument(
        "--map",
        dest="map_name",
        default="Town01",
        help="Map to load (default: Town01 for slower curves)",
    )
    parser.add_argument("--spawn-index", type=int, default=None)
    parser.add_argument(
        "--lane-change",
        action="store_true",
        help="Allow TM lane changes (default: stay in lane)",
    )
    parser.add_argument(
        "--ignore-lights",
        action="store_true",
        help="Ignore traffic lights while recording (default: obey)",
    )
    parser.add_argument(
        "--traffic",
        type=int,
        default=15,
        help="Spawn N NPC vehicles while recording (default: 15)",
    )
    parser.add_argument(
        "--no-traffic",
        action="store_true",
        help="Force empty road",
    )
    parser.add_argument(
        "--no-oversample-turns",
        action="store_true",
        help="Disable turn oversampling (default: on)",
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu", "auto"],
        default="cuda",
        help="Training / drive device (default: cuda)",
    )
    parser.add_argument(
        "--drive",
        action="store_true",
        help="After training, drive from the same map + spawn as the recording",
    )
    parser.add_argument(
        "--drive-seconds",
        type=float,
        default=60.0,
        help="How long to drive in matched eval (default: 60)",
    )
    args = parser.parse_args()

    print("=== 1/3 Record expert ===")
    print(f"Lane-keep defaults: map={args.map_name}, lane_change={args.lane_change}")
    run_dir = record_run(
        seconds=args.seconds,
        fps=args.fps,
        out=args.data_dir,
        tm_port=args.tm_port,
        map_name=args.map_name,
        spawn_index=args.spawn_index,
        lane_change=args.lane_change,
        ignore_lights=args.ignore_lights,
        traffic=0 if args.no_traffic else max(0, int(args.traffic)),
    )

    print("\n=== 2/3 Train model ===")
    model_path = train_model(
        run_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        out=args.model,
        device_name=args.device,
        oversample_turns=not args.no_oversample_turns,
    )

    if args.drive:
        print("\n=== 3/3 Matched drive (same spawn as recording) ===")
        print(f"Model: {model_path}")
        print(f"Run:   {run_dir}")
        try:
            input("Press Enter to start matched drive (Ctrl+C to skip) ... ")
        except KeyboardInterrupt:
            print("\nSkipped drive.")
            return
        drive_run(
            model_path=model_path,
            seconds=args.drive_seconds,
            fps=args.fps,
            from_run=run_dir,
            device_name=args.device,
            traffic=None,
            tm_port=args.tm_port,
        )
        print("\nDone. Matched eval finished.")
    else:
        print(f"\nDone. Model ready at {model_path}")
        print(
            f"Matched drive: python -m expert.drive --from-run {run_dir} "
            f"--model {model_path}"
        )
        print(f"Audit data:    python -m expert.audit_data --data {args.data_dir}")


if __name__ == "__main__":
    main()
