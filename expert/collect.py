"""
Collect a KEEP lane-keeping dataset on Town01, then train.

Uses spawn_scout.json if present (prefers high-turn spawns), otherwise
scouts first. Records with lights ignored and no lane changes.

Requires CarlaUE4.exe running.

Example:
  python -m expert.collect --seconds 180 --drive
  python -m expert.collect --spawn-index 1 --dest-index -1 --seconds 180
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from expert.audit_data import score_run
from expert.drive import drive_run
from expert.record import parse_camera_names, record_run
from expert.train import train_model


def ranked_spawns(scout_path: Path, map_name: str) -> list[int]:
    if not scout_path.exists():
        return []
    payload = json.loads(scout_path.read_text())
    if payload.get("map") != map_name:
        return []
    ranked = payload.get("ranked") or []
    return [
        int(row["spawn_index"])
        for row in ranked
        if float(row.get("turn_pct", 0)) >= 5.0
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", dest="map_name", default="Town01")
    parser.add_argument(
        "--keep-map",
        action="store_true",
        help="Skip load_world; use whatever map Carla already has",
    )
    parser.add_argument("--seconds", type=float, default=180.0)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--model", type=Path, default=Path("models/bc.pt"))
    parser.add_argument("--scout", type=Path, default=Path("data/spawn_scout.json"))
    parser.add_argument("--spawn-index", type=int, default=None)
    parser.add_argument(
        "--dest-index",
        type=int,
        default=None,
        help="Fixed A→B dest spawn (-1 = auto balanced). Omit to use --route-mode.",
    )
    parser.add_argument(
        "--route-mode",
        choices=["balanced", "junction", "farthest"],
        default="balanced",
        help="How to pick courses (default: balanced left/right A→B)",
    )
    parser.add_argument(
        "--no-shuttle",
        action="store_true",
        help="Do not reverse A↔B when destination is reached",
    )
    parser.add_argument("--device", choices=["cuda", "cpu", "auto"], default="cuda")
    parser.add_argument(
        "--drive",
        action="store_true",
        help="Matched-spawn drive after training (waits for Enter)",
    )
    parser.add_argument("--drive-seconds", type=float, default=30.0)
    parser.add_argument(
        "--cameras",
        default="all",
        help="Record cameras: 'all' (default: center+left+right) or 'center'",
    )
    parser.add_argument(
        "--require-multicam",
        action="store_true",
        help="KEEP only if center+left+right exist (use with --cameras all)",
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
    args = parser.parse_args()
    camera_names = parse_camera_names(args.cameras)
    require_multicam = args.require_multicam or set(camera_names) >= {
        "center",
        "left",
        "right",
    }

    candidates = (
        [args.spawn_index]
        if args.spawn_index is not None
        else ranked_spawns(args.scout, args.map_name)
    )
    # Point-to-point: one fixed start → dest across the map.
    if args.dest_index is not None:
        start = args.spawn_index if args.spawn_index is not None else 0
        candidates = [start]
        print(
            f"Point-to-point collection: spawn {start} → dest {args.dest_index} "
            f"(shuttle={'off' if args.no_shuttle else 'on'})"
        )
    elif not candidates:
        candidates = [230, 203, 159, 1, 76]
        print(
            "No scout ranking for this map; trying known Town01 high-turn spawns: "
            + ", ".join(str(i) for i in candidates)
        )
    else:
        print(f"Spawn candidates: {candidates}")
    print(f"Route mode: {args.route_mode}")
    print(f"Cameras: {', '.join(camera_names)}")
    if args.keep_map:
        print("Keeping Carla's current map (--keep-map).")
    else:
        print(f"Will ensure map: {args.map_name}")

    run_dir = None
    report = None
    for spawn_index in candidates:
        print(f"\n=== Record map={args.map_name} spawn={spawn_index} ===")
        run_dir = record_run(
            seconds=args.seconds,
            fps=args.fps,
            out=args.data_dir,
            map_name=None if args.keep_map else args.map_name,
            spawn_index=spawn_index,
            dest_index=args.dest_index,
            shuttle=not args.no_shuttle,
            route_mode=args.route_mode,
            lane_change=False,
            ignore_lights=False,
            cameras=camera_names,
            traffic=0 if args.no_traffic else max(0, int(args.traffic)),
        )
        report = score_run(run_dir, require_multicam=require_multicam)
        status = "KEEP" if report.ok else "FAIL"
        print(
            f"[{status}] turn={report.turn_pct:.1f}% "
            f"stopped={report.stopped_pct} reasons={report.reasons or 'ok'}"
        )
        if report.ok:
            break
        print("Trying next spawn ...")

    if run_dir is None or report is None or not report.ok:
        raise SystemExit(
            "No KEEP recording. Start CARLA, or run: "
            "python -m expert.scout_spawns --map Town01"
        )

    print("\n=== Train ===")
    model_path = train_model(
        run_dir,
        epochs=args.epochs,
        device_name=args.device,
        out=args.model,
        oversample_turns=True,
    )

    print(f"\nKEEP run: {run_dir}")
    print(f"Model:    {model_path}")
    if args.drive:
        print("\n=== Matched drive ===")
        try:
            input("Press Enter to start matched drive (Ctrl+C to skip) ... ")
        except KeyboardInterrupt:
            print("\nSkipped drive.")
            return
        drive_run(
            model_path=model_path,
            seconds=args.drive_seconds,
            from_run=run_dir,
            device_name=args.device,
        )
        print(
            "Did it stay in lane ~20–30s without crashing? "
            "If yes, the data-collection goal is met."
        )
    else:
        print(
            f"Matched drive: python -m expert.drive --from-run {run_dir} "
            f"--model {model_path} --seconds {args.drive_seconds:.0f}"
        )


if __name__ == "__main__":
    main()
