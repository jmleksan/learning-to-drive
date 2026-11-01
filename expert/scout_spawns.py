"""
Scout spawn points for high turn coverage (no camera — fast).

Ranks spawns by |steer|>=0.15 fraction under Traffic Manager.

Example:
  python -m expert.scout_spawns --map Town01 --probe-seconds 20 --top 8
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from utils.connection import connect_client
from utils.run_meta import ensure_map
from utils.state import speed_kmh
from utils.sync_mode import SynchronousMode
from utils.vehicles import spawn_vehicle


@dataclass
class SpawnScore:
    spawn_index: int
    frames: int
    turn_pct: float
    sharp_pct: float
    max_abs_steer: float
    mean_speed: float
    stopped_pct: float


def probe_spawn(
    client,
    world,
    tm,
    *,
    spawn_index: int,
    fps: float,
    probe_seconds: float,
    tm_port: int,
    lane_change: bool,
    ignore_lights: bool,
) -> SpawnScore:
    frames = int(probe_seconds * fps)
    vehicle = spawn_vehicle(world, spawn_index=spawn_index)
    try:
        vehicle.set_autopilot(True, tm_port)
        light_pct = 100.0 if ignore_lights else 0.0
        tm.ignore_lights_percentage(vehicle, light_pct)
        tm.ignore_signs_percentage(vehicle, light_pct)
        tm.auto_lane_change(vehicle, lane_change)

        steers: list[float] = []
        speeds: list[float] = []
        for _ in range(frames):
            world.tick()
            control = vehicle.get_control()
            steers.append(abs(control.steer))
            speeds.append(speed_kmh(vehicle))
    finally:
        if vehicle.is_alive:
            try:
                vehicle.destroy()
            except RuntimeError:
                pass

    n = len(steers) or 1
    turns = sum(1 for s in steers if s >= 0.15)
    sharp = sum(1 for s in steers if s >= 0.3)
    stopped = sum(1 for v in speeds if v < 1.0)
    return SpawnScore(
        spawn_index=spawn_index,
        frames=len(steers),
        turn_pct=100.0 * turns / n,
        sharp_pct=100.0 * sharp / n,
        max_abs_steer=max(steers) if steers else 0.0,
        mean_speed=sum(speeds) / n,
        stopped_pct=100.0 * stopped / n,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", dest="map_name", default="Town01")
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--probe-seconds", type=float, default=20.0)
    parser.add_argument("--tm-port", type=int, default=8000)
    parser.add_argument("--samples", type=int, default=12, help="How many spawns to probe")
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lane-change", action="store_true")
    parser.add_argument("--ignore-lights", action="store_true")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/spawn_scout.json"),
        help="Write ranked results JSON here",
    )
    args = parser.parse_args()

    client = connect_client()
    world = ensure_map(client, args.map_name)
    spawn_count = len(world.get_map().get_spawn_points())
    rng = random.Random(args.seed)
    indices = list(range(spawn_count))
    rng.shuffle(indices)
    indices = indices[: min(args.samples, spawn_count)]

    print(
        f"Probing {len(indices)}/{spawn_count} spawns for "
        f"{args.probe_seconds:.0f}s each (no camera) ..."
    )
    scores: list[SpawnScore] = []
    with SynchronousMode(client, world, fps=args.fps, tm_port=args.tm_port) as (
        world,
        tm,
    ):
        for i, spawn_index in enumerate(indices, start=1):
            score = probe_spawn(
                client,
                world,
                tm,
                spawn_index=spawn_index,
                fps=args.fps,
                probe_seconds=args.probe_seconds,
                tm_port=args.tm_port,
                lane_change=args.lane_change,
                ignore_lights=args.ignore_lights,
            )
            scores.append(score)
            flag = "KEEP?" if score.turn_pct >= 5.0 else "low"
            print(
                f"  [{i}/{len(indices)}] spawn={spawn_index:3d}  "
                f"turn={score.turn_pct:5.1f}% sharp={score.sharp_pct:4.1f}%  "
                f"max|steer|={score.max_abs_steer:.2f}  "
                f"speed={score.mean_speed:4.1f}  stopped={score.stopped_pct:4.1f}%  {flag}"
            )

    scores.sort(key=lambda s: (s.turn_pct, s.sharp_pct, s.max_abs_steer), reverse=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "map": args.map_name,
        "probe_seconds": args.probe_seconds,
        "ranked": [asdict(s) for s in scores],
    }
    args.out.write_text(json.dumps(payload, indent=2) + "\n")

    print(f"\nTop {args.top} spawns for {args.map_name}:")
    for score in scores[: args.top]:
        print(
            f"  spawn={score.spawn_index}  turn={score.turn_pct:.1f}%  "
            f"sharp={score.sharp_pct:.1f}%  max|steer|={score.max_abs_steer:.2f}"
        )
    if scores and scores[0].turn_pct >= 5.0:
        best = scores[0].spawn_index
        print(
            f"\nNext:\n  python -m expert.fit --map {args.map_name} "
            f"--spawn-index {best} --drive"
        )
    else:
        print(
            "\nNo spawn hit >=5% turns in this sample. "
            "Try --samples 20 or --probe-seconds 30."
        )
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
