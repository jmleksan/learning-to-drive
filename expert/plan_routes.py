"""
Preview balanced A→B courses without recording (needs CARLA running).

Examples:
  python -m expert.plan_routes --map Town01 --top 8
  python -m expert.plan_routes --map Town01 --spawn-index 1 --top 5
"""
from __future__ import annotations

import argparse
import random

from utils.connection import connect_client
from utils.routing import (
    analyze_route_turns,
    build_planner,
    farthest_spawn_index,
    pick_balanced_route,
    plan_spawn_to_spawn,
)
from utils.run_meta import ensure_map


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", dest="map_name", default="Town01")
    parser.add_argument("--spawn-index", type=int, default=None)
    parser.add_argument("--top", type=int, default=8)
    parser.add_argument("--pair-samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    client = connect_client(timeout=30.0)
    world = ensure_map(client, args.map_name)
    spawn_pts = world.get_map().get_spawn_points()
    print(f"Map {args.map_name}: {len(spawn_pts)} spawns. Scoring routes ...")
    grp = build_planner(world.get_map())
    rng = random.Random(args.seed)

    ranked: list[tuple[float, int, int, dict]] = []
    starts = (
        [args.spawn_index]
        if args.spawn_index is not None
        else rng.sample(range(len(spawn_pts)), min(12, len(spawn_pts)))
    )
    for oi in starts:
        far = [
            i
            for i, sp in enumerate(spawn_pts)
            if i != oi and spawn_pts[oi].location.distance(sp.location) >= 100.0
        ]
        if len(far) < 4:
            far = [i for i in range(len(spawn_pts)) if i != oi]
        sample = far if len(far) <= args.pair_samples else rng.sample(far, args.pair_samples)
        sample.append(farthest_spawn_index(spawn_pts, oi))
        for di in set(sample):
            try:
                _dest, route, _ = plan_spawn_to_spawn(grp, spawn_pts, oi, di)
            except Exception:
                continue
            stats = analyze_route_turns(route)
            ranked.append((float(stats["score"]), oi, di, stats))

    ranked.sort(key=lambda row: row[0], reverse=True)
    print(f"\nTop {args.top} balanced courses:")
    for score, oi, di, stats in ranked[: args.top]:
        print(
            f"  spawn {oi:3d} → {di:3d}  "
            f"L={stats['left']} R={stats['right']} S={stats['straight']}  "
            f"len={stats['length_m']:.0f}m  score={score:.2f}"
        )
        print(
            f"    record: python -m expert.record --map {args.map_name} "
            f"--spawn-index {oi} --dest-index {di} --seconds 180"
        )

    print("\nBest overall via picker:")
    oi, di, _dest, _route, stats = pick_balanced_route(
        grp,
        spawn_pts,
        start_index=args.spawn_index,
        pair_samples=args.pair_samples,
        rng=rng,
    )
    print(
        f"  spawn {oi} → {di}  L={stats['left']} R={stats['right']} "
        f"len={stats['length_m']:.0f}m score={stats['score']:.2f}"
    )


if __name__ == "__main__":
    main()
