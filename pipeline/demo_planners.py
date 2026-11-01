"""Offline teaching demos for grid A* and RRT* (no CARLA required)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pipeline.motion import astar_grid, demo_occupancy, rrt_star_grid


def _render(occ: np.ndarray, path: list[tuple[int, int]] | None) -> str:
    grid = np.where(occ, "#", ".")
    if path:
        for y, x in path:
            if 0 <= y < grid.shape[0] and 0 <= x < grid.shape[1]:
                grid[y, x] = "*"
        sy, sx = path[0]
        gy, gx = path[-1]
        grid[sy, sx] = "S"
        grid[gy, gx] = "G"
    return "\n".join("".join(row) for row in grid)


def run_demo(*, out: Path) -> dict:
    occ = demo_occupancy()
    start = (2, 2)
    goal = (37, 37)
    a_path = astar_grid(occ, start, goal)
    r_path = rrt_star_grid(occ, start, goal, iterations=1200, seed=1)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "astar.txt").write_text(_render(occ, a_path), encoding="utf-8")
    (out / "rrt_star.txt").write_text(_render(occ, r_path), encoding="utf-8")
    summary = {
        "astar_len": None if a_path is None else len(a_path),
        "rrt_star_len": None if r_path is None else len(r_path),
        "start": start,
        "goal": goal,
        "grid": list(occ.shape),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote ASCII maps under {out}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("models/pipeline_preview/planner_demo"),
    )
    args = parser.parse_args()
    run_demo(out=args.out)


if __name__ == "__main__":
    main()
