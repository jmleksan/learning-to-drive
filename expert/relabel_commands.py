"""
Rewrite command labels so they match expert steer + keep planner approach intent.

Uses the same rules as live recording (`label_command`): large |steer| → TURN_*,
FOLLOW_LANE never stays if the wheel is turned much. Low-steer TURN_*/GO_STRAIGHT
approach labels are kept.

Examples:
  python -m expert.relabel_commands --data data/run_20260913_171811
  python -m expert.relabel_commands --data data/
"""
from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

from expert.model import resolve_run_dirs
from utils.balance import (
    CONTROLS_HEADER,
    warn_if_follow_lane_heavy,
    write_balanced_controls,
    write_command_distribution,
)
from utils.commands import COMMAND_FOLLOW_LANE, label_command


def relabel_run(
    run_dir: Path,
    *,
    steer_turn: float = 0.15,
    steer_follow_max: float = 0.10,
) -> Path:
    controls_path = run_dir / "controls.csv"
    if not controls_path.exists():
        raise FileNotFoundError(controls_path)

    backup = run_dir / "controls_before_relabel.csv"
    if not backup.exists():
        shutil.copy2(controls_path, backup)
    else:
        # Relabel from the original backup so reruns are idempotent.
        shutil.copy2(backup, controls_path)

    rows = list(csv.DictReader(controls_path.open()))
    if not rows:
        raise RuntimeError(f"Empty controls in {run_dir}")

    changed = 0
    for row in rows:
        steer = float(row["steer"])
        old = row.get("command", COMMAND_FOLLOW_LANE)
        new = label_command(
            old,
            steer,
            steer_turn=steer_turn,
            steer_follow_max=steer_follow_max,
        )
        if new != old:
            changed += 1
        row["command"] = new

    fieldnames = list(rows[0].keys())
    for key in CONTROLS_HEADER:
        if key not in fieldnames:
            fieldnames.append(key)

    with controls_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    warn_if_follow_lane_heavy(rows)
    write_command_distribution(run_dir, rows)
    write_balanced_controls(run_dir, rows)
    print(f"{run_dir.name}: updated {changed} command labels → {controls_path}")
    return controls_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        nargs="+",
        required=True,
        help="Run folder(s) or parent data/ directory",
    )
    parser.add_argument(
        "--steer-turn",
        type=float,
        default=0.15,
        help="|steer| above this forces TURN_LEFT/RIGHT",
    )
    parser.add_argument(
        "--steer-follow-max",
        type=float,
        default=0.10,
        help="FOLLOW_LANE allowed only if |steer| <= this",
    )
    args = parser.parse_args()
    runs = resolve_run_dirs(args.data)
    for run in runs:
        relabel_run(
            run,
            steer_turn=args.steer_turn,
            steer_follow_max=args.steer_follow_max,
        )


if __name__ == "__main__":
    main()
