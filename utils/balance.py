"""Post-record command balancing and distribution stats."""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

from utils.commands import ALL_COMMANDS, COMMAND_FOLLOW_LANE, JUNCTION_COMMANDS


CONTROLS_HEADER = [
    "frame",
    "camera",
    "steer",
    "throttle",
    "brake",
    "speed_kmh",
    "command",
    "at_traffic_light",
    "light_state",
    "near_stop_sign",
    # Eval-only lane position (never fed to PilotNet).
    "cte_m",
    "lane_offset_frac",
    "heading_err_deg",
    # Nearest forward radar return in meters. Empty = nothing ahead.
    # Fed to throttle/brake only, never to steer.
    "radar_m",
]


def _center_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [r for r in rows if r.get("camera", "center") == "center"]


def command_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    counts = Counter(r.get("command", COMMAND_FOLLOW_LANE) for r in rows)
    return {cmd: int(counts.get(cmd, 0)) for cmd in ALL_COMMANDS}


def command_percents(counts: dict[str, int]) -> dict[str, float]:
    total = sum(counts.values()) or 1
    return {k: 100.0 * v / total for k, v in counts.items()}


def write_command_distribution(run_dir: Path, rows: list[dict[str, str]]) -> Path:
    center = _center_rows(rows)
    all_counts = command_counts(rows)
    center_counts = command_counts(center)
    payload = {
        "all_cameras": {
            "counts": all_counts,
            "percents": command_percents(all_counts),
            "n": sum(all_counts.values()),
        },
        "center": {
            "counts": center_counts,
            "percents": command_percents(center_counts),
            "n": sum(center_counts.values()),
        },
        "junction_center_pct": sum(
            command_percents(center_counts).get(c, 0.0) for c in JUNCTION_COMMANDS
        ),
        "follow_lane_center_pct": command_percents(center_counts).get(
            COMMAND_FOLLOW_LANE, 0.0
        ),
    }
    out = run_dir / "command_distribution.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    return out


def write_balanced_controls(
    run_dir: Path,
    rows: list[dict[str, str]],
    *,
    max_follow_lane_center_pct: float = 60.0,
) -> Path:
    """
    Downsample FOLLOW_LANE center rows until they are <= max_follow_lane_center_pct
    of center frames. Keep all turn/change-lane center rows and all left/right rows.
    """
    center = _center_rows(rows)
    side = [r for r in rows if r.get("camera") not in (None, "", "center")]

    follow = [r for r in center if r.get("command") == COMMAND_FOLLOW_LANE]
    other = [r for r in center if r.get("command") != COMMAND_FOLLOW_LANE]

    n_center = len(center)
    if n_center == 0:
        balanced = list(rows)
    else:
        # Target: follow / n_center_balanced <= max_pct/100
        # n_center_balanced = n_follow_keep + n_other
        # n_follow_keep / (n_follow_keep + n_other) <= p
        # n_follow_keep <= p/(1-p) * n_other
        p = max_follow_lane_center_pct / 100.0
        if p >= 1.0:
            follow_keep = follow
        else:
            max_follow = int(p / (1.0 - p) * len(other)) if other else 0
            if len(follow) > max_follow:
                # Evenly subsample unique frames
                if max_follow <= 0:
                    follow_keep = []
                else:
                    indices = [
                        min(len(follow) - 1, int(round(i * (len(follow) - 1) / (max_follow - 1))))
                        if max_follow > 1
                        else 0
                        for i in range(max_follow)
                    ]
                    # Deduplicate while preserving order
                    seen: set[int] = set()
                    follow_keep = []
                    for idx in indices:
                        if idx not in seen:
                            seen.add(idx)
                            follow_keep.append(follow[idx])
                    # If dedupe shrank the set, fill from remaining
                    if len(follow_keep) < max_follow:
                        for idx, row in enumerate(follow):
                            if idx not in seen:
                                follow_keep.append(row)
                                seen.add(idx)
                                if len(follow_keep) >= max_follow:
                                    break
            else:
                follow_keep = follow
        balanced_center = follow_keep + other
        # Preserve temporal order by frame
        balanced_center.sort(key=lambda r: int(r["frame"]))
        balanced = balanced_center + side
        balanced.sort(key=lambda r: (int(r["frame"]), r.get("camera", "")))

    out = run_dir / "controls_balanced.csv"
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CONTROLS_HEADER, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(balanced)
    return out


def warn_if_follow_lane_heavy(rows: list[dict[str, str]], *, threshold_pct: float = 70.0) -> None:
    center = _center_rows(rows)
    if not center:
        return
    pct = command_percents(command_counts(center)).get(COMMAND_FOLLOW_LANE, 0.0)
    if pct > threshold_pct:
        print(
            f"WARNING: center FOLLOW_LANE is {pct:.1f}% > {threshold_pct:.0f}% "
            "before downsample (bad dests / map)."
        )
