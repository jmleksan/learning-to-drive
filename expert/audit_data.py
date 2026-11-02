"""
Audit recorded runs for lane-keeping data quality.

Primary gate (when cte_m is present): stay in-lane through TURN_*.
Soft/hard |steer| buckets are informational only.

Examples:
  python -m expert.audit_data
  python -m expert.audit_data --data data/ --archive-bad
  python -m expert.audit_data data/run_... --lane-only
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from utils.commands import (
    COMMAND_FOLLOW_LANE,
    COMMAND_TURN_LEFT,
    COMMAND_TURN_RIGHT,
    JUNCTION_COMMANDS,
)

_TURN_COMMANDS = (COMMAND_TURN_LEFT, COMMAND_TURN_RIGHT)
_LANE_KEEP = (COMMAND_FOLLOW_LANE, "GO_STRAIGHT")

REQUIRED_COLUMNS = (
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
)


@dataclass
class LaneStats:
    n: int = 0
    mean_abs_cte: float | None = None
    p90_abs_cte: float | None = None
    max_abs_cte: float | None = None
    frac_over_0_7: float | None = None
    frac_over_0_85: float | None = None


@dataclass
class RunReport:
    path: Path
    map_name: str
    spawn_index: int | None
    frames: int
    turn_pct: float
    sharp_pct: float
    max_abs_steer: float
    mean_speed: float | None
    stopped_pct: float | None
    follow_lane_pct: float | None
    junction_pct: float | None
    multicam: bool
    has_cte: bool
    turn_lane: LaneStats = field(default_factory=LaneStats)
    follow_lane: LaneStats = field(default_factory=LaneStats)
    ok: bool = False
    reasons: list[str] = field(default_factory=list)


def _center_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    if rows and "camera" in rows[0]:
        return [r for r in rows if r.get("camera") == "center"]
    return rows


def _has_three_cameras(run_dir: Path) -> bool:
    base = run_dir / "images"
    return all(
        (base / name).is_dir() and any((base / name).glob("*.png"))
        for name in ("center", "left", "right")
    )


def _lane_stats(rows: list[dict[str, str]]) -> LaneStats:
    if not rows or "cte_m" not in rows[0]:
        return LaneStats()
    abs_cte = [abs(float(r["cte_m"])) for r in rows]
    abs_cte.sort()
    n = len(abs_cte)
    if n == 0:
        return LaneStats()
    fracs = [abs(float(r.get("lane_offset_frac", 0.0))) for r in rows]
    return LaneStats(
        n=n,
        mean_abs_cte=sum(abs_cte) / n,
        p90_abs_cte=abs_cte[min(n - 1, int(0.9 * n))],
        max_abs_cte=abs_cte[-1],
        frac_over_0_7=100.0 * sum(1 for f in fracs if f > 0.7) / n,
        frac_over_0_85=100.0 * sum(1 for f in fracs if f > 0.85) / n,
    )


def score_run(
    run_dir: Path,
    *,
    max_stopped_pct: float = 20.0,
    min_frames: int = 600,
    max_follow_lane_pct: float = 60.0,
    min_junction_pct: float = 15.0,
    require_multicam: bool = False,
    max_turn_p90_cte: float = 0.6,
    max_turn_edge_pct: float = 8.0,
    lane_gate: bool = True,
) -> RunReport:
    controls = run_dir / "controls.csv"
    balanced = run_dir / "controls_balanced.csv"
    meta_path = run_dir / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    rows = list(csv.DictReader(controls.open())) if controls.exists() else []
    center = _center_rows(rows)
    frames = len(center) if center else len(rows)
    steers = [abs(float(r["steer"])) for r in center] if center else [0.0]
    has_speed = bool(rows) and "speed_kmh" in rows[0]
    has_cte = bool(rows) and "cte_m" in rows[0]
    speeds = [float(r["speed_kmh"]) for r in center] if has_speed and center else []

    turns = sum(1 for s in steers if s >= 0.15)
    sharp = sum(1 for s in steers if s >= 0.3)
    turn_pct = 100.0 * turns / frames if frames else 0.0
    sharp_pct = 100.0 * sharp / frames if frames else 0.0
    mean_speed = sum(speeds) / len(speeds) if speeds else None
    # Idle at red/yellow is expected once lights are obeyed — don't FAIL on that.
    if speeds and center:
        idle = 0
        for r, v in zip(center, speeds):
            if v >= 1.0:
                continue
            state = (r.get("light_state") or "").strip()
            if state in ("Red", "Yellow"):
                continue
            idle += 1
        stopped = idle
    else:
        stopped = sum(1 for v in speeds if v < 1.0) if speeds else None
    stopped_pct = 100.0 * stopped / frames if stopped is not None and frames else None

    follow_lane_pct: float | None = None
    junction_pct: float | None = None
    mix_rows = list(csv.DictReader(balanced.open())) if balanced.exists() else rows
    mix_center = _center_rows(mix_rows)
    if mix_center and "command" in mix_center[0]:
        n = len(mix_center)
        follow_lane_pct = (
            100.0
            * sum(1 for r in mix_center if r.get("command") == COMMAND_FOLLOW_LANE)
            / n
        )
        junction_pct = (
            100.0
            * sum(1 for r in mix_center if r.get("command") in JUNCTION_COMMANDS)
            / n
        )

    turn_rows = [r for r in center if r.get("command") in _TURN_COMMANDS]
    follow_rows = [r for r in center if r.get("command") in _LANE_KEEP]
    turn_lane = _lane_stats(turn_rows) if has_cte else LaneStats()
    follow_lane = _lane_stats(follow_rows) if has_cte else LaneStats()

    multicam = _has_three_cameras(run_dir)
    reasons: list[str] = []
    if frames < min_frames:
        reasons.append(f"too short ({frames} < {min_frames} frames)")
    # Soft/hard steer% is informational — do not FAIL on low turn%.
    if stopped_pct is not None and stopped_pct > max_stopped_pct:
        reasons.append(f"stopped% {stopped_pct:.1f} > {max_stopped_pct}")
    if not has_speed:
        reasons.append("missing speed_kmh (re-record)")

    if lane_gate:
        if not has_cte:
            reasons.append("missing cte_m (re-record with lane metrics)")
        elif turn_lane.n == 0:
            reasons.append("no TURN_* frames for lane audit")
        else:
            if (
                turn_lane.p90_abs_cte is not None
                and turn_lane.p90_abs_cte > max_turn_p90_cte
            ):
                reasons.append(
                    f"TURN_* p90 |cte|={turn_lane.p90_abs_cte:.2f}m "
                    f"> {max_turn_p90_cte}"
                )
            if (
                turn_lane.frac_over_0_85 is not None
                and turn_lane.frac_over_0_85 > max_turn_edge_pct
            ):
                reasons.append(
                    f"TURN_* |lane_offset_frac|>0.85 "
                    f"{turn_lane.frac_over_0_85:.1f}% > {max_turn_edge_pct}%"
                )

    if require_multicam:
        if not multicam:
            reasons.append("missing images/center|left|right")
        if not rows or any(c not in rows[0] for c in REQUIRED_COLUMNS):
            missing = [c for c in REQUIRED_COLUMNS if not rows or c not in rows[0]]
            reasons.append(f"missing columns: {', '.join(missing)}")
        if follow_lane_pct is None or junction_pct is None:
            reasons.append("missing command mix (re-record)")
        else:
            if follow_lane_pct > max_follow_lane_pct:
                reasons.append(
                    f"FOLLOW_LANE {follow_lane_pct:.1f}% > {max_follow_lane_pct} "
                    "(after rebalance)"
                )
            if junction_pct < min_junction_pct:
                reasons.append(
                    f"junction cmds {junction_pct:.1f}% < {min_junction_pct}"
                )
    else:
        if not rows or "command" not in rows[0]:
            reasons.append("missing command column (re-record)")
        elif follow_lane_pct is not None and junction_pct is not None:
            if follow_lane_pct > max_follow_lane_pct:
                reasons.append(
                    f"FOLLOW_LANE {follow_lane_pct:.1f}% > {max_follow_lane_pct} "
                    "(after rebalance)"
                )
            if junction_pct < min_junction_pct:
                reasons.append(
                    f"junction cmds {junction_pct:.1f}% < {min_junction_pct}"
                )
        for col in ("at_traffic_light", "light_state", "near_stop_sign"):
            if not rows or col not in rows[0]:
                reasons.append(f"missing {col}")
                break

    map_name = meta.get("map_load_name") or meta.get("map") or "?"
    return RunReport(
        path=run_dir,
        map_name=str(map_name),
        spawn_index=meta.get("spawn_index"),
        frames=frames,
        turn_pct=turn_pct,
        sharp_pct=sharp_pct,
        max_abs_steer=max(steers) if steers else 0.0,
        mean_speed=mean_speed,
        stopped_pct=stopped_pct,
        follow_lane_pct=follow_lane_pct,
        junction_pct=junction_pct,
        multicam=multicam,
        has_cte=has_cte,
        turn_lane=turn_lane,
        follow_lane=follow_lane,
        ok=not reasons,
        reasons=reasons,
    )


def archive_run(run_dir: Path, archive_dir: Path) -> Path:
    archive_dir.mkdir(parents=True, exist_ok=True)
    dest = archive_dir / run_dir.name
    if dest.exists():
        shutil.rmtree(dest)
    shutil.move(str(run_dir), str(dest))
    return dest


def _fmt_lane(label: str, s: LaneStats) -> str:
    if s.n == 0 or s.p90_abs_cte is None:
        return f"{label}: n/a"
    return (
        f"{label}: n={s.n} mean|cte|={s.mean_abs_cte:.2f} "
        f"p90={s.p90_abs_cte:.2f} max={s.max_abs_cte:.2f}  "
        f"|frac|>0.7={s.frac_over_0_7:.1f}% "
        f"|frac|>0.85={s.frac_over_0_85:.1f}%"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run",
        nargs="?",
        type=Path,
        default=None,
        help="Optional single run dir (else scan --data)",
    )
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--max-stopped-pct", type=float, default=20.0)
    parser.add_argument("--min-frames", type=int, default=600)
    parser.add_argument("--max-follow-lane-pct", type=float, default=60.0)
    parser.add_argument("--min-junction-pct", type=float, default=15.0)
    parser.add_argument(
        "--max-turn-p90-cte",
        type=float,
        default=0.6,
        help="KEEP if TURN_* p90 |cte_m| is below this (meters)",
    )
    parser.add_argument(
        "--max-turn-edge-pct",
        type=float,
        default=8.0,
        help="KEEP if %% of TURN_* frames with |lane_offset_frac|>0.85 is below this",
    )
    parser.add_argument(
        "--no-lane-gate",
        action="store_true",
        help="Do not require cte_m / lane adherence (legacy soft/hard era)",
    )
    parser.add_argument(
        "--require-multicam",
        action="store_true",
        help="Require center+left+right image folders for KEEP",
    )
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="Skip command/light/lane KEEP gates (old single-camera runs)",
    )
    parser.add_argument(
        "--archive-bad",
        action="store_true",
        help="Move failing runs into data/archive/",
    )
    args = parser.parse_args()

    runs = [args.run] if args.run is not None else sorted(args.data.glob("run_*"))
    if not runs:
        print(f"No run_* folders in {args.data}")
        return

    lane_gate = not args.no_lane_gate and not args.legacy
    print(
        f"Auditing {len(runs)} run(s)  "
        f"(lane gate={'ON' if lane_gate else 'OFF'}; "
        f"TURN_* p90 |cte|≤{args.max_turn_p90_cte}m; "
        f"steer% is info only)\n"
    )
    good: list[RunReport] = []
    bad: list[RunReport] = []
    for run in runs:
        if args.legacy:
            report = score_run(
                run,
                max_stopped_pct=args.max_stopped_pct,
                min_frames=args.min_frames,
                require_multicam=False,
                lane_gate=False,
            )
            report.reasons = [
                r
                for r in report.reasons
                if r.startswith("too short")
                or r.startswith("stopped%")
                or r.startswith("missing speed")
            ]
            report.ok = not report.reasons
        else:
            report = score_run(
                run,
                max_stopped_pct=args.max_stopped_pct,
                min_frames=args.min_frames,
                max_follow_lane_pct=args.max_follow_lane_pct,
                min_junction_pct=args.min_junction_pct,
                require_multicam=args.require_multicam,
                max_turn_p90_cte=args.max_turn_p90_cte,
                max_turn_edge_pct=args.max_turn_edge_pct,
                lane_gate=lane_gate,
            )
        status = "KEEP" if report.ok else "FAIL"
        speed_bit = (
            f"speed={report.mean_speed:.1f} stopped={report.stopped_pct:.1f}%"
            if report.mean_speed is not None and report.stopped_pct is not None
            else "no-speed"
        )
        cmd_bit = ""
        if report.follow_lane_pct is not None and report.junction_pct is not None:
            cmd_bit = (
                f"  follow={report.follow_lane_pct:.1f}% "
                f"junction={report.junction_pct:.1f}%"
            )
        cams = "3cam" if report.multicam else "1cam"
        print(
            f"[{status}] {report.path.name}  map={report.map_name} "
            f"spawn={report.spawn_index}  frames={report.frames}  "
            f"steer≥0.15={report.turn_pct:.1f}% (info)  "
            f"max|steer|={report.max_abs_steer:.3f}  {speed_bit}  {cams}{cmd_bit}"
        )
        if report.has_cte:
            print(f"         {_fmt_lane('TURN_*', report.turn_lane)}")
            print(f"         {_fmt_lane('FOLLOW', report.follow_lane)}")
        else:
            print("         cte_m: missing")
        if report.reasons:
            print(f"         reasons: {', '.join(report.reasons)}")
        (good if report.ok else bad).append(report)

    print(f"\nSummary: {len(good)} keep / {len(bad)} fail")
    if args.archive_bad and bad:
        archive_dir = args.data / "archive"
        for report in bad:
            dest = archive_run(report.path, archive_dir)
            print(f"Archived {report.path.name} -> {dest}")
    elif bad:
        print("Tip: re-run with --archive-bad to move FAIL runs into data/archive/")
    if good:
        print("Train on keep runs:")
        print(
            "  python -m expert.train --data "
            + " ".join(str(r.path) for r in good)
            + " --stack-frames 3 --oversample-turns --device cuda"
        )
    else:
        print(
            "No KEEP runs yet. Record with lane metrics, e.g.:\n"
            "  python -m expert.record --map Town01 --seconds 300 --spawn-index 230\n"
            "  python -m expert.audit_data data/run_..."
        )


if __name__ == "__main__":
    main()
