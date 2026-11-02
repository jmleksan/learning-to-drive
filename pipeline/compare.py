"""Compare classical vs learned Step 1 drive metrics (offline JSON)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


_KEYS = (
    "collisions",
    "collisions_per_km",
    "lane_invasions",
    "route_completion_pct",
    "avg_speed_kmh",
    "distance_m",
    "cte_p50_m",
    "cte_p90_m",
    "milestone",
    "controller",
    "agent",
)


def load_metrics(path: Path) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if "drive_metrics.json" in str(path) or "collisions" in data:
        return data
    raise ValueError(f"Not a drive_metrics.json: {path}")


def compare(
    classical: Path,
    learned: Path,
    *,
    out: Path | None = None,
) -> dict:
    c = load_metrics(classical)
    l = load_metrics(learned)
    table = {
        "classical": {k: c.get(k) for k in _KEYS if k in c or k in ("controller", "agent")},
        "learned": {k: l.get(k) for k in _KEYS if k in l},
        "delta": {},
    }
    for key in (
        "collisions_per_km",
        "lane_invasions",
        "route_completion_pct",
        "avg_speed_kmh",
        "cte_p90_m",
    ):
        if key in c and key in l and isinstance(c[key], (int, float)) and isinstance(
            l[key], (int, float)
        ):
            table["delta"][key] = round(float(c[key]) - float(l[key]), 3)

    # Fill agent labels
    table["classical"].setdefault("agent", c.get("agent", "classical"))
    table["learned"].setdefault("agent", l.get("agent", "learned"))

    text_lines = [
        "# Classical vs learned (Step 4 milestone)",
        "",
        "| Metric | Classical | Learned | Δ (C−L) |",
        "|--------|-----------|---------|---------|",
    ]
    for key in (
        "collisions",
        "collisions_per_km",
        "lane_invasions",
        "route_completion_pct",
        "avg_speed_kmh",
        "cte_p90_m",
        "milestone",
    ):
        cv = table["classical"].get(key, "—")
        lv = table["learned"].get(key, "—")
        dv = table["delta"].get(key, "—")
        text_lines.append(f"| {key} | {cv} | {lv} | {dv} |")
    text_lines.append("")
    text_lines.append(
        f"Classical controller: `{table['classical'].get('controller', '?')}`"
    )
    text = "\n".join(text_lines) + "\n"
    print(text)
    if out is not None:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        out.with_suffix(".json").write_text(
            json.dumps(table, indent=2), encoding="utf-8"
        )
        print(f"Wrote {out}")
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--classical",
        type=Path,
        required=True,
        help="Path to classical drive_metrics.json",
    )
    parser.add_argument(
        "--learned",
        type=Path,
        default=Path("data/drive_log_20261009_164441/drive_metrics.json"),
        help="Path to learned Step 1 drive_metrics.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("models/pipeline_preview/compare.md"),
    )
    args = parser.parse_args()
    compare(args.classical, args.learned, out=args.out)


if __name__ == "__main__":
    main()
