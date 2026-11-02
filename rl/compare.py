"""Scorecard: GPS-RL vs camera-RL vs Step 1 IL (offline JSON)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline.compare import _KEYS, load_metrics

_METRIC_ROWS = (
    "collisions",
    "collisions_per_km",
    "lane_invasions",
    "route_completion_pct",
    "avg_speed_kmh",
    "cte_p90_m",
    "milestone",
)


def _col(metrics: dict | None, key: str) -> str:
    if metrics is None:
        return "—"
    value = metrics.get(key, "—")
    return "—" if value is None else str(value)


def compare(
    *,
    gps: Path | None,
    camera: Path | None,
    il: Path | None,
    out: Path | None = None,
) -> dict:
    columns: list[tuple[str, dict]] = []
    if gps is not None:
        columns.append(("GPS-RL", load_metrics(gps)))
    if camera is not None:
        columns.append(("camera-RL", load_metrics(camera)))
    if il is not None:
        columns.append(("IL", load_metrics(il)))
    if len(columns) < 2:
        raise ValueError("Need at least two of --gps, --camera, --il")

    header = "| Metric | " + " | ".join(name for name, _ in columns) + " |"
    divider = "|--------|" + "|".join("--------" for _ in columns) + "|"
    lines = [
        "# GPS-RL vs camera-RL vs IL",
        "",
        header,
        divider,
    ]
    for key in _METRIC_ROWS:
        cells = " | ".join(_col(data, key) for _, data in columns)
        lines.append(f"| {key} | {cells} |")
    lines.append("")
    for name, data in columns:
        agent = data.get("agent", "?")
        model = data.get("model") or data.get("controller") or ""
        extra = f" `{model}`" if model else ""
        lines.append(f"- {name}: `{agent}`{extra}")
    text = "\n".join(lines) + "\n"
    print(text)

    table = {
        name: {k: data.get(k) for k in _KEYS if k in data}
        for name, data in columns
    }
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
        "--gps",
        type=Path,
        default=None,
        help="Privileged GPS-RL drive_metrics.json",
    )
    parser.add_argument(
        "--camera",
        type=Path,
        default=None,
        help="Camera-RL drive_metrics.json",
    )
    parser.add_argument(
        "--il",
        type=Path,
        default=Path("data/drive_log_20261009_164441/drive_metrics.json"),
        help="Step 1 IL drive_metrics.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("models/rl_sac_cam/compare.md"),
    )
    args = parser.parse_args()
    compare(gps=args.gps, camera=args.camera, il=args.il, out=args.out)


if __name__ == "__main__":
    main()
