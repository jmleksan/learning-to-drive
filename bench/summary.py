"""Score table for one or more bench/run.sh result folders.

Bench2Drive's tools/merge_route_json.py divides by 220 whatever the route
count, so a partial set comes out wrong. This divides by the routes that ran
and prints a table per scenario.

  python bench/summary.py data/bench/dev10_transfuser
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path


def main() -> None:
    for folder in sys.argv[1:]:
        records = json.loads((Path(folder) / "results.json").read_text())["_checkpoint"]["records"]
        print(f"\n{folder}: {len(records)} routes")
        print(f"{'route':28s} {'town':9s} {'scenario':38s} {'DS':>6s} {'RC':>6s} {'IP':>5s}  status / infractions")
        infractions: Counter[str] = Counter()
        for r in records:
            scores = r["scores"]
            hits = {k: len(v) for k, v in r["infractions"].items() if v}
            infractions.update(hits)
            print(
                f"{r['route_id']:28s} {r.get('town_name', '?'):9s} {r.get('scenario_name', '?')[:38]:38s} "
                f"{scores['score_composed']:6.1f} {scores['score_route']:6.1f} {scores['score_penalty']:5.2f}  "
                f"{r['status']} {hits or ''}"
            )
        if not records:
            continue
        n = len(records)
        clean = sum(
            r["status"] in ("Completed", "Perfect")
            and not any(v for k, v in r["infractions"].items() if k != "min_speed_infractions")
            for r in records
        )
        print(
            f"\nDriving Score {sum(r['scores']['score_composed'] for r in records) / n:.1f}   "
            f"Route Completion {sum(r['scores']['score_route'] for r in records) / n:.1f}   "
            f"Infraction Penalty {sum(r['scores']['score_penalty'] for r in records) / n:.2f}   "
            f"Success {100 * clean / n:.0f}%"
        )
        print("Infractions:", dict(infractions.most_common()))


if __name__ == "__main__":
    main()
