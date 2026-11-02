"""Dumbbell chart of two Dev10 runs, one row per route, for the README.

  python bench/plot_dev10.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

BEFORE = Path("results/dev10_before_gpsfix/results.json")
AFTER = Path("results/dev10_gpsfix/results.json")
OUT = Path("docs/images/dev10_runs.png")
# The GPS bug only touched maps referenced away from latitude 0.
AFFECTED_TOWNS = {"Town11", "Town13", "Town15"}

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_MUTED = "#52514e"
GRID = "#e4e3df"
LINK = "#b5b4ae"
RUN_1 = "#86b6ef"
RUN_2 = "#184f95"


def load(path: Path) -> dict[str, dict]:
    records = json.loads(path.read_text())["_checkpoint"]["records"]
    return {r["route_id"]: r for r in records}


def main() -> None:
    before, after = load(BEFORE), load(AFTER)
    rows = [
        {
            "name": f"{after[k]['scenario_name'].rsplit('_', 1)[0]} ({after[k]['town_name']})",
            "before": before[k]["scores"]["score_composed"],
            "after": after[k]["scores"]["score_composed"],
            "affected": after[k]["town_name"] in AFFECTED_TOWNS,
        }
        for k in after
    ]
    groups = [
        ("Could not be affected by the GPS fix", [r for r in rows if not r["affected"]]),
        ("Affected by the GPS fix", [r for r in rows if r["affected"]]),
    ]

    fig, ax = plt.subplots(figsize=(10, 6.4), dpi=160, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    y = 0.0
    ticks, labels = [], []
    for title, group in groups:
        ax.text(0, y, title, fontsize=11, fontweight="bold", color=TEXT, va="center")
        y += 1
        for r in sorted(group, key=lambda r: -abs(r["after"] - r["before"])):
            ax.plot([r["before"], r["after"]], [y, y], color=LINK, linewidth=2, zorder=1)
            ax.scatter(r["before"], y, s=64, color=RUN_1, edgecolor=SURFACE, linewidth=1.5, zorder=2)
            ax.scatter(r["after"], y, s=64, color=RUN_2, edgecolor=SURFACE, linewidth=1.5, zorder=3)
            ticks.append(y)
            labels.append(r["name"])
            y += 1
        y += 0.6

    ax.set_yticks(ticks, labels, fontsize=9.5, color=TEXT)
    ax.set_ylim(y - 0.8, -0.8)
    ax.set_xlim(-3, 104)
    ax.set_xlabel("Driving Score", fontsize=10, color=TEXT_MUTED)
    ax.tick_params(axis="x", colors=TEXT_MUTED, length=0)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.suptitle(
        "Same model, two Dev10 runs: routes the fix could not touch moved too",
        x=0.02, ha="left", fontsize=13.5, fontweight="bold", color=TEXT,
    )
    handles = [
        plt.Line2D([], [], marker="o", linestyle="", markersize=8, color=RUN_1, label="Run 1, GPS bug (average 41.2)"),
        plt.Line2D([], [], marker="o", linestyle="", markersize=8, color=RUN_2, label="Run 2, GPS fixed (average 38.6)"),
    ]
    fig.legend(
        handles=handles, loc="upper left", bbox_to_anchor=(0.012, 0.955), ncols=2,
        frameon=False, fontsize=10, labelcolor=TEXT, handletextpad=0.2,
    )
    fig.text(0.02, 0.012, "One dot means both runs scored the same.", fontsize=9, color=TEXT_MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 0.91))
    fig.savefig(OUT, facecolor=SURFACE)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
