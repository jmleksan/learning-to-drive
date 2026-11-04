"""Dumbbell chart of fusion vs camera-only open-loop error, for the Step 6 write-up.

Reads the open_loop.json that transfuser.eval writes for both models.

  python -m transfuser.plot_compare data/tf/eval_compare/open_loop.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_MUTED = "#52514e"
GRID = "#e4e3df"
LINK = "#b5b4ae"
CAMERA = "#86b6ef"
FUSION = "#184f95"
METRICS = [
    ("stop_ade_m", "Stopped frames"),
    ("fde_m", "Final waypoint (2 s)"),
    ("ade_m", "All waypoints"),
    ("turn_ade_m", "Turn frames"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", type=Path)
    parser.add_argument("--out", type=Path, default=Path("docs/images/step6_fusion_vs_camera.png"))
    args = parser.parse_args()
    runs = {r["modality"]: r for r in json.loads(args.results.read_text())}
    fusion, camera = runs["both"], runs["camera"]

    fig, ax = plt.subplots(figsize=(8, 3.4), dpi=160, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for y, (key, label) in enumerate(METRICS):
        f, c = fusion[key], camera[key]
        ax.plot([f, c], [y, y], color=LINK, linewidth=2, zorder=1)
        ax.scatter(c, y, s=70, color=CAMERA, edgecolor=SURFACE, linewidth=1.5, zorder=2)
        ax.scatter(f, y, s=70, color=FUSION, edgecolor=SURFACE, linewidth=1.5, zorder=3)
        ax.text(c + 0.012, y, f"{c:.2f}", va="center", ha="left", fontsize=9, color=TEXT_MUTED)
        ax.text(f - 0.012, y, f"{f:.2f}", va="center", ha="right", fontsize=9, color=TEXT_MUTED)
    ax.set_yticks(range(len(METRICS)), [label for _, label in METRICS], fontsize=10, color=TEXT)
    ax.set_ylim(len(METRICS) - 0.4, -0.9)
    ax.set_xlim(0, 0.6)
    ax.set_xlabel("Waypoint error on the validation run (m), lower is better", fontsize=9, color=TEXT_MUTED)
    ax.tick_params(axis="x", colors=TEXT_MUTED, labelsize=9)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.scatter([], [], s=70, color=FUSION, label="Fusion (camera + lidar)")
    ax.scatter([], [], s=70, color=CAMERA, label="Camera only")
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 1.02), ncol=2, frameon=False, fontsize=9)
    ax.set_title("Fusion vs camera-only, open loop", loc="left", fontsize=12, fontweight="bold", color=TEXT)
    fig.tight_layout()
    fig.savefig(args.out, facecolor=SURFACE)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
