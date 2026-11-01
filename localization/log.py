"""Localization CSV schema and IO."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

LOCALIZATION_HEADER = [
    "frame",
    "t_s",
    "gt_e_m",
    "gt_n_m",
    "gt_yaw_rad",
    "gt_speed_mps",
    "gnss_e_m",
    "gnss_n_m",
    "gnss_alt_m",
    "imu_ax",
    "imu_ay",
    "imu_az",
    "imu_gx",
    "imu_gy",
    "imu_gz",
    "imu_compass",
    # Raw CARLA world pose (debug / overlays).
    "gt_x_m",
    "gt_y_m",
    "gt_z_m",
]


def localization_csv_path(run_dir: Path) -> Path:
    return Path(run_dir) / "localization.csv"


def load_localization(run_dir: Path) -> np.ndarray:
    """Load localization.csv as a structured array / dict of columns."""
    path = localization_csv_path(run_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"No localization.csv in {run_dir}. Re-record with --gnss --imu."
        )
    rows = list(csv.DictReader(path.open()))
    if not rows:
        raise ValueError(f"{path} is empty")
    out: dict[str, np.ndarray] = {}
    for key in LOCALIZATION_HEADER:
        if key == "frame":
            out[key] = np.array([int(r[key]) for r in rows], dtype=np.int32)
        else:
            out[key] = np.array(
                [float(r[key]) if r.get(key) not in (None, "") else np.nan for r in rows],
                dtype=np.float64,
            )
    return out  # type: ignore[return-value]


def open_localization_writer(run_dir: Path):
    path = localization_csv_path(run_dir)
    fh = path.open("w", newline="")
    writer = csv.DictWriter(fh, fieldnames=LOCALIZATION_HEADER)
    writer.writeheader()
    return fh, writer
