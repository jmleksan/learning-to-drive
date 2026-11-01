"""Synthetic localization log for offline EKF demos (no CARLA)."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

from localization.ekf import wrap_angle
from localization.log import LOCALIZATION_HEADER, localization_csv_path


def write_synth_run(
    run_dir: Path,
    *,
    seconds: float = 60.0,
    fps: float = 10.0,
    gnss_noise_m: float = 3.5,
    seed: int = 0,
) -> Path:
    """Drive a smooth arc; log GT + noisy GNSS + IMU yaw rate + speed."""
    rng = np.random.default_rng(seed)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    n_frames = int(seconds * fps)
    dt = 1.0 / fps

    # Gentle left curve then straight.
    speed = 8.0  # m/s
    yaw = 0.0  # ENU: north
    east = 0.0
    north = 0.0
    rows: list[dict[str, str]] = []
    for i in range(n_frames):
        t = i * dt
        if 5.0 < t < 25.0:
            yaw_rate = 0.12
        else:
            yaw_rate = 0.0
        east += speed * math.sin(yaw) * dt
        north += speed * math.cos(yaw) * dt
        yaw += yaw_rate * dt

        gnss_e = east + float(rng.normal(0.0, gnss_noise_m))
        gnss_n = north + float(rng.normal(0.0, gnss_noise_m))
        # Match CARLA conventions used by eval: gyro.z opposite ENU yaw rate,
        # compass such that yaw_enu = π - compass.
        imu_gz = -yaw_rate + float(rng.normal(0.0, 0.01))
        imu_compass = wrap_angle(math.pi - yaw)
        rows.append(
            {
                "frame": str(i),
                "t_s": f"{t:.4f}",
                "gt_e_m": f"{east:.6f}",
                "gt_n_m": f"{north:.6f}",
                "gt_yaw_rad": f"{yaw:.6f}",
                "gt_speed_mps": f"{speed:.6f}",
                "gnss_e_m": f"{gnss_e:.6f}",
                "gnss_n_m": f"{gnss_n:.6f}",
                "gnss_alt_m": "0.0",
                "imu_ax": "0.0",
                "imu_ay": "0.0",
                "imu_az": "9.81",
                "imu_gx": "0.0",
                "imu_gy": "0.0",
                "imu_gz": f"{imu_gz:.6f}",
                "imu_compass": f"{imu_compass:.6f}",
                "gt_x_m": f"{east:.6f}",
                "gt_y_m": f"{north:.6f}",
                "gt_z_m": "0.0",
            }
        )

    path = localization_csv_path(run_dir)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=LOCALIZATION_HEADER)
        writer.writeheader()
        writer.writerows(rows)
    (run_dir / "meta.json").write_text(
        '{"synth": true, "gnss_noise_m": %s, "fps": %s}\n'
        % (gnss_noise_m, fps)
    )
    return path
