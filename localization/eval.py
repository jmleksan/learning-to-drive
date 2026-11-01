"""Run EKF on a recorded localization.csv and score vs GT / raw GNSS.

Examples:
  python -m localization.eval --data data/run_...
  python -m localization.eval --synth --out models/loc_preview
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from localization.ekf import EgoEKF, compass_to_yaw_enu, wrap_angle
from localization.log import load_localization
from localization.synth import write_synth_run


def _finite_pairs(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Keep rows where both Nx2 arrays are finite."""
    mask = np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
    return a[mask], b[mask]


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    a, b = _finite_pairs(a, b)
    if len(a) == 0:
        return float("nan")
    d = a - b
    return float(np.sqrt(np.mean(np.sum(d * d, axis=1))))


def _mean_abs(a: np.ndarray, b: np.ndarray) -> float:
    a, b = _finite_pairs(a, b)
    if len(a) == 0:
        return float("nan")
    return float(np.mean(np.linalg.norm(a - b, axis=1)))


def _first_valid_index(*columns: np.ndarray) -> int:
    for i in range(len(columns[0])):
        if all(np.isfinite(col[i]) for col in columns):
            return i
    raise ValueError("localization.csv has no finite GNSS+IMU rows to seed the EKF")


def run_ekf_on_log(log: dict) -> dict[str, np.ndarray]:
    n = len(log["frame"])
    dt = np.diff(log["t_s"], prepend=log["t_s"][0])
    if n > 1:
        dt[0] = float(dt[1]) if np.isfinite(dt[1]) and dt[1] > 0 else 0.1

    # CARLA often delivers the first GNSS/IMU one tick late → frame 0 is NaN.
    i0 = _first_valid_index(log["gnss_e_m"], log["gnss_n_m"])
    speed0 = float(log["gt_speed_mps"][i0])
    if not np.isfinite(speed0):
        speed0 = 0.0
    # Seed heading from compass when present (inference-available), else GT.
    compass0 = float(log["imu_compass"][i0])
    if np.isfinite(compass0):
        yaw0 = compass_to_yaw_enu(compass0)
    else:
        yaw0 = float(log["gt_yaw_rad"][i0]) if np.isfinite(log["gt_yaw_rad"][i0]) else 0.0

    ekf = EgoEKF(
        x0=np.array(
            [
                float(log["gnss_e_m"][i0]),
                float(log["gnss_n_m"][i0]),
                yaw0,
                max(speed0, 0.0),
            ]
        )
    )
    est = np.full((n, 4), np.nan, dtype=np.float64)
    for i in range(i0, n):
        if i > i0:
            # CARLA gyro.z is opposite our ENU yaw derivative on Town maps.
            yaw_rate = float(log["imu_gz"][i])
            if not np.isfinite(yaw_rate):
                yaw_rate = 0.0
            else:
                yaw_rate = -yaw_rate
            speed = float(log["gt_speed_mps"][i])
            if not np.isfinite(speed):
                speed = None
            ekf.predict(float(dt[i]), yaw_rate=yaw_rate, speed=speed)
        ge = float(log["gnss_e_m"][i])
        gn = float(log["gnss_n_m"][i])
        if np.isfinite(ge) and np.isfinite(gn):
            ekf.update_gnss(ge, gn)
        compass = float(log["imu_compass"][i])
        if np.isfinite(compass):
            ekf.update_compass(compass)
        speed = float(log["gt_speed_mps"][i])
        if np.isfinite(speed):
            ekf.update_speed(speed)
        est[i] = ekf.x
        est[i, 2] = wrap_angle(est[i, 2])
    # Fill leading NaNs with the first estimate so plots stay continuous.
    if i0 > 0 and np.isfinite(est[i0]).all():
        est[:i0] = est[i0]
    return {
        "e": est[:, 0],
        "n": est[:, 1],
        "yaw": est[:, 2],
        "v": est[:, 3],
    }


def score(log: dict, est: dict[str, np.ndarray]) -> dict[str, float]:
    gt = np.stack([log["gt_e_m"], log["gt_n_m"]], axis=1)
    gnss = np.stack([log["gnss_e_m"], log["gnss_n_m"]], axis=1)
    filt = np.stack([est["e"], est["n"]], axis=1)
    yaw_err = np.array(
        [
            wrap_angle(float(a) - float(b))
            for a, b in zip(est["yaw"], log["gt_yaw_rad"])
            if np.isfinite(a) and np.isfinite(b)
        ]
    )
    gnss_valid = int(np.isfinite(gnss).all(axis=1).sum())
    return {
        "gnss_rmse_m": _rmse(gnss, gt),
        "ekf_rmse_m": _rmse(filt, gt),
        "gnss_mae_m": _mean_abs(gnss, gt),
        "ekf_mae_m": _mean_abs(filt, gt),
        "ekf_yaw_mae_deg": (
            float(np.mean(np.abs(yaw_err)) * 180.0 / np.pi) if len(yaw_err) else float("nan")
        ),
        "n_frames": int(len(log["frame"])),
        "n_gnss_valid": gnss_valid,
    }


def draw_trajectory(
    log: dict,
    est: dict[str, np.ndarray],
    out_path: Path,
    *,
    size: int = 800,
    margin: int = 40,
) -> Path:
    gt_e, gt_n = log["gt_e_m"], log["gt_n_m"]
    g_e, g_n = log["gnss_e_m"], log["gnss_n_m"]
    e_e, e_n = est["e"], est["n"]
    all_e = np.concatenate([gt_e, g_e, e_e])
    all_n = np.concatenate([gt_n, g_n, e_n])
    finite = np.isfinite(all_e) & np.isfinite(all_n)
    if not np.any(finite):
        raise ValueError("No finite positions to plot in localization log")
    min_e, max_e = float(np.nanmin(all_e)), float(np.nanmax(all_e))
    min_n, max_n = float(np.nanmin(all_n)), float(np.nanmax(all_n))
    span = max(max_e - min_e, max_n - min_n, 1.0)
    pad = 0.05 * span
    min_e -= pad
    max_e += pad
    min_n -= pad
    max_n += pad
    span_e = max(max_e - min_e, 1e-6)
    span_n = max(max_n - min_n, 1e-6)

    img = Image.new("RGB", (size, size), (18, 18, 22))
    draw = ImageDraw.Draw(img)

    def to_px(e: float, n: float) -> tuple[int, int] | None:
        if not (np.isfinite(e) and np.isfinite(n)):
            return None
        u = margin + (e - min_e) / span_e * (size - 2 * margin)
        # Image y down; north up.
        v = margin + (1.0 - (n - min_n) / span_n) * (size - 2 * margin)
        return int(u), int(v)

    def draw_path(es, ns, color, width: int = 2) -> None:
        pts = [p for e, n in zip(es, ns) if (p := to_px(float(e), float(n))) is not None]
        if len(pts) >= 2:
            draw.line(pts, fill=color, width=width)

    # GNSS as scattered dots.
    for e, n in zip(g_e[::2], g_n[::2]):
        px = to_px(float(e), float(n))
        if px is None:
            continue
        x, y = px
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(220, 80, 80))
    draw_path(gt_e, gt_n, (80, 220, 120), width=3)
    draw_path(e_e, e_n, (80, 160, 255), width=2)

    draw.text((12, 10), "green=GT  blue=EKF  red=GNSS", fill=(220, 220, 220))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    return out_path


def gate_passed(metrics: dict[str, float]) -> bool:
    """EKF must beat raw GNSS on position RMSE."""
    ekf = metrics["ekf_rmse_m"]
    gnss = metrics["gnss_rmse_m"]
    if not (np.isfinite(ekf) and np.isfinite(gnss)):
        return False
    return ekf < gnss * 0.75 and ekf < gnss - 0.25


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, help="Recorded run_dir with localization.csv")
    parser.add_argument(
        "--synth",
        action="store_true",
        help="Generate a synthetic noisy-GNSS run and evaluate (no CARLA needed)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("models/loc_preview"),
        help="Output directory for plot + metrics JSON",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.synth:
        run_dir = args.out / "synth_run"
        write_synth_run(run_dir, seed=args.seed)
        data_dir = run_dir
    elif args.data is not None:
        data_dir = args.data
    else:
        parser.error("Pass --data run_dir or --synth")

    log = load_localization(data_dir)
    est = run_ekf_on_log(log)
    metrics = score(log, est)
    metrics["gate_passed"] = gate_passed(metrics)

    args.out.mkdir(parents=True, exist_ok=True)
    plot_path = draw_trajectory(log, est, args.out / "trajectory.png")
    metrics_path = args.out / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")

    print(f"frames:     {metrics['n_frames']}")
    print(f"GNSS RMSE:  {metrics['gnss_rmse_m']:.3f} m")
    print(f"EKF  RMSE:  {metrics['ekf_rmse_m']:.3f} m")
    print(f"EKF yaw MAE:{metrics['ekf_yaw_mae_deg']:.2f} deg")
    print(f"gate:       {'PASS' if metrics['gate_passed'] else 'FAIL'}")
    print(f"plot:       {plot_path}")
    print(f"metrics:    {metrics_path}")


if __name__ == "__main__":
    main()
