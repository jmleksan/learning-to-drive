"""Scored Town01 drive for a trained PPO/SAC lane-keep policy.

Writes drive_metrics.json with the same fields as Stage 1.

  python -m rl.eval_drive --model models/rl_ppo/best_model.zip --algo ppo --seconds 120
  python -m rl.eval_drive --model models/rl_sac_cam/best_model.zip --algo sac --camera --seconds 180
  # checkpoints trained before raw steer became the default
  python -m rl.eval_drive --model models/rl_sac/best_model.zip --algo sac --steer-smooth
  # BEV / steer-only detected from the checkpoint; BEV road is SegNet unless --bev-source map
  python -m rl.eval_drive --model models/rl_sac_bev/best_model.zip --algo sac
  python -m rl.eval_drive --model models/rl_sac_bev/best_model.zip --algo sac --bev-source map
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path

from stable_baselines3 import PPO, SAC

from rl.env import CarlaLaneKeepEnv, env_kwargs_for_model
from utils.drive_score import collisions_per_km, route_completion_pct

_DRIVE_LOG_HEADER = [
    "frame",
    "steer",
    "throttle",
    "brake",
    "speed_kmh",
    "cte_m",
    "lane_offset_frac",
    "heading_err_deg",
    "lane_invasions",
    "collisions",
    "reward",
]


def eval_drive(
    *,
    model_path: Path,
    algo: str,
    seconds: float,
    fps: float,
    map_name: str,
    spawn_index: int,
    host: str | None,
    port: int,
    out: Path | None,
    device: str,
    slow: float = 1.0,
    warmup_seconds: float = 3.0,
    camera: bool = False,
    steer_smooth: bool = False,
    bev_source: str = "perceived",
    seg_path: Path = Path("models/seg.pt"),
    lights: bool = False,
    npc_vehicles: int = 0,
    walkers: int = 0,
) -> Path:
    frames = max(1, int(seconds * fps))
    algo = algo.lower()
    load_cls = PPO if algo == "ppo" else SAC
    model = load_cls.load(str(model_path), device=device)
    obs_kwargs = env_kwargs_for_model(model)
    camera = camera or obs_kwargs.get("camera", False)
    bev = obs_kwargs.get("bev", False)

    env = CarlaLaneKeepEnv(
        host=host,
        port=port,
        map_name=map_name,
        spawn_index=spawn_index,
        fps=fps,
        max_steps=frames,
        turn_start_prob=0.0,
        camera=camera,
        bev=bev,
        speed_control=obs_kwargs["speed_control"],
        speed_setpoint=obs_kwargs["speed_setpoint"],
        bev_objects=obs_kwargs.get("bev_objects", False),
        lights=lights,
        npc_vehicles=npc_vehicles,
        walkers=walkers,
        steer_smooth=steer_smooth,
        bev_source=bev_source if bev else "map",
        seg_path=seg_path,
    )
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(out) if out else Path(f"data/rl_drive_{algo}_{stamp}")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    red_runs = 0
    last_info: dict = {}
    warmup_info: dict = {}
    warmup_frames = max(0, int(float(warmup_seconds) * fps))
    pace = max(0.0, float(slow))
    if pace > 0:
        print(f"Eval pacing {pace:.1f}× slower than realtime (use --slow 0 for uncapped)")
    if warmup_frames:
        print(f"Skipping first {warmup_seconds:.1f}s ({warmup_frames} frames) from the scorecard")
    step_wall = (1.0 / max(1e-6, fps)) * pace
    try:
        obs, last_info = env.reset()
        next_t = time.perf_counter()
        for frame_idx in range(frames):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            if pace > 0:
                next_t += step_wall
                delay = next_t - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                else:
                    next_t = time.perf_counter()
            last_info = info
            red_runs += int(bool(info.get("ran_red")))
            if frame_idx + 1 == warmup_frames:
                warmup_info = dict(info)
            if frame_idx >= warmup_frames:
                rows.append(
                    {
                        "frame": str(frame_idx),
                        "steer": f"{info.get('steer', 0.0):.4f}",
                        "throttle": f"{info.get('throttle', 0.0):.4f}",
                        "brake": f"{info.get('brake', 0.0):.4f}",
                        "speed_kmh": f"{info.get('speed_kmh', 0.0):.3f}",
                        "cte_m": f"{info.get('cte_m', 0.0):.4f}",
                        "lane_offset_frac": f"{info.get('lane_offset_frac', 0.0):.4f}",
                        "heading_err_deg": f"{info.get('heading_err_deg', 0.0):.3f}",
                        "lane_invasions": str(info.get("lane_invasions", 0)),
                        "collisions": str(info.get("collisions", 0)),
                        "reward": f"{float(reward):.4f}",
                    }
                )
            if terminated or truncated:
                break
    finally:
        env.close()

    n_col = max(
        0,
        int(last_info.get("collisions", 0)) - int(warmup_info.get("collisions", 0)),
    )
    n_inv = max(
        0,
        int(last_info.get("lane_invasions", 0))
        - int(warmup_info.get("lane_invasions", 0)),
    )
    odom_m = max(
        0.0,
        float(last_info.get("odom_m", 0.0)) - float(warmup_info.get("odom_m", 0.0)),
    )
    progress_m = float(last_info.get("route_progress_m", 0.0))
    planned_m = float(last_info.get("route_length_m", 0.0))
    speeds = [float(r["speed_kmh"]) for r in rows]
    ctes = [abs(float(r["cte_m"])) for r in rows]
    per_km = collisions_per_km(n_col, odom_m)
    metrics = {
        "agent": f"rl_{algo}_cam" if camera else (f"rl_{algo}_bev" if bev else f"rl_{algo}"),
        "obs": "camera" if camera else ("bev" if bev else "gps"),
        "speed_control": obs_kwargs["speed_control"],
        "bev_source": bev_source if bev else None,
        "model": str(model_path),
        "lane_invasions": n_inv,
        "red_light_runs": red_runs,
        "traffic": {"lights": lights, "npc_vehicles": npc_vehicles, "walkers": walkers},
        "collisions": n_col,
        "distance_m": round(odom_m, 2),
        "route_progress_m": round(progress_m, 2),
        "route_length_m": round(planned_m, 2),
        "route_completion_pct": round(
            route_completion_pct(progress_m, planned_m), 2
        ),
        "avg_speed_kmh": round(
            (sum(speeds) / max(1, len(speeds))), 2
        ),
        "collisions_per_km": round(per_km, 3),
        "max_collisions_per_km": 1.0,
        "spawn_index": spawn_index,
        "map": map_name,
        "seconds_requested": seconds,
        "warmup_seconds": warmup_seconds,
        "frames": len(rows),
        "milestone": "PASS" if per_km <= 1.0 else "FAIL",
    }
    if ctes:
        ctes_sorted = sorted(ctes)
        metrics["cte_p50_m"] = round(ctes_sorted[len(ctes_sorted) // 2], 3)
        metrics["cte_p90_m"] = round(
            ctes_sorted[int(0.9 * (len(ctes_sorted) - 1))], 3
        )

    log_path = out_dir / "drive_log.csv"
    with log_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_DRIVE_LOG_HEADER)
        writer.writeheader()
        writer.writerows(rows)
    metrics_path = out_dir / "drive_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Wrote {log_path} and {metrics_path}")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--algo", choices=["ppo", "sac"], required=True)
    parser.add_argument("--seconds", type=float, default=120.0)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--map", dest="map_name", default="Town01")
    parser.add_argument("--spawn-index", type=int, default=0)
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--slow",
        type=float,
        default=1.0,
        help="Wall-clock pace vs sim time. 1=realtime (default), 2=half speed, 0=uncapped.",
    )
    parser.add_argument(
        "--warmup-seconds",
        type=float,
        default=3.0,
        help="Drop this many seconds from the scorecard (spawn settle).",
    )
    parser.add_argument(
        "--camera",
        action="store_true",
        help="Match a --camera trained policy (stacked RGB + speed/command).",
    )
    parser.add_argument(
        "--steer-smooth",
        action="store_true",
        help="Old EMA steer filter. Use for models trained before raw steer "
        "was the default (e.g. models/rl_sac, models/rl_sac_cam).",
    )
    parser.add_argument(
        "--lights",
        action="store_true",
        help="Traffic lights matter: red/yellow stop lines in the BEV, hazard "
        "speed cap and a red-light penalty. Model must be a traffic (setpoint) model.",
    )
    parser.add_argument("--npc-vehicles", type=int, default=0, help="Autopilot cars (needs --bev).")
    parser.add_argument("--walkers", type=int, default=0, help="AI pedestrians (needs --bev).")
    parser.add_argument(
        "--bev-source",
        choices=["map", "perceived"],
        default="perceived",
        help="BEV road channel: SegNet road from center/left/right cameras "
        "(default, needs rendering) or map truth (privileged, works with "
        "--no-render). Route channel always comes from the planner.",
    )
    parser.add_argument("--seg", type=Path, default=Path("models/seg.pt"), help="SegNet for --bev-source perceived.")
    args = parser.parse_args()
    if not args.model.exists():
        raise SystemExit(f"Model not found: {args.model}")
    eval_drive(
        model_path=args.model,
        algo=args.algo,
        seconds=args.seconds,
        fps=args.fps,
        map_name=args.map_name,
        spawn_index=args.spawn_index,
        host=args.host,
        port=args.port,
        out=args.out,
        device=args.device,
        slow=args.slow,
        warmup_seconds=args.warmup_seconds,
        camera=args.camera,
        steer_smooth=args.steer_smooth,
        bev_source=args.bev_source,
        seg_path=args.seg,
        lights=args.lights,
        npc_vehicles=args.npc_vehicles,
        walkers=args.walkers,
    )


if __name__ == "__main__":
    main()
