"""Deterministic turn test for camera / BEV policies: no exploration noise.

Training's turns/clear_rate_* is measured with gSDE noise on, so it can
hide (or fake) progress. This replays the same seeded set of turn
approaches for every checkpoint, so rows are directly comparable.

  python -m rl.eval_turns --model models/rl_sac_cam_sde/checkpoints/sac_lane_300000_steps.zip \\
      --model models/rl_sac_cam_sde/checkpoints/sac_lane_480000_steps.zip --episodes 40
  # BEV road is SegNet by default; --bev-source map for map truth
  python -m rl.eval_turns --model models/rl_sac_bev/best_model.zip --bev-source map
  # add the waypoint follower as a ceiling
  python -m rl.eval_turns --model models/rl_sac_cam_sde/best_model.zip --follower
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO, SAC

from rl.env import CarlaLaneKeepEnv, env_kwargs_for_model
from utils.waypoint_follower import WaypointFollower

_KINDS = ("left", "right", "straight")


def _reset_to_kind(env: CarlaLaneKeepEnv, kinds: set[str], seed: int | None):
    """Reset until the episode approaches one of `kinds`. Uses only env RNG."""
    obs, _ = env.reset(seed=seed)
    while env._turn_kind not in kinds:
        obs, _ = env.reset()
    return obs


def _run_episode(env: CarlaLaneKeepEnv, obs, act) -> dict:
    """Drive until the corner is cleared or the episode ends. act(obs) -> action."""
    speeds: list[float] = []
    while True:
        obs, _r, terminated, truncated, info = env.step(act(obs))
        speeds.append(float(info["speed_kmh"]))
        cleared = env._route_index >= int(env._turn_exit_i)
        if cleared or terminated or truncated:
            break
    if cleared:
        end = "cleared"
    else:
        end = info.get("end_reason", "timeout")
    return {
        "kind": env._turn_kind,
        "cleared": cleared,
        "end": end,
        "miss": None if cleared else env._turn_miss(info["cte_m"]),
        "mean_speed_kmh": float(np.mean(speeds)) if speeds else 0.0,
    }


def _evaluate(env, name: str, make_act, episodes: int, kinds: set[str], seed: int) -> dict:
    results = []
    for ep in range(episodes):
        obs = _reset_to_kind(env, kinds, seed if ep == 0 else None)
        results.append(_run_episode(env, obs, make_act()))
        r = results[-1]
        print(
            f"  {name} ep {ep:3d}  {r['kind']:8s}  "
            f"{'cleared' if r['cleared'] else 'MISS ' + str(r['miss']) + ' / ' + r['end']}"
            f"  {r['mean_speed_kmh']:.1f} km/h"
        )
    summary: dict = {"name": name, "episodes": len(results)}
    for kind in _KINDS:
        rows = [r for r in results if r["kind"] == kind]
        if rows:
            summary[f"clear_{kind}"] = round(sum(r["cleared"] for r in rows) / len(rows), 3)
            summary[f"n_{kind}"] = len(rows)
    misses = Counter(r["miss"] for r in results if not r["cleared"] and r["kind"] != "straight")
    summary["misses"] = dict(misses)
    summary["ends"] = dict(Counter(r["end"] for r in results))
    summary["mean_speed_kmh"] = round(float(np.mean([r["mean_speed_kmh"] for r in results])), 2)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, action="append", default=[])
    parser.add_argument("--algo", choices=["ppo", "sac"], default="sac")
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument(
        "--kinds",
        default="left,right",
        help="Comma list of left,right,straight to test (default left,right).",
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--follower", action="store_true", help="Also score the waypoint follower.")
    parser.add_argument("--steer-smooth", action="store_true", help="For models trained with the old EMA filter.")
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
    parser.add_argument("--map", dest="map_name", default="Town01")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    kinds = {k.strip() for k in args.kinds.split(",") if k.strip()}
    if not kinds <= set(_KINDS):
        raise SystemExit(f"--kinds must be from {_KINDS}")
    if not args.model and not args.follower:
        raise SystemExit("Give at least one --model or --follower")
    for path in args.model:
        if not path.exists():
            raise SystemExit(f"Model not found: {path}")

    load_cls = PPO if args.algo == "ppo" else SAC
    models = [load_cls.load(str(path), device=args.device) for path in args.model]
    # One env for every row, so all models must share obs/action spaces.
    obs_kwargs = env_kwargs_for_model(models[0]) if models else {"camera": True}
    for path, model in zip(args.model, models):
        if env_kwargs_for_model(model) != obs_kwargs:
            raise SystemExit(f"{path} uses different obs/actions than {args.model[0]}")
    env = CarlaLaneKeepEnv(
        host=args.host,
        port=args.port,
        map_name=args.map_name,
        fps=args.fps,
        turn_start_prob=1.0,
        steer_smooth=args.steer_smooth,
        **obs_kwargs,
        bev_source=args.bev_source if obs_kwargs.get("bev") else "map",
        seg_path=args.seg,
        lights=args.lights,
        npc_vehicles=args.npc_vehicles,
        walkers=args.walkers,
    )
    summaries = []
    try:
        for path, model in zip(args.model, models):

            def make_act(model=model):
                return lambda obs: model.predict(obs, deterministic=True)[0]

            print(f"== {path}")
            summaries.append(
                _evaluate(env, str(path), make_act, args.episodes, kinds, args.seed)
            )

        if args.follower:
            follower = WaypointFollower(cruise_kmh=18.0, turn_kmh=14.0)

            def make_follower_act():
                follower.reset()
                follower.route_index = env._route_index

                def act(_obs):
                    ctrl, _t, _c = follower.control(env._vehicle, env._route)
                    return env.follower_action(ctrl)

                return act

            print("== follower")
            summaries.append(
                _evaluate(env, "follower", make_follower_act, args.episodes, kinds, args.seed)
            )
    finally:
        env.close()

    print()
    cols = [f"clear_{k}" for k in _KINDS if k in kinds]
    print(f"{'model':60s} " + " ".join(f"{c:>14s}" for c in cols) + "   misses (wide/tight/approach)   km/h")
    for s in summaries:
        m = s["misses"]
        print(
            f"{s['name'][-60:]:60s} "
            + " ".join(f"{s.get(c, float('nan')):14.2f}" for c in cols)
            + f"   {m.get('wide', 0):>4d}/{m.get('tight', 0)}/{m.get('approach', 0):<14d}"
            + f"   {s['mean_speed_kmh']:.1f}"
        )
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = args.out or Path(f"data/rl_eval_turns_{stamp}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
