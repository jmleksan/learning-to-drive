"""Drive the RL env with the hand-coded waypoint follower instead of a policy.

Answers "is the env + reward fair?" before blaming the learner. If the
follower clears corners here and the reward terms look sensible in turns, the
env is fine and the problem is learning. If it fails or turns reward strangely,
the env/reward is the bug.

  python -m rl.check_follower --episodes 10 --fixed-route
"""
from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np

from rl.env import CTE_FAIL_M, CarlaLaneKeepEnv
from utils.waypoint_follower import WaypointFollower

_TERMS = (
    "r_progress",
    "r_cte",
    "r_heading",
    "r_steer",
    "r_jerk",
    "r_speed",
    "r_hold",
    "r_flip",
    "r_match",
    "reward",
)


def run(args: argparse.Namespace) -> None:
    env = CarlaLaneKeepEnv(
        host=args.host,
        port=args.port,
        spawn_index=args.spawn_index,
        camera=args.camera,
        fixed_route=args.fixed_route,
        no_rendering=args.no_render,
        turn_start_prob=args.turn_mix,
    )
    follower = WaypointFollower(cruise_kmh=18.0, turn_kmh=14.0)
    sums = {True: defaultdict(float), False: defaultdict(float)}
    counts = {True: 0, False: 0}
    try:
        for ep in range(args.episodes):
            env.reset()
            follower.reset()
            follower.route_index = env._route_index
            steps = 0
            ret = 0.0
            max_cte = 0.0
            turn_steps = 0
            info: dict = {}
            reason = "?"
            while True:
                turning = env._upcoming_turn()
                ctrl, _target, _cmd = follower.control(env._vehicle, env._route)
                action = np.array([ctrl.steer, ctrl.throttle], dtype=np.float32)
                _obs, reward, terminated, truncated, info = env.step(action)
                steps += 1
                ret += reward
                max_cte = max(max_cte, abs(info["cte_m"]))
                turn_steps += int(turning)
                counts[turning] += 1
                for k in _TERMS:
                    sums[turning][k] += info.get(k, 0.0)
                if terminated or truncated:
                    if info["collisions"]:
                        reason = "collision"
                    elif abs(info["cte_m"]) > CTE_FAIL_M:
                        reason = "cte>limit"
                    elif info["route_completion_pct"] >= 95.0:
                        reason = "route done"
                    else:
                        reason = "timeout"
                    break
            print(
                f"ep {ep:2d}  steps {steps:4d}  return {ret:7.2f}  "
                f"turn-steps {turn_steps:3d}  max|cte| {max_cte:4.2f}  "
                f"route {info['route_completion_pct']:5.1f}%  end: {reason}"
            )
    finally:
        env.close()

    print("\nmean reward terms per step")
    print(f"{'term':12s} {'turning':>9s} {'straight':>9s}")
    for k in _TERMS:
        t = sums[True][k] / max(1, counts[True])
        s = sums[False][k] / max(1, counts[False])
        print(f"{k:12s} {t:9.3f} {s:9.3f}")
    print(f"steps: turning={counts[True]} straight={counts[False]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--spawn-index", type=int, default=0)
    parser.add_argument("--fixed-route", action="store_true")
    parser.add_argument("--camera", action="store_true", help="Camera obs (renderer on).")
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--turn-mix", type=float, default=None)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
