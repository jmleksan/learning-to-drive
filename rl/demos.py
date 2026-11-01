"""Warm-start a camera SAC from the hand-coded waypoint follower.

Pure RL from pixels never found the corners, but the follower clears them in
the same env (see rl/check_follower.py). So: drive the env with the follower,
store those transitions in the SAC replay buffer, then behaviour-clone the
actor on them before normal SAC takes over.
"""
from __future__ import annotations

import numpy as np
import torch as th
import torch.nn.functional as F
from stable_baselines3.common.logger import configure

from utils.waypoint_follower import WaypointFollower

# Pull the cloned actor's std in so early rollouts stay near the demo.
BC_LOG_STD_TARGET = -1.5
BC_LOG_STD_WEIGHT = 0.1


def _batched(obs: dict) -> dict:
    return {k: np.asarray(v)[None] for k, v in obs.items()}


def collect_follower_demos(env, model, steps: int) -> int:
    """Fill model.replay_buffer with follower-driven transitions.

    env is the (Monitor-wrapped) CarlaLaneKeepEnv, not the VecEnv.
    Returns the number of transitions stored.
    """
    raw = env.unwrapped
    follower = WaypointFollower(cruise_kmh=18.0, turn_kmh=14.0)
    obs, _ = env.reset()
    follower.reset()
    follower.route_index = raw._route_index
    stored = 0
    while stored < steps:
        ctrl, _target, _cmd = follower.control(raw._vehicle, raw._route)
        action = raw.follower_action(ctrl)
        action = np.clip(action, env.action_space.low, env.action_space.high)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        model.replay_buffer.add(
            _batched(obs),
            _batched(next_obs),
            model.policy.scale_action(action)[None],
            np.array([reward], dtype=np.float32),
            np.array([done]),
            [{"TimeLimit.truncated": bool(truncated and not terminated)}],
        )
        stored += 1
        obs = next_obs
        if done:
            obs, _ = env.reset()
            follower.reset()
            follower.route_index = raw._route_index
        if stored % 2000 == 0:
            print(f"  demos: {stored}/{steps}")
    return stored


def bc_pretrain(model, steps: int, batch_size: int = 128) -> None:
    """Behaviour-clone the SAC actor on whatever is in the replay buffer."""
    actor = model.policy.actor
    actor.set_training_mode(True)
    for i in range(steps):
        batch = model.replay_buffer.sample(batch_size, env=model._vec_normalize_env)
        mean, log_std, _ = actor.get_action_dist_params(batch.observations)
        loss = F.mse_loss(th.tanh(mean), batch.actions)
        # Under gSDE log_std is the noise-weight matrix, not a per-action std,
        # so the -1.5 target doesn't mean the same thing. Leave it alone.
        if not getattr(model, "use_sde", False):
            loss = loss + BC_LOG_STD_WEIGHT * ((log_std - BC_LOG_STD_TARGET) ** 2).mean()
        actor.optimizer.zero_grad()
        loss.backward()
        actor.optimizer.step()
        if (i + 1) % 500 == 0:
            print(f"  bc step {i + 1}/{steps}  loss {loss.item():.4f}")
    actor.set_training_mode(False)


def critic_warmup(model, steps: int, batch_size: int = 128) -> None:
    """Train only the critic on the demo buffer, actor and ent_coef frozen.

    A fresh critic is noise, and SAC's first actor updates chase it, which
    erases the cloned actor. Let the critic learn the demo values first.
    """
    # train() logs, but SB3 only builds the logger inside learn(). Use a
    # throwaway one; learn() replaces it with the real TensorBoard logger.
    model._logger = configure(None, [])
    actor_opt = model.policy.actor.optimizer
    ent_opt = getattr(model, "ent_coef_optimizer", None)
    actor_opt.step = lambda *a, **k: None
    if ent_opt is not None:
        ent_opt.step = lambda *a, **k: None
    try:
        model.train(gradient_steps=steps, batch_size=batch_size)
    finally:
        del actor_opt.step
        if ent_opt is not None:
            del ent_opt.step


def check_policy(env, model, episodes: int = 3) -> None:
    """Drive the deterministic actor and print how each episode ends."""
    raw = env.unwrapped
    for ep in range(episodes):
        obs, _ = env.reset()
        steps = 0
        max_cte = 0.0
        steer_abs = 0.0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, _r, terminated, truncated, info = env.step(action)
            steps += 1
            max_cte = max(max_cte, abs(info["cte_m"]))
            steer_abs += abs(info["steer"])
            if terminated or truncated:
                break
        print(
            f"  cloned actor ep {ep}: steps {steps}  max|cte| {max_cte:.2f}  "
            f"mean|steer| {steer_abs / steps:.3f}  "
            f"route {info['route_completion_pct']:.1f}%  "
            f"col {info['collisions']}  turn_ep {raw._turn_episode}"
        )
