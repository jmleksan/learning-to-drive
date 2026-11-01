"""Train PPO or SAC on the CARLA lane-keeping env.

Examples:
  python -m rl.train --algo ppo --timesteps 2000 --map Town01
  python -m rl.train --algo sac --timesteps 100000 --out models/rl_sac
  python -m rl.train --algo sac --timesteps 100000 --resume models/rl_sac/best_model.zip
  python -m rl.train --algo sac --camera --timesteps 100000 --out models/rl_sac_cam
  python -m rl.train --algo sac --camera --sde 20 --timesteps 300000 --out models/rl_sac_cam_sde
  python -m rl.train --algo sac --bev --bev-source map --speed-control --sde 20 --no-render --timesteps 300000 --out models/rl_sac_bev
  # fine-tune the map-BEV policy on SegNet road (perceived is the default)
  python -m rl.train --algo sac --bev --speed-control \
      --resume models/rl_sac_bev/best_model.zip --fresh-replay --timesteps 100000 --out models/rl_sac_bev_perc
"""
from __future__ import annotations

import argparse
import math
import re
import signal
from collections import deque
from pathlib import Path

import torch as th

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.type_aliases import TrainFreq, TrainFrequencyUnit

from rl.demos import (
    bc_pretrain,
    check_policy,
    collect_follower_demos,
    critic_warmup,
)
from rl.env import CAM_TURN_START_PROB, CarlaLaneKeepEnv

_ALGOS = {"ppo": PPO, "sac": SAC}


class EntropyFloor(BaseCallback):
    """Keep SAC's auto ent_coef from collapsing before turns are learned."""

    def __init__(self, floor: float) -> None:
        super().__init__()
        self.log_floor = math.log(floor)

    def _on_step(self) -> bool:
        log_ent = getattr(self.model, "log_ent_coef", None)
        if log_ent is not None:
            with th.no_grad():
                log_ent.clamp_(min=self.log_floor)
        return True


END_REASONS = ("collision", "off_lane", "red_light", "stalled", "route_done", "timeout")
TURN_MISSES = ("wide", "tight", "approach")


class EpisodeStats(BaseCallback):
    """Rolling per-episode stats ep_len/ep_rew can't show.

    turns/clear_rate_*  camera turn episodes that got past their corner
    turns/miss_*        how missed lefts/rights failed (wide = too little steer)
    ends/rate_*         how every episode ended
    ends/mean_speed_kmh mean speed over the episode
    """

    def __init__(self, window: int = 50, end_window: int = 100) -> None:
        super().__init__()
        self.window = window
        self.results: dict[str, deque[float]] = {}
        self.misses: deque[str] = deque(maxlen=window)
        self.ends: deque[str] = deque(maxlen=end_window)
        self.speeds: deque[float] = deque(maxlen=end_window)

    def _on_step(self) -> bool:
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if not done:
                continue
            reason = info.get("end_reason")
            if reason is not None:
                self.ends.append(reason)
                self.speeds.append(float(info.get("ep_mean_speed_kmh", 0.0)))
                for key in END_REASONS:
                    self.logger.record(
                        f"ends/rate_{key}", self.ends.count(key) / len(self.ends)
                    )
                self.logger.record(
                    "ends/mean_speed_kmh", sum(self.speeds) / len(self.speeds)
                )
            kind = info.get("turn_kind")
            if kind is None:
                continue
            for key in (kind, "all"):
                hist = self.results.setdefault(key, deque(maxlen=self.window))
                hist.append(1.0 if info.get("turn_cleared") else 0.0)
            for key, hist in self.results.items():
                self.logger.record(f"turns/clear_rate_{key}", sum(hist) / len(hist))
            miss = info.get("turn_miss")
            if miss is not None:
                self.misses.append(miss)
                for key in TURN_MISSES:
                    self.logger.record(
                        f"turns/miss_{key}", self.misses.count(key) / len(self.misses)
                    )
        return True


def _replay_path(out: Path) -> Path:
    return out / "replay_buffer.pkl"


def _load_model(
    algo: str,
    path: Path,
    env,
    out: Path,
    device: str,
    *,
    load_replay: bool = True,
):
    cls = _ALGOS[algo]
    model = cls.load(str(path), env=env, device=device, tensorboard_log=str(out))
    replay = _replay_path(out)
    if algo == "sac" and load_replay and replay.is_file():
        model.load_replay_buffer(str(replay))
        print(f"Loaded SAC replay buffer ({replay})")
    elif algo == "sac" and not load_replay:
        print("Skipping old replay buffer (fresh turn scenarios)")
    if algo == "sac":
        model.train_freq = TrainFreq(1, TrainFrequencyUnit.STEP)
        model.gradient_steps = 1
    return model


def _latest_checkpoint(out: Path, algo: str) -> tuple[Path, int] | None:
    """Newest {algo}_lane_<N>_steps[_interrupted].zip under out/checkpoints."""
    best = None
    for path in (out / "checkpoints").glob(f"{algo}_lane_*_steps*.zip"):
        m = re.fullmatch(rf"{algo}_lane_(\d+)_steps(_interrupted)?\.zip", path.name)
        if m and (best is None or int(m.group(1)) > best[1]):
            best = (path, int(m.group(1)))
    return best


def _save_model(algo: str, model, dest: Path, out: Path) -> Path:
    model.save(dest)
    if algo == "sac":
        model.save_replay_buffer(str(_replay_path(out)))
    zip_path = Path(str(dest) + ".zip")
    print(f"Saved {zip_path}")
    return zip_path


def _build_model(
    algo: str,
    env,
    out: Path,
    seed: int,
    device: str,
    camera: bool,
    warm: bool = False,
    sde_freq: int = 0,
):
    tb = str(out)
    # gSDE holds the exploration offset for sde_freq steps, so the car tries
    # sustained steer (a corner) instead of per-step jitter that cancels out.
    # SB3's log_std_init=-3 gives ~0.04 steer noise; -1 gives ~0.2, enough
    # that some 1 s windows hold >0.3 steer, about what a Town01 corner needs.
    sde = (
        {
            "use_sde": True,
            "sde_sample_freq": sde_freq,
            "use_sde_at_warmup": True,
            "policy_kwargs": {"log_std_init": -1.0},
        }
        if sde_freq > 0
        else {}
    )
    policy = "MultiInputPolicy" if camera else "MlpPolicy"
    if algo == "ppo":
        return PPO(
            policy,
            env,
            n_steps=512 if camera else 1024,
            batch_size=64,
            n_epochs=10,
            learning_rate=3e-4,
            gamma=0.99,
            verbose=1,
            seed=seed,
            device=device,
            tensorboard_log=tb,
        )
    if camera:
        # Vision SAC: auto entropy (fixed 0.1 swamped ~0.3/step rewards), slower CNN lr.
        # Do not resume the first RGB run — obs and hyperparams changed.
        return SAC(
            policy,
            env,
            buffer_size=50_000,
            batch_size=128,
            learning_starts=5_000,
            learning_rate=1e-4,
            gamma=0.99,
            tau=0.005,
            # warm start (demos): begin with little entropy bonus, not 1.0.
            ent_coef="auto_0.05" if warm else "auto",
            train_freq=1,
            gradient_steps=1,
            verbose=1,
            seed=seed,
            device=device,
            tensorboard_log=tb,
            **sde,
        )
    return SAC(
        policy,
        env,
        buffer_size=50_000,
        batch_size=256,
        learning_starts=1_000,
        learning_rate=3e-4,
        gamma=0.99,
        verbose=1,
        seed=seed,
        device=device,
        tensorboard_log=tb,
        **sde,
    )


def train(
    *,
    algo: str,
    timesteps: int,
    out: Path,
    map_name: str,
    spawn_index: int,
    host: str | None,
    port: int,
    fps: float,
    max_steps: int,
    seed: int,
    device: str,
    resume: Path | None,
    no_rendering: bool,
    camera: bool,
    turn_start_prob: float | None,
    fresh_replay: bool,
    fixed_route: bool,
    ent_floor: float,
    demo_steps: int,
    bc_steps: int,
    critic_steps: int,
    sde_freq: int = 0,
    steer_smooth: bool = False,
    bev: bool = False,
    bev_source: str = "perceived",
    seg_path: Path = Path("models/seg.pt"),
    speed_control: bool = False,
    speed_setpoint: bool = False,
    lights: bool = False,
    npc_vehicles: int = 0,
    walkers: int = 0,
    checkpoint_every: int = 10_000,
) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    image_obs = camera or bev
    env = CarlaLaneKeepEnv(
        host=host,
        port=port,
        map_name=map_name,
        spawn_index=spawn_index,
        fps=fps,
        max_steps=max_steps,
        no_rendering=no_rendering,
        camera=camera,
        bev=bev,
        speed_control=speed_control,
        speed_setpoint=speed_setpoint,
        lights=lights,
        npc_vehicles=npc_vehicles,
        walkers=walkers,
        fixed_route=fixed_route,
        steer_smooth=steer_smooth,
        bev_source=bev_source if bev else "map",
        seg_path=seg_path,
        turn_start_prob=(
            turn_start_prob
            if turn_start_prob is not None
            else (CAM_TURN_START_PROB if image_obs else None)
        ),
    )
    env = Monitor(env)
    eyes = "camera" if camera else (f"bev/{bev_source}" if bev else "gps")
    if speed_control:
        eyes += ", steer-only"
    if speed_setpoint:
        eyes += ", speed setpoint"
    if lights or npc_vehicles or walkers:
        eyes += f", traffic(lights={lights}, cars={npc_vehicles}, walkers={walkers})"
    if resume is not None:
        if not resume.exists():
            env.close()
            raise FileNotFoundError(f"Resume checkpoint not found: {resume}")
        model = _load_model(
            algo, resume, env, out, device, load_replay=not fresh_replay
        )
        already = int(model.num_timesteps)
        print(
            f"Resuming {algo.upper()} ({eyes}) from {resume} "
            f"({already} steps done, +{timesteps} more) → {out} "
            f"(render={not no_rendering})"
        )
    else:
        if sde_freq > 0 and algo != "sac":
            env.close()
            raise SystemExit("--sde is wired for SAC only")
        model = _build_model(
            algo, env, out, seed, device, image_obs, warm=demo_steps > 0, sde_freq=sde_freq
        )
        already = 0
        print(
            f"Training {algo.upper()} ({eyes}) for {timesteps} steps → {out} "
            f"(map={map_name}, spawn={spawn_index}, render={not no_rendering})"
        )
    if demo_steps > 0:
        if algo != "sac" or resume is not None:
            env.close()
            raise SystemExit("--demo-steps needs a fresh SAC run (no --resume)")
        print(f"Collecting {demo_steps} follower demo steps ...")
        n = collect_follower_demos(env, model, demo_steps)
        print(f"Behaviour-cloning actor ({bc_steps} steps on {n} transitions)")
        bc_pretrain(model, bc_steps)
        print("Cloned actor on its own:")
        check_policy(env, model, episodes=3)
        print(f"Critic warm-up ({critic_steps} steps, actor frozen)")
        critic_warmup(model, critic_steps)
        # Otherwise SB3 plays uniform-random actions until learning_starts,
        # which would throw the cloned actor away.
        model.learning_starts = 0
    checkpoint = CheckpointCallback(
        save_freq=max(1_000, checkpoint_every),
        save_path=str(out / "checkpoints"),
        name_prefix=f"{algo}_lane",
    )
    callbacks = [checkpoint, EpisodeStats()]
    if algo == "sac" and ent_floor > 0.0:
        callbacks.append(EntropyFloor(ent_floor))
    try:
        model.learn(
            total_timesteps=int(timesteps),
            callback=callbacks,
            tb_log_name=algo,
            reset_num_timesteps=resume is None,
        )
        return _save_model(algo, model, out / "best_model", out)
    except KeyboardInterrupt:
        # Ctrl+C: keep the weights (and SAC replay) instead of losing
        # everything since the last checkpoint. A second Ctrl+C mid-save
        # would leave a truncated replay pickle, so ignore it until done.
        previous = signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            dest = out / "checkpoints" / f"{algo}_lane_{model.num_timesteps}_steps_interrupted"
            print(f"\nInterrupted at {model.num_timesteps} steps; saving (Ctrl+C ignored until done) ...")
            return _save_model(algo, model, dest, out)
        finally:
            signal.signal(signal.SIGINT, previous)
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algo", choices=["ppo", "sac"], required=True)
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--map", dest="map_name", default="Town01")
    parser.add_argument("--spawn-index", type=int, default=0)
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device",
        default=None,
        help="cpu, cuda, or auto. PPO defaults to cpu (MLP is slower on GPU).",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Continue from a .zip (weights + SAC replay buffer if present). "
        "--timesteps is additional steps, not a new start from zero.",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="CARLA no-rendering mode (faster ticks, blank UE4 view).",
    )
    parser.add_argument(
        "--camera",
        action="store_true",
        help="Policy sees stacked center RGB + speed/command (same reward). "
        "Do not --resume a GPS checkpoint into this.",
    )
    parser.add_argument(
        "--turn-mix",
        type=float,
        default=None,
        help="Fraction of train episodes that start on a turn approach "
        "(camera default 0.85). Eval should leave this unset.",
    )
    parser.add_argument(
        "--fresh-replay",
        action="store_true",
        help="Resume weights only; do not load the old SAC replay (straights).",
    )
    parser.add_argument(
        "--fixed-route",
        action="store_true",
        help="Camera train: keep the --spawn-index route (no random origins), "
        "like the GPS runs. Turn-approach starts still apply.",
    )
    parser.add_argument(
        "--ent-floor",
        type=float,
        default=0.04,
        help="SAC: minimum ent_coef (0 disables). Stops exploration collapsing.",
    )
    parser.add_argument(
        "--demo-steps",
        type=int,
        default=0,
        help="Fresh SAC only: first drive this many steps with the waypoint "
        "follower, fill the replay buffer, and behaviour-clone the actor.",
    )
    parser.add_argument(
        "--sde",
        type=int,
        default=0,
        metavar="N",
        help="Fresh SAC only: gSDE exploration, resampled every N steps "
        "(20 = 1 s at 20 fps). 0 = plain per-step noise.",
    )
    parser.add_argument(
        "--steer-smooth",
        action="store_true",
        help="Old EMA steer filter (adds ~0.15 s lag). Off by default; only "
        "for resuming runs that were trained with it.",
    )
    parser.add_argument(
        "--bev",
        action="store_true",
        help="Policy sees a ground-truth bird's-eye road + route grid (privileged, "
        "no camera needed, works with --no-render). Same reward.",
    )
    parser.add_argument(
        "--speed-setpoint",
        action="store_true",
        help="Action is (steer, target speed 0-25 km/h); a P controller does the "
        "pedals. Needed for traffic so the agent can stop.",
    )
    parser.add_argument(
        "--lights",
        action="store_true",
        help="Traffic lights matter: red/yellow stop lines in the BEV, hazard "
        "speed cap and a red-light penalty. Needs --bev; implies --speed-setpoint.",
    )
    parser.add_argument("--npc-vehicles", type=int, default=0, help="Autopilot cars (needs --bev).")
    parser.add_argument("--walkers", type=int, default=0, help="AI pedestrians (needs --bev).")
    parser.add_argument(
        "--speed-control",
        action="store_true",
        help="Steer-only action; a P controller holds the follower's 18/14 km/h "
        "speed profile.",
    )
    parser.add_argument(
        "--bev-source",
        choices=["map", "perceived"],
        default="perceived",
        help="BEV road channel: SegNet road from center/left/right cameras "
        "(default, needs rendering) or map truth (privileged, works with "
        "--no-render). Route channel always comes from the planner.",
    )
    parser.add_argument("--seg", type=Path, default=Path("models/seg.pt"), help="SegNet for --bev-source perceived.")
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=10_000,
        help="Save a checkpoint every N steps (each ~20 MB for BEV SAC).",
    )
    parser.add_argument(
        "--auto-resume",
        action="store_true",
        help="Resume from the newest checkpoint in --out (start fresh if none) and "
        "treat --timesteps as the run's TOTAL. Exits at once if already done, so "
        "a shell loop can restart it after a crash.",
    )
    parser.add_argument("--bc-steps", type=int, default=3_000)
    parser.add_argument("--critic-steps", type=int, default=3_000)
    args = parser.parse_args()
    if args.camera and args.bev:
        raise SystemExit("pick one of --camera or --bev")
    if args.lights or args.npc_vehicles or args.walkers:
        if not args.bev:
            raise SystemExit("--lights / --npc-vehicles / --walkers need --bev")
        if args.speed_control:
            raise SystemExit("traffic needs --speed-setpoint, not --speed-control")
        if not args.speed_setpoint:
            print("Traffic on: using --speed-setpoint so the agent can stop")
            args.speed_setpoint = True
    suffix = "_cam" if args.camera else ("_bev" if args.bev else "")
    out = args.out or Path(f"models/rl_{args.algo}{suffix}")
    timesteps = args.timesteps
    if args.auto_resume:
        if args.resume is not None:
            raise SystemExit("pick one of --resume or --auto-resume")
        if (out / "best_model.zip").exists():
            print(f"{out / 'best_model.zip'} exists: run already finished")
            return
        latest = _latest_checkpoint(out, args.algo)
        if latest is not None:
            args.resume, done = latest
            timesteps = args.timesteps - done
            if timesteps <= 0:
                print(f"{args.resume} already has {done} >= {args.timesteps} steps")
                return
        else:
            print(f"No checkpoints in {out}; starting fresh")
    if args.device:
        device = args.device
    elif args.camera or args.bev:
        device = "auto"
    elif args.algo == "ppo":
        device = "cpu"
    else:
        device = "auto"
    train(
        algo=args.algo,
        timesteps=timesteps,
        out=out,
        map_name=args.map_name,
        spawn_index=args.spawn_index,
        host=args.host,
        port=args.port,
        fps=args.fps,
        max_steps=args.max_steps,
        seed=args.seed,
        device=device,
        resume=args.resume,
        no_rendering=args.no_render,
        camera=args.camera,
        turn_start_prob=args.turn_mix,
        fresh_replay=args.fresh_replay,
        fixed_route=args.fixed_route,
        ent_floor=args.ent_floor,
        demo_steps=args.demo_steps,
        bc_steps=args.bc_steps,
        critic_steps=args.critic_steps,
        sde_freq=args.sde,
        steer_smooth=args.steer_smooth,
        bev=args.bev,
        bev_source=args.bev_source,
        seg_path=args.seg,
        speed_control=args.speed_control,
        speed_setpoint=args.speed_setpoint,
        lights=args.lights,
        npc_vehicles=args.npc_vehicles,
        walkers=args.walkers,
        checkpoint_every=args.checkpoint_every,
    )


if __name__ == "__main__":
    main()
