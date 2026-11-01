"""Behavioral cloning model and dataset."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset

from utils.commands import (
    ALL_COMMANDS,
    COMMAND_DIM,
    COMMAND_FOLLOW_LANE,
    COMMAND_GO_STRAIGHT,
    COMMAND_TURN_LEFT,
    COMMAND_TURN_RIGHT,
    command_to_onehot,
)
from utils.lights import ALL_LIGHTS, LIGHT_DIM, LIGHT_OFF, light_to_onehot
from utils.radar import RADAR_RANGE_M, encode_radar_cell
from utils.state import MAX_SPEED_KMH, normalize_speed

# Must match the camera size used while recording / driving.
IMAGE_WIDTH = 160
IMAGE_HEIGHT = 120
# Temporal context: stack this many RGB frames (oldest → newest) as channels.
STACK_FRAMES = 3
IN_CHANNELS = 3 * STACK_FRAMES


def _run_has_traffic(run_dir: Path) -> bool:
    """True if meta.json records NPC traffic > 0."""
    meta_path = Path(run_dir) / "meta.json"
    if not meta_path.exists():
        return False
    try:
        meta = json.loads(meta_path.read_text())
    except Exception:
        return False
    return int(meta.get("traffic", 0) or 0) > 0


def _frame_stack_paths(
    run_dir: Path,
    frame: int,
    camera: str | None,
    *,
    stack_frames: int,
) -> list[Path]:
    """Paths for [t-(n-1), ..., t], repeating the earliest existing frame if needed."""
    n = max(1, stack_frames)
    current = _image_path_for_row(run_dir, frame, camera)
    paths: list[Path] = []
    for offset in range(n - 1, -1, -1):
        f = frame - offset
        if f < 0:
            paths.append(paths[-1] if paths else current)
            continue
        candidate = _image_path_for_row(run_dir, f, camera)
        if candidate.exists():
            paths.append(candidate)
        else:
            paths.append(paths[-1] if paths else current)
    return paths


def _load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def stack_rgb_arrays(frames: list[np.ndarray]) -> np.ndarray:
    """List of HxWx3 → HxWx(3n) oldest→newest."""
    if not frames:
        raise ValueError("Need at least one frame to stack")
    return np.concatenate(frames, axis=2)


def _is_dataset_dir(path: Path) -> bool:
    """True if path looks like a recorded run (any name with controls CSV)."""
    if not path.is_dir():
        return False
    return (path / "controls.csv").exists() or (
        path / "controls_balanced.csv"
    ).exists()


def resolve_run_dirs(paths: list[Path]) -> list[Path]:
    """Expand a data/ folder into dataset children, or accept a run path directly."""
    runs: list[Path] = []
    for path in paths:
        path = Path(path)
        if _is_dataset_dir(path):
            runs.append(path)
            continue
        if not path.is_dir():
            raise FileNotFoundError(f"Not a dataset directory: {path}")
        children = sorted(
            p for p in path.iterdir() if p.is_dir() and _is_dataset_dir(p)
        )
        if not children:
            raise FileNotFoundError(f"No dataset (controls.csv) found at {path}")
        runs.extend(children)
    if not runs:
        raise FileNotFoundError("No datasets found.")
    return runs


def _controls_csv_path(run_dir: Path) -> Path:
    """Prefer rebalanced CSV when present (new multi-cam capture)."""
    balanced = run_dir / "controls_balanced.csv"
    if balanced.exists():
        return balanced
    return run_dir / "controls.csv"


def _image_path_for_row(run_dir: Path, frame: int, camera: str | None) -> Path:
    """Resolve image path for legacy flat images/ or images/{camera}/."""
    if camera:
        candidate = run_dir / "images" / camera / f"{frame:06d}.png"
        if candidate.exists():
            return candidate
    flat = run_dir / "images" / f"{frame:06d}.png"
    return flat


class DrivingDataset(Dataset):
    """Loads stacked RGB + speed + command + light one-hot + controls."""

    def __init__(
        self,
        run_dirs: Path | str | list[Path | str],
        *,
        center_only: bool = False,
        max_follow_lane_steer: float = 0.40,
        stack_frames: int = STACK_FRAMES,
    ) -> None:
        if isinstance(run_dirs, (str, Path)):
            run_dirs = resolve_run_dirs([Path(run_dirs)])
        else:
            run_dirs = resolve_run_dirs([Path(p) for p in run_dirs])

        self.run_dirs = run_dirs
        self.stack_frames = max(1, int(stack_frames))
        self.traffic_runs = {d for d in run_dirs if _run_has_traffic(d)}
        # run_dir, frame, camera, speed, command_oh, light_oh, controls, radar
        self.samples: list[
            tuple[
                Path,
                int,
                str | None,
                float,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
            ]
        ] = []
        self.missing_speed_runs: list[str] = []
        self.missing_command_runs: list[str] = []
        self.missing_light_runs: list[str] = []
        self.missing_radar_traffic_runs: list[str] = []
        dropped_follow_steer = 0

        for run_dir in run_dirs:
            csv_path = _controls_csv_path(run_dir)
            if not csv_path.exists():
                continue
            lines = csv_path.read_text().strip().splitlines()
            if not lines:
                continue
            header = lines[0].split(",")
            has_speed = "speed_kmh" in header
            has_camera = "camera" in header
            has_command = "command" in header
            has_light = "light_state" in header
            has_radar = "radar_m" in header
            col = {name: i for i, name in enumerate(header)}
            if not has_speed:
                self.missing_speed_runs.append(run_dir.name)
            if not has_command:
                self.missing_command_runs.append(run_dir.name)
            if not has_light:
                self.missing_light_runs.append(run_dir.name)
            if not has_radar and run_dir in self.traffic_runs:
                self.missing_radar_traffic_runs.append(run_dir.name)

            for row in lines[1:]:
                parts = row.split(",")
                if len(parts) < 4:
                    continue
                frame = int(parts[col["frame"]])
                steer = float(parts[col["steer"]])
                throttle = float(parts[col["throttle"]])
                brake = float(parts[col["brake"]])
                if has_speed:
                    speed = normalize_speed(float(parts[col["speed_kmh"]]))
                else:
                    speed = 0.0
                if has_command:
                    command = parts[col["command"]].strip() or COMMAND_FOLLOW_LANE
                else:
                    command = COMMAND_FOLLOW_LANE
                # Tight waypoint expert often mislabels big steers as FOLLOW_LANE.
                if command == COMMAND_FOLLOW_LANE and abs(steer) > max_follow_lane_steer:
                    if abs(steer) >= 0.12:
                        command = (
                            COMMAND_TURN_RIGHT if steer > 0 else COMMAND_TURN_LEFT
                        )
                    else:
                        dropped_follow_steer += 1
                        continue
                light_state = (
                    parts[col["light_state"]].strip() if has_light else LIGHT_OFF
                )
                camera = parts[col["camera"]] if has_camera else None
                if center_only and camera is not None and camera != "center":
                    continue
                path = _image_path_for_row(run_dir, frame, camera)
                if path.exists():
                    controls = torch.tensor(
                        [steer, throttle, brake],
                        dtype=torch.float32,
                    )
                    command_oh = torch.tensor(
                        command_to_onehot(command),
                        dtype=torch.float32,
                    )
                    light_oh = torch.tensor(
                        light_to_onehot(light_state),
                        dtype=torch.float32,
                    )
                    radar_cell = parts[col["radar_m"]] if has_radar else ""
                    radar = torch.tensor(
                        encode_radar_cell(radar_cell, max_range_m=RADAR_RANGE_M),
                        dtype=torch.float32,
                    )
                    self.samples.append(
                        (
                            run_dir,
                            frame,
                            camera,
                            speed,
                            command_oh,
                            light_oh,
                            controls,
                            radar,
                        )
                    )

        if not self.samples:
            raise RuntimeError("No samples found in dataset.")
        if dropped_follow_steer:
            print(
                f"Dropped {dropped_follow_steer} FOLLOW_LANE frames with "
                f"|steer|>{max_follow_lane_steer} (noisy path-track labels)"
            )
        if self.missing_speed_runs:
            print(
                "WARNING: these runs have no speed_kmh column (re-record for best results): "
                + ", ".join(self.missing_speed_runs)
            )
        if self.missing_command_runs:
            print(
                "WARNING: these runs have no command column "
                f"(defaulting to {COMMAND_FOLLOW_LANE}): "
                + ", ".join(self.missing_command_runs)
            )
        if self.missing_light_runs:
            print(
                "WARNING: these runs have no light_state column "
                f"(defaulting to {LIGHT_OFF}): "
                + ", ".join(self.missing_light_runs)
            )
        print(
            f"Frame stack: {self.stack_frames} "
            f"({3 * self.stack_frames} input channels)"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self, index: int
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        return self.get_item(index, augment=False)

    def get_item(
        self,
        index: int,
        *,
        augment: bool = False,
        max_shift_px: int = 24,
        max_steer_nudge: float = 0.20,
        command_dropout_p: float = 0.0,
        command_dropout_steer_min: float = 0.12,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        run_dir, frame, camera, speed, command, light, controls, radar = self.samples[
            index
        ]
        paths = _frame_stack_paths(
            run_dir, frame, camera, stack_frames=self.stack_frames
        )
        frames = [_load_rgb(p) for p in paths]
        steer, throttle, brake = (
            float(controls[0]),
            float(controls[1]),
            float(controls[2]),
        )
        command = command.clone()
        light = light.clone()

        if augment and max_shift_px > 0:
            # dx > 0 shifts pixels right → scene looks like "car drifted left"
            # → nudge steer right (positive), matching left-camera recovery labels.
            dx = int(np.random.randint(-max_shift_px, max_shift_px + 1))
            if dx != 0:
                frames = [_shift_image_x(f, dx) for f in frames]
                steer = float(
                    np.clip(
                        steer + (dx / max_shift_px) * max_steer_nudge,
                        -1.0,
                        1.0,
                    )
                )

        # Command dropout: keep expert steer/throttle, but sometimes lie that
        # the command is FOLLOW_LANE. Forces the FOLLOW_LANE branch to recover
        # from off-lane / mid-turn images when the turn signal is missing.
        if (
            augment
            and command_dropout_p > 0.0
            and abs(steer) >= command_dropout_steer_min
            and float(np.random.random()) < command_dropout_p
        ):
            command = torch.tensor(
                command_to_onehot(COMMAND_FOLLOW_LANE), dtype=torch.float32
            )

        stacked = stack_rgb_arrays(frames)
        image_tensor = torch.from_numpy(stacked).permute(2, 0, 1)
        speed_tensor = torch.tensor([speed], dtype=torch.float32)
        controls_tensor = torch.tensor(
            [steer, throttle, brake], dtype=torch.float32
        )
        return image_tensor, speed_tensor, command, light, radar, controls_tensor

    def turn_sample_weights(
        self,
        *,
        steer_threshold: float = 0.15,
        boost: float = 5.0,
    ) -> torch.Tensor:
        weights: list[float] = []
        for *_, controls, _radar in self.samples:
            weight = boost if abs(controls[0].item()) >= steer_threshold else 1.0
            weights.append(weight)
        return torch.tensor(weights, dtype=torch.float64)

    def dagger_sample_weights(self, *, boost: float = 4.0) -> torch.Tensor:
        """Upweight frames from dagger_* correction runs."""
        weights: list[float] = []
        for run_dir, *_rest in self.samples:
            weights.append(boost if run_dir.name.startswith("dagger_") else 1.0)
        return torch.tensor(weights, dtype=torch.float64)

    def dagger_stats(self) -> tuple[int, int]:
        """Return (dagger_samples, total_samples)."""
        n = sum(
            1 for run_dir, *_ in self.samples if run_dir.name.startswith("dagger_")
        )
        return n, len(self.samples)

    def traffic_follow_sample_weights(self, *, weight: float = 0.25) -> torch.Tensor:
        """Downweight cruise FOLLOW/GO_STRAIGHT frames from traffic runs.

        Keeps traffic turns + braking (stop/go behind cars). Empty-road runs
        stay at weight 1 so they own lane-keeping.
        """
        follow_idx = ALL_COMMANDS.index(COMMAND_FOLLOW_LANE)
        straight_idx = ALL_COMMANDS.index(COMMAND_GO_STRAIGHT)
        weights: list[float] = []
        for (
            run_dir,
            _frame,
            _cam,
            _speed,
            command,
            _light,
            controls,
            _radar,
        ) in self.samples:
            if run_dir not in self.traffic_runs:
                weights.append(1.0)
                continue
            cmd_i = int(command.argmax().item())
            brake = float(controls[2].item())
            # Keep stop/go and turns from traffic; thin out cruise lane labels.
            if cmd_i in (follow_idx, straight_idx) and brake < 0.25:
                weights.append(weight)
            else:
                weights.append(1.0)
        return torch.tensor(weights, dtype=torch.float64)

    def traffic_follow_stats(self) -> tuple[int, int]:
        """(traffic-run samples, total samples)."""
        n = sum(1 for run_dir, *_ in self.samples if run_dir in self.traffic_runs)
        return n, len(self.samples)

    def light_stop_sample_weights(
        self,
        *,
        red_approach_boost: float = 2.0,
        red_wait_weight: float = 0.35,
        go_boost: float = 1.8,
    ) -> torch.Tensor:
        """Balance lights without drowning the set in red-wait copies.

        - Red/yellow while moving (approach): mild boost.
        - Red/yellow while stopped: downsample (many near-duplicate frames).
        - Green/Off while moving: mild go boost (fights ghost stops).
        """
        red_idx = ALL_LIGHTS.index("Red")
        yellow_idx = ALL_LIGHTS.index("Yellow")
        green_idx = ALL_LIGHTS.index("Green")
        off_idx = ALL_LIGHTS.index("Off")
        weights: list[float] = []
        for *_run, _frame, _cam, speed, _command, light, controls, _radar in self.samples:
            idx = int(light.argmax().item())
            brake = float(controls[2].item())
            throttle = float(controls[1].item())
            # speed is normalized 0..1 (see normalize_speed); ~0 ≈ stopped.
            speed_km = float(speed) * MAX_SPEED_KMH
            if idx in (red_idx, yellow_idx):
                if speed_km < 2.0 and brake >= 0.3:
                    weights.append(red_wait_weight)
                else:
                    weights.append(red_approach_boost)
            elif idx in (green_idx, off_idx) and throttle >= 0.15 and brake < 0.15:
                weights.append(go_boost)
            else:
                weights.append(1.0)
        return torch.tensor(weights, dtype=torch.float64)

    def turn_stats(self, *, steer_threshold: float = 0.15) -> tuple[int, int]:
        turns = sum(
            1
            for *_, c, _radar in self.samples
            if abs(c[0].item()) >= steer_threshold
        )
        return turns, len(self.samples)

    def command_stats(self) -> dict[str, int]:
        counts = {name: 0 for name in ALL_COMMANDS}
        for _run, _frame, _cam, _speed, command, _light, _controls, _radar in self.samples:
            idx = int(command.argmax().item())
            counts[ALL_COMMANDS[idx]] += 1
        return counts

    def light_stats(self) -> dict[str, int]:
        counts = {name: 0 for name in ALL_LIGHTS}
        for *_rest, light, _controls, _radar in self.samples:
            idx = int(light.argmax().item())
            counts[ALL_LIGHTS[idx]] += 1
        return counts


def _shift_image_x(array: np.ndarray, dx: int) -> np.ndarray:
    """Shift image horizontally; fill exposed edge by replication."""
    if dx == 0:
        return array
    out = np.empty_like(array)
    if dx > 0:
        out[:, dx:] = array[:, :-dx]
        out[:, :dx] = array[:, :1]
    else:
        out[:, :dx] = array[:, -dx:]
        out[:, dx:] = array[:, -1:]
    return out


class AugmentedSubset(Dataset):
    """Subset with optional train-time perspective shift + command dropout."""

    def __init__(
        self,
        dataset: DrivingDataset,
        indices: list[int],
        *,
        augment: bool = False,
        max_shift_px: int = 24,
        max_steer_nudge: float = 0.20,
        command_dropout_p: float = 0.0,
    ) -> None:
        self.dataset = dataset
        self.indices = list(indices)
        self.augment = augment
        self.max_shift_px = max_shift_px
        self.max_steer_nudge = max_steer_nudge
        self.command_dropout_p = command_dropout_p

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(
        self, index: int
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        return self.dataset.get_item(
            self.indices[index],
            augment=self.augment,
            max_shift_px=self.max_shift_px,
            max_steer_nudge=self.max_steer_nudge,
            command_dropout_p=self.command_dropout_p if self.augment else 0.0,
        )


class PilotNet(nn.Module):
    """CNN on stacked RGB + speed [+ command] [+ light] -> steer, throttle, brake.

    When use_command and branched=True (default), uses CIL-style heads:
    one steer/throttle/brake logit per command, selected by the one-hot.
    That forces FOLLOW_LANE and TURN_* to be different policies instead of
    letting image features drown a tiny concatenated command vector.

    Light state (Red/Yellow/Green/Off) is concatenated into the shared trunk
    like speed — sim GT now, detector later — so stop/go does not rely on
    resolving a few red pixels in the image.

    Radar range is not in that trunk. It is fused only into the throttle and
    brake heads, so a car ahead can change speed without moving the wheel.

    stack_frames>1 concatenates oldest→newest RGB as input channels so the
    net sees short-term motion without a recurrent head.
    """

    def __init__(
        self,
        *,
        use_command: bool = True,
        n_commands: int = COMMAND_DIM,
        branched: bool = True,
        stack_frames: int = STACK_FRAMES,
        use_light: bool = True,
        n_lights: int = LIGHT_DIM,
        use_radar: bool = False,
    ) -> None:
        super().__init__()
        self.use_command = use_command
        self.branched = bool(branched and use_command)
        self.n_commands = n_commands if use_command else 0
        self.use_light = bool(use_light)
        self.n_lights = n_lights if self.use_light else 0
        self.use_radar = bool(use_radar)
        self.radar_dim = 2 if self.use_radar else 0
        if self.use_radar and not self.branched:
            raise ValueError("Radar input requires branched control heads")
        self.stack_frames = max(1, int(stack_frames))
        self.in_channels = 3 * self.stack_frames
        # Branched: extras = speed (+ light). Concat cmd path: + commands too.
        light_extra = self.n_lights
        if self.branched:
            self.n_extra = 1 + light_extra
        else:
            self.n_extra = 1 + (self.n_commands if use_command else 0) + light_extra
        self.features = nn.Sequential(
            nn.Conv2d(self.in_channels, 24, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 36, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(36, 48, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, self.in_channels, IMAGE_HEIGHT, IMAGE_WIDTH)
            n_features = self.features(dummy).shape[1]

        if self.branched:
            self.trunk = nn.Sequential(
                nn.Linear(n_features + self.n_extra, 100),
                nn.ReLU(inplace=True),
                nn.Linear(100, 50),
                nn.ReLU(inplace=True),
            )
            self.steer_branches = nn.Linear(50, self.n_commands)
            if self.use_radar:
                # h stays image+speed+light. Radar joins only this longitudinal map.
                self.long_fuse = nn.Sequential(
                    nn.Linear(50 + self.radar_dim, 50),
                    nn.ReLU(inplace=True),
                )
            else:
                self.long_fuse = None
            self.throttle_branches = nn.Linear(50, self.n_commands)
            self.brake_branches = nn.Linear(50, self.n_commands)
            self.head = None
        else:
            self.trunk = None
            self.steer_branches = None
            self.throttle_branches = None
            self.brake_branches = None
            self.long_fuse = None
            self.head = nn.Sequential(
                nn.Linear(n_features + self.n_extra, 100),
                nn.ReLU(inplace=True),
                nn.Linear(100, 50),
                nn.ReLU(inplace=True),
                nn.Linear(50, 3),
            )

    def _speed_light(
        self,
        speed: torch.Tensor,
        light: torch.Tensor | None,
    ) -> torch.Tensor:
        if not self.use_light:
            return speed
        if light is None:
            # Default Off one-hot.
            batch = speed.shape[0]
            light = speed.new_zeros((batch, self.n_lights))
            light[:, ALL_LIGHTS.index(LIGHT_OFF)] = 1.0
        return torch.cat([speed, light], dim=1)

    def _long_features(
        self,
        h: torch.Tensor,
        radar: torch.Tensor | None,
    ) -> torch.Tensor:
        """Throttle/brake features. Steer keeps reading ``h`` with no radar."""
        if not self.use_radar or self.long_fuse is None:
            return h
        if radar is None:
            radar = h.new_ones((h.shape[0], self.radar_dim))
            radar[:, 1] = 0.0
        return self.long_fuse(torch.cat([h, radar], dim=1))

    def forward(
        self,
        image: torch.Tensor,
        speed: torch.Tensor,
        command: torch.Tensor | None = None,
        light: torch.Tensor | None = None,
        radar: torch.Tensor | None = None,
    ) -> torch.Tensor:
        feats = self.features(image)
        if self.branched:
            if command is None:
                raise ValueError("Branched PilotNet expects a command one-hot")
            extra = self._speed_light(speed, light)
            h = self.trunk(torch.cat([feats, extra], dim=1))
            long_h = self._long_features(h, radar)
            steer = torch.tanh((self.steer_branches(h) * command).sum(dim=1, keepdim=True))
            throttle = torch.sigmoid(
                (self.throttle_branches(long_h) * command).sum(dim=1, keepdim=True)
            )
            brake = torch.sigmoid(
                (self.brake_branches(long_h) * command).sum(dim=1, keepdim=True)
            )
            return torch.cat([steer, throttle, brake], dim=1)

        parts = [speed]
        if self.use_command:
            if command is None:
                raise ValueError("PilotNet expects a command one-hot tensor")
            parts.append(command)
        if self.use_light:
            if light is None:
                batch = speed.shape[0]
                light = speed.new_zeros((batch, self.n_lights))
                light[:, ALL_LIGHTS.index(LIGHT_OFF)] = 1.0
            parts.append(light)
        extra = torch.cat(parts, dim=1)
        x = torch.cat([feats, extra], dim=1)
        out = self.head(x)
        steer = torch.tanh(out[:, 0:1])
        throttle = torch.sigmoid(out[:, 1:2])
        brake = torch.sigmoid(out[:, 2:3])
        return torch.cat([steer, throttle, brake], dim=1)


def save_checkpoint(model: PilotNet, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "n_extra": model.n_extra,
            "use_command": model.use_command,
            "n_commands": model.n_commands,
            "branched": model.branched,
            "use_light": model.use_light,
            "n_lights": model.n_lights,
            "stack_frames": model.stack_frames,
            "in_channels": model.in_channels,
            "use_radar": model.use_radar,
            "radar_dim": model.radar_dim,
            "radar_range_m": RADAR_RANGE_M,
            "command_names": list(ALL_COMMANDS) if model.use_command else [],
            "light_names": list(ALL_LIGHTS) if model.use_light else [],
            "max_speed_kmh": MAX_SPEED_KMH,
        },
        path,
    )


def load_checkpoint(path: Path, device: torch.device) -> PilotNet:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
        use_command = bool(ckpt.get("use_command", ckpt.get("n_extra", 1) > 1))
        n_commands = int(ckpt.get("n_commands", COMMAND_DIM if use_command else 0))
        branched = bool(ckpt.get("branched", "steer_branches.weight" in state))
        use_light = bool(ckpt.get("use_light", False))
        n_lights = int(ckpt.get("n_lights", LIGHT_DIM if use_light else 0))
        # Legacy speed-only checkpoints stored n_extra=1.
        if ckpt.get("n_extra", 1) == 1 and "use_command" not in ckpt:
            use_command = False
            n_commands = 0
            branched = False
        stack_frames = int(ckpt.get("stack_frames", 0))
        if stack_frames <= 0:
            weight = state.get("features.0.weight")
            in_ch = int(weight.shape[1]) if weight is not None else 3
            stack_frames = max(1, in_ch // 3)
        use_radar = bool(ckpt.get("use_radar", False))
        model = PilotNet(
            use_command=use_command,
            n_commands=n_commands,
            branched=branched and use_command,
            stack_frames=stack_frames,
            use_light=use_light,
            n_lights=n_lights,
            use_radar=use_radar,
        ).to(device)
        model.load_state_dict(state)
    else:
        model = PilotNet(
            use_command=False, branched=False, stack_frames=1, use_light=False
        ).to(device)
        model.load_state_dict(ckpt)
    model.eval()
    return model


def exclusive_controls(
    steer: float,
    throttle: float,
    brake: float,
) -> tuple[float, float, float]:
    """Keep throttle/brake mutually exclusive."""
    if throttle >= brake:
        brake = 0.0
    else:
        throttle = 0.0
    return float(steer), float(throttle), float(brake)
