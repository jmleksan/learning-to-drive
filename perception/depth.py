"""CARLA depth-camera decode and the center/right_wide stereo geometry.

The depth camera is the teacher. It sits on the center mount and is not an
input to the network. The network sees the center photo and the right-wide photo.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

# Center and right_wide: 160 px, 90° horizontal field of view, parallel.
IMAGE_WIDTH = 160
IMAGE_HEIGHT = 120
FOV_DEG = 90.0
# CARLA +y is right. right_wide sits on the outer corner of the roof.
STEREO_CAMERA = "right_wide"
BASELINE_M = 1.00
# fx = (width / 2) / tan(fov / 2) = 80 / tan(45°) = 80 px.
FOCAL_PX = 80.0
# Closest range the search can represent: Z = fx * B / (MAX_DISP - 1) ≈ 1.3 m.
MAX_DISP = 64
# Sky and the far clip are not useful stereo targets.
MAX_DEPTH_M = 80.0

__all__ = [
    "BASELINE_M",
    "STEREO_CAMERA",
    "FOCAL_PX",
    "FOV_DEG",
    "IMAGE_HEIGHT",
    "IMAGE_WIDTH",
    "MAX_DEPTH_M",
    "MAX_DISP",
    "depth_to_meters",
    "load_depth_meters",
    "save_depth_meters",
]


def depth_to_meters(image) -> np.ndarray:
    """CARLA depth-camera image → HxW meters.

    RGB encodes a fraction of the 1000 m far plane:
    (R + G*256 + B*256*256) / (256**3 - 1) * 1000.
    """
    raw = np.frombuffer(image.raw_data, dtype=np.uint8)
    frame = raw.reshape((image.height, image.width, 4))
    b = frame[:, :, 0].astype(np.float32)
    g = frame[:, :, 1].astype(np.float32)
    r = frame[:, :, 2].astype(np.float32)
    normalized = (r + g * 256.0 + b * 256.0 * 256.0) / (256.0**3 - 1.0)
    return normalized * 1000.0


def save_depth_meters(path: Path, meters: np.ndarray) -> None:
    np.save(path, np.ascontiguousarray(meters, dtype=np.float16))


def load_depth_meters(path: Path) -> np.ndarray:
    meters = np.load(path).astype(np.float32)
    if meters.shape != (IMAGE_HEIGHT, IMAGE_WIDTH):
        raise ValueError(
            f"{path} is {meters.shape[1]}x{meters.shape[0]}, "
            f"expected {IMAGE_WIDTH}x{IMAGE_HEIGHT}"
        )
    return meters
