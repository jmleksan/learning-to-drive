"""CARLA semantic-segmentation tags and image decode.

The semantic camera writes the tag id into the red channel of a BGRA frame.
Tag ids match ``carla.CityObjectLabel`` in 0.9.16 (0..28). ``Any`` (255) is a
query filter, not a pixel tag.
"""
from __future__ import annotations

import numpy as np

# Index is the raw tag id stored in the red channel.
CLASS_NAMES: tuple[str, ...] = (
    "Unlabeled",
    "Road",
    "Sidewalk",
    "Building",
    "Wall",
    "Fence",
    "Pole",
    "TrafficLight",
    "TrafficSign",
    "Vegetation",
    "Terrain",
    "Sky",
    "Pedestrian",
    "Rider",
    "Car",
    "Truck",
    "Bus",
    "Train",
    "Motorcycle",
    "Bicycle",
    "Static",
    "Dynamic",
    "Other",
    "Water",
    "RoadLine",
    "Ground",
    "Bridge",
    "RailTrack",
    "GuardRail",
)
NUM_CLASSES = len(CLASS_NAMES)

# RGB colors used by CARLA's CityScapesPalette, indexed by tag id.
PALETTE = np.array(
    [
        (0, 0, 0),
        (128, 64, 128),
        (244, 35, 232),
        (70, 70, 70),
        (102, 102, 156),
        (190, 153, 153),
        (153, 153, 153),
        (250, 170, 30),
        (220, 220, 0),
        (107, 142, 35),
        (152, 251, 152),
        (70, 130, 180),
        (220, 20, 60),
        (255, 0, 0),
        (0, 0, 142),
        (0, 0, 70),
        (0, 60, 100),
        (0, 80, 100),
        (0, 0, 230),
        (119, 11, 32),
        (110, 190, 160),
        (170, 120, 50),
        (55, 90, 80),
        (45, 60, 150),
        (157, 234, 50),
        (81, 0, 81),
        (150, 100, 100),
        (230, 150, 140),
        (180, 165, 180),
    ],
    dtype=np.uint8,
)


def labels_from_bgra(raw: bytes | np.ndarray, height: int, width: int) -> np.ndarray:
    """Red channel of a BGRA buffer → HxW uint8 tag ids."""
    array = np.frombuffer(raw, dtype=np.uint8)
    array = array.reshape((height, width, 4))
    return np.ascontiguousarray(array[:, :, 2])


def semantic_to_labels(image) -> np.ndarray:
    """CARLA semantic-camera image → HxW uint8 tag ids."""
    return labels_from_bgra(image.raw_data, image.height, image.width)


def sanitize_labels(labels: np.ndarray) -> np.ndarray:
    """Map anything outside 0..NUM_CLASSES-1 to Unlabeled."""
    out = np.asarray(labels)
    if out.dtype != np.uint8:
        out = out.astype(np.uint8, copy=False)
    invalid = out >= NUM_CLASSES
    if invalid.any():
        out = out.copy()
        out[invalid] = 0
    return out


def colorize(labels: np.ndarray) -> np.ndarray:
    """HxW tag ids → HxWx3 RGB preview."""
    safe = sanitize_labels(labels)
    return PALETTE[safe]
