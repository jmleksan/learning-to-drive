"""Traffic-light state as a CIL-style conditioning signal (sim GT / later detector)."""
from __future__ import annotations

LIGHT_RED = "Red"
LIGHT_YELLOW = "Yellow"
LIGHT_GREEN = "Green"
LIGHT_OFF = "Off"

ALL_LIGHTS = (
    LIGHT_RED,
    LIGHT_YELLOW,
    LIGHT_GREEN,
    LIGHT_OFF,
)

LIGHT_DIM = len(ALL_LIGHTS)
_LIGHT_INDEX = {name: i for i, name in enumerate(ALL_LIGHTS)}


def normalize_light_state(state: str | None) -> str:
    """Map logged / CARLA names onto ALL_LIGHTS (None/Unknown → Off)."""
    name = (state or "").strip()
    if not name or name in ("None", "Unknown"):
        return LIGHT_OFF
    if "." in name:
        name = name.split(".")[-1]
    if name in _LIGHT_INDEX:
        return name
    return LIGHT_OFF


def light_to_onehot(state: str | None) -> list[float]:
    """One-hot over ALL_LIGHTS."""
    vec = [0.0] * LIGHT_DIM
    idx = _LIGHT_INDEX[normalize_light_state(state)]
    vec[idx] = 1.0
    return vec


def is_stop_light(state: str | None) -> bool:
    return normalize_light_state(state) in (LIGHT_RED, LIGHT_YELLOW)
