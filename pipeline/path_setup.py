"""Ensure vendored CARLA agents are importable as ``agents.*``."""
from __future__ import annotations

import sys
from pathlib import Path

_VENDOR_ROOT = Path(__file__).resolve().parents[1] / "vendor" / "carla_agents"


def ensure_carla_agents() -> Path:
    root = _VENDOR_ROOT.resolve()
    root_s = str(root)
    if root_s not in sys.path:
        sys.path.insert(0, root_s)
    return root
