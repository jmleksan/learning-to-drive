"""Town01 drive scorecard: route completion, speed, invasions, collisions."""

from __future__ import annotations

import threading
from typing import Sequence


def path_length_m(
    points: Sequence[tuple[float, float, float]],
    start: int,
    end: int,
) -> float:
    """Polyline length in meters from index start through index end, inclusive."""
    if not points:
        return 0.0
    a = max(0, min(start, len(points) - 1))
    b = max(0, min(end, len(points) - 1))
    if b < a:
        a, b = b, a
    total = 0.0
    px, py, pz = points[a]
    for i in range(a + 1, b + 1):
        x, y, z = points[i]
        total += ((x - px) ** 2 + (y - py) ** 2 + (z - pz) ** 2) ** 0.5
        px, py, pz = x, y, z
    return total


def collisions_per_km(collisions: int, distance_m: float) -> float:
    """Collisions per kilometer of distance actually driven."""
    if distance_m <= 1.0:
        return float(collisions) * 1000.0
    return collisions / (distance_m / 1000.0)


def route_completion_pct(progress_m: float, planned_m: float) -> float:
    if planned_m <= 1.0:
        return 0.0
    return max(0.0, min(100.0, 100.0 * progress_m / planned_m))


class EventCounter:
    """Count collision and lane-invasion sensor callbacks, debounced by frame."""

    def __init__(self, *, collision_gap_frames: int = 20) -> None:
        self.collision_gap_frames = max(1, int(collision_gap_frames))
        self.collisions = 0
        self.lane_invasions = 0
        self._last_collision_frame = -10_000
        self._last_invasion_frame = -10_000
        self._lock = threading.Lock()

    def on_collision(self, event) -> None:
        frame = int(getattr(event, "frame", 0))
        with self._lock:
            if frame - self._last_collision_frame < self.collision_gap_frames:
                return
            self._last_collision_frame = frame
            self.collisions += 1

    def on_lane_invasion(self, event) -> None:
        frame = int(getattr(event, "frame", 0))
        with self._lock:
            if frame == self._last_invasion_frame:
                return
            self._last_invasion_frame = frame
            self.lane_invasions += 1

    def snapshot(self) -> tuple[int, int]:
        with self._lock:
            return self.collisions, self.lane_invasions
