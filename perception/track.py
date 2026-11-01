"""Associate bird's-eye objects across frames so each keeps an id and speed."""
from __future__ import annotations

from dataclasses import dataclass

from perception.bev import BevObject


@dataclass(frozen=True)
class TrackedObject:
    """A BevObject plus a stable id and ego-frame velocity."""

    track_id: int
    class_id: int
    name: str
    x_m: float
    y_m: float
    length_m: float
    width_m: float
    cells: int
    vx_mps: float = 0.0
    vy_mps: float = 0.0
    age: int = 1


@dataclass
class _Track:
    track_id: int
    class_id: int
    name: str
    x_m: float
    y_m: float
    length_m: float
    width_m: float
    cells: int
    vx_mps: float = 0.0
    vy_mps: float = 0.0
    age: int = 1
    misses: int = 0


class ObjectTracker:
    """Greedy nearest-neighbor tracker in the ego frame.

    Same class, closest predicted position within ``max_match_m`` wins.
    Tracks that miss for ``max_misses`` frames are dropped.
    """

    def __init__(
        self,
        *,
        dt_s: float = 0.1,
        max_match_m: float = 5.0,
        max_misses: int = 8,
        velocity_alpha: float = 0.4,
        min_confirm: int = 3,
    ) -> None:
        self.dt_s = max(float(dt_s), 1e-3)
        self.max_match_m = float(max_match_m)
        self.max_misses = int(max_misses)
        self.velocity_alpha = float(velocity_alpha)
        # Don't emit a track until it has been seen this many frames in a row.
        self.min_confirm = max(1, int(min_confirm))
        self._next_id = 1
        self._tracks: list[_Track] = []

    def reset(self) -> None:
        self._next_id = 1
        self._tracks = []

    def update(self, detections: list[BevObject]) -> list[TrackedObject]:
        old = list(self._tracks)
        predicted = [
            (track.x_m + track.vx_mps * self.dt_s, track.y_m + track.vy_mps * self.dt_s)
            for track in old
        ]
        used_tracks: set[int] = set()
        used_dets: set[int] = set()
        pairs: list[tuple[float, int, int]] = []
        for ti, track in enumerate(old):
            px, py = predicted[ti]
            for di, det in enumerate(detections):
                if det.class_id != track.class_id:
                    continue
                dist = ((det.x_m - px) ** 2 + (det.y_m - py) ** 2) ** 0.5
                if dist <= self.max_match_m:
                    pairs.append((dist, ti, di))
        pairs.sort(key=lambda item: item[0])

        next_tracks: list[_Track] = []
        for _dist, ti, di in pairs:
            if ti in used_tracks or di in used_dets:
                continue
            used_tracks.add(ti)
            used_dets.add(di)
            track = old[ti]
            det = detections[di]
            raw_vx = (det.x_m - track.x_m) / self.dt_s
            raw_vy = (det.y_m - track.y_m) / self.dt_s
            a = self.velocity_alpha
            next_tracks.append(
                _Track(
                    track_id=track.track_id,
                    class_id=det.class_id,
                    name=det.name,
                    x_m=det.x_m,
                    y_m=det.y_m,
                    length_m=det.length_m,
                    width_m=det.width_m,
                    cells=det.cells,
                    vx_mps=(1.0 - a) * track.vx_mps + a * raw_vx,
                    vy_mps=(1.0 - a) * track.vy_mps + a * raw_vy,
                    age=track.age + 1,
                    misses=0,
                )
            )

        for di, det in enumerate(detections):
            if di in used_dets:
                continue
            next_tracks.append(
                _Track(
                    track_id=self._next_id,
                    class_id=det.class_id,
                    name=det.name,
                    x_m=det.x_m,
                    y_m=det.y_m,
                    length_m=det.length_m,
                    width_m=det.width_m,
                    cells=det.cells,
                )
            )
            self._next_id += 1

        for ti, track in enumerate(old):
            if ti in used_tracks:
                continue
            misses = track.misses + 1
            if misses > self.max_misses:
                continue
            next_tracks.append(
                _Track(
                    track_id=track.track_id,
                    class_id=track.class_id,
                    name=track.name,
                    x_m=track.x_m + track.vx_mps * self.dt_s,
                    y_m=track.y_m + track.vy_mps * self.dt_s,
                    length_m=track.length_m,
                    width_m=track.width_m,
                    cells=track.cells,
                    vx_mps=track.vx_mps,
                    vy_mps=track.vy_mps,
                    age=track.age + 1,
                    misses=misses,
                )
            )

        self._tracks = next_tracks
        return [
            TrackedObject(
                track_id=t.track_id,
                class_id=t.class_id,
                name=t.name,
                x_m=t.x_m,
                y_m=t.y_m,
                length_m=t.length_m,
                width_m=t.width_m,
                cells=t.cells,
                vx_mps=t.vx_mps,
                vy_mps=t.vy_mps,
                age=t.age,
            )
            for t in self._tracks
            if t.misses == 0 and t.age >= self.min_confirm
        ]
