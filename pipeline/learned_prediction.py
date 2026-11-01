"""Optional stretch: linear residual on top of constant-velocity rollouts.

Not used by the default classical drive. Train offline on track histories if/when
you record ego-frame track sequences; until then prefer ``constant_velocity_predict``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pipeline.prediction import PredictedAgent, constant_velocity_predict


@dataclass
class LinearResidualPredictor:
    """Predict Δ(x,y) residual vs CV using a fixed linear map on recent history.

    ``weights`` shape (4, 2): [x, y, vx, vy] → residual offset at horizon.
    Defaults to zeros → identical to constant velocity.
    """

    weights: np.ndarray
    horizon_s: float = 3.0
    dt_s: float = 0.1

    @classmethod
    def identity(cls, **kwargs) -> "LinearResidualPredictor":
        return cls(weights=np.zeros((4, 2), dtype=np.float64), **kwargs)

    def predict(self, tracks: list) -> list[PredictedAgent]:
        base = constant_velocity_predict(
            tracks, horizon_s=self.horizon_s, dt_s=self.dt_s
        )
        out: list[PredictedAgent] = []
        for track, pred in zip(tracks, base):
            feat = np.array(
                [
                    float(getattr(track, "x_m", 0.0)),
                    float(getattr(track, "y_m", 0.0)),
                    float(getattr(track, "vx_mps", 0.0)),
                    float(getattr(track, "vy_mps", 0.0)),
                ],
                dtype=np.float64,
            )
            residual = feat @ self.weights  # (2,)
            traj = pred.trajectory_xy.copy()
            # Blend residual toward the end of the horizon.
            n = len(traj)
            for i in range(n):
                a = i / max(1, n - 1)
                traj[i] = traj[i] + a * residual
            out.append(
                PredictedAgent(
                    track_id=pred.track_id,
                    class_id=pred.class_id,
                    name=pred.name,
                    trajectory_xy=traj,
                    length_m=pred.length_m,
                    width_m=pred.width_m,
                )
            )
        return out
