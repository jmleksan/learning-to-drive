"""Draw classical decisions in the CARLA debug HUD (spectator, not RGB cams)."""
from __future__ import annotations

import math

import carla

from pipeline.agent import AgentStep


def _draw_string(
    world: carla.World,
    location: carla.Location,
    text: str,
    *,
    life_time: float,
    color: carla.Color,
) -> None:
    debug = world.debug
    try:
        if hasattr(debug, "draw_hud_string"):
            debug.draw_hud_string(location, text, False, color, life_time)
        else:
            debug.draw_string(location, text, False, color, life_time)
    except Exception:
        return


def _ego_to_world(
    ego_tf: carla.Transform, x_m: float, y_m: float, *, z: float = 0.5
) -> carla.Location:
    yaw = math.radians(ego_tf.rotation.yaw)
    c, s = math.cos(yaw), math.sin(yaw)
    # Ego: +x forward, +y right → world.
    wx = ego_tf.location.x + c * x_m - s * y_m
    wy = ego_tf.location.y + s * x_m + c * y_m
    return carla.Location(x=wx, y=wy, z=ego_tf.location.z + z)


def draw_decisions(
    world: carla.World,
    vehicle: carla.Vehicle,
    step: AgentStep,
    *,
    life_time: float = 0.12,
    draw_predictions: bool = True,
    draw_frenet: bool = True,
) -> None:
    """HUD: behavior reason, predicted agent futures, Frenet sample polyline."""
    tf = vehicle.get_transform()
    loc = tf.location
    d = step.decision
    lead = "—" if step.lead_m is None else f"{step.lead_m:.1f}m"
    sense = step.sense_mode or "?"
    lines = [
        f"BEH {d.state.value}  ({d.reason})",
        f"cmd={d.command}  v*={d.target_speed_kmh:.0f}  light={step.light}",
        f"sense={sense}  pred={step.n_predictions}  lead={lead}",
        f"ctrl steer={step.control.steer:+.2f} "
        f"thr={step.control.throttle:.2f} brk={step.control.brake:.2f}",
    ]
    color = {
        "STOP": carla.Color(255, 60, 60),
        "YIELD": carla.Color(255, 180, 40),
        "TURN": carla.Color(80, 160, 255),
        "LANE_FOLLOW": carla.Color(80, 220, 120),
        "LANE_CHANGE_LEFT": carla.Color(200, 120, 255),
        "LANE_CHANGE_RIGHT": carla.Color(200, 120, 255),
    }.get(d.state.value, carla.Color(220, 220, 220))

    for i, line in enumerate(lines):
        _draw_string(
            world,
            carla.Location(loc.x, loc.y, loc.z + 2.2 + 0.35 * i),
            line,
            life_time=life_time,
            color=color,
        )

    debug = world.debug
    if draw_predictions and step.predictions:
        for pred in step.predictions:
            traj = pred.trajectory_xy
            if traj is None or len(traj) < 2:
                continue
            for i in range(0, len(traj) - 1, 2):
                a = _ego_to_world(tf, float(traj[i, 0]), float(traj[i, 1]))
                b = _ego_to_world(tf, float(traj[i + 1, 0]), float(traj[i + 1, 1]))
                try:
                    if hasattr(debug, "draw_hud_line"):
                        debug.draw_hud_line(
                            a, b, 2.0, carla.Color(255, 200, 40), life_time, False
                        )
                    else:
                        debug.draw_line(
                            a, b, 0.05, carla.Color(255, 200, 40), life_time, False
                        )
                except Exception:
                    break

    if draw_frenet and step.frenet_xy:
        pts = step.frenet_xy
        for i in range(len(pts) - 1):
            a = carla.Location(pts[i][0], pts[i][1], loc.z + 0.4)
            b = carla.Location(pts[i + 1][0], pts[i + 1][1], loc.z + 0.4)
            try:
                if hasattr(debug, "draw_hud_line"):
                    debug.draw_hud_line(
                        a, b, 3.0, carla.Color(40, 200, 255), life_time, False
                    )
                else:
                    debug.draw_line(
                        a, b, 0.08, carla.Color(40, 200, 255), life_time, False
                    )
            except Exception:
                break


def draw_ekf_pose(
    world: carla.World,
    location: carla.Location,
    *,
    life_time: float = 0.12,
) -> None:
    """Cyan marker for filtered ego pose (vs GT vehicle)."""
    debug = world.debug
    try:
        if hasattr(debug, "draw_hud_point"):
            debug.draw_hud_point(
                carla.Location(location.x, location.y, location.z + 1.0),
                8.0,
                carla.Color(40, 220, 255),
                life_time,
                False,
            )
        else:
            debug.draw_point(
                carla.Location(location.x, location.y, location.z + 1.0),
                0.2,
                carla.Color(40, 220, 255),
                life_time,
                False,
            )
        _draw_string(
            world,
            carla.Location(location.x, location.y, location.z + 1.4),
            "EKF",
            life_time=life_time,
            color=carla.Color(40, 220, 255),
        )
    except Exception:
        return
