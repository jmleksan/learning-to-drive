"""Classical modular agent: predict → behavior → Frenet → controller."""
from __future__ import annotations

from dataclasses import dataclass, field

import carla

from pipeline.behavior import (
    BehaviorDecision,
    BehaviorFSM,
    crossing_threat,
    lead_distance_from_predictions,
)
from pipeline.control import Controller, ControlTarget, make_controller
from pipeline.motion import frenet_plan, plan_to_control_target
from pipeline.prediction import actors_to_ego_tracks, constant_velocity_predict
from utils.commands import COMMAND_FOLLOW_LANE
from utils.lights import normalize_light_state
from utils.routing import upcoming_command
from utils.traffic_signs import traffic_light_fields


@dataclass
class AgentStep:
    control: carla.VehicleControl
    decision: BehaviorDecision
    target: carla.Location
    route_index: int
    n_predictions: int
    lead_m: float | None
    light: str = "Off"
    sense_mode: str = "oracle"
    predictions: list = field(default_factory=list)
    frenet_xy: list[tuple[float, float]] | None = None


class ClassicalAgent:
    """Perception→Prediction→Planning→Control without NN planning/control."""

    def __init__(
        self,
        *,
        controller: str = "hybrid",
        cruise_kmh: float = 28.0,
        turn_kmh: float = 16.0,
        predict_horizon_s: float = 3.0,
        use_frenet: bool = True,
    ) -> None:
        self.fsm = BehaviorFSM(cruise_kmh=cruise_kmh, turn_kmh=turn_kmh)
        self.controller: Controller = make_controller(controller)
        self.cruise_kmh = cruise_kmh
        self.turn_kmh = turn_kmh
        self.predict_horizon_s = predict_horizon_s
        self.use_frenet = use_frenet
        self.route_index = 0

    def reset(self) -> None:
        self.fsm.reset()
        self.controller.reset()
        self.route_index = 0

    def step(
        self,
        vehicle: carla.Vehicle,
        route: list,
        *,
        world: carla.World | None = None,
        other_actors: list | None = None,
        tracks: list | None = None,
        pose_location: carla.Location | None = None,
        sense_mode: str = "oracle",
    ) -> AgentStep:
        if not route:
            ctrl = carla.VehicleControl(brake=1.0)
            decision = BehaviorDecision(
                self.fsm.state, 0.0, COMMAND_FOLLOW_LANE, "empty_route"
            )
            return AgentStep(
                ctrl, decision, vehicle.get_location(), 0, 0, None, sense_mode=sense_mode
            )

        # Planning pose: EKF estimate when provided, else GT vehicle location.
        loc = pose_location if pose_location is not None else vehicle.get_location()
        command, self.route_index = upcoming_command(
            loc, route, start_hint=self.route_index
        )

        try:
            _at, raw_light = traffic_light_fields(vehicle)
        except Exception:
            raw_light = "Off"
        light = normalize_light_state(raw_light)

        if tracks is not None:
            ego_tracks = list(tracks)
            sense = sense_mode
        else:
            actors = other_actors
            if actors is None and world is not None:
                actors = list(world.get_actors().filter("vehicle.*"))
                actors += list(world.get_actors().filter("walker.pedestrian.*"))
            actors = list(actors or [])
            ego_tracks = actors_to_ego_tracks(vehicle, actors)
            sense = "oracle"

        preds = constant_velocity_predict(
            ego_tracks, horizon_s=self.predict_horizon_s, dt_s=0.1
        )
        lead_m = lead_distance_from_predictions(preds)
        threat = crossing_threat(preds)

        decision = self.fsm.step(
            route_command=command,
            light_state=light,
            lead_distance_m=lead_m,
            crossing_agent_ahead=threat,
        )

        frenet_xy = None
        if self.use_frenet:
            plan = frenet_plan(
                vehicle,
                route,
                route_index_hint=self.route_index,
                desired_speed_kmh=decision.target_speed_kmh,
                predictions=preds,
            )
            self.route_index = plan.route_index
            goal = plan_to_control_target(plan)
            target = plan.target
            frenet_xy = plan.samples_xy
        else:
            from utils.routing import lookahead_target

            target, self.route_index = lookahead_target(
                loc, route, start_hint=self.route_index, lookahead_m=8.0
            )
            wp = route[self.route_index][0]
            heading_wp = route[min(self.route_index + 1, len(route) - 1)][0]
            goal = ControlTarget(
                target=target,
                path_transform=wp.transform,
                desired_speed_kmh=decision.target_speed_kmh,
                heading_yaw_deg=heading_wp.transform.rotation.yaw,
            )

        control = self.controller.compute(vehicle, goal)
        if decision.target_speed_kmh <= 0.5:
            control.throttle = 0.0
            control.brake = max(control.brake, 0.65)
        return AgentStep(
            control=control,
            decision=decision,
            target=target,
            route_index=self.route_index,
            n_predictions=len(preds),
            lead_m=lead_m,
            light=light,
            sense_mode=sense,
            predictions=preds,
            frenet_xy=frenet_xy,
        )
