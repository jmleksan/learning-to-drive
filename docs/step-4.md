# Step 4 — Classical Modular Pipeline

Step 4 rebuilds the Town01 driving task as a **classical** stack:
**Perception → Prediction → Planning → Control**, with no neural net for
planning or control. Perception may use oracle CARLA actors (like BasicAgent)
or Step 2 nets; localization may use the Step 3 EKF.

The illuminating part is the comparison: same scorecard as Step 1
(`collisions/km`, lane invasions, route %, speed, CTE) against
`models/radar_dagger.pt`.

## Architecture

```
OpenDRIVE map ──► GlobalRoutePlanner (lane graph)
                         │
Tracks / actors ──► CV prediction ──► Behavior FSM ──► Frenet + costs
                                                         │
                                              PID / PP / Stanley / MPC / hybrid
                                                         │
                                              VehicleControl (steer/throttle/brake)
```

CARLA’s own miniature stack lives under `vendor/carla_agents/` (0.9.16 MIT):
BasicAgent, BehaviorAgent, LocalPlanner, VehiclePIDController. They are the
production modular pattern in miniature.

| Layer | This repo | CARLA agents.navigation |
|-------|-----------|-------------------------|
| Global route | `utils/route_planner.py` / GRP | `global_route_planner.py` |
| Behavior | `pipeline/behavior.py` FSM | `behavior_agent.py` rules |
| Local path | `pipeline/motion.py` Frenet | LocalPlanner waypoint queue |
| Control | `pipeline/control.py` | `controller.py` PID |
| Prediction | `pipeline/prediction.py` CV | implicit / none |

## Gates

1. **Agents readable** — `vendor/carla_agents` imports (`BasicAgent`, `LocalPlanner`).
2. **Map literacy** — `python -m pipeline.map_dump` writes OpenDRIVE stats.
3. **Planner demos** — `python -m pipeline.demo_planners` runs A* + RRT* offline.
4. **Classical drive** — `python -m pipeline.drive` completes a Town01 route and
   writes `drive_metrics.json` with `collisions_per_km ≤ 1.0` (same gate spirit as Step 1).
5. **Comparison** — `python -m pipeline.compare` tables classical vs
   `data/drive_log_20261009_164441`.

## Commands

```bash
# Study vendored CARLA agents (after path_setup)
python -c "from pipeline.path_setup import ensure_carla_agents; ensure_carla_agents(); from agents.navigation.basic_agent import BasicAgent; print(BasicAgent)"

# OpenDRIVE literacy (no CARLA) + live Town01 dump (CARLA running)
python -m pipeline.map_dump --offline --out models/pipeline_preview/map_stats.json
python -m pipeline.map_dump --map Town01 --out models/pipeline_preview/map_stats_live.json

# A* / RRT* teaching demos (no CARLA)
python -m pipeline.demo_planners --out models/pipeline_preview/planner_demo

# Classical drive — swap controllers: pid | pure_pursuit | stanley | hybrid | mpc
python -m pipeline.drive --seconds 120 --controller hybrid --map Town01

# See decisions in the spectator HUD (behavior why / preds / Frenet)
python -m pipeline.drive --seconds 120 --traffic 10 --debug-decisions

# Your camera+lidar nets instead of CARLA actor GT (+ optional EKF ego for planning)
python -m pipeline.drive --seconds 120 --traffic 15 --perceive --debug-decisions \
  --seg models/seg.pt --occupancy models/occupancy.pt
python -m pipeline.drive --seconds 120 --ekf --plan-from-ekf --debug-decisions

# Compare to Step 1 learned agent
python -m pipeline.compare \
  --classical data/classical_drive_XXXX/drive_metrics.json \
  --learned data/drive_log_20261009_164441/drive_metrics.json \
  --out models/pipeline_preview/compare.md
```

## Control ladder

| Name | Idea |
|------|------|
| `pid` | Lateral CTE/heading PID + speed PID (CARLA LocalPlanner style) |
| `pure_pursuit` | Geometric chase of a lookahead point |
| `stanley` | Heading + crosstrack arctan correction |
| `hybrid` | Blend of PP + Stanley (close to the old waypoint follower) |
| `mpc` | Short-horizon kinematic bicycle search |

PID is the baseline. MPC pays off once there is a path and predictions worth
looking ahead at.

## Behavior FSM states

`LANE_FOLLOW`, `LANE_CHANGE_*`, `TURN`, `STOP` (red/yellow or close lead),
`YIELD` (soft gap / crossing threat). Inputs: route `RoadOption` command,
traffic light, constant-velocity predictions.

## Motion planning

- **Live:** Frenet lateral samples along the GRP centerline, costed for comfort,
  progress, and clearance vs predicted agents.
- **Teaching:** grid A* and RRT* in `pipeline/demo_planners.py` (ASCII maps under
  `models/pipeline_preview/planner_demo/`).

## Comparison protocol

Match conditions as closely as practical:

- Map: Town01
- Scorecard: same fields as Step 1 (`utils/drive_score.py`)
- Learned baseline: `data/drive_log_20261009_164441` / `models/radar_dagger.pt`
- Classical: `pipeline.drive` with chosen `--controller`

The table is in `models/pipeline_preview/compare.md`.

## Status

Implemented: vendored agents, OpenDRIVE sample dump, A*/RRT* demos,
prediction / FSM / Frenet / control ladder, compare tooling, learned baseline
table under `models/pipeline_preview/compare.md`.

**Live classical Town01 drive:** one 120 s drive with the `hybrid` controller
had 0 collisions and 0 lane invasions over 462 m. `models/pipeline_preview/compare.md`
tables it against Step 1, but Step 1 drove for 600 s, so the route percentages
(56.8% and 89.8%) are not like for like. The raw drive log was not kept.

## What was learned

- Why production stacks stay modular even when components are learned.
- Where classical wins (lights, explicit stop distance, tunable costs) and where
  the learned agent still feels smoother or more “human.”
- Why hybrids dominate: classical structure + learned perception/prediction.

## Package map

| Path | Role |
|------|------|
| `vendor/carla_agents/` | Upstream BasicAgent / BehaviorAgent / LocalPlanner / PID |
| `pipeline/prediction.py` | Constant-velocity rollouts |
| `pipeline/behavior.py` | Behavior FSM |
| `pipeline/motion.py` | Frenet + A* / RRT* |
| `pipeline/control.py` | Swappable controllers |
| `pipeline/agent.py` | Wired classical agent |
| `pipeline/drive.py` | Scored Town01 drive + optional EKF |
| `pipeline/map_dump.py` | OpenDRIVE topology dump |
| `pipeline/compare.py` | Classical vs learned metrics |
