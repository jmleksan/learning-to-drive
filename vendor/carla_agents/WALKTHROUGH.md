# Reading CARLA `agents.navigation` (0.9.16)

Skim in this order — it mirrors a production modular stack:

1. **`global_route_planner.py`** — Builds a NetworkX graph from OpenDRIVE
   lane topology; `trace_route(origin, dest)` returns `(Waypoint, RoadOption)`.
   Same idea as our `utils/route_planner.py`.

2. **`local_planner.py`** — Holds a queue of upcoming waypoints and calls the
   PID controller each tick. `RoadOption` lives here.

3. **`controller.py`** — `VehiclePIDController` = lateral PID + longitudinal
   PID. This is the textbook PID layer Stage 4 reimplements in
   `pipeline/control.py` (`--controller pid`).

4. **`basic_agent.py`** — Wires GRP + LocalPlanner + traffic-light / obstacle
   checks. Closest analogue to our `pipeline/agent.py` + `pipeline/behavior.py`
   STOP rules.

5. **`behavior_agent.py`** — Extends BasicAgent with Cautious / Normal /
   Aggressive parameter sets and richer rule-based decisions (lane change,
   following distance). Compare to our FSM states.

Import via `pipeline.path_setup.ensure_carla_agents()` then
`from agents.navigation.basic_agent import BasicAgent`.
