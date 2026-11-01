# Vendored CARLA 0.9.16 `agents`

Source: [carla-simulator/carla @ 0.9.16](https://github.com/carla-simulator/carla/tree/0.9.16/PythonAPI/carla/agents)
License: MIT (CVC)

The pip `carla` wheel is the simulator API only. This tree is the miniature
modular stack shipped with CARLA:

| Module | Role |
|--------|------|
| `navigation/global_route_planner.py` | Lane-graph A* over OpenDRIVE topology |
| `navigation/local_planner.py` | Waypoint queue + PID tracking |
| `navigation/controller.py` | Lateral + longitudinal PID |
| `navigation/basic_agent.py` | Route follow + lights + simple obstacle stop |
| `navigation/behavior_agent.py` | Rule-based behaviors (cautious/normal/aggressive) |

Put `vendor/carla_agents` on `sys.path` (see `pipeline.path_setup`) then:

```python
from agents.navigation.basic_agent import BasicAgent
from agents.navigation.behavior_agent import BehaviorAgent
```

This repo also keeps a slightly adapted GRP in `utils/route_planner.py` used by
the expert record/drive loop; both are the same algorithm.
