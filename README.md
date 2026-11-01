# Learning Autonomous Driving in CARLA

A staged, hands-on build of a self-driving stack in the [CARLA](https://carla.org/) simulator, from a behavior-cloned CNN to a simplified TransFuser scored on a public benchmark. Each stage has its own code, its own pass/fail gate, and a write-up of what broke and why.

**Latest result:** TransFuser-lite, trained only on Town01, scored **Driving Score 41.2** on the 10-route Bench2Drive Dev10 subset (Route Completion 65.0%, Infraction Penalty 0.67). It drove 3.3 km of Town01 traffic with one collision, then got stuck or blocked on four of the ten hard scenarios.

## Stages

| # | Stage | What was built | Result | Write-up |
|---|---|---|---|---|
| 1 | Imitation learning | Branched CNN with radar and DAgger | 0 collisions, 89.8% of a 2.6 km route | [step-1](docs/step-1.md) |
| 2 | Perception | SegNet, learned occupancy, BEV, tracking | Live scene view with no simulator truth | [step-2](docs/step-2.md) |
| 3 | Localization | EKF on GNSS, IMU, and wheel speed | 0.69 m RMSE against 4.66 m for raw GNSS, on a synthetic noisy log | [step-3](docs/step-3.md) |
| 4 | Classical pipeline | Prediction, behavior FSM, Frenet planning, five controllers | 0 collisions and 0 lane invasions in a 462 m, 120 s drive | [step-4](docs/step-4.md) |
| 5 | Reinforcement learning | SAC on a perceived BEV | 95.1% of a 711 m route, 0 collisions, no traffic | [step-5](docs/step-5.md) |
| 6 | TransFuser-lite | Camera + lidar fusion to waypoints | 1 collision in 3.3 km with 30 cars | [step-6](docs/step-6.md) |
| 7 | Benchmarking | Agent running on Bench2Drive | Driving Score 41.2 on Dev10 | [step-7](docs/step-7.md) |

The stages are not all scored the same way, so compare results within a stage rather than across this table.

## Benchmark results

`models/transfuser.pt` on Bench2Drive Dev10, using CARLA 0.9.15. Raw results are in [`results/`](results/dev10_transfuser/results.json).

| Scenario | Town | Driving Score | Outcome |
|---|---|---:|---|
| HazardAtSideLane | 05 | 100.0 | Clean finish |
| SequentialLaneChange | 12 | 100.0 | Clean finish |
| YieldToEmergencyVehicle | 03 | 70.0 | Finished, failed to yield |
| SignalizedJunctionLeftTurnEnterFlow | 15 | 52.8 | Finished, one collision |
| ConstructionObstacleTwoWays | 11 | 31.2 | Finished, left the route lanes |
| ParkingCrossingPedestrian | 13 | 30.0 | Finished, hit a pedestrian and a vehicle |
| NonSignalizedJunctionLeftTurn | 12 | 12.0 | Stuck, timed out |
| BlockedIntersection | 04 | 8.8 | Hit scenery, timed out |
| StaticCutIn | 15 | 7.3 | Collision, then blocked |
| ParkingExit | 13 | 0.0 | Never left the parking spot |

**How to read this score**

- Dev10 is a subset, so 41.2 is not comparable to published 220-route Bench2Drive scores or to the official CARLA Leaderboard. The full 220 routes were not run.
- With ten routes, one route can move the average by up to 10 points.
- The model was trained on Low quality images but tested on Epic, because Low quality crashes CARLA 0.9.15 on this setup. The cost of that mismatch was not measured.
- The car drives slowly, with 124 minimum-speed infractions. They did not lower the score here.

## What the project showed

- **Good open-loop metrics can hide a broken driver.** TransFuser-lite's first version had low waypoint error but ignored its route target at junctions. A test that mirrored the target exposed it, and drawing the targets into the lidar view fixed it ([Step 6](docs/step-6.md)).
- **A little of the right data beats a lot of the same data.** In Step 1, 120 s of DAgger fixed failures that 600 s of expert driving had not ([Step 5](docs/step-5.md)).
- **Average driving says little about hard scenarios.** The same model that handled Town01 traffic scored 0–12 on four Dev10 routes ([Step 7](docs/step-7.md)).
- **Input details decide whether a model works at all.** At the wrong lidar density, the car never moved on the benchmark ([Step 7](docs/step-7.md)).

## Repository layout

| Path | Contents |
|---|---|
| `expert/` | Step 1: rule-based expert, data recording, imitation learning, DAgger |
| `segmentation/`, `perception/` | Step 2: SegNet, BEV, occupancy, tracking |
| `localization/` | Step 3: EKF and localization evaluation |
| `pipeline/` | Step 4: classical planning and control |
| `rl/` | Step 5: Gymnasium environment, PPO and SAC training |
| `transfuser/` | Step 6: model, training, evaluation, closed-loop drive |
| `bench/` | Step 7: Bench2Drive agent, runner, score table, patch |
| `utils/` | Shared CARLA helpers: connection, routing, sensors, scoring |
| `vendor/carla_agents/` | CARLA's MIT-licensed `agents` module, version 0.9.16 |
| `scripts/` | Connection check, map picker, vehicle spawn, GPU reset |
| `results/` | Benchmark results kept in git |
| `docs/` | One write-up per stage |

Datasets, drive logs, and checkpoints are about 35 GB and are not in git. `data/` and `models/` must be regenerated with the commands in each stage's write-up, so results here cannot be reproduced from a fresh clone without recording and training again.

## Setup

Tested with CARLA running on Windows and the Python code in WSL2, on a 6 GB RTX 3060 laptop GPU. The client finds the Windows host automatically, or you can set `CARLA_HOST`.

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt

# With CarlaUE4.exe (0.9.16) running on Windows
.venv/bin/python -m scripts.connect
```

### Example commands

```bash
# Record expert data
python -m expert.record --semantic --lidar --gnss --cameras center,left,right --seconds 180

# Train and drive TransFuser-lite
python -m transfuser.train --data data/run_A data/run_B --val data/run_C --out models/transfuser.pt
python -m transfuser.drive --model models/transfuser.pt --traffic 30 --spawn-index 42
```

### Running the benchmark

The benchmark uses a separate install, because it needs CARLA 0.9.15 and Python 3.10 at most:

1. Install CARLA 0.9.15 and its additional maps to `C:\CARLA_0.9.15`.
2. Clone [Bench2Drive](https://github.com/Thinklab-SJTU/Bench2Drive) at commit `7ec25d1` into `~/workspace/bench`, then run `git apply` with `bench/bench2drive.patch`.
3. Create `~/workspace/bench/.venv` with Python 3.10, then install `bench/requirements.txt`.
4. Copy `PythonAPI/carla/agents` from the CARLA install into `~/workspace/bench/carla_pythonapi/`.

```bash
# run.sh starts CARLA itself, so leave any other CARLA closed
bash bench/run.sh dev10 models/transfuser.pt
~/workspace/bench/.venv/bin/python bench/summary.py data/bench/dev10_transfuser
```

## Known limitations

- Training data comes from Town01 only.
- The model has no training on the benchmark's scenarios, and NoCrash was not run.
- The exact cause of the Low quality crash on CARLA 0.9.15 was not identified.
- Results come from single runs. Only one Dev10 route was repeated.

## Acknowledgments

TransFuser-lite is a simplified reimplementation of [TransFuser](https://arxiv.org/abs/2205.15997) (Chitta et al.). Evaluation uses [Bench2Drive](https://github.com/Thinklab-SJTU/Bench2Drive), the [CARLA Leaderboard](https://leaderboard.carla.org/), and Scenario Runner.
