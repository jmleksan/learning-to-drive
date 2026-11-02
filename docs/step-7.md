# Step 7 — Benchmarking with Bench2Drive

Step 7 replaces our own drive scorecard with a standard benchmark. The Stage 6
model runs unchanged on Bench2Drive, which uses Leaderboard 2.0 scoring on short
routes that each contain one hard scenario.

## Metrics

- **Route Completion (RC)**: percent of the route driven.
- **Infraction Penalty (IP)**: starts at 1.0 and is multiplied down per infraction.
- **Driving Score (DS)**: RC × IP per route, then averaged.
- **Success**: finished the route with no infractions except minimum speed.

Each route is capped at 4,000 ticks (200 s). That shows up as `Failed - TickRuntime`.

## Setup

| Piece | Where |
|---|---|
| CARLA 0.9.15 + additional maps | `C:\CARLA_0.9.15\WindowsNoEditor` |
| Bench2Drive (patched, see below) | `~/workspace/bench/Bench2Drive` |
| Python 3.10 env (`bench/requirements.txt`) | `~/workspace/bench/.venv` |
| CARLA `agents` module | `~/workspace/bench/carla_pythonapi` |
| Agent, runner, score table | `bench/` |

The project's own `.venv` stays on Python 3.12 and CARLA 0.9.16: the 0.9.15 client
has no 3.12 wheel, and 0.9.16 lacks Towns 11, 12, 13 and 15.

```bash
bash bench/run.sh dev10 models/transfuser.pt            # about 40 minutes
bash bench/run.sh bench2drive220 models/transfuser.pt   # roughly 12 hours (estimate)
~/workspace/bench/.venv/bin/python bench/summary.py results/dev10_gpsfix
```

`run.sh` starts the server itself, restarts it after a crash, and resumes. Use
`bench/summary.py` for partial sets: Bench2Drive's `tools/merge_route_json.py`
always divides by 220.

### Patches to Bench2Drive

They are saved as `bench/bench2drive.patch`, against Bench2Drive commit `7ec25d1`:

- `leaderboard_evaluator.py`: `CARLA_EXTERNAL_SERVER` skips launching a Linux
  server. `SIGUSR1` dumps all thread stacks.
- `route_parser.py` (both copies), `scenario_parser.py`: `getchildren()` →
  `list()`, removed in Python 3.9.
- `scenario_manager.py`: chase camera instead of the 70 m overhead view.

## Agent (`bench/tf_agent.py`)

The benchmark's sensors differ from training, and each gap is bridged here:

| | Training | Benchmark | Bridge |
|---|---|---|---|
| Lidar spin | full spin per frame | half spin per frame | merge two frames, ego-motion corrected |
| Lidar density | about 8,000 points | about 28,000 after merging | keep 27%, measured |
| Lidar type | semantic | ray-cast only | net ignores tags |
| Position | simulator truth | noisy GPS + compass | filter on speed and compass |
| Route | planner | GPS points | lookahead, extended past the end |
| Speed | unsigned | signed | `abs()` |
| Ego car | Tesla Model 3 | Lincoln MKZ 2020 | none |
| Rendering | Low | Epic | none (see below) |

Measured on Town05: position error 0.15 m mean, near target error 0.25 m mean.
Measured on Town13 after the GPS fix below: position error 0.49 m mean, near
target error 0.48 m mean.
Lidar density mattered a lot. At 1/6 (about 4,950 points) the car never moved.

## Low quality crashes CARLA 0.9.15 here

`-quality-level=Low` crashes the Windows server as soon as a route starts.

| Test | Result |
|---|---|
| All sensors, no scenario, Low | passes |
| Scenario, no sensors, Low | passes |
| Center or right camera alone + scenario, Low | passes |
| Left camera alone + scenario, Low | crash |
| All sensors + scenario, Low, Town03 | crash, "Shader compilation failures are Fatal" |
| All sensors + scenario, Epic | passes, all 10 Dev10 routes |

Ruled out: `-RenderOffScreen`, `-dx11`, sensor spawn timing, world settings, the
`hero` role name, and CUDA. The exact cause is unknown. VRAM is not the likely
reason: Low crashed at about 2.6 GB, and Epic runs at up to 5.9 GB.

The left camera faces oncoming traffic, so a vehicle material missing from the
Low shader set is the likely trigger, but that is not proven.

**So the benchmark runs on Epic, and the model was trained on Low.** That shift is
part of every score below and has not been measured separately.

## GPS bug in the first run

Towns 11, 13, and 15 are referenced near 41 degrees latitude, not 0. The first
version of `gps_to_xy` ignored that, so east-west distances read 33% too long on
five of the ten routes. Longitude is now scaled by the cosine of the route's
latitude. That needs no map access and leaves the other towns unchanged.

## Results: Dev10 (`models/transfuser.pt`)

| Route | Town | Scenario | Before fix | After fix | RC | IP | Result after fix |
|---|---|---|---:|---:|---:|---:|---|
| 3514 | 13 | ParkingExit | 0.0 | 0.0 | 0 | 1.00 | never left the parking spot |
| 3255 | 13 | ParkingCrossingPedestrian | 30.0 | 30.0 | 100 | 0.30 | hit a pedestrian and a vehicle |
| 26405 | 15 | StaticCutIn | 7.3 | 2.2 | 100 | 0.02 | finished with six collisions |
| 25381 | 05 | HazardAtSideLane | 100.0 | 100.0 | 100 | 1.00 | clean |
| 25378 | 03 | YieldToEmergencyVehicle | 70.0 | 70.0 | 100 | 0.70 | failed to yield |
| 25424 | 11 | ConstructionObstacleTwoWays | 31.2 | 41.6 | 42 | 1.00 | timed out |
| 2091 | 12 | NonSignalizedJunctionLeftTurn | 12.0 | 12.0 | 12 | 1.00 | stuck, timed out |
| 27494 | 04 | BlockedIntersection | 8.8 | 13.6 | 16 | 0.86 | timed out, left the route lanes |
| 17569 | 12 | SequentialLaneChange | 100.0 | 36.0 | 100 | 0.36 | two collisions |
| 28198 | 15 | SignalizedJunctionLeftTurnEnterFlow | 52.8 | 81.0 | 100 | 0.81 | left the route lanes |

| | Driving Score | Route Completion | Infraction Penalty | Success |
|---|---:|---:|---:|---:|
| Before fix | 41.2 | 65.0 | 0.67 | 20% |
| After fix | 38.6 | 66.9 | 0.70 | 10% |

**The fix had no measurable effect on the score.** The five affected routes rose
from 24.3 to 31.0 on average, but the five unaffected routes fell from 58.2 to
46.3 with no meaningful code change, so the difference is within run-to-run
variation. One unaffected route went from 100 to 36. Comparing two models on Dev10
needs several runs each, not one.

This is a small subset, so it can't be compared with published Bench2Drive 220
scores.

- The car is slow. Nearly every finished route has about 20 minimum-speed
  infractions (127 in total). They did not lower IP in this scoring.
- Four routes timed out. The model has no training for parking exits or for moving
  around a blockage.

No crashes or server restarts in either run. Windows memory over the last three
routes of the first run: at least 1.5 GB available, paging bursts up to 7,000
pages/s, no failures.

The full 220 routes were not run on this model: at roughly 12 hours it would add
little. Dev10 stays as the yardstick for new models, and the 220 is for a final one.
