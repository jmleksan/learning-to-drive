# Step 1 — Imitation Learning

Step 1 is a Town01 drive where the network owns the controls. The route only supplies a high-level command. The model outputs steer, throttle, and brake. The gate is fewer than 1 collision per kilometer, with the car staying in its lane through turns.

That gate passed on `data/drive_log_20261009_164441`.

## What the drive has to do

- Output throttle and brake, not steer alone.
- Condition on the command (`FOLLOW_LANE`, `TURN_LEFT`, `TURN_RIGHT`, `GO_STRAIGHT`) so a turn command is a different policy from lane following.
- Avoid causal confusion: the image, speed, and recent motion have to explain the controls, not a delayed frame or a command bit the image can drown out.
- See off-center views during training, so a small drift has a recovery label.
- Score the drive: collisions, lane invasions, route completion, speed, and cross-track error.
- Stay under 1 collision/km on Town01.

## How the model is set up

The network is a branched PilotNet (`expert/model.py`). A camera CNN, speed, and the traffic-light state share one trunk. The command is not concatenated into that trunk. It picks a separate steer, throttle, and brake head, so `TURN_LEFT` and `FOLLOW_LANE` do not have to share one set of weights.

Three camera frames are stacked so the net sees a short bit of motion. Training shifts the image sideways and nudges the steer label, which is the off-center recovery. Red and yellow lights, and frames where the wheel is actually turning, are sampled more often than straight cruising.

Radar is a bumper sensor, 30 m forward. Road returns below the horizon are dropped. The range is concatenated only onto the throttle and brake heads. Steer never reads it. A car ahead is allowed to change speed, and it is not allowed to mean "do not turn."

At drive time there is no path follower and no lead-car override. If the checkpoint was trained with `--radar`, the live range is fed to throttle and brake. Older checkpoints ignore it. There is no `--radar` flag on drive.

## What we changed to get here

Early drives could hold a lane and then failed in a few repeatable ways.

**The command arrived, and the wheel stayed straight.** `TURN_RIGHT` means the upcoming maneuver, not "steer now." Most recorded turn frames are still the straight approach, so the turn head outputs about 0 until the corner is in view. On a T-junction the command showed up about a second before a wall. More of the same curving-lane footage would not show that case. `expert/record_turns.py` starts a few meters before each exit and records one short clip per turn, with the command fixed for the whole clip.

**The car drove into the vehicle ahead.** The expert brakes from the simulator's distance to that car. The model only had the camera, so extra pictures of the same crash did not add the missing range. Radar was added for throttle and brake only. A first recording was unusable: the beam hit the road at about 12 m on every frame, including while stopped. The altitude filter removes those returns. Train `--radar` only on recordings made after that filter.

**Stopping meant touching the lead car.** The shared lead hold fully brakes inside 4.5 m center-to-center, which is bumper contact. `expert/record_leads.py` records a stop near 8 m. Some of those clips are a red light at an intersection: the command is the turn, a car is waiting, and the expert stops behind it instead of driving the turn. The label is "this command, car ahead, brake."

The set that landed is `data/leads_20261009_151128`: 32 clips, average stop 8.2 m. 20 are `FOLLOW_LANE`. 12 are a turn command held at a red light (4 left, 4 right, 4 straight).

**DAgger was the wrong place to put car-following.** DAgger here labels frames when the policy leaves the lane. Pointing it at every recovery, and oversampling those frames back into the lane model, made lane-keeping worse. Traffic and radar can be on during DAgger, but those frames stay out of the lane-following mix. The stop behavior comes from the lead clips.

**The scorecard columns were swapped.** `EventCounter.snapshot()` returns collisions, then lane invasions. The drive loop used to unpack that backwards, so the live `inv` / `col` line and `drive_log.csv` disagreed with `drive_metrics.json`. The json was the correct count. The unpack in `expert/drive.py` now matches the scorecard. Logs written before that fix still have the columns swapped.

## The run that passed

`models/radar_dagger.pt` was saved at 16:41. The drive in `data/drive_log_20261009_164441` started a few minutes later.

| | |
|---|---|
| Collisions | 0 (0.00 per km, limit 1.00) |
| Lane invasions | 4 |
| Route | 89.8% (2334 m of 2598 m) |
| Average speed | 14.0 km/h |
| Turn cross-track error | p90 0.51 m (limit 0.60 m) |
| Lane-follow cross-track error | p90 0.22 m |

The 600 s timer ended the run with about 260 m of route left. On the last left turn the car stopped for a red light in lane, with a car on radar, and then took the turn when the light changed.

## What is still open

Brightness and weather augmentation were never added. This pass is one Town01 route, not a repeated loop of the town. Radar stays off the steering trunk until there are demonstrations of steering around a car, which the expert does not do today.

## Other tools in `expert/`

| Script | Role |
|---|---|
| `collect.py` | Collect a lane-keeping dataset on Town01, then train |
| `fit.py` | Record and train, with an optional matched drive |
| `waypoint_drive.py` | Drive planned routes with a pure-pursuit controller |
| `scout_spawns.py` | Find spawn points with high turn coverage |
| `plan_routes.py` | Preview balanced routes without recording |
| `relabel_commands.py` | Rewrite command labels to match the expert's steering |
| `audit_data.py` | Audit recorded runs for lane-keeping data quality |
