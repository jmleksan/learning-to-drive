# Step 6 — TransFuser-lite (camera + lidar → waypoints)

Step 6 reimplements a simplified TransFuser: one network reads the RGB strip
and a lidar BEV, fuses them with attention at four scales, and outputs four
future waypoints. A PID controller turns those into steer, throttle, and brake.
Like Step 1 it learns from the expert, but the output is a trajectory instead
of raw controls, and auxiliary heads make the features inspectable.

## Architecture

```
left|center|right RGB (3×120×480) ─► ResNet stage 1..4 ──┐
                                        ▲│ fusion ×4     ├─► avg-pool + sum ─► MLP ─► GRU ×4 ─► waypoints (0.5 s apart)
lidar BEV + goals (4×128×128)    ─► ResNet stage 1..4 ──┘                       ▲
                                                                     2 target points (25 m, 50 m on route)
speed ─► embedded into every fusion block

aux:  camera branch → image semantics (7 classes)
      lidar branch  → BEV semantics from lidar tags (road / other / vehicle / pedestrian)
      fused vector  → red light for ego, expert braking
```

- **Fusion** (`transfuser/model.py:Fusion`): each branch is pooled to a token
  grid (image 4×16, lidar 8×8), tokens get a learned position embedding plus a
  speed embedding, one transformer layer attends over all of them, and the
  result is upsampled and added back to both branches.
- **Waypoint decoder**: GRUCell, 4 steps, input = previous waypoint + both
  target points. Waypoints are ego frame (x forward, y right), meters. L1 loss.
- **Goal channels**: the two targets are also drawn as blobs into the lidar BEV,
  one channel each, as TransFuser rasterizes its goal. They are not sensor data:
  they come from the route, so they exist at inference too. A target outside
  the grid is clamped to the nearest edge.
- **Controller** (`transfuser/controller.py`): TransFuser's PID, with one change.
  Steer aims at the midpoint of waypoints 1–2. Target speed = their spacing /
  0.5 s. Stopping comes from bunched waypoints. Braking is proportional instead
  of all-or-nothing, and full braking needs 3 frames of "stop" in a row.
- **Backbone** (`--backbone`, default `resnet18`): torchvision ResNet-18 for
  both branches. The camera branch starts from ImageNet
  weights (`--no-pretrained` for random init). The lidar branch always starts
  random: its input has 4 channels, and ImageNet features don't transfer to a
  BEV histogram. `--backbone small` is a narrower ResNet-10-ish from scratch.
  The default learning rate is 1e-4 for resnet18 (so early updates don't wreck
  the pretrained features) and 3e-4 for small.

| backbone | params | peak GPU (batch 32) | train step | inference |
|---|---|---|---|---|
| resnet18 (default) | 27.0M | 1.6 GB | ~300 ms | ~18 ms/frame |
| small | 7.0M | 0.9 GB | ~170 ms | ~18 ms/frame |

Pretraining makes each training step slower. The benefit is fewer epochs to a
good error, and image features that hold up better on weather and lighting not
in the data.

### What differs from the paper

| TransFuser | Here |
|---|---|
| ImageNet ResNet-34 image / ResNet-18 lidar | ImageNet ResNet-18 image / ResNet-18 lidar from scratch |
| 256×256 lidar, 32 m | 128×128, 0.25 m cells, 32 m ahead ±16 m |
| BEV aux from the HD map | BEV aux from semantic-lidar tags (only cells with hits) |
| depth + semantics + 3D boxes aux | image semantics + BEV semantics + red light/brake |
| hundreds of thousands of frames, many towns | Town01, a few runs |

The lidar-tag BEV label is the weakest aux: road vs obstacle is close to
reading the height channels. Vehicle vs other obstacle is the real signal.

## Data

The existing recorder already logs everything. `--gnss` is only there for
`localization.csv`. The waypoint labels use its ground-truth pose columns,
not the noisy GNSS.

```bash
python -m expert.record --cameras center,left,right --semantic --lidar --gnss \
  --traffic 30 --walkers 20 --seconds 600 --spawn-index 42
```

Record several runs (different `--spawn-index`, a couple with `--weather`),
and keep one whole run for validation. Frames within a run are near-duplicates,
so a random frame split would leak. Each 10-minute run is about 12k frames.
At the default `--stride 2` that is about 6k samples.

Labels (`transfuser/data.py`):
- Waypoints are the poses 0.5 / 1 / 1.5 / 2 s later, in the current ego frame.
- The target points are the poses 25 m and 50 m further along the driven path.
  At drive time they are `lookahead_target` on the planned route.
- `to_ego` assumes this repo's Town01 ENU (e = +x, n = +y), a mirror of true
  east/north. Check any new label against real steering: corr(steer, waypoint y)
  should be about +0.97.
- Frames where the near target is over 4 m sideways (32%) are sampled twice as
  often, so junctions are about half of training (`--junction-weight`).
- A sample is dropped if its future runs past the end of the log or across a
  pose jump.
- Long stops are capped. The recorder's expert can deadlock behind a stuck NPC
  for minutes, which would be thousands of identical "stand still" samples. Of
  each stop, only the first 10 s (a normal light or queue wait) and the last
  4 s (pulling away) are kept.
- The red-light label is `at_traffic_light` with Red/Yellow. The brake label is
  expert brake > 0.1.

## Run it

```bash
# train (fusion + aux)
python -m transfuser.train --data data/run_* --val data/run_VAL --out models/transfuser.pt

# ablations: same data, one change each
python -m transfuser.train --data data/run_* --val data/run_VAL --modality camera --out models/tf_camera.pt
python -m transfuser.train --data data/run_* --val data/run_VAL --modality lidar  --out models/tf_lidar.pt
python -m transfuser.train --data data/run_* --val data/run_VAL --no-aux         --out models/tf_noaux.pt
python -m transfuser.train --data data/run_* --val data/run_VAL --no-pretrained  --out models/tf_scratch.pt

# open-loop table + BEV/waypoint PNGs
python -m transfuser.eval --data data/run_VAL --models models/transfuser.pt models/tf_*.pt --viz 20

# closed loop (CARLA running), same scorecard as pipeline.drive
python -m transfuser.drive --model models/transfuser.pt --seconds 300 --traffic 30
```

About 150 s per epoch on the 3060 with ~34k samples. On long runs CUDA can die
with "unknown error", as in Step 5, sometimes several times within one epoch.
`--auto-resume` continues from `<out>_last.pt` (full optimizer state, saved
every `--save-every` batches, about 35 s), so loop it. A resumed epoch reshuffles
and runs only the batches it still owes. Pinned memory is off because the
aborts surfaced in the pinned allocator.

```bash
until python -m transfuser.train --data data/run_* --val data/run_VAL --auto-resume; do sleep 30; done
```

Checkpoint selection uses val waypoint L1 only, so the aux and no-aux runs are
picked the same way. `*_last.pt` is the final epoch and `.json` is the per-epoch
history.

## Evaluation

Open-loop (`transfuser.eval`): ADE/FDE, turn-frame ADE (expert's 2 s waypoint
more than 1.5 m sideways), stop-frame ADE, stop accuracy (does the PID brake
from the predicted waypoints exactly when the expert stops), aux mIoU, and red
light / brake accuracy.

Closed-loop (`transfuser.drive`): route completion, collisions/km (gate ≤ 1),
lane invasions, CTE p50/p90, and stopped %. The drive log keeps the predicted
waypoints, target point, and affordance probabilities for every frame.

Open-loop numbers do not predict closed-loop driving well. A low ADE can still
drift, because the model has only seen the expert's states (the Step 5
argument). The recorder's mid-turn nudges give some recovery data. If closed
loop drifts, the next step is TransFuser++-style shift/rotation augmentation.

## Results

Data: 9 Town01 runs, about 38k samples after the stop cap. Validation is the
whole spawn 120 run. The best epoch was picked on that same run, so open-loop
numbers are slightly optimistic.

### Open loop

| model | ADE | FDE | turn ADE | stopped ADE | stop acc | pre-turn, mirrored target |
|---|---|---|---|---|---|---|
| v1: target in GRU only | 0.21 | 0.46 | 0.10 | 0.36 | 0.92 | +0.44 m |
| v2: goal channels, 50 m target, junction sampling | 0.17 | 0.36 | 0.11 | 0.24 | 0.92 | -1.21 m |

The last column is the 2 s waypoint's sideways position on frames just before a
turn, with both targets mirrored (+ = the expert's turn side, the expert is
+1.90 m). A model that follows the route should go negative. v1 did not: it read
the turn from the scene as well as the target. Open-loop ADE cannot show this,
because road and route agree in recorded data. Goal channels, far target, and
sampling changed together, so their shares are unknown.

Most error is along-track (speed and when to go). Lateral error is about 3 cm.

### Closed loop (no traffic, 4 drives of 150 s: spawns 0, 230, 76, 203)

| setup | distance | collisions | lane invasions | turns taken / missed | false stops |
|---|---|---|---|---|---|
| v1, rain (spawns' first drives) | 572 m | 55 | 17 | 3 of 4 turn failures were left turns | - |
| v2, rain | 1,135 m | 0 | 8 | 10 / 0 | - |
| v2, ClearNoon, full brake | 1,594 m | 0 | 16 | 14 / 0 | 49 |
| v2, ClearNoon, proportional brake | 1,522 m | 0 | 16 | 14 / 0 | 15 |
| same, 30 NPC cars | 1,753 m | 1 | 14 | 14 / 0 | 16 |
| same, 30 NPC cars, firm-brake fix | 1,548 m | 0 | 10 | 12 / 0 | 15 |

Across both traffic batches that is 1 collision in 3.3 km (0.30/km), under the
gate of 1. It is one town and eight 150 s drives, so a thin sample. In traffic
some "false stops" may be real queueing, since the count only checks for a light.

The one collision was the car's fault: it rear-ended a stopped ambulance at about
9 km/h. It slowed correctly, but the soft brake cap (0.3) meant it arrived too
close, about 14 → 6 km/h while the target speed read 2 km/h. Right behind the
ambulance the model's target jumped to 14 km/h and it drove in. The expert stops
over 4 m back, so this close-up view is not in the data. The controller now brakes
firmly when the target stays under 60% of current speed for 3 frames. That avoids
getting so close, but the model's blind spot up close is still there.

v1's collisions came after leaving the route, and that count is inflated by the
car staying pinned. Turns "taken" means the target swung back ahead with the car
within 2.5 m of the route. A false stop is under 3 s with no light governing the
car, so a stop short of a light's trigger volume counts too.

### Fusion vs camera-only

Same data and settings. Camera-only has no lidar branch, so it also loses the
goal channels and gets the route only through the decoder.

| open loop | fusion | camera-only |
|---|---|---|
| ADE / FDE | 0.17 / 0.36 m | 0.23 / 0.46 m |
| Stopped-frame error | 0.24 m | 0.51 m |
| Stop accuracy | 91.8% | 88.7% |
| Red-light / brake accuracy | 99.8 / 99.5% | 97.8 / 97.6% |
| Image mIoU | 0.81 | 0.76 |
| Pre-turn, mirrored target | -1.21 m | -1.38 m |

Closed loop with 30 NPC cars (firm-brake controller, 4 drives of 150 s):

| | fusion | camera-only |
|---|---|---|
| Distance | 1,548 m | 1,420 m |
| Collisions | 0 | 2 (one incident, side-swiping a firetruck) |
| Short stops, no light | 15 | 8 |
| Lane invasions | 10 | 8 |

Fusion wins open loop, mostly on stop and go timing. Closed loop is too thin to
separate them: one collision incident in each model's traffic drives, over 3.3 km
for fusion and 1.4 km for camera-only. Camera-only was if anything smoother.

Camera-only follows the target as well as fusion with no goal channels, so those
channels are probably not what fixed route following. The 50 m target or junction
sampling is more likely. Not isolated.

### Pedestrians (fusion, 30 cars + 40 pedestrians, 4 drives of 150 s)

1,436 m, 0 pedestrians hit, 3 collisions in one incident, 12 lane invasions, 14 of
15 turns clean (one 2.73 m wide corner, recovered). How many pedestrians the car
actually met is unknown because the drive does not log it, so this is weak
evidence. The incident: stopped close behind a tiny Microlino car, it pulled away
into it when traffic moved. Same close-range blind spot as the ambulance. Three
counted hits in 2 s of contact gives 2.09/km, 0.70/km per incident.

Pedestrians slow CARLA about 3× in real time (150 s takes 380 to 500 s). The sim
steps are fixed, so results are not affected.

Two more collisions were reported for that batch, with a wall. They came after the
drive ended, when sync mode was off and the car rolled on. They are not counted,
and `transfuser.drive` now reads collisions from the logged frames.

### Bugs found by driving

None of these were visible in open-loop numbers:

- **Mirrored labels.** `to_ego` had y flipped. Target and waypoints were wrong in
  the same way, so validation looked perfect while the car turned away from its
  target. Fixed by negating y in the decoder's weights, no retraining.
- **Weather.** `transfuser.drive` never set it, so early drives ran in rain the
  model never saw. The worst stall (125 s at one light) was in that weather and did not recur in ClearNoon, though the cause was not isolated.
- **All-or-nothing brake.** A small target-speed dip became a full brake. The
  expert rarely slows on open road, so low speed itself seems to trigger "stop".
  The car stopped, restarted, and stopped again every 3 s or so.

Open: close-range blind spot behind stopped cars (2 of 2 fusion collision
incidents), false stops, line crossings on corners (about 1 m from center), weather
generalization, and no recovery data. Lidar-only, no-aux, and no-ImageNet were not
run. The bar is one town, short drives, so this is a working baseline, not a
robust driver.
