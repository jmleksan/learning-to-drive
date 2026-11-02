# Step 2 — Perception Fundamentals

Step 2 decouples **seeing** from **driving**. The car needs a scene representation (road, objects, motion) that a policy can consume later. CARLA supplies free ground truth for training; at inference the stack uses only trained models plus lidar points.

Gate for this step: a perception module that outputs a bird’s-eye scene (road + objects + tracks) **without** reading CARLA semantic/depth GT at run time. That gate passed on the live/compare videos under `models/bev_preview/`.

## What this step has to do

- Train semantic segmentation from CARLA tags, then run it on RGB alone.
- Get range somehow (depth camera, stereo attempt, then lidar).
- Project the world into a bird’s-eye grid (BEV).
- Detect vehicles/people as objects on that grid and track them across frames.
- Replace hand-tuned lidar voting with a learned occupancy net.
- Prove the whole chain works **live** on recorded drives, with only RGB, lidar, and checkpoints (SegNet + OccupancyNet + tracker), then compare against CARLA truth so quality is measurable.

## How the stack is set up

**Road.** Center RGB → `SegNet` (`segmentation/`) → tag map. Road/sidewalk are painted into BEV by projecting those tags onto the ground plane (`perception/bev.py`). Training uses `seg/center` GT; live mode does not.

**Objects.** Roof semantic lidar → geometry **pillars** (`perception/pillars.py`) → `OccupancyNet` (`perception/occupancy.py`) → occupied cells + class. Small blobs are dropped. Driving output keeps **vehicles/people only**, raises the occupancy threshold, ignores detections closer than 4 m (ego body), and only shows tracks after **3 frames** of confirmation (`perception/track.py`).

**Grid.** 40 m forward, ±12 m lateral, 0.25 m cells. Ego is the white mark at the near edge.

**Truth for training/eval.** Camera tags + depth → `bev_from_labels` (answer key). Hand lidar voting (`bev_from_lidar`) was the first object path and remains useful for debugging.

**Recording.** `expert.record --semantic --depth --lidar --cameras center`, optional `--walkers`, `--weather` / time groups (`noon` / `sunset` / `night`). Weather is stored in `meta.json`.

## What we changed to get here

**Stereo depth was weak for BEV objects.** Center + `right_wide` stereo could estimate meters, but cars and people were noisy on the grid. Lidar hits were clearer for objects, so BEV objects moved to lidar; depth stayed as the training answer key for camera-projected truth.

**Hand lidar voting needs thresholds.** A cell needed enough hits of the same tag; tiny blobs were erased. That works until you tire of tuning `min_hits` and dust. Learned occupancy reads pillar stats (count, heights, offsets) and predicts free vs class — same job, trained decision boundary.

**Road from lidar rings looked bad.** Camera ground fill (segmenter on z = 0) gives readable road shape. Objects still come from lidar/occupancy on top.

**Pedestrians needed expert brake.** With walkers in the scene, the expert had to stop for people ahead or the run recorded collisions. Perception then had people-tagged lidar to learn from.

**Tracking is greedy association.** Nearest same-class blob within a few meters keeps an id and smoothed ego-frame velocity. Good enough for a scene list; not a production Kalman tracker.

**False positives vs misses.** Empty-road ghosts (fake cars/poles) were cut with threshold, blob size, vehicle-only allowlist, track confirm, and min range. A weather-heavy retrain then went **too shy** (missed the lead fire truck). Keeping the original busy clear run in the mix and retraining again restored detection: ~93% veh@present and ~6/199 empty-FP on `data/run_20261016_200616`.

**Live mode.** `--mode live` uses only RGB + lidar + checkpoints. `--mode compare` adds CARLA truth BEV and a green/red/blue match panel.

## Commands that matter

```bash
# Train
python -m segmentation.train --data data/run_...
python -m perception.train_occupancy --data data/run_A data/run_B --epochs 12 --device cuda

# Live perception (no CARLA GT)
python -m perception.bev --data data/run_... --video --mode live \
  --seg models/seg.pt --occupancy models/occupancy.pt --device cuda

# Score against truth
python -m perception.bev --data data/run_20261016_200616 --video --mode compare \
  --device cuda --out models/bev_preview --stride 2
```

Checkpoints: `models/seg.pt`, `models/occupancy.pt`.  
Reference compare: `models/bev_preview/run_20261016_200616_compare.mp4`.

## What was learned

- **Perception is its own product.** A policy should eventually eat a BEV/track list, not reinvent seeing from raw pixels every time — even if end-to-end RGB can still work for demos.
- **GT is for labels, not for runtime.** Train on CARLA tags/depth; ship SegNet + lidar pillars + OccupancyNet.
- **Sensor choice is empirical.** Stereo was the “right” textbook path for depth; lidar won for object BEV in this town and resolution.
- **Hand rules ≈ what the net should learn.** Hit counts and blob cleanup are occupancy in disguise. The net cleans noise and fills gaps when the data mix is honest.
- **Data mix beats more of the same.** Too much empty/weather → shy net. Too little empty → ghosts. Busy clear traffic + a little empty + a little weather is the useful recipe.
- **Metrics that match the goal.** Val IoU (~0.2–0.3 here) is a training signal. For “usable,” watch **veh@present**, **empty-FP**, and whether the lead vehicle stays boxed on the compare video.
- **Filters are not cheating for Step 2.** Threshold, class allowlist, and track confirm are how you turn a noisy grid into a driving-facing scene list. Retrain when filters stop helping.

## What is still open (not required to leave Step 2)

- A/B: drive policy on segmented/BEV input vs raw RGB.
- Stable API: dump per-frame tensors (BEV + tracks) for imitation learning.
- Stronger tracker / camera–lidar fusion / other towns.
- Chasing production occ IoU (0.5+) — polish, not a new concept.

## Verdict

Step 2 is complete: segmentation, depth/lidar exploration, BEV, learned occupancy, tracking, and a live scene representation independent of the driving policy. Further occupancy tweaks are optional; the learning move is to **consume** this representation in Step 1-style driving (or the next stage), not to grind another weather retrain.
