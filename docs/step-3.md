# Step 3 — Sensor Fusion & Localization

Step 3 adds a second product next to perception: **where is the ego car?** Perception (Step 2) still answers “what’s around me” with late fusion on a BEV grid. Localization fuses **noisy GNSS + IMU + wheel speed** with a classical EKF so pose stays smooth.

Gate for this step: under injected GNSS noise, an EKF ego-pose beats raw GPS (lower position RMSE) on the same log. Multi-cam late road fusion and a radar BEV cue are the perception stretch — not a BEVFusion reimplementation.

## What this step has to do

- Record GNSS + IMU (+ GT pose) into `localization.csv` with a shared ENU meter frame.
- Run an EKF: predict with speed + yaw-rate, update with GNSS.
- Prove the filter on a synthetic noisy run (and on CARLA when available).
- Extend BEV road with left/center/right SegNet projection (late / geometric merge).
- Optionally mark forward radar range on the live BEV.

## How the stack is set up

**Localization.**

| Sensor | Role |
|--------|------|
| GNSS | Absolute position, noisy — EKF measurement |
| IMU gyro.z | Yaw rate — EKF predict |
| Wheel speed | Forward speed — EKF predict / soft update |
| Vehicle transform | GT for scoring only |

State: `[e, n, yaw, v]` in local east/north meters (yaw: 0 = north, + toward east). Code: `localization/ekf.py`, sensors in `localization/sensors.py`, record wiring in `expert/record.py`.

Predict uses wheel speed + IMU yaw rate (CARLA `gyro.z` is negated into ENU). Updates: GNSS position, wheel speed, and **IMU compass** (`yaw = π - compass`) so heading does not drift — available at inference, not GT.

**Perception (still late fusion).**

- Road: SegNet → project onto ground plane (`perception/bev.py`). `--multicam` runs left/center/right and OR-merges into one BEV.
- Objects: lidar pillars → OccupancyNet (unchanged).
- Radar: scalar range from `controls.csv` drawn as a cyan cue (`--radar`) — late confirmation, not a new net.

Left/right RGB mounts in `utils/vehicles.py` yaw **±55°** so side coverage is real (wide cams stay yaw 0 for stereo).

**Out of scope.** Soft lift / BEVFusion mid-level feature fusion. SegNet and OccupancyNet stay.

## Commands that matter

```bash
# Record localization (CARLA running)
python -m expert.record --gnss --imu --no-traffic --seconds 60 --cameras center

# Offline EKF gate (no CARLA) — synthetic noisy GPS
python -m localization.eval --synth --out models/loc_preview

# EKF on a recorded run
python -m localization.eval --data data/run_... --out models/loc_preview

# Multi-cam live BEV (+ optional radar cue)
python -m perception.bev --data data/run_... --video --mode live \
  --seg models/seg.pt --occupancy models/occupancy.pt --device cuda \
  --multicam --radar --out models/bev_preview

# Retrain SegNet on yawed left/right (mix with older center runs)
python -m expert.record --semantic --lidar --cameras center,left,right --seconds 180
python -m segmentation.train --data data/run_OLD data/run_NEW --epochs 20 --out models/seg.pt
```

Reference synth gate: `models/loc_preview/metrics.json` and `trajectory.png` (green=GT, blue=EKF, red=GNSS).

## What was learned

- **Fusion ≠ concat.** Localization fuses by trust (process vs measurement noise). Perception fuses by geometry on a shared BEV.
- **Late fusion** keeps SegNet / OccupancyNet as separate decisions; the grid is the meeting place.
- **BEVFusion** mixes *features* per cell before detecting — powerful, but not required for this gate. Read it; don’t rebuild it here.
- **GNSS alone jitters; IMU/odom alone drifts.** Together they make a usable pose.
- **Multi-cam** is the same ground-plane paint with different extrinsics — more road cells on the sides, same OccupancyNet objects.

## Gate checklist

- [x] GNSS/IMU plumbing + `localization.csv` schema and `--gnss` / `--imu` on record.
- [x] EKF RMSE beats raw GNSS on synthetic noisy log (`python -m localization.eval --synth`).
- [x] Multi-cam BEV road merge (`--multicam`) with yawed left/right mounts.
- [x] Radar range overlay on live BEV (`--radar`).
- [ ] Optional: re-record a CARLA run with `--gnss --imu` and confirm the same gate on real sensors.

## Verdict

Step 3 localization is classical and **not trained**. Perception nets from Step 2 remain; Stage 3 teaches how modalities meet (EKF for pose, projection for multi-cam road) without replacing them with a unified mid-level fusion net.
