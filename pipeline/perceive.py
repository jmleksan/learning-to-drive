"""Live camera + lidar perception for the classical stack (SegNet + OccupancyNet)."""
from __future__ import annotations

import queue
from dataclasses import dataclass
from pathlib import Path

import carla
import numpy as np
import torch

from perception.bev import (
    DRIVE_OBJECT_IDS,
    FORWARD_M,
    LATERAL_M,
    CELL_M,
    keep_drive_objects,
    objects_from_grid,
)
from perception.lidar import attach_semantic_lidar, points_from_measurement
from perception.occupancy import bev_from_occupancy, load_occupancy_model
from perception.track import ObjectTracker, TrackedObject
from segmentation.model import IMAGE_HEIGHT, IMAGE_WIDTH, load_seg_model
from utils.images import image_to_array
from utils.vehicles import attach_rgb_camera


def _pop_latest(q: queue.Queue):
    latest = None
    while True:
        try:
            latest = q.get_nowait()
        except queue.Empty:
            break
    return latest


def enu_to_world_location(
    world_map: carla.Map,
    e_m: float,
    n_m: float,
    up_m: float,
    origin,
    ref: carla.Location,
) -> carla.Location:
    """Local ENU → approximate CARLA Location via Jacobian at ``ref``."""
    from localization.pose_enu import location_to_enu

    eps = 1.0
    e0, n0, u0 = location_to_enu(world_map, ref, origin)
    e_x, n_x, _ = location_to_enu(
        world_map,
        carla.Location(ref.x + eps, ref.y, ref.z),
        origin,
    )
    e_y, n_y, _ = location_to_enu(
        world_map,
        carla.Location(ref.x, ref.y + eps, ref.z),
        origin,
    )
    # [de, dn] = J [dx, dy]
    j00, j01 = e_x - e0, e_y - e0
    j10, j11 = n_x - n0, n_y - n0
    det = j00 * j11 - j01 * j10
    if abs(det) < 1e-8:
        return carla.Location(ref.x, ref.y, ref.z + up_m - u0)
    de, dn = e_m - e0, n_m - n0
    dx = (j11 * de - j01 * dn) / det
    dy = (-j10 * de + j00 * dn) / det
    return carla.Location(ref.x + dx, ref.y + dy, ref.z + (up_m - u0))


@dataclass
class LivePerception:
    """RGB center + semantic lidar → tracked objects in ego frame."""

    device: str
    seg_model: object
    occ_model: object
    tracker: ObjectTracker
    cam_q: queue.Queue
    lidar_q: queue.Queue
    camera: carla.Sensor
    lidar: carla.Sensor

    @classmethod
    def attach(
        cls,
        world: carla.World,
        vehicle: carla.Vehicle,
        actors: list,
        *,
        seg_path: Path,
        occ_path: Path,
        fps: float = 20.0,
        device: str | None = None,
    ) -> "LivePerception":
        from utils.device import resolve_device

        dev = resolve_device(device)
        seg = load_seg_model(seg_path, device=dev)
        occ = load_occupancy_model(occ_path, device=dev)
        cam = attach_rgb_camera(
            world, vehicle, width=IMAGE_WIDTH, height=IMAGE_HEIGHT
        )
        lidar = attach_semantic_lidar(world, vehicle, fps=fps)
        actors.extend([cam, lidar])
        cam_q: queue.Queue = queue.Queue()
        lidar_q: queue.Queue = queue.Queue()
        cam.listen(cam_q.put)
        lidar.listen(lidar_q.put)
        tracker = ObjectTracker(dt_s=1.0 / max(fps, 1e-3), min_confirm=3)
        print(f"Live perception ON: SegNet={seg_path} OccupancyNet={occ_path} ({dev})")
        return cls(
            device=str(dev),
            seg_model=seg,
            occ_model=occ,
            tracker=tracker,
            cam_q=cam_q,
            lidar_q=lidar_q,
            camera=cam,
            lidar=lidar,
        )

    def reset(self) -> None:
        self.tracker.reset()

    def update(self) -> list[TrackedObject]:
        image = _pop_latest(self.cam_q)
        lidar_meas = _pop_latest(self.lidar_q)
        if image is None or lidar_meas is None:
            # Sensors not ready this tick — keep confirmed tracks aging out.
            return self.tracker.update([])

        rgb = image_to_array(image).astype(np.float32) / 255.0
        # SegNet
        tensor = (
            torch.from_numpy(rgb)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(self.device)
        )
        with torch.no_grad():
            logits = self.seg_model(tensor)
            labels = logits.argmax(dim=1)[0].detach().cpu().numpy().astype(np.int16)

        points = points_from_measurement(lidar_meas)
        grid = bev_from_occupancy(
            points,
            self.occ_model,
            labels=labels,
            device=self.device,
            forward_m=FORWARD_M,
            lateral_m=LATERAL_M,
            cell_m=CELL_M,
        )
        grid = keep_drive_objects(grid)
        objs = objects_from_grid(
            grid,
            class_ids=DRIVE_OBJECT_IDS,
            forward_m=FORWARD_M,
            lateral_m=LATERAL_M,
            cell_m=CELL_M,
        )
        # Drop very near ego (hood / self).
        objs = [o for o in objs if o.x_m >= 4.0]
        return self.tracker.update(objs)
