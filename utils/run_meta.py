"""Save/load per-run metadata (map, spawn) for matched evaluation."""
from __future__ import annotations

import json
from pathlib import Path

import carla


def carla_map_to_load_name(full_name: str) -> str:
    """Carla/Maps/Town10HD_Opt -> Town10HD_Opt"""
    return full_name.rsplit("/", 1)[-1]


def find_spawn_index(vehicle: carla.Vehicle, world: carla.World) -> int:
    location = vehicle.get_transform().location
    spawn_points = world.get_map().get_spawn_points()
    best_index = 0
    best_dist = float("inf")
    for index, transform in enumerate(spawn_points):
        dist = location.distance(transform.location)
        if dist < best_dist:
            best_dist = dist
            best_index = index
    return best_index


def save_run_meta(
    run_dir: Path,
    *,
    map_name: str,
    spawn_index: int,
    fps: float,
    lane_change: bool,
    traffic: int = 0,
    walkers: int = 0,
    semantic: bool = False,
    depth: bool = False,
    lidar: bool = False,
    gnss: bool = False,
    imu: bool = False,
    weather: str | None = None,
    geo_origin: dict | None = None,
    gnss_noise_lat_lon_deg: float | None = None,
) -> Path:
    meta = {
        "map": map_name,
        "map_load_name": carla_map_to_load_name(map_name),
        "spawn_index": spawn_index,
        "fps": fps,
        "lane_change": lane_change,
        "traffic": int(traffic),
        "walkers": int(walkers),
        "semantic": bool(semantic),
        "depth": bool(depth),
        "lidar": bool(lidar),
        "gnss": bool(gnss),
        "imu": bool(imu),
    }
    if weather is not None:
        meta["weather"] = weather
    if geo_origin is not None:
        meta["geo_origin"] = geo_origin
    if gnss_noise_lat_lon_deg is not None:
        meta["gnss_noise_lat_lon_deg"] = float(gnss_noise_lat_lon_deg)
    path = run_dir / "meta.json"
    path.write_text(json.dumps(meta, indent=2) + "\n")
    return path


def load_run_meta(run_dir: Path) -> dict:
    path = Path(run_dir) / "meta.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No meta.json in {run_dir}. Re-record, or pass --spawn-index manually."
        )
    return json.loads(path.read_text())


def apply_run_meta(client: carla.Client, meta: dict) -> carla.World:
    load_name = meta.get("map_load_name") or carla_map_to_load_name(meta["map"])
    return ensure_map(client, load_name)


def ensure_map(
    client: carla.Client,
    map_name: str,
    *,
    settle_seconds: float | None = None,
) -> carla.World:
    """Load map_name only if the server is not already on that town."""
    import time

    world = client.get_world()
    wanted = carla_map_to_load_name(map_name)
    current = carla_map_to_load_name(world.get_map().name)
    if current == wanted:
        print(f"Already on {current}; skipping map load.")
        return world

    heavy = (
        "Town10" in current
        or "HD" in current
        or "Town10" in wanted
        or "HD" in wanted
    )
    if heavy:
        print(
            f"WARNING: switching {current} → {wanted} is hard on VRAM. "
            "Prefer starting CarlaUE4 already on Town01."
        )
    if settle_seconds is None:
        # HD↔normal swaps need longer for VRAM to settle before cameras.
        settle_seconds = 30.0 if heavy else 8.0

    print(f"Loading map {wanted} (was {current}) ...")
    client.set_timeout(60.0)
    client.load_world(wanted)
    world = client.get_world()
    client.set_timeout(10.0)
    if settle_seconds > 0:
        print(f"Waiting {settle_seconds:.0f}s after map load ...")
        time.sleep(settle_seconds)
    return client.get_world()
