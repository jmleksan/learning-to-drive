"""Attach CARLA GNSS and IMU with configurable noise."""

from __future__ import annotations

import queue

import carla

# Default GNSS noise ~ few meters of horizontal jitter (degrees).
# 1e-5 deg lat ≈ 1.1 m; 3e-5 ≈ 3.3 m.
DEFAULT_GNSS_STDDEV_DEG = 3.0e-5
DEFAULT_GNSS_ALT_STDDEV_M = 1.0
DEFAULT_IMU_ACCEL_STDDEV = 0.05
DEFAULT_IMU_GYRO_STDDEV = 0.01


def attach_gnss(
    world: carla.World,
    vehicle: carla.Vehicle,
    *,
    fps: float = 10.0,
    noise_lat_lon_deg: float = DEFAULT_GNSS_STDDEV_DEG,
    noise_alt_m: float = DEFAULT_GNSS_ALT_STDDEV_M,
) -> carla.Sensor:
    """Roof GNSS. noise_* stddevs make raw fixes visibly jitter."""
    bp = world.get_blueprint_library().find("sensor.other.gnss")
    if bp.has_attribute("sensor_tick"):
        bp.set_attribute("sensor_tick", str(1.0 / max(fps, 1e-3)))
    # CARLA 0.9.x attribute names.
    for key, value in (
        ("noise_lat_stddev", str(noise_lat_lon_deg)),
        ("noise_lon_stddev", str(noise_lat_lon_deg)),
        ("noise_alt_stddev", str(noise_alt_m)),
        ("noise_lat_bias", "0.0"),
        ("noise_lon_bias", "0.0"),
        ("noise_alt_bias", "0.0"),
    ):
        if bp.has_attribute(key):
            bp.set_attribute(key, value)
    transform = carla.Transform(carla.Location(x=0.0, y=0.0, z=2.4))
    return world.spawn_actor(bp, transform, attach_to=vehicle)


def attach_imu(
    world: carla.World,
    vehicle: carla.Vehicle,
    *,
    fps: float = 10.0,
    accel_stddev: float = DEFAULT_IMU_ACCEL_STDDEV,
    gyro_stddev: float = DEFAULT_IMU_GYRO_STDDEV,
) -> carla.Sensor:
    """Cabin IMU (accel + gyro + compass)."""
    bp = world.get_blueprint_library().find("sensor.other.imu")
    if bp.has_attribute("sensor_tick"):
        bp.set_attribute("sensor_tick", str(1.0 / max(fps, 1e-3)))
    for key, value in (
        ("noise_accel_stddev_x", str(accel_stddev)),
        ("noise_accel_stddev_y", str(accel_stddev)),
        ("noise_accel_stddev_z", str(accel_stddev)),
        ("noise_gyro_stddev_x", str(gyro_stddev)),
        ("noise_gyro_stddev_y", str(gyro_stddev)),
        ("noise_gyro_stddev_z", str(gyro_stddev)),
    ):
        if bp.has_attribute(key):
            bp.set_attribute(key, value)
    transform = carla.Transform(carla.Location(x=0.0, y=0.0, z=1.0))
    return world.spawn_actor(bp, transform, attach_to=vehicle)


def try_attach_gnss(
    world: carla.World,
    vehicle: carla.Vehicle,
    actors: list,
    gnss_queue: queue.Queue,
    *,
    fps: float = 10.0,
    noise_lat_lon_deg: float = DEFAULT_GNSS_STDDEV_DEG,
    noise_alt_m: float = DEFAULT_GNSS_ALT_STDDEV_M,
) -> bool:
    try:
        sensor = attach_gnss(
            world,
            vehicle,
            fps=fps,
            noise_lat_lon_deg=noise_lat_lon_deg,
            noise_alt_m=noise_alt_m,
        )
    except Exception as exc:
        print(f"GNSS unavailable ({exc})")
        return False
    actors.append(sensor)
    sensor.listen(gnss_queue.put)
    print(
        f"GNSS ON: noise lat/lon std={noise_lat_lon_deg:.1e} deg "
        f"(~{noise_lat_lon_deg * 111_320:.1f} m), alt std={noise_alt_m:.1f} m"
    )
    return True


def try_attach_imu(
    world: carla.World,
    vehicle: carla.Vehicle,
    actors: list,
    imu_queue: queue.Queue,
    *,
    fps: float = 10.0,
    accel_stddev: float = DEFAULT_IMU_ACCEL_STDDEV,
    gyro_stddev: float = DEFAULT_IMU_GYRO_STDDEV,
) -> bool:
    try:
        sensor = attach_imu(
            world,
            vehicle,
            fps=fps,
            accel_stddev=accel_stddev,
            gyro_stddev=gyro_stddev,
        )
    except Exception as exc:
        print(f"IMU unavailable ({exc})")
        return False
    actors.append(sensor)
    sensor.listen(imu_queue.put)
    print(
        f"IMU ON: accel std={accel_stddev:.3f} m/s², "
        f"gyro std={gyro_stddev:.3f} rad/s"
    )
    return True


def pop_gnss(gnss_queue: queue.Queue) -> carla.GnssMeasurement | None:
    """Drain to the newest GNSS sample."""
    latest = None
    while True:
        try:
            latest = gnss_queue.get_nowait()
        except queue.Empty:
            break
    return latest


def pop_imu(imu_queue: queue.Queue) -> carla.IMUMeasurement | None:
    """Drain to the newest IMU sample."""
    latest = None
    while True:
        try:
            latest = imu_queue.get_nowait()
        except queue.Empty:
            break
    return latest
