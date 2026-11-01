"""Latitude/longitude ↔ local east-north-up meters."""

from __future__ import annotations

import math
from dataclasses import dataclass


# Mean meters per degree (good enough for a town-sized run).
_METERS_PER_DEG_LAT = 111_320.0


@dataclass(frozen=True)
class GeoOrigin:
    """ENU origin: lat/lon in degrees, alt in meters."""

    lat_deg: float
    lon_deg: float
    alt_m: float = 0.0


def geo_to_enu(
    lat_deg: float,
    lon_deg: float,
    alt_m: float,
    origin: GeoOrigin,
) -> tuple[float, float, float]:
    """Geodetic → local east, north, up (meters)."""
    lat0 = math.radians(origin.lat_deg)
    dlat = lat_deg - origin.lat_deg
    dlon = lon_deg - origin.lon_deg
    east = dlon * _METERS_PER_DEG_LAT * math.cos(lat0)
    north = dlat * _METERS_PER_DEG_LAT
    up = alt_m - origin.alt_m
    return east, north, up


def enu_to_geo(
    east_m: float,
    north_m: float,
    up_m: float,
    origin: GeoOrigin,
) -> tuple[float, float, float]:
    """Local ENU meters → geodetic degrees / meters."""
    lat0 = math.radians(origin.lat_deg)
    lat = origin.lat_deg + north_m / _METERS_PER_DEG_LAT
    lon = origin.lon_deg + east_m / (
        _METERS_PER_DEG_LAT * max(math.cos(lat0), 1e-6)
    )
    alt = origin.alt_m + up_m
    return lat, lon, alt


def enu_to_xy(
    east_m: float,
    north_m: float,
    *,
    yaw0_rad: float = 0.0,
) -> tuple[float, float]:
    """Rotate ENU into a local x-forward / y-right frame aligned with yaw0.

    yaw0 is the ground-truth heading at the origin (radians, CARLA convention:
    0 = +x world). Used so GNSS meters line up with GT x/y when both share
    the same origin pose.
    """
    c = math.cos(yaw0_rad)
    s = math.sin(yaw0_rad)
    # ENU east/north → rotate so +x matches vehicle-forward at origin.
    # CARLA yaw 0 points +x; map north is often not +x, so we align using GT.
    x = c * north_m + s * east_m
    y = -s * north_m + c * east_m
    return x, y


def xy_to_enu(
    x_m: float,
    y_m: float,
    *,
    yaw0_rad: float = 0.0,
) -> tuple[float, float]:
    """Inverse of enu_to_xy."""
    c = math.cos(yaw0_rad)
    s = math.sin(yaw0_rad)
    north = c * x_m - s * y_m
    east = s * x_m + c * y_m
    return east, north
