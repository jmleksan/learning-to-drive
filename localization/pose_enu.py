"""CARLA world pose → ENU meters / ENU yaw using the map geo reference."""

from __future__ import annotations

import math

import carla

from localization.geo import GeoOrigin, geo_to_enu


def geolocation_of(world_map: carla.Map, location: carla.Location):
    return world_map.transform_to_geolocation(location)


def origin_from_location(world_map: carla.Map, location: carla.Location) -> GeoOrigin:
    geo = geolocation_of(world_map, location)
    return GeoOrigin(
        lat_deg=float(geo.latitude),
        lon_deg=float(geo.longitude),
        alt_m=float(geo.altitude),
    )


def location_to_enu(
    world_map: carla.Map,
    location: carla.Location,
    origin: GeoOrigin,
) -> tuple[float, float, float]:
    geo = geolocation_of(world_map, location)
    return geo_to_enu(
        float(geo.latitude),
        float(geo.longitude),
        float(geo.altitude),
        origin,
    )


def yaw_enu_rad(
    world_map: carla.Map,
    transform: carla.Transform,
    origin: GeoOrigin,
) -> float:
    """Heading in ENU: 0 = north, + toward east (same convention as EgoEKF)."""
    loc = transform.location
    fwd = transform.get_forward_vector()
    p1 = carla.Location(
        x=loc.x + fwd.x,
        y=loc.y + fwd.y,
        z=loc.z + fwd.z,
    )
    e0, n0, _ = location_to_enu(world_map, loc, origin)
    e1, n1, _ = location_to_enu(world_map, p1, origin)
    return math.atan2(e1 - e0, n1 - n0)
