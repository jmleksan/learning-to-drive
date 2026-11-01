"""CARLA weather + time-of-day presets for recording diversity."""
from __future__ import annotations

import random

# Names match carla.WeatherParameters class attributes (0.9.16).
WEATHER_PRESETS: tuple[str, ...] = (
    # Noon
    "ClearNoon",
    "CloudyNoon",
    "WetNoon",
    "WetCloudyNoon",
    "SoftRainNoon",
    "MidRainyNoon",
    "HardRainNoon",
    # Sunset / dusk
    "ClearSunset",
    "CloudySunset",
    "WetSunset",
    "WetCloudySunset",
    "SoftRainSunset",
    "MidRainSunset",
    "HardRainSunset",
    # Night
    "ClearNight",
    "CloudyNight",
    "WetNight",
    "WetCloudyNight",
    "SoftRainNight",
    "MidRainyNight",
    "HardRainNight",
    # Extra
    "DustStorm",
)

# Shorthand → any preset whose name ends with that suffix (or DustStorm alone).
TIME_GROUPS: dict[str, tuple[str, ...]] = {
    "noon": tuple(p for p in WEATHER_PRESETS if p.endswith("Noon")),
    "sunset": tuple(p for p in WEATHER_PRESETS if p.endswith("Sunset")),
    "dusk": tuple(p for p in WEATHER_PRESETS if p.endswith("Sunset")),
    "night": tuple(p for p in WEATHER_PRESETS if p.endswith("Night")),
}

__all__ = [
    "TIME_GROUPS",
    "WEATHER_PRESETS",
    "apply_weather",
    "list_weather_names",
    "resolve_weather_name",
]


def list_weather_names() -> list[str]:
    return list(WEATHER_PRESETS)


def resolve_weather_name(name: str) -> str:
    """Normalize a CLI weather / time string to a preset attribute name."""
    key = name.strip()
    if not key:
        raise ValueError("Empty weather name")

    lower = key.lower().replace("-", "").replace("_", "")

    if lower in ("random", "rand"):
        return random.choice(WEATHER_PRESETS)

    # Time-of-day: pick a random condition at that time.
    if lower in TIME_GROUPS:
        return random.choice(TIME_GROUPS[lower])
    if lower.startswith("random") and lower[6:] in TIME_GROUPS:
        return random.choice(TIME_GROUPS[lower[6:]])

    compact = lower
    for preset in WEATHER_PRESETS:
        if preset.lower() == compact or preset.lower() == key.lower():
            return preset
        # Allow "clear night" / "hard rain noon" style
        if preset.lower() == compact:
            return preset

    raise ValueError(
        f"Unknown weather {name!r}. "
        f"Presets: {', '.join(WEATHER_PRESETS)}. "
        f"Or time groups: noon, sunset, night, random."
    )


def apply_weather(world, name: str):
    """Set world weather from a preset / time group. Returns resolved name."""
    import carla

    resolved = resolve_weather_name(name)
    params = getattr(carla.WeatherParameters, resolved, None)
    if params is None:
        raise ValueError(f"This CARLA build has no WeatherParameters.{resolved}")
    world.set_weather(params)
    return resolved
