"""Data-only validation shared by CARLA recipes and simulator configuration."""

import math
from collections.abc import Mapping
from typing import Any

# A curated set of built-in ``carla.WeatherParameters`` presets for world-level robustness sweeps
# (pass one as the scenario ``weather`` key). Validation accepts the full native preset set
# below; this curated sequence gives robustness sweeps a smaller clear-to-adverse spread.
CARLA_WEATHER_PRESETS = (
    "ClearNoon", "CloudyNoon", "WetNoon", "WetCloudyNoon", "MidRainyNoon", "HardRainNoon",
    "ClearSunset", "CloudySunset", "WetSunset", "HardRainSunset",
)


# Native enum names can be validated without importing the CARLA SDK or opening a world.
_MAP_LAYERS = frozenset({"Buildings", "Decals", "Foliage", "Ground", "ParkedVehicles",
                         "Particles", "Props", "StreetLights", "Walls"})
_LIGHT_GROUPS = frozenset({"Street", "Building", "Other", "None"})
_VEHICLE_LIGHTS = frozenset({"NONE", "Position", "LowBeam", "HighBeam", "Brake", "RightBlinker",
                             "LeftBlinker", "Reverse", "Fog", "Interior", "Special1", "Special2", "All"})
_WEATHER_PRESETS = frozenset({
    *CARLA_WEATHER_PRESETS, "SoftRainNoon", "WetCloudySunset", "MidRainSunset", "SoftRainSunset",
    "ClearNight", "CloudyNight", "WetNight", "WetCloudyNight", "SoftRainNight", "MidRainyNight",
    "HardRainNight", "DustStorm",
})
_WEATHER_RANGES: dict[str, tuple[float, float]] = {
    **{name: (0.0, 100.0) for name in (
        "cloudiness", "precipitation", "precipitation_deposits", "wind_intensity", "fog_density",
        "wetness", "dust_storm",
    )},
    "sun_azimuth_angle": (0.0, 360.0), "sun_altitude_angle": (-90.0, 90.0),
    **{name: (0.0, math.inf) for name in (
        "fog_distance", "fog_falloff", "scattering_intensity", "mie_scattering_scale",
        "rayleigh_scattering_scale",
    )},
}


def _settings_mapping(value: Any, allowed: set[str] | frozenset[str], name: str) -> Mapping[str, Any]:
    """Check a native-setting structure before any world operations."""
    if not isinstance(value, Mapping) or set(value) - allowed:
        raise ValueError(f"{name} supports only {', '.join(sorted(allowed))}")
    return value


def _name_list(value: Any, allowed: frozenset[str] | None, name: str) -> None:
    """Validate enum/actor names without needing native enum objects or spawned actors."""
    if (not isinstance(value, (list, tuple))
            or any(not isinstance(item, str) or not item for item in value)
            or (allowed is not None and any(item not in allowed for item in value))):
        raise ValueError(f"{name} requires supported nonempty names")


def _finite_number(value: Any, name: str, low: float, high: float = math.inf) -> None:
    """Reject booleans, coercible strings, NaNs and values outside the native range."""
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not low <= value <= high):
        raise ValueError(f"{name} must be a finite number in [{low}, {high}]")


def _validate_map_layers(settings: Any) -> None:
    if settings is None:
        return
    settings = _settings_mapping(settings, {"load", "unload"}, "map_layers")
    for action in ("unload", "load"):
        _name_list(settings.get(action, []), _MAP_LAYERS, f"map_layers.{action}")


def _validate_lighting(settings: Any) -> None:
    if settings is None:
        return
    settings = _settings_mapping(settings, {"street_lights", "vehicle_lights"}, "lighting")
    scene = settings.get("street_lights")
    if scene is not None:
        scene = _settings_mapping(scene, {"enabled", "intensity", "color", "group"}, "street_lights")
        group = scene.get("group", "Street")
        if not isinstance(group, str) or group not in _LIGHT_GROUPS:
            raise ValueError("street_lights.group must be Street, Building, Other or None")
        if "enabled" in scene and not isinstance(scene["enabled"], bool):
            raise ValueError("street_lights.enabled must be a boolean")
        if "intensity" in scene:
            _finite_number(scene["intensity"], "street_lights.intensity", 0)
        if "color" in scene:
            color = scene["color"]
            if (not isinstance(color, (list, tuple)) or len(color) != 3
                    or any(isinstance(c, bool) or not isinstance(c, int) or not 0 <= c <= 255
                           for c in color)):
                raise ValueError("street_lights.color requires three integers in [0,255]")
    vehicles = settings.get("vehicle_lights", {})
    if not isinstance(vehicles, Mapping) or any(not isinstance(key, str) or not key for key in vehicles):
        raise ValueError("vehicle_lights maps stable actor IDs (or all) to light flag lists")
    for flags in vehicles.values():
        _name_list(flags, _VEHICLE_LIGHTS, "vehicle_lights values")


def _validate_road_friction(settings: Any) -> None:
    if settings is None:
        return
    settings = _settings_mapping(settings, {"tire_friction", "actors"}, "road_friction")
    value = settings.get("tire_friction")
    _finite_number(value, "road_friction.tire_friction", 0)
    if value == 0:
        raise ValueError("road_friction.tire_friction must be positive")
    if "actors" in settings:
        _name_list(settings["actors"], None, "road_friction.actors")


def _validate_weather(settings: Any) -> None:
    if settings is None:
        return
    if isinstance(settings, str):
        if settings not in _WEATHER_PRESETS:
            raise ValueError(f"unknown CARLA weather preset {settings!r} (see CARLA_WEATHER_PRESETS)")
        return
    settings = _settings_mapping(settings, set(_WEATHER_RANGES), "weather")
    for name, value in settings.items():
        _finite_number(value, f"weather.{name}", *_WEATHER_RANGES[name])


_CARLA_WORLD_VALIDATORS = {
    "weather": _validate_weather, "lighting": _validate_lighting,
    "map_layers": _validate_map_layers, "road_friction": _validate_road_friction,
}
CARLA_WORLD_SETTINGS = frozenset(_CARLA_WORLD_VALIDATORS)


def validate_carla_world_settings(settings: Mapping[str, Any]) -> None:
    """Validate data-only CARLA world settings without SDK imports or a live connection."""
    validators = _CARLA_WORLD_VALIDATORS
    if not isinstance(settings, Mapping):
        raise TypeError("CARLA world settings must be a mapping")
    unsupported = set(settings) - validators.keys()
    if unsupported:
        raise ValueError(f"Unsupported CARLA world settings: {sorted(unsupported)}")
    for name, value in settings.items():
        validators[name](value)
