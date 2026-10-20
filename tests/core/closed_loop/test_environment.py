"""Environment validation and repeatable image transforms through the public API."""

import numpy as np
import pytest

from avsectester.closed_loop.datatypes import Observation
from avsectester.closed_loop.environment import (
    SensorEnvironment, merge_environments, normalize_environment, validate_environment,
)
from tests.fakes import MemorySimulator


def test_validate_environment():
    base = {"world": {"weather": "ClearNoon"}, "sensors": [
        {"type": "GaussianNoise", "options": {"severity": .2}, "camera": "front"},
    ]}
    override = {"world": {"weather": "WetNoon"}}
    merged = merge_environments(base, override)
    assert merged["world"] == {"weather": "WetNoon"}
    assert merged["sensors"] == base["sensors"]
    assert validate_environment(merged, MemorySimulator()) == normalize_environment(merged)
    assert merge_environments(base, {"sensors": []})["sensors"] == []
    with pytest.raises(ValueError, match="Unknown environment"):
        normalize_environment({"mistyped": True})
    with pytest.raises(ValueError, match="does not support"):
        validate_environment({"world": {"unknown": 1}}, MemorySimulator())


TRANSFORMS = [
    ("GaussianNoise", {}), ("Fog", {}), ("Rain", {}), ("Snow", {}),
    ("Brightness", {}), ("LowLight", {}), ("Contrast", {}), ("Gamma", {}),
    ("ColorTemp", {}), ("ShotNoise", {}), ("MotionBlur", {}), ("DefocusBlur", {}),
    ("JPEGCompression", {}), ("ChromaticAberration", {}),
]


@pytest.mark.parametrize("kind,options,dependencies", [
    *[pytest.param(kind, options, (), id=kind) for kind, options in TRANSFORMS],
    pytest.param("AlbumentationsCorruption", {"transform": "Fog"}, ("albumentations",),
                 id="AlbumentationsCorruption"),
])
def test_sensor_transforms(kind, options, dependencies):
    for dependency in dependencies:
        pytest.importorskip(dependency)
    image = np.random.default_rng(4).integers(25, 200, (40, 48, 3), dtype=np.uint8)
    observation = Observation(.3, 3, {"front": image})
    environment = {"sensors": [{"type": kind, "camera": "front",
                                "options": {"severity": .6, **options}}]}
    first = SensorEnvironment(environment, seed=7).apply(observation, position=3)
    again = SensorEnvironment(environment, seed=7).apply(observation, position=3)
    assert first.sensor_data["front"].shape == image.shape
    assert first.sensor_data["front"].dtype == np.uint8
    np.testing.assert_array_equal(first.sensor_data["front"], again.sensor_data["front"])
    assert np.any(first.sensor_data["front"] != image)
    np.testing.assert_array_equal(observation.sensor_data["front"], image)
    with pytest.raises(ValueError, match="unavailable camera"):
        SensorEnvironment(environment).apply(Observation(0., 0), position=0)
