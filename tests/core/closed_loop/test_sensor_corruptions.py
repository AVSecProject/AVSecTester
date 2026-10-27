"""Sensor corruptions share shape, copying and deterministic-seed behaviour."""
import importlib.util

import numpy as np
import pytest

from avsectester.closed_loop.environment import SensorEnvironment
from avsectester.data_types import Observation


CORRUPTION_CASES = [
    ("GaussianNoise", {}), ("Fog", {}), ("Rain", {}), ("Snow", {}),
    ("Brightness", {}), ("LowLight", {}), ("Contrast", {}), ("Gamma", {}),
    ("ColorTemp", {}), ("ShotNoise", {}), ("MotionBlur", {}), ("DefocusBlur", {}),
    ("JPEGCompression", {}), ("ChromaticAberration", {}),
]


@pytest.mark.parametrize("kind,options", [
    *[pytest.param(kind, options, id=kind) for kind, options in CORRUPTION_CASES],
    pytest.param("AlbumentationsCorruption", {"transform": "Fog"}, marks=pytest.mark.skipif(importlib.util.find_spec("albumentations") is None,
                                             reason="albumentations is not installed"),
                 id="AlbumentationsCorruption"),
    pytest.param("avsectester.closed_loop.sensor_corruptions:Brightness", {},
                 id="custom-corruption"),
])
def test_sensor_corruption_contract(kind, options):
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
