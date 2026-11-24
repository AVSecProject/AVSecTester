"""World/sensor environment merging, normalization and validation through the public API."""

import pytest
import numpy as np

from avsectester.closed_loop.environment import (
    SensorEnvironment,
    merge_environments, normalize_environment, validate_environment,
)
from avsectester.data_types import Observation
from tests.fakes import MemorySimulator


def test_merge_environments():
    base = {"world": {"weather": "ClearNoon"}, "sensors": [
        {"type": "GaussianNoise", "options": {"severity": .2}, "camera": "front"},
    ]}
    merged = merge_environments(base, {"world": {"weather": "WetNoon"}})
    assert merged["world"] == {"weather": "WetNoon"}
    assert merged["sensors"] == base["sensors"]
    assert merge_environments(base, {"sensors": []})["sensors"] == []


def test_normalize_environment():
    assert normalize_environment(None) == {"world": {}, "sensors": []}
    assert normalize_environment({"world": {"weather": "ClearNoon"}})["world"] == {
        "weather": "ClearNoon"}
    with pytest.raises(ValueError, match="Unknown environment"):
        normalize_environment({"mistyped": True})


def test_validate_environment():
    environment = {"world": {"weather": "ClearNoon"}}
    assert validate_environment(environment, MemorySimulator()) == normalize_environment(environment)
    with pytest.raises(ValueError, match="does not support"):
        validate_environment({"world": {"unknown": 1}}, MemorySimulator())


def test_sensor_environment():
    environment = SensorEnvironment({"sensors": [{"type": "Brightness", "camera": "front"}]})
    with pytest.raises(ValueError, match="unavailable camera"):
        environment.apply(Observation(0., 0), position=0)
    pixels = np.full((4, 5, 3), 100, np.uint8)
    source = Observation(0., 0, {"front": pixels})
    environment.bind(source)
    batch = Observation(.033333, 1, {"rear": pixels.copy()})
    skipped = environment.apply(batch, position=0, available_sensors=frozenset({"front", "rear"}))
    assert tuple(skipped.sensor_data) == ("rear",)
    np.testing.assert_array_equal(skipped.sensor_data["rear"], pixels)
    with pytest.raises(ValueError, match="unavailable camera"):
        environment.apply(batch, position=0, available_sensors=frozenset({"rear"}))
    seed = 2**40 + 7
    noisy = SensorEnvironment({"sensors": [{"type": "GaussianNoise", "camera": "front",
                                             "options": {"severity": .2}}]}, seed)
    first = noisy.apply(source, 3).sensor_data["front"]
    intermediate = noisy.apply(source, 3, interval_capture_index={"front": 1, "rear": 9})
    again = noisy.apply(source, 3, interval_capture_index={"rear": 9, "front": 1})
    assert not np.array_equal(first, intermediate.sensor_data["front"])
    np.testing.assert_array_equal(intermediate.sensor_data["front"], again.sensor_data["front"])
