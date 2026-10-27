"""World/sensor environment merging, normalization and validation through the public API."""

import pytest

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
