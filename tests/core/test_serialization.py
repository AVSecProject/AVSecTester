"""Records reconstruct by registration; native data remains ordinary values."""

from dataclasses import dataclass

import numpy as np
import pytest

from avsectester.datasets.base import FrameReference
from avsectester.serialization import DEFAULT_REGISTRY, SerializationRegistry
from avsectester.data_types import Control, Detection, TargetSpec, Unavailable
from avsectester.runlog import Recorder, RunLog, StepRecord


@dataclass
class Measurement:
    values: np.ndarray
    record: str = "measurement"


@dataclass
class Marker:
    pass


@dataclass
class UnregisteredMeasurement:
    value: float


class SensorResult:
    def __init__(self, values):
        self.values = np.asarray(values)


class ClassifiedSensorResult(SensorResult):
    pass


def test_register_dataclass(tmp_path):
    from avsectester.closed_loop.simulators.ego_dynamics import PlanarEgoState

    for record in (Control(throttle=.4), Unavailable("no map"), TargetSpec((1., 2., 3., 4.)),
                   Detection((8., 0., 1.), (4., 2., 2.), 0., "car", .9),
                   FrameReference("scene", 0, "first", 0.), PlanarEgoState(x=2., speed=3., lateral_speed=.4,
                       yaw_rate=.2, acceleration=-1.5)):
        encoded = DEFAULT_REGISTRY.encode(record)
        assert encoded["_avsectester_type"] == "dataclass"
        assert encoded["type"] == f"{type(record).__module__}:{type(record).__qualname__}"
        assert DEFAULT_REGISTRY.decode(encoded) == record
    user_data = {"_avsectester_type": "user-value", "_avsectester_keys": ["item"],
                 "record": "user-record"}
    assert DEFAULT_REGISTRY.decode(DEFAULT_REGISTRY.encode(user_data)) == user_data
    DEFAULT_REGISTRY.register_dataclass(Measurement)
    DEFAULT_REGISTRY.register_dataclass(Marker)
    assert DEFAULT_REGISTRY.encode(Marker())["fields"] == {}
    pixels = np.arange(12, dtype=np.float32).reshape(3, 4)[:, ::2]
    with Recorder(tmp_path / "run") as recorder:
        recorder.record_step(StepRecord(
            0, components={"user": Measurement(pixels), "marker": Marker()},
        ))
    components = RunLog.read(tmp_path / "run").steps[0].components
    assert isinstance(components["marker"], Marker)
    restored = components["user"]
    assert isinstance(restored, Measurement) and restored.record == "measurement"
    np.testing.assert_array_equal(restored.values, pixels)
    with pytest.raises(TypeError, match="register_dataclass"):
        DEFAULT_REGISTRY.encode(UnregisteredMeasurement(.5))


def test_register_dataclass_path(tmp_path, monkeypatch):
    registry = SerializationRegistry()
    registry.register_dataclass("user_records:Sample")
    (tmp_path / "user_records.py").write_text(
        "from dataclasses import dataclass\n@dataclass\nclass Sample:\n    value: float\n    record: str\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    data = {"_avsectester_type": "dataclass", "type": "user_records:Sample",
            "fields": {"value": .75, "record": "sample"}}
    decoded = registry.decode(data)
    assert decoded.value == .75 and decoded.record == "sample"
    assert registry.encode(decoded) == data
    with pytest.raises(TypeError, match="register_dataclass"):
        registry.decode({"_avsectester_type": "dataclass", "type": "unregistered_records:Sample",
                         "fields": {"value": .5}})


def test_register_native():
    registry = SerializationRegistry()
    assert registry.snapshot([1, "initial capture"]) == [1, "initial capture"]
    registry.register_native(__name__, "SensorResult", lambda result: {"values": result.values})
    source = SensorResult([1, 2])
    snapshot = registry.snapshot(source)
    source.values[0] = 9
    assert isinstance(snapshot, SensorResult)
    np.testing.assert_array_equal(snapshot.values, [1, 2])
    registry.register_native(__name__, "ClassifiedSensorResult",
        lambda result: {"category": "vehicle", "values": result.values})
    decoded = registry.decode(registry.encode({
        "base": source, "specific": ClassifiedSensorResult([3]),
    }))
    np.testing.assert_array_equal(decoded["base"]["values"], [9, 2])
    assert decoded["specific"]["category"] == "vehicle"
    np.testing.assert_array_equal(decoded["specific"]["values"], [3])
    with pytest.raises(ValueError, match="Duplicate"):
        registry.register_native(__name__, "SensorResult", lambda result: result.values)
    with pytest.raises(TypeError, match="register_native"):
        registry.encode(object())
