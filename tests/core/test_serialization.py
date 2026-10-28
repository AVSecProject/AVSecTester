"""Serialization contracts use independent registries and detached value snapshots."""

from collections import OrderedDict, defaultdict, namedtuple
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

from avsectester.data_types import Control, Detection, TargetSpec, Unavailable, WorldSnapshot
from avsectester.datasets import FrameReference
from avsectester.closed_loop.simulators.ego_dynamics import PlanarEgoState
from avsectester.geometry import FThetaCamera, PinholeCamera, WindshieldDistortion
from avsectester.rendering import ObjectVisibility
from avsectester.serialization import DEFAULT_REGISTRY, SerializationRegistry


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


Point = namedtuple("Point", "x y")
VALUES = [
    pytest.param(None, None, id="none"),
    pytest.param({"_avsectester_type": "user", "_avsectester_keys": ["x"]},
                 {"_avsectester_type": "user", "_avsectester_keys": ["x"]}, id="reserved-keys"),
    pytest.param(torch.tensor([[1., 2.]], requires_grad=True),
                 np.array([[1., 2.]], np.float32), id="tensor"),
    pytest.param(3.25, 3.25, id="scalar"),
    pytest.param(np.int64(7), 7, id="numpy-scalar"),
    pytest.param(Path("artifacts/patch.png"), Path("artifacts/patch.png"), id="path"),
    pytest.param(b"sensor-bytes", b"sensor-bytes", id="bytes"),
    pytest.param(WorldSnapshot(.1, 3, traffic_lights={"signal-1": "red"}),
                 WorldSnapshot(.1, 3, traffic_lights={"signal-1": "red"}), id="world"),
    pytest.param(ObjectVisibility("actor_cuboids", ("actor",), 12),
                 ObjectVisibility("actor_cuboids", ("actor",), 12), id="visibility"),
    pytest.param({"list": [1, None], "tuple": ("car", .9)},
                 {"list": [1, None], "tuple": ("car", .9)}, id="nested"),
    pytest.param(Point(1, [2, 3]), (1, [2, 3]), id="namedtuple"),
    pytest.param(defaultdict(list, host=[1]), {"host": [1]}, id="defaultdict"),
    pytest.param(OrderedDict(second=2, first=1), {"second": 2, "first": 1}, id="ordered-dict"),
    pytest.param(np.arange(24, dtype=np.float32).reshape(4, 6)[:, ::2],
                 np.array([[0, 2, 4], [6, 8, 10], [12, 14, 16], [18, 20, 22]], np.float32),
                 id="strided-array"),
    pytest.param(np.arange(6, dtype=np.int16).reshape(2, 3).T,
                 np.array([[0, 3], [1, 4], [2, 5]], np.int16), id="transposed-array"),
    pytest.param(np.empty((0, 4), dtype=np.uint8), np.empty((0, 4), np.uint8), id="empty-array"),
    pytest.param(PinholeCamera(np.array([[50., 0., 16.], [0., 55., 12.], [0., 0., 1.]]), 32, 24),
                 PinholeCamera(np.array([[50., 0., 16.], [0., 55., 12.], [0., 0., 1.]]), 32, 24),
                 id="pinhole"),
    pytest.param(FThetaCamera(16., 12., (0., 20.), 32, 24, (0., .05), max_angle=1.4),
                 FThetaCamera(16., 12., (0., 20.), 32, 24, (0., .05), max_angle=1.4), id="ftheta"),
    pytest.param(WindshieldDistortion((0., 1., 0., 0., 0., 0.), (0., 0., 0., 1., 0., 0.),
                                     (0., 1., 0., 0., 0., 0.), (0., 0., 0., 1., 0., 0.)),
                 WindshieldDistortion((0., 1., 0., 0., 0., 0.), (0., 0., 0., 1., 0., 0.),
                                     (0., 1., 0., 0., 0., 0.), (0., 0., 0., 1., 0., 0.)),
                 id="windshield"),
]


def assert_value(actual, expected):
    if isinstance(expected, np.ndarray):
        assert actual.shape == expected.shape and actual.dtype == expected.dtype
        np.testing.assert_array_equal(actual, expected)
    elif is_dataclass(expected):
        assert type(actual) is type(expected)
        for item in fields(expected):
            assert_value(getattr(actual, item.name), getattr(expected, item.name))
    elif isinstance(expected, dict):
        assert list(actual) == list(expected)
        for key, value in expected.items():
            assert_value(actual[key], value)
    elif isinstance(expected, (tuple, list)):
        assert isinstance(actual, type(expected)) and len(actual) == len(expected)
        for value, other in zip(actual, expected):
            assert_value(value, other)
    else:
        assert actual == expected


@pytest.mark.parametrize("value,expected", VALUES)
def test_encode_and_decode(value, expected):
    registry = SerializationRegistry()
    for record in (WorldSnapshot, ObjectVisibility, PinholeCamera, FThetaCamera, WindshieldDistortion):
        registry.register_dataclass(record)
    assert_value(registry.decode(registry.encode(value)), expected)


def test_default_registry():
    records = (Control(throttle=.2), Unavailable("absent"), TargetSpec((0., 0., 2., 2.), label="car"),
               Detection((1., 0., 0.), (4., 2., 1.), 0., "car", .9),
               FrameReference("scene", 1, "frame-1", .1), PlanarEgoState())
    for record in records:
        assert_value(DEFAULT_REGISTRY.decode(DEFAULT_REGISTRY.encode(record)), record)


def test_register_dataclass():
    registry = SerializationRegistry()
    for record in (Control, Unavailable, TargetSpec, Detection, FrameReference, Measurement, Marker):
        registry.register_dataclass(record)
    for value in (Control(throttle=.4), Unavailable("no map"), TargetSpec((1., 2., 3., 4.)),
                  Detection((8., 0., 1.), (4., 2., 2.), 0., "car", .9),
                  FrameReference("scene", 0, "first", 0.), Marker(),
                  Measurement(np.arange(12, dtype=np.float32).reshape(3, 4)[:, ::2])):
        assert_value(registry.decode(registry.encode(value)), value)
    with pytest.raises(TypeError, match="register_dataclass"):
        registry.encode(UnregisteredMeasurement(.5))
    with pytest.raises(ValueError, match="Duplicate"):
        registry.register_dataclass(Measurement)


def test_register_dataclass_path(tmp_path, monkeypatch):
    registry = SerializationRegistry()
    registry.register_dataclass("user_records:Sample")
    (tmp_path / "user_records.py").write_text(
        "from dataclasses import dataclass\n@dataclass\nclass Sample:\n    value: float\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "user_records", raising=False)
    from user_records import Sample
    module = sys.modules.pop("user_records")
    monkeypatch.setitem(sys.modules, "user_records", module)

    source = Sample(.75)
    restored = registry.decode(registry.encode(source))
    assert type(restored) is Sample and restored.value == .75
    with pytest.raises(TypeError, match="register_dataclass"):
        SerializationRegistry().decode(registry.encode(source))


def test_register_native():
    registry = SerializationRegistry()
    registry.register_native(__name__, "SensorResult", lambda result: {"values": result.values})
    registry.register_native(__name__, "ClassifiedSensorResult",
                             lambda result: {"category": "vehicle", "values": result.values})
    decoded = registry.decode(registry.encode({
        "base": SensorResult([1, 2]), "specific": ClassifiedSensorResult([3]),
    }))
    np.testing.assert_array_equal(decoded["base"]["values"], [1, 2])
    assert decoded["specific"]["category"] == "vehicle"
    np.testing.assert_array_equal(decoded["specific"]["values"], [3])
    with pytest.raises(ValueError, match="Duplicate"):
        registry.register_native(__name__, "SensorResult", lambda result: result.values)
    with pytest.raises(TypeError, match="register_native"):
        registry.encode(object())


def test_snapshot():
    registry = SerializationRegistry()
    source = SensorResult([1, 2])
    ordinary = registry.snapshot({"sensor": source, "same": source})
    source.values[0] = 9
    assert ordinary["sensor"] is ordinary["same"] and ordinary["sensor"] is not source
    np.testing.assert_array_equal(ordinary["sensor"].values, [1, 2])
    registry.register_native(
        __name__, "SensorResult", lambda value: value.values,
        snapshot=lambda value, snapshot: SensorResult(snapshot(value.values) + 1),
    )
    snapshot = registry.snapshot(source)
    np.testing.assert_array_equal(snapshot.values, [10, 3])
    snapshot.values[1] = 99
    np.testing.assert_array_equal(source.values, [9, 2])
    invalid = SerializationRegistry()
    invalid.register_native(__name__, "SensorResult", lambda value: value.values,
                            snapshot=lambda value, snapshot: value)
    with pytest.raises(TypeError, match="independent snapshot"):
        invalid.snapshot(source)


def test_decode():
    registry = SerializationRegistry()
    encoded = registry.encode(np.arange(3, dtype=np.float32))
    np.testing.assert_array_equal(registry.decode(encoded), [0., 1., 2.])
    unsafe = {**encoded, "dtype": "O"}
    with pytest.raises(ValueError, match="Unsafe array dtype"):
        registry.decode(unsafe)
