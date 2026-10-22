"""Recorder and RunLog contracts, including numerical payload round trips."""

from collections import OrderedDict, defaultdict, namedtuple
from dataclasses import fields, is_dataclass
from pathlib import Path

import numpy as np
import pytest

from avsectester.closed_loop.datatypes import Control, Detection, Observation, WorldSnapshot
from avsectester.geometry import FThetaCamera, PinholeCamera
from avsectester.rendering import ObjectVisibility
from avsectester.runlog import Recorder, RunLog, StepRecord, read_table

Point = namedtuple("Point", "x y")
VALUES = [
    pytest.param(None, None, id="none"),
    pytest.param(3.25, 3.25, id="scalar"),
    pytest.param(np.int64(7), 7, id="numpy-scalar"),
    pytest.param(Path("artifacts/patch.png"), Path("artifacts/patch.png"), id="path"),
    pytest.param(b"sensor-bytes", b"sensor-bytes", id="bytes"),
    pytest.param(WorldSnapshot(.1, 3, traffic_lights={"signal-1": "red"}),
                 WorldSnapshot(.1, 3, traffic_lights={"signal-1": "red"}), id="traffic-lights"),
    pytest.param(ObjectVisibility("actor_cuboids", ("unmeasured-actor",), 12),
                 ObjectVisibility("actor_cuboids", ("unmeasured-actor",), 12), id="object-visibility"),
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
                 id="pinhole-camera"),
    pytest.param(FThetaCamera(16., 12., (0., 20.), 32, 24, (0., .05), max_angle=1.4),
                 FThetaCamera(16., 12., (0., 20.), 32, 24, (0., .05), max_angle=1.4),
                 id="fisheye-camera"),
]


def assert_value(actual, expected):
    if isinstance(expected, np.ndarray):
        assert actual.shape == expected.shape and actual.dtype == expected.dtype
        np.testing.assert_array_equal(actual, expected)
    elif is_dataclass(expected):
        assert type(actual) is type(expected)
        for field in fields(expected):
            assert_value(getattr(actual, field.name), getattr(expected, field.name))
    elif isinstance(expected, dict):
        assert list(actual) == list(expected)
        for key, item in expected.items():
            assert_value(actual[key], item)
    elif isinstance(expected, (tuple, list)):
        assert isinstance(actual, type(expected)) and len(actual) == len(expected)
        for item, other in zip(actual, expected):
            assert_value(item, other)
    else:
        assert actual == expected


@pytest.mark.parametrize("value,expected", VALUES)
def test_record_and_read(tmp_path, value, expected):
    pixels = np.full((2, 3, 3), 40, np.uint8)
    observation = Observation(.1, 4, {"front": pixels})
    command = Control(throttle=.5)
    detection = Detection((10, 0, 1), (4, 2, 2), 0, "car", .9)
    recorder = Recorder(tmp_path / "run", metadata={"seed": 4})
    recorder.record_event("initialization", scenario="scene")
    recorder.record_step(StepRecord(
        0, raw_observation=observation, produced_command=command, executed_command=command,
        detections=[detection], components={"result": value},
    ))
    pixels.fill(0)
    command.throttle = 0
    memory = recorder.finish()
    decoded = RunLog.read(memory.path)
    assert decoded.status == "completed" and decoded.metadata == {"seed": 4}
    assert [event.kind for event in decoded.events] == ["initialization", "termination"]
    step = decoded.steps[0]
    assert_value(step.components["result"], expected)
    np.testing.assert_array_equal(step.raw_observation.sensor_data["front"], 40)
    assert step.produced_command.throttle == step.executed_command.throttle == .5
    assert step.detections == [detection]
    with pytest.raises(RuntimeError, match="closed"):
        recorder.record_step(StepRecord(1))


def test_read_selected_columns(tmp_path):
    with Recorder(tmp_path / "run") as recorder:
        recorder.record_step(StepRecord(0, components={"image": np.zeros((3, 4), np.uint8)},
                                        executed_command=Control(brake=.75)))
    table = read_table(tmp_path / "run", columns=["step", "executed_command"])
    assert table.column_names == ["step", "executed_command"]
    assert table.to_pylist()[0]["executed_command"]["brake"] == .75
    with pytest.raises(ValueError, match="Unknown RunLog columns"):
        read_table(tmp_path / "run", columns=["missing"])


def test_failed_run_status(tmp_path):
    with pytest.raises(ValueError, match="sensor disconnected"):
        with Recorder(tmp_path / "failed") as recorder:
            recorder.record_step(StepRecord(0))
            raise ValueError("sensor disconnected")
    failed = RunLog.read(tmp_path / "failed")
    assert failed.status == "failed" and [step.step for step in failed.steps] == [0]
    assert failed.events[-1].data["error"] == "sensor disconnected"
    with pytest.raises(KeyboardInterrupt):
        with Recorder(tmp_path / "cancelled"):
            raise KeyboardInterrupt
    assert RunLog.read(tmp_path / "cancelled").status == "cancelled"


def test_chunks_and_schema_evolution(tmp_path):
    recorder = Recorder(tmp_path / "run", chunk_steps=1, retain_steps=False)
    recorder.record_step(StepRecord(0, components={"score": 1}))
    recorder.record_step(StepRecord(1, components={"score": 2., "trajectory": [(1., 2.)]}))
    recorder.abort()
    recovered = RunLog.read(tmp_path / "run")
    assert recovered.status == "incomplete"
    assert [step.step for step in recovered.steps] == [0, 1]
    assert recovered.steps[0].components["score"] == 1
    assert recovered.steps[1].components["trajectory"] == [(1., 2.)]
    with pytest.raises(ValueError, match="positive"):
        Recorder(chunk_steps=0)
