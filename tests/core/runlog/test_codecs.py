"""Public codec registration supports late additions and specific subclasses."""

import numpy as np
import pytest

from avsectester.runlog import CodecRegistry, Recorder, RunLog, StepRecord


class SensorResult:
    def __init__(self, values):
        self.values = np.asarray(values)


class ClassifiedSensorResult(SensorResult):
    pass


def test_codec_registry(tmp_path):
    registry = CodecRegistry()
    recorder = Recorder(codecs=registry)
    assert recorder.capture([1, "initial capture"]) == [1, "initial capture"]
    registry.register(SensorResult, lambda result: {"values": result.values}, name="sensor.v1")
    source = SensorResult([1, 2])
    snapshot = recorder.capture(source)
    source.values[0] = 9
    np.testing.assert_array_equal(snapshot.values, [1, 2])
    registry.register(ClassifiedSensorResult, lambda result: {"class": "vehicle", "values": result.values},
                      name="classified-sensor.v1")
    with Recorder(tmp_path / "run", codecs=registry) as writer:
        writer.record_step(StepRecord(0, components={"base": source,
                                                     "specific": ClassifiedSensorResult([3])}))
    decoded = RunLog.read(tmp_path / "run")
    assert decoded.steps[0].components["base"]["values"].tolist() == [9, 2]
    assert decoded.steps[0].components["specific"]["class"] == "vehicle"
    assert decoded.steps[0].components["specific"]["values"].tolist() == [3]
    with pytest.raises(ValueError, match="Duplicate"):
        registry.register(SensorResult, lambda result: result.values)
