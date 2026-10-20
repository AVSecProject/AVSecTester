"""Actual AVStack payloads retain useful native snapshots and readable persistent values."""
import numpy as np
import pytest

pytest.importorskip("avstack.config")

from avsectester.closed_loop.datatypes import Observation
from avsectester.runlog import Recorder, RunLog, StepRecord
from avsectester.runlog.codecs import default_codecs


def test_native_codec(native_sensor_case, tmp_path, make_ego, make_detections):
    from avstack.geometry import GlobalOrigin3D, ReferenceFrame

    make_sensor, payload_array, _ = native_sensor_case
    body = ReferenceFrame(np.array([10., 0., 0.]), np.quaternion(1), GlobalOrigin3D)
    mount = ReferenceFrame(np.array([1., 0., 1.6]), np.quaternion(1), body)
    sensor = make_sensor(mount)
    detections = make_detections([(5., 0., 0.)], reference=mount, frame=10)
    original = payload_array(sensor.data).copy()
    observation = Observation(.5, 10, {"sensor": sensor}, vehicle_state=make_ego(xyz=(10., 0., 0.)))
    with Recorder(tmp_path, codecs=default_codecs()) as recorder:
        captured = recorder.record_step(StepRecord(0, raw_observation=observation,
                                                   components={"perception": detections}))
        detections[0].score = .1
        payload_array(sensor.data)[:] = 0
    assert captured.components["perception"][0].score == .8
    saved = RunLog.read(tmp_path).steps[0]
    assert saved.components["perception"]["data"][0]["score"] == .8
    restored = saved.raw_observation.sensor_data["sensor"]["data"]
    np.testing.assert_array_equal(payload_array(restored), original)
    assert saved.raw_observation.vehicle_state["position"]["x"] == [10., 0., 0.]
