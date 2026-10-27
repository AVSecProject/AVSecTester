"""RunLog transports one captured decision and owns recording lifecycle checks."""

from pathlib import Path
from collections import OrderedDict, namedtuple

import numpy as np
import pytest

from avsectester.data_types import ActorState, Control, Detection, Observation, RecordedOutput, WorldSnapshot
from avsectester.closed_loop.simulators.base import ClockSnapshot
from avsectester.geometry import CameraCalibration, FThetaCamera, PinholeCamera, WindshieldDistortion
from avsectester.geometry.cameras import BODY_FROM_OPTICAL
from avsectester.rendering import ObjectVisibility
from avsectester.runlog import Recorder, RunLog, StepRecord, read_table


def test_record_and_read(tmp_path):
    pixels = np.full((2, 3, 3), 40, np.uint8)
    observation = Observation(.1, 4, {"front": pixels})
    command = Control(throttle=.5)
    detection = Detection((10, 0, 1), (4, 2, 2), 0, "car", .9)
    components = {
        "none": None, "integer": np.int64(7),
        "path": Path("artifacts/patch.png"), "bytes": b"sensor-bytes",
        "namedtuple": namedtuple("Point", "x y")(1, [2, 3]),
        "ordered": OrderedDict(second=2, first=1),
        "pinhole": PinholeCamera(np.array([[50., 0., 16.], [0., 55., 12.], [0., 0., 1.]]), 32, 24),
    }
    pose = np.eye(4)
    pose[:3, 3] = (4., 2., 1.)
    obj = ActorState(pose, extent=(4., 2., 1.))
    world = WorldSnapshot(.1, 3, objects={"sign": obj})
    visibility = {"sign": {"front": ObjectVisibility("depth", (), 12)}}
    recorder = Recorder(tmp_path / "run", metadata={"seed": 4})
    recorder.record_event("initialization", scenario="scene", clock=ClockSnapshot(.1, 4))
    outputs = [RecordedOutput("ttc", 1.5, 0, "observation", 4, .1),
               RecordedOutput("summary", {"counts": [2, 3], "missing": None},
                              0, "observation", 4, .1)]
    recorder.record_step(StepRecord(
        0, raw_observation=observation, produced_command=command, executed_command=command,
        detections=[detection], components=components, outputs=outputs,
        object_visibility=visibility, physical_post=world,
    ))
    pixels.fill(0)
    command.throttle = 0
    memory = recorder.finish()
    decoded = RunLog.read(memory.path)
    assert decoded.status == "completed" and decoded.metadata == {"seed": 4}
    assert [event.kind for event in decoded.events] == ["initialization", "termination"]
    assert decoded.events[0].data["clock"] == ClockSnapshot(.1, 4)
    step, = decoded.steps
    assert step.components.keys() == components.keys()
    assert step.components["none"] is None
    assert step.components["integer"] == 7
    assert step.components["path"] == Path("artifacts/patch.png")
    assert step.components["bytes"] == b"sensor-bytes"
    assert step.components["namedtuple"] == (1, [2, 3])
    assert list(step.components["ordered"].items()) == [("second", 2), ("first", 1)]
    camera = step.components["pinhole"]
    assert isinstance(camera, PinholeCamera) and (camera.width, camera.height) == (32, 24)
    np.testing.assert_array_equal(camera.K, components["pinhole"].K)
    assert step.object_visibility == visibility
    stored = step.physical_post.objects["sign"]
    np.testing.assert_array_equal(stored.pose, obj.pose)
    assert stored.extent == obj.extent
    np.testing.assert_array_equal(step.raw_observation.sensor_data["front"], 40)
    assert step.produced_command.throttle == step.executed_command.throttle == .5
    assert step.detections == [detection] and step.outputs == outputs


def test_observation_calibration(tmp_path):
    lens = FThetaCamera(16., 12., (0., 20.), 32, 24, (0., .05), max_angle=1.4,
                        windshield=WindshieldDistortion(
                            (0., 1., 0., 0., 0., 0.), (0., 0., 0., 1., 0., 0.),
                            (0., 1., 0., 0., 0., 0.), (0., 0., 0., 1., 0., 0.)))
    observation = Observation(.1, 1, calibration={"front": CameraCalibration(lens, BODY_FROM_OPTICAL)})
    with Recorder(tmp_path / "calibrated") as recorder:
        recorder.record_step(StepRecord(0, model_observation=observation))
    decoded = RunLog.read(tmp_path / "calibrated").steps[0].model_observation
    calibration = decoded.calibration["front"]
    assert isinstance(calibration, CameraCalibration) and isinstance(calibration.camera, FThetaCamera)
    assert calibration.camera.windshield == lens.windshield
    np.testing.assert_array_equal(calibration.rig_from_camera, BODY_FROM_OPTICAL)
    references = RunLog.read(tmp_path / "calibrated", load_payloads=False)
    calibration = references.steps[0].model_observation.calibration["front"]
    assert isinstance(calibration, CameraCalibration) and isinstance(calibration.camera, FThetaCamera)


def test_read_selected_columns(tmp_path):
    with Recorder(tmp_path / "run") as recorder:
        recorder.record_step(StepRecord(0, components={"image": np.zeros((3, 4), np.uint8)},
                                        executed_command=Control(brake=.75)))
    table = read_table(tmp_path / "run", columns=["step", "executed_command"])
    assert table.column_names == ["step", "executed_command"]
    assert table.to_pylist()[0]["executed_command"]["fields"]["brake"] == .75
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


def test_record_step():
    recorder = Recorder()
    record = recorder.record_step(StepRecord(0, components={"reading": 1.}))
    assert record.step == 0 and record.components == {"reading": 1.}
    with pytest.raises(TypeError, match="requires StepRecord"):
        recorder.record_step(object())
    with pytest.raises(ValueError, match="strictly increasing"):
        recorder.record_step(StepRecord(0))
    recorder.finish()
    with pytest.raises(RuntimeError, match="closed"):
        recorder.record_step(StepRecord(1))


def test_flush(tmp_path):
    directory = tmp_path / "flushed"
    recorder = Recorder(directory)
    recorder.record_step(StepRecord(0, components={"reading": 1.}))
    recorder.flush()
    pending = directory / "steps" / "part-99999.parquet.open"
    pending.write_bytes(b"an incomplete chunk")
    visible = RunLog.read(directory)
    assert visible.status == "incomplete"
    assert [step.components["reading"] for step in visible.steps] == [1.]
    recorder.finish()
    assert len(RunLog.read(directory).steps) == 1


def test_record_without_path():
    recorder = Recorder()
    recorder.record_event("initialization", note="in memory")
    recorder.record_step(StepRecord(0, components={"score": 1.}))
    recorder.flush()
    log = recorder.finish()
    assert log.path is None and log.status == "completed"
    assert len(log.steps) == 1 and log.steps[0].components["score"] == 1.
    assert [event.kind for event in log.events] == ["initialization", "termination"]
    with pytest.raises(ValueError, match="persistent output path"):
        Recorder(retain_steps=False)


def test_finish():
    recorder = Recorder()
    with pytest.raises(ValueError, match="Final status"):
        recorder.finish("unknown")
    log = recorder.finish("failed", error="a sensor stopped")
    assert log.status == "failed" and log.events[-1].data["error"] == "a sensor stopped"
    assert recorder.finish() is log
