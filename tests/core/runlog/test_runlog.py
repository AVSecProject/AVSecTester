"""RunLog transports one captured decision and owns recording lifecycle checks."""

import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from avsectester.data_types import (
    ActorState, Control, Detection, DeliveredAcquisition, Observation, PointCloud,
    RecordedOutput, RosMessage, TimedStateEstimate, WorldSnapshot,
)
from avsectester.closed_loop.simulators.base import ClockSnapshot
from avsectester.geometry import (
    CameraCalibration, FThetaCamera, WindshieldDistortion,
)
from avsectester.geometry.cameras import BODY_FROM_OPTICAL
from avsectester.geometry.lidars import LidarCalibration
from avsectester.rendering import ObjectVisibility
from avsectester.runlog import CaptureRecord, MessageRecord, Recorder, RunLog, StepRecord, read_table
from avsectester.serialization import DEFAULT_REGISTRY


def test_record_capture(tmp_path):
    pixels = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    recorder = Recorder(tmp_path / "shared", payload_shard_bytes=10)
    payload = recorder.snapshot(pixels)
    raw = Observation(0., 0, {"front": payload}, calibration={"front": {"gain": 2}})
    record = CaptureRecord(0, 0, ("front",), raw, raw)
    captured = recorder.record_capture(record)
    with pytest.raises(ValueError, match="Duplicate capture_id"):
        recorder.record_capture(record)
    step = recorder.record_step(StepRecord(0, raw_observation=raw, model_observation=raw))
    raw.calibration["front"]["gain"] = 9
    raw.ego_speed = 12.
    record.frame_ranges_us = {"front": (1, 2)}
    assert captured.raw_observation.sensor_data["front"] is payload
    np.testing.assert_array_equal(step.model_observation.sensor_data["front"], payload)
    assert captured.raw_observation.calibration == {"front": {"gain": 2}}
    assert step.raw_observation.calibration == {"front": {"gain": 2}}
    assert captured.raw_observation.ego_speed == step.raw_observation.ego_speed == 0.
    assert captured.frame_ranges_us == {}
    recorder.finish()
    restored = RunLog.read(tmp_path / "shared")
    np.testing.assert_array_equal(
        restored.captures[0].model_observation.sensor_data["front"], payload)


def test_record_message(tmp_path):
    dtype = np.dtype({"names": ["x", "returns"], "formats": [">f4", ("<i2", (2,))],
                      "offsets": [0, 8], "itemsize": 16})
    slots = bytearray(range(96))
    data = np.frombuffer(slots, dtype=dtype)[::2]
    cloud_bytes = bytes(slots[:16] + slots[32:48] + slots[64:80])
    cloud = RosMessage("sensor_msgs/msg/PointCloud2", {"data": data, "header": {"frame_id": "lidar"}})
    grid = RosMessage("nav_msgs/msg/OccupancyGrid", {"data": np.array([1, -1], dtype=np.int8)})
    output = RosMessage(grid.type, {"data": np.array([0, -1], dtype=np.int8)})
    imu = RosMessage("sensor_msgs/msg/Imu", {"header": {"frame_id": "imu"}})
    records = (
        MessageRecord("/cloud", cloud.type, 0, 5, 10, 100, 120, 130, .001, 200, False, cloud),
        MessageRecord("/grid", grid.type, 0, 6, 10, 200, 220, 230, .002, 300, True, grid, output),
        MessageRecord("/metadata", imu.type, 1, None, 11, 300, 320, 330, .003, 400, True),
        MessageRecord(
            "/imu", imu.type, 1, None, 11, 400, 420, 430, .004, None, False, imu,
            modifiable=False,
        ),
    )
    baseline = tmp_path / "without_interception"
    with Recorder(baseline) as recorder:
        with pytest.raises(ValueError, match="record_messages=True"):
            recorder.record_message(records[0])
    assert RunLog.read(baseline).messages == []

    path = tmp_path / "with_interception"
    with Recorder(path, record_messages=True, retain_sensor_data=False, chunk_steps=2) as recorder:
        captured = [recorder.record_message(record) for record in records]
        data["x"] = 0
        assert captured[0].original.fields["data"].tobytes() == cloud_bytes
        assert all(item.original is None and item.forwarded is None for item in recorder.log.messages)

    assert RunLog.read(path, load_records=False).messages == []
    metadata = RunLog.read(path, load_sensor_data=False).messages
    assert [(item.topic, item.modifiable, item.modified, item.relay_sent_ns) for item in metadata] == [
        ("/cloud", True, False, 200), ("/grid", True, True, 300),
        ("/metadata", True, True, 400), ("/imu", False, False, None),
    ]
    assert all(item.original is None and item.forwarded is None for item in metadata)
    original, changed, omitted, observed = RunLog.read(path).messages
    assert original.topic == "/cloud" and original.forwarded is None
    assert original.original.type == cloud.type
    assert original.original.fields["header"] == cloud.fields["header"]
    restored = original.original.fields["data"]
    assert restored.dtype == dtype and restored.tobytes() == cloud_bytes and restored.flags.writeable
    np.testing.assert_array_equal(changed.original.fields["data"], [1, -1])
    np.testing.assert_array_equal(changed.forwarded.fields["data"], [0, -1])
    assert omitted == records[2]
    assert observed == records[3]
    assert read_table(path, ["topic", "step"], stream="messages").to_pydict() == {
        "topic": ["/cloud", "/grid", "/metadata", "/imu"], "step": [0, 0, 1, 1],
    }


@pytest.mark.parametrize("representations", [("model_input",), ("raw", "model_input")])
def test_payload_retention(tmp_path, representations):
    names = {"raw": "raw_observation", "model_input": "model_observation"}
    pixels = np.arange(1200, dtype=np.uint8).reshape(20, 20, 3)
    with Recorder(tmp_path / "run", retain_sensor_data=False, chunk_steps=1,
                  payload_shard_bytes=512) as recorder:
        for index in range(2):
            payload = recorder.snapshot(pixels + index)
            observations = {names[name]: Observation(index / 10, index, {"front": payload})
                            for name in representations}
            recorder.record_capture(CaptureRecord(
                index, index * 100000, ("front",), **observations))
            recorder.record_step(StepRecord(index, **observations,
                delivered_captures={"front": (index, None)}))
    assert all(step.model_observation.sensor_data == {} for step in recorder.log.steps)
    restored = RunLog.read(tmp_path / "run")
    assert [record.step for record in restored.steps] == [0, 1]
    assert [record.capture_id for record in restored.captures] == [0, 1]
    assert [record.kind for record in restored.events] == ["termination"]
    for index, step in enumerate(restored.steps):
        assert step.delivered_captures == {"front": (index, None)}
        np.testing.assert_array_equal(step.model_observation.sensor_data["front"], pixels + index)
        for name in representations:
            np.testing.assert_array_equal(
                getattr(restored.captures[index], names[name]).sensor_data["front"], pixels + index)


def test_read_version_one(tmp_path):
    root = tmp_path / "legacy"
    (root / "steps").mkdir(parents=True)
    observation = DEFAULT_REGISTRY.encode(Observation(.1, 2))
    observation["fields"].pop("deliveries")
    observation["fields"].pop("state_estimates")
    pq.write_table(pa.Table.from_pylist([{"step": 0, "raw_observation": observation}]),
                   root / "steps/part-00000.parquet")
    (root / "manifest.json").write_text(json.dumps({
        "metadata": {"seed": 7}, "status": "completed"}))
    log = RunLog.read(root)
    assert log.format_version == 1 and log.status == "completed" and log.metadata == {"seed": 7}
    assert log.captures == [] and log.steps[0].raw_observation.t == .1
    assert log.steps[0].raw_observation.deliveries == {}


def test_record_and_read(tmp_path):
    pixels = np.full((2, 3, 3), 40, np.uint8)
    observation = Observation(.1, 4, {"front": pixels})
    command = Control(throttle=.5)
    detection = Detection((10, 0, 1), (4, 2, 2), 0, "car", .9)
    components = {"distance": 7.5}
    pose = np.eye(4)
    pose[:3, 3] = (4., 2., 1.)
    obj = ActorState(pose, extent=(4., 2., 1.))
    world = WorldSnapshot(.1, 3, objects={"sign": obj})
    visibility = {"sign": {"front": ObjectVisibility("depth", (), 12)}}
    recorder = Recorder(tmp_path / "run", metadata={"seed": 4})
    observation.sensor_data["front"] = recorder.snapshot(pixels)
    recorder.record_event("initialization", scenario="scene", clock=ClockSnapshot(.1, 4))
    estimate = TimedStateEstimate(100000, .1, np.eye(4), 2.)
    observation.vehicle_state, observation.ego_speed = estimate.vehicle_state, estimate.ego_speed
    observation.state_estimates = (estimate,)
    observation.deliveries = {"front": (DeliveredAcquisition(
        100000, .1, 4, pixels, None, state=estimate, capture_id=3, frame_start_us=83000),)}
    recorder.record_capture(CaptureRecord(
        3, 100000, ("front",), raw_observation=observation,
        physical_state=world, object_visibility=visibility,
        frame_ranges_us={"front": (83000, 100000)}))
    outputs = [RecordedOutput("ttc", 1.5, 0, "observation", 4, .1),
               RecordedOutput("summary", {"counts": [2, 3], "missing": None},
                              0, "observation", 4, .1)]
    recorder.record_step(StepRecord(
        0, raw_observation=observation, produced_command=command, executed_command=command,
        detections=[detection], components=components, outputs=outputs,
        object_visibility=visibility, physical_post=world,
        executed_command_issue_time_us=100000, delivered_captures={"front": (3,)},
    ))
    pixels.fill(0)
    command.throttle = 0
    memory = recorder.finish()
    decoded = RunLog.read(memory.path)
    assert decoded.status == "completed" and decoded.metadata == {"seed": 4}
    manifest = RunLog.read(memory.path, load_records=False)
    assert manifest.metadata == decoded.metadata and manifest.status == decoded.status
    assert manifest.path == decoded.path and manifest.format_version == decoded.format_version
    assert not (manifest.steps or manifest.events or manifest.captures)
    assert [event.kind for event in decoded.events] == ["initialization", "termination"]
    assert decoded.events[0].data["clock"] == ClockSnapshot(.1, 4)
    step, = decoded.steps
    assert step.components == components
    assert step.object_visibility == visibility
    stored = step.physical_post.objects["sign"]
    np.testing.assert_array_equal(stored.pose, obj.pose)
    assert stored.extent == obj.extent
    np.testing.assert_array_equal(step.raw_observation.sensor_data["front"], 40)
    assert step.produced_command.throttle == step.executed_command.throttle == .5
    assert step.detections == [detection] and step.outputs == outputs
    assert decoded.format_version == 2 and step.executed_command_issue_time_us == 100000
    assert step.delivered_captures == {"front": (3,)}
    np.testing.assert_array_equal(step.raw_observation.delivered("front")[-1].state.vehicle_state,
                                  step.raw_observation.vehicle_state)
    assert memory.steps[0].raw_observation.deliveries["front"][-1].data is None
    assert step.raw_observation.deliveries["front"][-1].data is None
    capture, = decoded.captures
    assert capture.capture_id == 3
    assert capture.object_visibility == visibility
    assert capture.frame_ranges_us == {"front": (83000, 100000)}
    np.testing.assert_array_equal(capture.raw_observation.delivered("front")[-1].data, 40)
    assert capture.raw_observation.delivered("front")[-1].time_us == 100000
    stripped = RunLog.read(memory.path, load_sensor_data=False)
    stripped_observation = stripped.captures[0].raw_observation
    assert stripped_observation.sensor_data == {}
    assert stripped_observation.delivered("front")[-1].data is None
    assert stripped_observation.delivered("front")[-1].frame_start_us == 83000
    assert stripped_observation.delivered("front")[-1].time_us == 100000
    np.testing.assert_array_equal(stripped_observation.vehicle_state, np.eye(4))
    assert stripped_observation.estimated_states()[0].time_us == 100000


@pytest.mark.parametrize("sensor,payload,calibration", [
    pytest.param("front", np.full((2, 3, 3), 40, np.uint8),
                 CameraCalibration(FThetaCamera(
                     16., 12., (0., 20.), 32, 24, (0., .05), max_angle=1.4,
                     windshield=WindshieldDistortion(
                         (0., 1., 0., 0., 0., 0.), (0., 0., 0., 1., 0., 0.),
                         (0., 1., 0., 0., 0., 0.), (0., 0., 0., 1., 0., 0.))),
                     BODY_FROM_OPTICAL), id="camera"),
    pytest.param("lidar", PointCloud(np.array([[1., 2., 3.], [4., 5., 6.]], np.float32),
                                    np.array([.2, .8], np.float32)),
                 LidarCalibration(np.array([[1., 0., 0., 1.17], [0., 1., 0., 0.],
                                            [0., 0., 1., 1.95], [0., 0., 0., 1.]]), .1),
                 id="lidar"),
    pytest.param("lidar", PointCloud(np.array([[7., 8., 9.]], np.float32)),
                 LidarCalibration(np.eye(4), .1), id="lidar-no-intensity"),
])
def test_observation_calibration(tmp_path, sensor, payload, calibration):
    observation = Observation(.1, 1, {sensor: payload}, calibration={sensor: calibration})
    with Recorder(tmp_path / "calibrated") as recorder:
        recorder.record_step(StepRecord(0, model_observation=observation))
    decoded = RunLog.read(tmp_path / "calibrated").steps[0].model_observation
    assert DEFAULT_REGISTRY.encode(decoded) == DEFAULT_REGISTRY.encode(observation)
    references = RunLog.read(tmp_path / "calibrated", load_payloads=False)
    restored = references.steps[0].model_observation
    assert type(restored.calibration[sensor]) is type(calibration)


def test_sensor_data_on_disk_only(tmp_path):
    calibration = {"lidar": LidarCalibration(np.eye(4), .1)}
    observation = Observation(
        .1, 1, {"front": np.full((2, 3, 3), 40, np.uint8)}, calibration=calibration)
    observation.deliveries = {"front": (DeliveredAcquisition(
        100000, .1, 1, observation.sensor_data["front"], None,
        TimedStateEstimate(100000, .1, observation.vehicle_state, observation.ego_speed),
        capture_id=0),)}
    recorder = Recorder(tmp_path / "run", retain_sensor_data=False)
    returned = recorder.record_step(StepRecord(
        0, raw_observation=observation, model_observation=observation, components={"score": 1.}))
    recorder.record_capture(CaptureRecord(0, 100000, ("front",),
                                         raw_observation=observation))
    np.testing.assert_array_equal(returned.raw_observation.sensor_data["front"], 40)
    memory = recorder.finish()
    for log in (memory, RunLog.read(tmp_path / "run", load_sensor_data=False)):
        step, = log.steps
        assert step.raw_observation.sensor_data == step.model_observation.sensor_data == {}
        assert type(step.model_observation.calibration["lidar"]) is LidarCalibration
        assert step.components == {"score": 1.}
        assert log.captures[0].raw_observation.sensor_data == {}
        sample, = log.captures[0].raw_observation.deliveries["front"]
        assert sample.data is None and sample.time_us == 100000
    saved, = RunLog.read(tmp_path / "run").steps
    np.testing.assert_array_equal(saved.model_observation.sensor_data["front"], 40)
    np.testing.assert_array_equal(
        RunLog.read(tmp_path / "run").captures[0].raw_observation.sensor_data["front"], 40)
    with pytest.raises(ValueError, match="persistent output path"):
        Recorder(retain_sensor_data=False)


def test_read_selected_columns(tmp_path):
    with Recorder(tmp_path / "run") as recorder:
        recorder.record_step(StepRecord(0, components={"image": np.zeros((3, 4), np.uint8)},
                                        executed_command=Control(brake=.75)))
    table = read_table(tmp_path / "run", columns=["step", "executed_command"])
    assert table.column_names == ["step", "executed_command"]
    row, = table.to_pylist()
    assert DEFAULT_REGISTRY.decode(row["executed_command"]) == Control(brake=.75)
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
    recorder = Recorder(tmp_path / "run", chunk_steps=2, retain_steps=False)
    recorder.record_step(StepRecord(0, components={"score": 1}))
    recorder.record_step(StepRecord(1, components={"score": 2}))
    recorder.record_step(StepRecord(2, components={"score": 3., "trajectory": [(1., 2.)]}))
    recorder.record_step(StepRecord(3, components={"score": 4.}))
    recorder.flush()
    recorder.record_step(StepRecord(4, components={"image": np.ones((2, 3), np.uint8)}))
    recorder.abort()
    recovered = RunLog.read(tmp_path / "run")
    assert recovered.status == "incomplete"
    assert [step.step for step in recovered.steps] == [0, 1, 2, 3]
    assert recovered.steps[0].components["score"] == 1
    assert recovered.steps[2].components["trajectory"] == [(1., 2.)]
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


def test_finish(tmp_path, monkeypatch):
    recorder = Recorder(tmp_path / "completed")
    with pytest.raises(ValueError, match="Final status"):
        recorder.finish("unknown")
    log = recorder.finish("failed", error="a sensor stopped")
    assert log.status == "failed" and log.events[-1].data["error"] == "a sensor stopped"
    stats = log.events[-1].data["recording"]
    assert stats == {"backpressure_wait_s": 0., "backpressure_batches": 0}
    assert RunLog.read(log.path).events[-1].data["recording"] == stats
    assert recorder.finish() is log

    directory = tmp_path / "blocked"
    recorder = Recorder(directory, chunk_steps=1)
    write_error = OSError("Parquet storage unavailable")

    def fail_write(*args, **kwargs):
        raise write_error

    monkeypatch.setattr(pq, "write_table", fail_write)
    try:
        recorder.record_step(StepRecord(0))
        with pytest.raises(RuntimeError, match="RunLog background write failed") as failure:
            recorder.finish()
        assert failure.value.__cause__ is write_error
        with pytest.raises(RuntimeError, match="RunLog background write failed") as repeated:
            recorder.record_event("failure")
        assert repeated.value.__cause__ is failure.value.__cause__
    finally:
        recorder.abort()
    recovered = RunLog.read(directory)
    assert recovered.status == "incomplete" and not recovered.steps
