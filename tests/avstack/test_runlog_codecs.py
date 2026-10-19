"""Persist real adapter outputs as queryable data, with independent snapshots."""

import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from avsectester.runlog.codecs import default_codecs
from avsectester.closed_loop.datatypes import Observation
from avsectester.runlog import Recorder, RunLog, StepRecord, read_table


def test_registry_does_not_load_adapter_dependencies():
    subprocess.run(
        [sys.executable, "-c", "from avsectester.runlog.codecs import default_codecs; "
         "r = default_codecs(); assert r.encoder(object()) is None; "
         "import sys; assert not any(m in sys.modules for m in "
         "('avstack', 'torch', 'carla', 'alpasim_driver', 'pyarrow'))"],
        check=True,
    )


def test_native_observation_and_detection_round_trip(tmp_path, make_ego, make_detections):
    from avcarla.geometry import CarlaReferenceFrame
    from avstack.calibration import CameraCalibration
    from avstack.geometry import GlobalOrigin3D
    from avstack.sensors import ImageData

    body = CarlaReferenceFrame(GlobalOrigin3D, location=(10, 0, 0))
    sensor = CarlaReferenceFrame(body, location=(1, 0, 1.6))
    calibration = CameraCalibration(sensor, np.eye(3, 4), (2, 3, 3), channel_order="rgb")
    pixels = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    image = ImageData(0.5, 10, pixels, calibration, "front")
    detections = make_detections([(5, 0, 0)], reference=sensor, frame=10)
    ego = make_ego(xyz=(10, 0, 0), speed=5, t=0.5)
    observation = Observation(t=0.5, frame=10, vehicle_state=ego, sensor_data={"camera": image})
    recorder = Recorder(tmp_path, codecs=default_codecs(), chunk_steps=1)
    snapshot = recorder.record_step(StepRecord(
        0, raw_observation=observation, components={"perception": detections},
    ))
    pixels[:] = 0
    body.x = np.array([99.0, 0, 0])
    detections[0].score = 0.1
    recorder.finish()

    # Memory remains useful to existing AVStack consumers and independent of
    # subsequent world mutation, while disk requires no native reconstruction.
    assert type(snapshot.components["perception"]) is type(detections)
    assert snapshot.components["perception"][0].score == pytest.approx(0.8)
    saved = RunLog.read(tmp_path).steps[0]
    obj = saved.components["perception"]["data"][0]
    assert obj["score"] == pytest.approx(0.8)
    assert obj["data"]["position"]["x"] == [5, 0, 0]
    assert obj["reference"]["reference"]["x"] == [10, 0, 0]
    saved_image = saved.raw_observation.sensor_data["camera"]
    np.testing.assert_array_equal(saved_image["data"], np.arange(18).reshape(2, 3, 3))
    assert saved_image["calibration"]["channel_order"] == "rgb"
    assert saved.raw_observation.vehicle_state["velocity"]["x"] == [5, 0, 0]

    import pyarrow as pa
    table = read_table(tmp_path, columns=["components"])
    position = (table.schema.field("components").type.field("perception").type
                .field("value").type.field("data").type.value_type.field("value").type
                .field("data").type.field("value").type.field("position").type
                .field("value").type.field("x").type)
    assert pa.types.is_list(position)
    without_payloads = RunLog.read(tmp_path, load_payloads=False).steps[0]
    assert without_payloads.raw_observation.sensor_data["camera"]["data"]["chunks"]


def test_real_pipeline_records_tracking_planning_and_control(
    tmp_path, pipeline_config, make_detections, make_ego,
):
    from avsectester.closed_loop.avstacks.modular import ModularAVStack

    # Build the registry before the model imports its planning/tracking types.
    recorder = Recorder(tmp_path, codecs=default_codecs(), chunk_steps=3)
    stack = ModularAVStack(pipeline_config)
    stack.instrument()
    for i in range(8):
        observation = Observation(
            t=i * 0.05, frame=i, ego_speed=5,
            sensor_data={"lidar": make_detections([(20 - i * 1.5, 0, 0)], frame=i)},
            vehicle_state=make_ego(speed=5, t=i * 0.05),
        )
        if i == 0:
            stack.reset(observation)
        command = stack(observation)
        recorder.record_step(StepRecord(
            i, model_observation=observation, produced_command=command,
            components=stack.component_log(),
        ))
    recorder.finish()

    saved = RunLog.read(tmp_path)
    assert len(saved.steps) == 8
    tracking = saved.steps[-1].components["tracking"]["data"]
    assert len(tracking) == 1
    assert len(tracking[0]["x"]) == 9
    assert tracking[0]["box"]["hwl"] == [1.5, 1.8, 4.0]
    waypoints = saved.steps[-1].components["planning"]["waypoints"]
    assert waypoints[0]["waypoint"]["target_speed"] == 0
    assert saved.steps[0].components["planning"]["waypoints"][0]["waypoint"]["target_speed"] == 6
    assert saved.steps[-1].components["control"]["brake"] > 0
    assert saved.steps[-1].produced_command.brake > 0


def test_lidar_imu_gps_payloads_are_not_stringified(tmp_path):
    from avstack.calibration import GpsCalibration, ImuCalibration, LidarCalibration
    from avstack.geometry import GlobalOrigin3D, PointMatrix3D
    from avstack.sensors import GpsData, ImuData, LidarData

    calibration = LidarCalibration(GlobalOrigin3D)
    points = np.arange(12, dtype=float).reshape(3, 4)
    sensors = {
        "lidar": LidarData(0, 0, PointMatrix3D(points, calibration), calibration, "lidar"),
        "imu": ImuData(0, 0, {"dt": 0.05, "dv": np.ones(3)},
                       ImuCalibration(GlobalOrigin3D), "imu"),
        "gps": GpsData(0, 0, np.array([1.0, 2.0, 3.0]),
                       GpsCalibration(GlobalOrigin3D), "gps", levar=np.zeros(3)),
    }
    with Recorder(tmp_path, codecs=default_codecs()) as recorder:
        recorder.record_step(StepRecord(0, components=sensors))
    saved = RunLog.read(tmp_path).steps[0].components
    np.testing.assert_array_equal(saved["lidar"]["data"]["x"], points)
    np.testing.assert_array_equal(saved["imu"]["data"]["dv"], np.ones(3))
    np.testing.assert_array_equal(saved["gps"]["data"], [1, 2, 3])


def test_unregistered_native_objects_fail_without_attribute_traversal(tmp_path):
    class Unknown:
        secret = "must not be serialized"

    with pytest.raises(TypeError, match="No RunLog codec"):
        with Recorder(tmp_path, codecs=default_codecs()) as recorder:
            recorder.record_step(StepRecord(0, components={"unknown": Unknown()}))


def test_noncopyable_carla_capture_is_detached_inside_real_sensor(tmp_path, monkeypatch):
    import carla
    from avstack.calibration import LidarCalibration
    from avstack.geometry import GlobalOrigin3D
    from avstack.sensors import LidarData

    # CARLA measurement constructors are server-only. Exercise its documented
    # raw-buffer boundary with a non-copyable capture and a real AVStack wrapper.
    class Measurement:
        frame = 12
        timestamp = 0.6
        channels = 4
        horizontal_angle = 0.2

        def __init__(self, buffer):
            self.raw_data = memoryview(buffer)

        def __deepcopy__(self, memo):
            raise TypeError("SDK measurement cannot be pickled")

    monkeypatch.setattr(carla, "LidarMeasurement", Measurement)
    points = np.arange(12, dtype=np.float32).reshape(3, 4)
    buffer = bytearray(points.tobytes())
    sensor = LidarData(0.6, 12, Measurement(buffer), LidarCalibration(GlobalOrigin3D), "0",
                       flipy=True)
    with Recorder(tmp_path / "log", codecs=default_codecs()) as recorder:
        saved = recorder.record_step(StepRecord(0, raw_observation=Observation(
            0.6, 12, sensor_data={"lidar": sensor},
        )))
        buffer[:] = bytes(len(buffer))
    memory_sensor = saved.raw_observation.sensor_data["lidar"]
    assert isinstance(memory_sensor, LidarData)
    assert memory_sensor.flipy is True
    np.testing.assert_array_equal(np.frombuffer(memory_sensor.data.raw_data, dtype=np.float32),
                                  points.reshape(-1))
    # Existing native exporters still apply their own coordinate conversion.
    memory_sensor.save_to_file(str(tmp_path / "points.bin"))
    exported = np.fromfile(tmp_path / "points.bin", dtype=np.float32).reshape(3, 4)
    expected = points.copy()
    expected[:, 1] *= -1
    np.testing.assert_array_equal(exported, expected)
    disk = RunLog.read(tmp_path / "log").steps[0].raw_observation.sensor_data["lidar"]["data"]
    assert disk["raw_bytes"] == points.tobytes()
    assert disk["metadata"]["channels"] == 4


def test_alpasim_public_output_fields_and_late_registration(tmp_path, monkeypatch):
    # Protocol doubles for the inspected NVIDIA types. This does not claim to
    # execute AlpaSim or load Alpamayo; their full environment is separate.
    class Pose:
        def to_proto(self):
            return [1.0, 2.0, 0.0], [1.0, 0.0, 0.0, 0.0]

    class Trajectory:
        timestamps_us = np.array([100_000, 200_000], dtype=np.uint64)
        positions = np.array([[1.0, 0, 0], [2.0, 0, 0]])
        quaternions = np.array([[0.0, 0, 0, 1], [0.0, 0, 0, 1]])

    class ModelPrediction:
        candidate_positions = np.array([[[1.0, 0, 0], [2.0, 0, 0]]])
        candidate_rotations = np.tile(np.eye(3), (1, 2, 1, 1))
        selected_index = 0
        reasoning_text = "brake for lead"
        selected_plan = Trajectory()
        model_t0_us = 50_000
        pose_local_to_rig_t0 = Pose()
        waypoint_timestamps_us = np.array([100_000, 200_000], dtype=np.uint64)

    recorder = Recorder(tmp_path, codecs=default_codecs())
    monkeypatch.setitem(sys.modules, "alpasim_driver.models.base",
                        SimpleNamespace(ModelPrediction=ModelPrediction))
    monkeypatch.setitem(sys.modules, "alpasim_utils.geometry",
                        SimpleNamespace(Pose=Pose, Trajectory=Trajectory))
    recorder.record_step(StepRecord(0, components={"policy": ModelPrediction()}))
    recorder.finish()
    result = RunLog.read(tmp_path).steps[0].components["policy"]
    np.testing.assert_array_equal(result["candidate_positions"], ModelPrediction.candidate_positions)
    np.testing.assert_array_equal(result["selected_plan"]["timestamps_us"], Trajectory.timestamps_us)
    assert result["pose_local_to_rig_t0"]["q_wxyz"] == [1, 0, 0, 0]
    assert result["reasoning_text"] == "brake for lead"
