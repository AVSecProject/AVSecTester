"""Recorded replay bounds and explicit mapping of native camera names to outputs."""

from dataclasses import replace

import numpy as np
import pytest

from avsectester.closed_loop.simulators.recorded import RecordedReplaySimulator
from avsectester.closed_loop.timing import TimingConfiguration
from avsectester.data_types import Control
from avsectester.execution.runner import ClosedLoopRunner, RunRequest
from tests.fakes import ConstantAVStack


def test_sensor_aliases(nuscenes_dataset):
    start = nuscenes_dataset.scenario_start("scene", nuscenes_dataset.frames("scene")[0])
    with RecordedReplaySimulator().create_runtime() as runtime:
        runtime.reset(start)
        native = runtime.sensor_view("CAM_FRONT")
        assert runtime.camera_sensors == {"CAM_FRONT"}
        assert set(runtime.observe().sensor_data) == {"CAM_FRONT"}
    with RecordedReplaySimulator(camera_aliases={"CAM_FRONT": "front"}).create_runtime() as runtime:
        runtime.reset(start)
        assert runtime.camera_sensors == {"front"}
        observation = runtime.observe()
        assert set(observation.sensor_data) == set(observation.calibration) == {"front"}
        assert runtime.sensor_view("CAM_FRONT").sensor == "front"
        assert runtime.sensor_geometry("CAM_FRONT").sensor == "front"
        np.testing.assert_array_equal(runtime.sensor_view("front").data, native.data)
        np.testing.assert_allclose(runtime.sensor_view("front").cam_from_world, native.cam_from_world)
    first, second = start.payload["frame_data"][:2]
    front = first.sensor_references["CAM_FRONT"]
    mount = front.metadata["ego_from_sensor"].copy()
    mount[1, 3] += 1.
    left = replace(front, sensor="CAM_LEFT", uri=second.sensor_references["CAM_FRONT"].uri,
                   metadata={**front.metadata, "ego_from_sensor": mount})
    two_camera_start = replace(start, payload={
        "frame_data": (replace(first, sensor_references={"CAM_FRONT": front, "CAM_LEFT": left}),),
    })
    with RecordedReplaySimulator().create_runtime() as runtime:
        runtime.reset(two_camera_start)
        original = runtime.observe()
        geometry = {name: runtime.sensor_geometry(name) for name in runtime.camera_sensors}
    aliases = {"CAM_FRONT": "CAM_LEFT", "CAM_LEFT": "CAM_FRONT"}
    with RecordedReplaySimulator(camera_aliases=aliases).create_runtime() as runtime:
        runtime.reset(two_camera_start)
        observation = runtime.observe()
        for native_name, output_name in aliases.items():
            np.testing.assert_array_equal(observation.sensor_data[output_name],
                                          original.sensor_data[native_name])
            np.testing.assert_array_equal(observation.calibration[output_name]["ego_from_sensor"],
                                          original.calibration[native_name]["ego_from_sensor"])
            view = runtime.sensor_view(output_name)
            assert view.sensor == output_name
            np.testing.assert_array_equal(view.data, original.sensor_data[native_name])
            np.testing.assert_allclose(view.cam_from_world, geometry[native_name].cam_from_world)
            assert runtime.sensor_geometry(output_name).sensor == output_name
            np.testing.assert_allclose(runtime.sensor_geometry(output_name).cam_from_world,
                                       geometry[native_name].cam_from_world)


def test_terminal_frame(nuscenes_dataset):
    frames = nuscenes_dataset.frames("scene")
    with RecordedReplaySimulator(camera_aliases={"CAM_FRONT": "front"}).create_runtime() as runtime:
        runtime.reset(nuscenes_dataset.scenario_start("scene", frames[1]))
        runtime.step(Control(throttle=1.))
        assert runtime.observe().frame == frames[-1].position
        assert runtime.observe().sensor_data["front"][0, 0, 0] == 52
        assert runtime.state().ego.extent is None
        with pytest.raises(ValueError, match="terminal recorded support"):
            runtime.step(Control(throttle=1.))


def test_epoch_timestamps(nuscenes_dataset):
    start = nuscenes_dataset.scenario_start("scene", nuscenes_dataset.frames("scene")[0])
    timestamps = (1500000000.123456, 1500000000.624173, 1500000001.122812)
    records = tuple(replace(data, frame=replace(data.frame, timestamp=time))
                    for data, time in zip(start.payload["frame_data"], timestamps))
    start = replace(start, frame=records[0].frame, payload={"frame_data": records})
    with RecordedReplaySimulator(start).create_runtime() as runtime:
        log = ClosedLoopRunner().run(runtime, ConstantAVStack(), RunRequest(2))
        assert runtime.decision_period_us == 500000
    assert log.status == "completed"
    assert [step.raw_observation.t for step in log.steps] == list(timestamps[:2])
    assert [step.physical_post.t for step in log.steps] == list(timestamps[1:])


def test_validate_timing():
    from avsectester.closed_loop.timing import SensorTiming

    simulator = RecordedReplaySimulator()
    simulator.validate_timing(TimingConfiguration())
    simulator.validate_timing(TimingConfiguration(decision_period_us=500000,
                                                 sensors={"CAM_FRONT": SensorTiming(500000)}))
    with pytest.raises(ValueError, match="default timing"):
        simulator.validate_timing(TimingConfiguration(planner_delay_us=100000))
