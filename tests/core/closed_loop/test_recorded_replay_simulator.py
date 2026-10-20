"""Recorded replay preserves RGB pixels, advances source frames and isolates callers."""
import numpy as np
import pytest
from PIL import Image
from avsectester.datasets.base import ActorState, FrameReference, FrameData, ScenarioStart, SensorDataReference
from avsectester.closed_loop.datatypes import Control
from avsectester.closed_loop.simulators.recorded import RecordedReplaySimulator


def source(tmp_path):
    refs = []
    for i in range(3):
        path = tmp_path / f"{i}.png"
        Image.fromarray(np.full((2, 3, 3), 100 + i, np.uint8)).save(path)
        frame = FrameReference("s", 12+i, str(i), 4+i*.5)
        refs.append(FrameData(frame, sensor_data={"front": SensorDataReference(str(path), "front", frame.timestamp)},
                              ego=ActorState(np.eye(4), (2., 0., 0.), "vehicle", (4., 2., 1.5))))
    return ScenarioStart("NuScenesDataset", "s", refs[0].frame, {"frame_data": tuple(refs)})


def test_recorded_simulator_runtime_advances_actual_images_and_resets_configured_origin(tmp_path):
    simulator_runtime = RecordedReplaySimulator(source(tmp_path)).create_runtime()
    simulator_runtime.reset()
    assert np.all(simulator_runtime.observe().sensor_data["front"] == 100)
    simulator_runtime.step(Control(throttle=1, steer=.5))
    advanced = simulator_runtime.observe()
    assert advanced.frame == 13 and advanced.t == 4.5
    assert np.all(advanced.sensor_data["front"] == 101)
    simulator_runtime.step(Control())
    with pytest.raises(ValueError, match="terminal recorded"):
        simulator_runtime.step(Control())
    simulator_runtime.reset()
    assert simulator_runtime.observe().frame == 12


def test_recorded_observation_edits_do_not_change_source_or_clock(tmp_path):
    simulator_runtime = RecordedReplaySimulator(source(tmp_path)).create_runtime()
    simulator_runtime.reset()
    observation = simulator_runtime.observe()
    observation.sensor_data["front"][:] = 0
    observation.frame, observation.t = -1, -2
    assert simulator_runtime.state().frame == 12
    assert simulator_runtime.state().t == 4
    assert simulator_runtime.state().ego.extent == (4., 2., 1.5)
    assert simulator_runtime.state().ego.velocity == (2., 0., 0.)
    assert np.all(simulator_runtime.observe().sensor_data["front"] == 100)


def calibrated_source(tmp_path):
    from dataclasses import replace
    from avsectester.geometry.cameras import PinholeCamera
    from avsectester.geometry.poses import rotation_matrix

    start = source(tmp_path)
    ego_pose = np.eye(4)
    ego_pose[:3, :3] = rotation_matrix((0, 0, 30))
    ego_pose[:3, 3] = (5, -2, 1)
    ego_from_sensor = np.eye(4)
    ego_from_sensor[:3, 3] = (1, .2, 1.5)
    records = []
    for record in start.payload['frame_data']:
        sensor = replace(record.sensor_data['front'], metadata={
            'camera': PinholeCamera(np.eye(3), 3, 2), 'ego_from_sensor': ego_from_sensor,
        })
        records.append(replace(record, sensor_data={'front': sensor},
            ego=ActorState(ego_pose, extent=(4, 2, 1.5), box_center=(.2, 0, .5)),
            actors={'host': ActorState(np.eye(4), extent=(2, 1, 1), box_center=(0, .4, .8)),
                    'ego': ActorState(ego_pose, extent=(4, 2, 1.5))}))
    return replace(start, payload={'frame_data': tuple(records)})


def test_recorded_sensor_view_uses_physical_extrinsics_cuboid_centres_and_excludes_ego(tmp_path):
    start = calibrated_source(tmp_path)
    simulator_runtime = RecordedReplaySimulator(start).create_runtime()
    assert simulator_runtime.can_render
    simulator_runtime.reset()
    view = simulator_runtime.sensor_view('front')
    record = start.payload['frame_data'][0]
    expected = np.linalg.inv(record.ego.pose @ record.sensor_data['front'].metadata['ego_from_sensor'])
    np.testing.assert_allclose(view.cam_from_world, expected)
    np.testing.assert_allclose(view.ego.transform, record.ego.box_transform())
    np.testing.assert_allclose(view.actors['host'].transform[:3, 3], [0, .4, .8])
    assert 'ego' in record.actors  # Malformed provider input is normalized once at reset.
    assert 'ego' not in view.actors
    assert 'ego' not in simulator_runtime.state().actors
    assert simulator_runtime.camera_sensors == frozenset({'front'})
    # Geometry reads never copy an entire observation.
    simulator_runtime.observe = lambda: pytest.fail('geometry must not read/copy observations')
    np.testing.assert_allclose(simulator_runtime.sensor_geometry('front').cam_from_world, expected)


def test_recorded_sensor_view_rejects_uninitialized_unknown_uncalibrated_and_missing_images(tmp_path):
    simulator_runtime = RecordedReplaySimulator(calibrated_source(tmp_path)).create_runtime()
    with pytest.raises(RuntimeError, match='Initialize recorded replay'):
        simulator_runtime.sensor_view('front')
    simulator_runtime.reset()
    with pytest.raises(ValueError, match='Unknown recorded sensor'):
        simulator_runtime.sensor_view('rear')
    simulator_runtime._capture.sensor_data.pop('front')
    with pytest.raises(ValueError, match='no RGB capture'):
        simulator_runtime.sensor_view('front')
    simulator_runtime = RecordedReplaySimulator(source(tmp_path)).create_runtime()
    simulator_runtime.reset()
    with pytest.raises(ValueError, match='no camera calibration'):
        simulator_runtime.sensor_view('front')


def test_recorded_replay_missing_ego_has_explicit_error(tmp_path):
    from dataclasses import replace

    start = source(tmp_path)
    records = list(start.payload['frame_data'])
    records[1] = replace(records[1], ego=None)
    simulator_runtime = RecordedReplaySimulator(replace(start, payload={'frame_data': records})).create_runtime()
    with pytest.raises(ValueError, match='ego state'):
        simulator_runtime.reset()


def test_recorded_replay_renders_through_the_closed_loop_service_boundary(tmp_path):
    from copy import deepcopy
    from avsectester.closed_loop.services import ClosedLoopServices, RenderRequest
    from avsectester.datasets.base import Dataset

    start = calibrated_source(tmp_path)
    class RecordedData(Dataset):
        def scenarios(self):
            return iter(())
        def frames(self, scenario_id):
            return tuple(record.frame for record in start.payload['frame_data'])
        def frame_data(self, scenario_id, frame):
            return deepcopy(next(record for record in start.payload['frame_data'] if record.frame == frame))
        def native(self, scenario_id):
            return None
        def scenario_start(self, scenario_id, frame):
            assert frame == start.frame
            return deepcopy(start)
    with ClosedLoopServices(RecordedData(), RecordedReplaySimulator()) as services:
        with services.session(0) as session:
            result = session.render(RenderRequest('s', start.frame, 'front'))
    assert result.view.data.shape == (2, 3, 3)
    assert np.all(result.observation.sensor_data['front'] == 100)
