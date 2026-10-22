"""Simulator components share one calibrated capture and world-object contract."""

import numpy as np
import pytest

from avsectester.execution.services import ClosedLoopServices, RenderRequest, SensorRequest
from avsectester.geometry.cameras import BODY_FROM_OPTICAL
from avsectester.closed_loop.simulators.base import SimulatorRuntime
from avsectester.data_types import Control


def test_simulators(simulator_case):
    _dataset, simulator, start, initialization, expected_shape, _initial, _initial_state = simulator_case
    with simulator.create_runtime(seed=3) as runtime:
        assert runtime.can_replay is simulator.can_replay
        assert simulator.can_replay == (type(runtime)._replay_tick is not SimulatorRuntime._replay_tick)
        runtime.reset(start, initialization)
        original = runtime.observe()
        state = runtime.state()
        assert state.ego is not None and "ego" not in state.actors
        view = runtime.sensor_view("front")
        assert runtime.camera_sensors <= runtime.sensors
        assert view.sensor in runtime.sensors
        assert view.data.shape == expected_shape
        assert view.time == original.t and view.frame == original.frame
        assert view.cam_from_world.shape == (4, 4)
        assert view.sensor_from_world.shape == (4, 4)
        np.testing.assert_allclose(view.sensor_from_world, BODY_FROM_OPTICAL @ view.cam_from_world)
        assert view.camera.project(np.array([[0., 0., 4.]])).shape == (1, 2)
        runtime.step(Control(throttle=.1))
        advanced = runtime.observe()
        assert advanced.frame == state.frame + 1
        assert advanced.t - state.t == pytest.approx(.1)
        assert runtime.state().frame == advanced.frame
        with pytest.raises(ValueError):
            runtime.sensor_view("missing-camera")
    with pytest.raises(RuntimeError, match="Reset"):
        runtime.observe()


def test_read_sensors_views(simulator_case, assert_camera_views_equal):
    dataset, simulator, start, _initialization, _expected_shape, _initial, _initial_state = simulator_case
    frames = dataset.frames(start.scenario_id)[:3]
    with ClosedLoopServices(dataset, simulator) as services:
        with services.session(3) as session:
            samples = [session.read_sensors(SensorRequest(start.scenario_id, frame, ("front",),
                                                        views=True)) for frame in frames]
        for sample in samples:
            with services.session(3) as session:
                rendered = session.render(RenderRequest(start.scenario_id, sample.frame, "front"))
            assert_camera_views_equal(sample.views["front"], rendered.view)


@pytest.mark.parametrize("simulator_case", ["carla_case", "recorded_case"], indirect=True)
def test_insert_object(simulator_case, assert_inserted_world_object):
    _dataset, simulator, _start, _initialization, _shape, initial, initial_state = simulator_case
    with simulator.create_runtime(seed=3) as runtime:
        original, inserted, _expected = assert_inserted_world_object(runtime, initial, initial_state)
        assert original.depth is None and inserted.depth is None


def test_insert_object_with_depth(nurec_case, assert_inserted_world_object):
    _dataset, simulator, _start, _initialization, _shape, initial, initial_state = nurec_case
    with simulator.create_runtime(seed=3) as runtime:
        original, inserted, expected = assert_inserted_world_object(runtime, initial, initial_state)
        assert original.depth is not None
        np.testing.assert_array_equal(inserted.depth, expected.depth)
