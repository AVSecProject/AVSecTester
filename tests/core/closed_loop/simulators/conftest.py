"""Native-boundary simulator cases shared by the component contract tests."""

from types import SimpleNamespace

import numpy as np
import pytest

from avsectester.closed_loop.simulators.base import Initialization
from avsectester.closed_loop.simulators.carla import CarlaSimulator
from avsectester.closed_loop.simulators.nurec import NuRecRenderer, NuRecSimulator
from avsectester.closed_loop.simulators.recorded import RecordedReplaySimulator
from tests.fakes import CarlaNativeBoundary, NuRecRpcBoundary


@pytest.fixture
def carla_case(carla_dataset, monkeypatch):
    CarlaNativeBoundary().install(monkeypatch, None)
    simulator = CarlaSimulator({"rate": 10}, settle_iters=0)
    frames = carla_dataset.frames("scene")
    with simulator.create_runtime(seed=3) as runtime:
        sequence = runtime.generate_frame_data(carla_dataset.scenario_start("scene", frames[0]), frames)
    start = carla_dataset.scenario_start("scene", frames[2])
    initialization = Initialization(sequence.at(frames[2]), sequence.resolved_scenario, sequence.provenance)
    initial = carla_dataset.scenario_start("scene", frames[0])
    initial_state = Initialization(sequence.at(frames[0]), sequence.resolved_scenario, sequence.provenance)
    return SimpleNamespace(dataset=carla_dataset, simulator=simulator, start=start,
                           initialization=initialization, shape=(12, 16, 3),
                           initial=initial, initial_state=initial_state)


@pytest.fixture
def nurec_case(nurec_dataset, monkeypatch):
    NuRecRpcBoundary().install(monkeypatch)
    # Shared simulator contracts compare inserted objects with geometric composites.
    simulator = NuRecSimulator(NuRecRenderer(cameras=["front"]), object_occlusion="geometry", time_step=.1)
    start = nurec_dataset.scenario_start("scene", nurec_dataset.frames("scene")[1])
    initial = nurec_dataset.scenario_start("scene", nurec_dataset.frames("scene")[0])
    return SimpleNamespace(dataset=nurec_dataset, simulator=simulator, start=start,
                           initialization=None, shape=(12, 16, 3), initial=initial, initial_state=None)


@pytest.fixture
def recorded_case(nuscenes_dataset):
    simulator = RecordedReplaySimulator(camera_aliases={"CAM_FRONT": "front"})
    start = nuscenes_dataset.scenario_start("scene", nuscenes_dataset.frames("scene")[1])
    initial = nuscenes_dataset.scenario_start("scene", nuscenes_dataset.frames("scene")[0])
    return SimpleNamespace(dataset=nuscenes_dataset, simulator=simulator, start=start,
                           initialization=None, shape=(80, 100, 3), initial=initial, initial_state=None)


@pytest.fixture(params=["carla_case", "nurec_case", "recorded_case"])
def simulator_case(request):
    return request.getfixturevalue(request.param)


@pytest.fixture
def assert_inserted_world_object(make_object_in_front):
    """The inserted face follows the simulator's composition, depth and projection geometry."""
    from avsectester.data_types import Control
    from avsectester.geometry.projection import project_box
    from avsectester.rendering import PlacedObject, composite_objects

    def check(runtime, start=None, initialization=None):
        runtime.reset(start, initialization)
        asset, pose = make_object_in_front(runtime.sensor_view("front"))
        runtime.step(Control())
        clean_first = runtime.sensor_view("front")
        runtime.step(Control())
        clean_second = runtime.sensor_view("front")
        runtime.reset(start, initialization)
        runtime.insert_object("sign", asset, pose)
        runtime.step(Control())
        view = runtime.sensor_view("front")
        expected = composite_objects(
            clean_first.data, clean_first.camera, clean_first.cam_from_world,
            (PlacedObject("sign", asset, pose),), scene_depth=clean_first.depth,
            occluders=clean_first.actors, occlusion_tolerance_m=clean_first.occlusion_tolerance_m,
        )
        np.testing.assert_array_equal(view.data, expected.image)
        planned = clean_first.composite_plan((PlacedObject("sign", asset, pose),))
        np.testing.assert_array_equal(planned.apply(clean_first.data).image, view.data)
        np.testing.assert_array_equal(view.composition.background, clean_first.data)
        np.testing.assert_array_equal(view.composition.plan.visible_mask, planned.visible_mask)
        assert runtime.object_visibility("sign")["front"] == expected.object_visibility["sign"]
        changed = np.any(view.data != clean_first.data, axis=2)
        assert np.any(changed)
        world_from_box = pose.copy()
        world_from_box[:3, 3] += pose[:3, :3] @ np.array([0., 0., 1.])
        x0, y0, x1, y1 = project_box(view.camera, view.cam_from_world, world_from_box, asset.extent)
        ys, xs = np.nonzero(changed)
        assert xs.min() >= np.floor(x0) and xs.max() < np.ceil(x1)
        assert ys.min() >= np.floor(y0) and ys.max() < np.ceil(y1)
        runtime.remove_object("sign")
        runtime.step(Control())
        np.testing.assert_array_equal(runtime.sensor_view("front").data, clean_second.data)
        return clean_first, view, expected
    return check


@pytest.fixture
def assert_camera_views_equal():
    def compare(first, second):
        assert first.sensor == second.sensor
        assert first.native == second.native == {}
        np.testing.assert_array_equal(first.data, second.data)
        np.testing.assert_equal(first.depth, second.depth)
        np.testing.assert_allclose(first.cam_from_world, second.cam_from_world)
        np.testing.assert_allclose(first.sensor_from_world, second.sensor_from_world)
        assert type(first.camera) is type(second.camera)
        assert (first.camera.width, first.camera.height) == (second.camera.width, second.camera.height)
        pixels = np.array([[.5, .5], [first.camera.width / 2, first.camera.height / 2]])
        np.testing.assert_allclose(first.camera.unproject(pixels), second.camera.unproject(pixels))
        assert first.actors.keys() == second.actors.keys()
        for identity in first.actors:
            np.testing.assert_allclose(first.actors[identity].box_transform(), second.actors[identity].box_transform())
            assert first.actors[identity].extent == second.actors[identity].extent
        np.testing.assert_allclose(first.ego.box_transform(), second.ego.box_transform())
        assert first.ego.extent == second.ego.extent
    return compare
