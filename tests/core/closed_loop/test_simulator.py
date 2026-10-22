"""The base simulator runtime's public capture, replay and ownership contract."""

from dataclasses import replace
from threading import Lock

import numpy as np
import pytest

from avsectester.closed_loop.datatypes import Control
from avsectester.closed_loop.hooks import Hook, HookSet
from avsectester.closed_loop.simulator import DeploymentPlan, SensorGeometry, can_deploy_at
from tests.fakes import CAMERA, CAM_FROM_WORLD, ConstantAVStack, MemoryDataset, MemorySimulator, actor_state


def test_reset():
    data = MemoryDataset()
    frames = data.frames("scene")
    with MemorySimulator().create_runtime() as simulator_runtime:
        simulator_runtime.reset(data.scenario_start("scene", frames[3]))
        assert simulator_runtime.observe().frame == 3
        assert simulator_runtime.state().t == pytest.approx(frames[3].timestamp)
        simulator_runtime.reset()
        assert simulator_runtime.observe().frame == 0
        assert simulator_runtime.clock.t == 0.
        with pytest.raises(ValueError, match="does not support"):
            simulator_runtime.reset(world={"sensors": []})


def test_step():
    with MemorySimulator().create_runtime() as simulator_runtime:
        with pytest.raises(RuntimeError, match="Reset"):
            simulator_runtime.step(Control())
        simulator_runtime.reset()
        before = simulator_runtime.clock
        simulator_runtime.step(Control(throttle=.5))
        assert simulator_runtime.clock.frame == before.frame + 1
        assert simulator_runtime.clock.t == pytest.approx(before.t + .1)
        assert simulator_runtime.observe().frame == simulator_runtime.clock.frame
        with pytest.raises(ValueError, match="Simulator does not support trajectory commands"):
            simulator_runtime.step(Control(trajectory=np.zeros((2, 3))))


def test_observe_and_state():
    with MemorySimulator().create_runtime() as simulator_runtime:
        with pytest.raises(RuntimeError, match="Reset"):
            simulator_runtime.observe()
        simulator_runtime.reset()
        first = simulator_runtime.observe()
        original = first.sensor_data["front"].copy()
        first.sensor_data["front"][:] = 0
        np.testing.assert_array_equal(simulator_runtime.observe().sensor_data["front"], original)
        state = simulator_runtime.state()
        state.ego.pose[0, 3] = 1000.
        assert simulator_runtime.state().ego.pose[0, 3] != 1000.
        assert simulator_runtime.clock.frame == first.frame


def test_setup_tick():
    data = MemoryDataset()
    deployment = DeploymentPlan(2)
    starts = (0, 4)

    class Deploy:
        setup_ticks = deployment.ticks

        def __call__(self, value, context):
            context.simulator_runtime.setup_tick()

    for position in starts:
        with MemorySimulator(capture_ticks=1).create_runtime() as simulator_runtime:
            hooks = HookSet([Hook("simulator.setup", Deploy())])
            with hooks.session(simulator_runtime, ConstantAVStack()).activate():
                simulator_runtime.reset(data.scenario_start("scene", data.frames("scene")[position]),
                                        deployment=deployment)
            assert simulator_runtime.observe().frame == position
            assert simulator_runtime.clock.t == pytest.approx(position * .1)
            with pytest.raises(RuntimeError, match="simulator.setup"):
                simulator_runtime.setup_tick()
            assert can_deploy_at(position, deployment, 1)
            with pytest.raises(ValueError, match="deployment window"):
                simulator_runtime.reset(data.scenario_start("scene", data.frames("scene")[1]),
                                        deployment=deployment)


def test_generate_frame_data():
    data = MemoryDataset()
    frames = data.frames("scene")[:4]
    with MemorySimulator().create_runtime() as simulator_runtime:
        sequence = simulator_runtime.generate_frame_data(data.scenario_start("scene", frames[0]), frames)
        assert tuple(record.frame for record in sequence.frames) == tuple(frames)
        for record in sequence.frames:
            simulator_runtime.reset(data.scenario_start("scene", record.frame))
            np.testing.assert_allclose(record.ego.pose, simulator_runtime.state().ego.pose)
        with pytest.raises(ValueError, match="consecutive"):
            simulator_runtime.generate_frame_data(data.scenario_start("scene", frames[0]), frames[::2])


def test_exclusive():
    simulator = MemorySimulator(resource_key=("memory", "shared-world"))
    with simulator.create_runtime() as first, simulator.create_runtime() as second:
        with first.exclusive():
            with pytest.raises(RuntimeError, match="already"):
                with second.exclusive():
                    pytest.fail("two independent users acquired one world")
        with second.exclusive():
            second.reset()
            assert second.observe().frame == 0


def test_insert_object(assert_inserted_world_object, make_object_in_front):
    class LockedHarmonizer:
        def __init__(self):
            self.lock = Lock()

        def __call__(self, composite, mask, background):
            with self.lock:
                return composite

    points = np.array([[3., -1., .5, .8]], np.float32)
    with MemorySimulator(extra_sensors={"lidar": points}).create_runtime() as simulator_runtime:
        original, inserted, _expected = assert_inserted_world_object(simulator_runtime)
        assert original.depth is None and inserted.depth is None
        asset, world_from_object = make_object_in_front(simulator_runtime.sensor_view("front"))
        harmonizer = LockedHarmonizer()
        simulator_runtime.insert_object("harmonized", asset, world_from_object,
                                        harmonizer=harmonizer)
        simulator_runtime.step(Control())
        # Views share immutable placed objects, so a harmonizer holding a lock is never copied.
        assert simulator_runtime.sensor_view("front").objects["harmonized"].harmonizer is harmonizer

    class NoObjectSimulator(MemorySimulator):
        can_insert_objects = False

    with NoObjectSimulator().create_runtime() as simulator_runtime:
        simulator_runtime.reset()
        asset, world_from_object = make_object_in_front(simulator_runtime.sensor_view("front"))
        with pytest.raises(NotImplementedError, match="cannot insert objects"):
            simulator_runtime.insert_object("sign", asset, world_from_object)
        with pytest.raises(NotImplementedError, match="cannot insert objects"):
            simulator_runtime.remove_object("sign")


def test_insert_object_with_depth(assert_inserted_world_object):
    depth = np.full((CAMERA.height, CAMERA.width), np.inf)
    with MemorySimulator(sensor_depths={"front": depth}).create_runtime() as simulator_runtime:
        original, inserted, expected = assert_inserted_world_object(simulator_runtime)
        np.testing.assert_array_equal(original.depth, depth)
        np.testing.assert_array_equal(inserted.depth, expected.depth)
        assert np.any(np.isfinite(inserted.depth))


@pytest.mark.parametrize("sensor_depths, occlusion_source, ignored_occluders", [
    ({}, "actor_cuboids", ("unbounded",)),
    ({name: np.full((CAMERA.height, CAMERA.width), np.inf)
      for name in ("front", "second", "away")}, "depth", ()),
], ids=["actor-cuboids", "measured-depth"])
def test_object_visibility(sensor_depths, occlusion_source, ignored_occluders, make_object_in_front):
    away = np.diag([-1., 1., -1., 1.]) @ CAM_FROM_WORLD
    geometries = {
        "front": SensorGeometry("front", CAMERA, CAM_FROM_WORLD),
        "second": SensorGeometry("second", CAMERA, CAM_FROM_WORLD),
        "away": SensorGeometry("away", CAMERA, away),
    }
    unbounded = actor_state((10., 0., 0.))
    simulator = MemorySimulator(camera_geometries=geometries, sensor_depths=sensor_depths,
                                actors={"unbounded": replace(unbounded, extent=None)})
    with simulator.create_runtime() as simulator_runtime:
        simulator_runtime.reset()
        asset, world_from_object = make_object_in_front(simulator_runtime.sensor_view("front"))
        simulator_runtime.insert_object("sign", asset, world_from_object)
        offscreen = world_from_object.copy()
        offscreen[2, 3] += 10000.
        simulator_runtime.insert_object("offscreen", asset, offscreen)
        with pytest.raises(KeyError, match="next capture"):
            simulator_runtime.object_visibility("sign")
        with pytest.raises(KeyError, match="unknown"):
            simulator_runtime.object_visibility("unknown")
        simulator_runtime.step(Control())
        visibility = simulator_runtime.object_visibility("sign")
        assert visibility.keys() == {"front", "second"}
        assert simulator_runtime.object_visibility("offscreen") == {}
        for sensor in ("front", "second"):
            assert visibility[sensor].occlusion_source == occlusion_source
            assert visibility[sensor].ignored_occluders == ignored_occluders
            assert visibility[sensor].visible_pixels > 0
            view = simulator_runtime.sensor_view(sensor)
            assert view.objects.keys() == {"sign"}
            assert view.objects["sign"].object_id == "sign"
            assert view.objects["sign"].asset.category == "stop_sign"
            np.testing.assert_array_equal(view.objects["sign"].world_from_object, world_from_object)
            assert visibility[sensor].visible_pixels == np.count_nonzero(np.any(view.data != 21, axis=2))
        assert visibility["front"].visible_pixels == visibility["second"].visible_pixels
        assert simulator_runtime.sensor_view("away").objects == {}
        simulator_runtime.remove_object("sign")
        # Visibility describes the latest capture, which still contains the removed object.
        assert simulator_runtime.object_visibility("sign") == visibility
        simulator_runtime.step(Control())
        with pytest.raises(KeyError, match="sign"):
            simulator_runtime.object_visibility("sign")


def test_object_visibility_hidden(make_object_in_front):
    # Measured scene depth in front of the object hides every pixel; the camera still has an entry.
    depth = np.full((CAMERA.height, CAMERA.width), .5)
    with MemorySimulator(sensor_depths={"front": depth}).create_runtime() as simulator_runtime:
        simulator_runtime.reset()
        original = simulator_runtime.sensor_view("front")
        asset, world_from_object = make_object_in_front(original)
        simulator_runtime.insert_object("sign", asset, world_from_object)
        simulator_runtime.step(Control())
        visibility = simulator_runtime.object_visibility("sign")
        assert visibility.keys() == {"front"} and visibility["front"].visible_pixels == 0
        assert simulator_runtime.sensor_view("front").objects.keys() == {"sign"}
