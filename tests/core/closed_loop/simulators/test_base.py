"""The base simulator runtime's public capture, replay and ownership contract."""

from dataclasses import replace
from contextlib import ExitStack
from threading import Lock

import numpy as np
import pytest

from avsectester.data_types import Control
from avsectester.rendering import MeshAsset, MeshPart, PlacedObject, composite_objects
from avsectester.closed_loop.hooks import Hook, HookSet
from avsectester.closed_loop.simulators.base import DeploymentPlan, SensorGeometry, UnsupportedStartError
from avsectester.closed_loop.timing import TimingConfiguration
from avsectester.closed_loop.simulators.nurec import NuRecRenderer, NuRecSimulator
from avsectester.closed_loop.simulators.recorded import RecordedReplaySimulator
from tests.fakes import CAMERA, CAM_FROM_WORLD, ConstantAVStack, MemoryDataset, MemorySimulator, actor_state


def test_validate_start():
    simulator = MemorySimulator()
    dataset = MemoryDataset()
    start = dataset.scenario_start("scene", dataset.frames("scene")[0])
    simulator.validate_start(None)
    simulator.validate_start(start)
    with pytest.raises(TypeError, match="ScenarioStart"):
        simulator.validate_start(object())
    with pytest.raises(ValueError, match="requires memory ScenarioStart"):
        simulator.validate_start(replace(start, kind="unsupported"))
    with pytest.raises(ValueError, match="time_step"):
        simulator.validate_start(replace(start, payload={**start.payload, "time_step": .2}))


def test_validate_startup():
    dataset = MemoryDataset()
    start = dataset.scenario_start("scene", dataset.frames("scene")[4])
    simulator = MemorySimulator()
    simulator.validate_startup(None, 0)
    simulator.validate_startup(start, 200000)
    with pytest.raises(UnsupportedStartError, match="does not cover"):
        simulator.validate_startup(start, 300000)
    with pytest.raises(ValueError, match="does not support force_gt_duration_us"):
        MemorySimulator(can_replay=False).validate_timing(TimingConfiguration(force_gt_duration_us=100000))


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
        def reject_step(value, context):
            raise RuntimeError("step post failed")

        hooks = HookSet([Hook("simulator.step.post", reject_step)])
        with hooks.session(simulator_runtime, ConstantAVStack()).activate():
            with pytest.raises(RuntimeError, match="step post failed"):
                simulator_runtime.step(Control())
        with pytest.raises(RuntimeError, match="Reset"):
            simulator_runtime.observe()


def test_advance_and_capture():
    with MemorySimulator().create_runtime() as runtime:
        runtime.reset()
        runtime.advance_interval(100000, Control(throttle=.5), report_times_us=(50000,))
        assert runtime.clock.time_us == 0
        with pytest.raises(RuntimeError, match="observing"):
            runtime.observe()
        runtime.capture_at(50000, ())
        middle = runtime.state()
        assert middle.t == .05 and middle.ego_speed == 1.025
        assert middle.ego.pose[0, 3] == pytest.approx(.0525)
        captured = runtime.capture_at(100000, ("front",))
        assert runtime.state().ego_speed == 1.05
        assert runtime.clock.time_us == 100000
        assert captured.frame == runtime.clock.frame == 1
        assert set(captured.sensor_data) == {"front"}
        captured.sensor_data["front"][:] = 0
        assert runtime.capture_at(100000).sensor_data["front"].min() == 21
        assert runtime.sensor_view("front").time == captured.t
        with pytest.raises(ValueError, match="next decision"):
            runtime.advance_interval(250000, Control())
        with pytest.raises(ValueError, match="current time"):
            runtime.capture_at(0)


def test_pose_only_capture():
    with MemorySimulator(captures_between_decisions=False).create_runtime() as runtime:
        runtime.reset()
        runtime.advance_interval(100000, Control(throttle=.5), report_times_us=(50000,))
        observation = runtime.capture_at(50000, ())
        assert observation.sensor_data == {} and observation.t == .05
        assert observation.ego_speed == runtime.state().ego_speed == 1.025
        with pytest.raises(ValueError, match="cannot acquire between decisions"):
            runtime.capture_at(50000, ("front",))
        assert runtime.clock.time_us == 50000
        runtime.capture_at(100000)
        assert runtime.observe().sensor_data.keys() == {"front"}


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


def test_start_replay():
    data = MemoryDataset()
    frames = data.frames("scene")
    with MemorySimulator().create_runtime() as simulator_runtime:
        simulator_runtime.start_replay(data.scenario_start("scene", frames[2]))
        assert simulator_runtime.observe().frame == 2
        assert simulator_runtime.can_replay
        simulator_runtime.reset()
        assert simulator_runtime.observe().frame == 0


def test_advance_replay():
    data = MemoryDataset()
    frames = data.frames("scene")
    with MemorySimulator().create_runtime() as simulator_runtime:
        with pytest.raises(RuntimeError, match="initialized runtime"):
            simulator_runtime.advance_replay(1)
        simulator_runtime.start_replay(data.scenario_start("scene", frames[1]))
        simulator_runtime.advance_replay(2)
        assert simulator_runtime.observe().frame == 3
        assert simulator_runtime.clock.t == pytest.approx(frames[3].timestamp)
        simulator_runtime.advance_interval(simulator_runtime.clock.time_us + 100000, Control())
        with pytest.raises(RuntimeError, match="pending interval endpoint"):
            simulator_runtime.advance_replay(1)
        with pytest.raises(ValueError, match="nonnegative integer"):
            simulator_runtime.advance_replay(-1)
    with MemorySimulator(can_replay=False).create_runtime() as simulator_runtime:
        with pytest.raises(NotImplementedError, match="replay"):
            simulator_runtime.advance_replay(1)


def test_claim_world():
    simulator = MemorySimulator(world_key=("memory", "shared-world"))
    owner, other_owner = object(), object()
    with simulator.create_runtime() as first, simulator.create_runtime() as second:
        assert first.world_key == second.world_key
        with first.claim_world(owner):
            with first.claim_world(owner), second.claim_world(owner):
                with first.exclusive():
                    first.reset()
                    assert first.observe().frame == 0
            with pytest.raises(RuntimeError, match="already in use"):
                with second.claim_world(other_owner):
                    pytest.fail("an independent owner acquired a claimed world")
        with second.claim_world(other_owner), second.exclusive():
            second.reset()
            assert second.observe().frame == 0
        with first.exclusive():
            first.reset()

        # Runner cleanup can close a session's outer claim before exclusive() exits.
        claim_resources = ExitStack()
        claim_resources.enter_context(first.claim_world(owner))
        with first.exclusive():
            claim_resources.close()
        with second.claim_world(owner):
            with pytest.raises(RuntimeError, match="already in use"):
                with first.exclusive():
                    pytest.fail("an expired claim was reused by another operation")


def test_exclusive():
    simulator = MemorySimulator(world_key=("memory", "shared-world"))
    with simulator.create_runtime() as first, simulator.create_runtime() as second:
        with first.exclusive():
            with pytest.raises(RuntimeError, match="already"):
                with second.exclusive():
                    pytest.fail("two independent users acquired one world")
            with pytest.raises(RuntimeError, match="active operation"):
                with first.exclusive():
                    pytest.fail("one runtime acquired two operations")
        with second.exclusive():
            second.reset()
            assert second.observe().frame == 0


def test_insert_object(make_object_in_front):
    class LockedHarmonizer:
        def __init__(self):
            self.lock = Lock()

        def __call__(self, composite, mask, background):
            with self.lock:
                return composite

    points = np.array([[3., -1., .5, .8]], np.float32)
    with MemorySimulator(extra_sensors={"lidar": points}).create_runtime() as runtime:
        runtime.reset()
        clean = runtime.sensor_view("front")
        assert clean.composition is None
        assert clean.object_masks == {}
        asset, pose = make_object_in_front(clean)
        runtime.insert_object("sign", asset, pose)
        world_object = runtime.state().objects["sign"]
        np.testing.assert_array_equal(world_object.pose, pose)
        assert world_object.velocity == (0., 0., 0.)
        assert world_object.category == asset.category and world_object.extent == asset.extent
        assert world_object.box_center == (0., 0., asset.extent[2] / 2)
        np.testing.assert_array_equal(runtime.observe().sensor_data["front"], clean.data)
        with pytest.raises(KeyError, match="next capture"):
            runtime.object_visibility("sign")
        with pytest.raises(ValueError, match="already inserted"):
            runtime.insert_object("sign", asset, pose)
        part = MeshPart(np.array([[0., -1., 0.], [0., 1., 0.], [0., 0., 2.]]),
                        np.array([[0, 1, 2]]), color=np.full(4, 255, np.uint8))
        with pytest.raises(TypeError, match="MemorySimulator does not support asset type MeshAsset"):
            runtime.insert_object("mesh", MeshAsset((part,), (0., 2., 2.), "vehicle"), pose)
        assert "mesh" not in runtime.state().objects
        runtime.step(Control())
        composed = runtime.sensor_view("front")
        assert composed.occlusion_tolerance_m == runtime.simulator.occlusion_tolerance_m
        with MemorySimulator(extra_sensors={"lidar": points}).create_runtime() as clean_runtime:
            clean_runtime.reset()
            clean_runtime.step(Control())
            capture = clean_runtime.sensor_view("front")
        expected = composite_objects(
            capture.data, capture.camera, capture.cam_from_world,
            (PlacedObject("sign", asset, pose),), occluders=capture.actors,
            occlusion_tolerance_m=runtime.simulator.occlusion_tolerance_m)
        np.testing.assert_array_equal(composed.data, expected.image)
        assert composed.composition is not None
        np.testing.assert_array_equal(composed.composition.plan.apply(
            composed.composition.background).image, composed.data)
        np.testing.assert_array_equal(runtime.observe().sensor_data["front"], expected.image)
        with pytest.raises(ValueError, match="already"):
            composed.composite_plan(composed.objects.values())
        np.testing.assert_array_equal(runtime.observe().sensor_data["lidar"], points)
        runtime.remove_object("sign")
        assert "sign" not in runtime.state().objects
        assert "sign" in runtime.sensor_view("front").objects
        runtime.step(Control())
        assert "sign" not in runtime.sensor_view("front").objects
        with pytest.raises(KeyError, match="Unknown"):
            runtime.remove_object("missing")
    with MemorySimulator(extra_sensors={"lidar": points}).create_runtime() as simulator_runtime:
        simulator_runtime.reset()
        asset, world_from_object = make_object_in_front(simulator_runtime.sensor_view("front"))
        harmonizer = LockedHarmonizer()
        simulator_runtime.insert_object("harmonized", asset, world_from_object,
                                        harmonizer=harmonizer)
        simulator_runtime.step(Control())
        # Views share immutable placed objects, so a harmonizer holding a lock is never copied.
        assert simulator_runtime.sensor_view("front").objects["harmonized"].harmonizer is harmonizer

    with MemorySimulator(can_insert_objects=False).create_runtime() as simulator_runtime:
        simulator_runtime.reset()
        asset, world_from_object = make_object_in_front(simulator_runtime.sensor_view("front"))
        with pytest.raises(NotImplementedError, match="cannot insert objects"):
            simulator_runtime.insert_object("sign", asset, world_from_object)
        with pytest.raises(NotImplementedError, match="cannot insert objects"):
            simulator_runtime.remove_object("sign")


def test_insert_object_with_depth(make_object_in_front):
    depth = np.full((CAMERA.height, CAMERA.width), np.inf)
    with MemorySimulator(sensor_depths={"front": depth}).create_runtime() as runtime:
        runtime.reset()
        original = runtime.sensor_view("front")
        asset, pose = make_object_in_front(original)
        runtime.insert_object("sign", asset, pose)
        runtime.step(Control())
        inserted = runtime.sensor_view("front")
        np.testing.assert_array_equal(original.depth, depth)
        assert np.isfinite(inserted.depth).any()
        assert inserted.depth.shape == original.depth.shape
        with MemorySimulator(sensor_depths={"front": depth}).create_runtime() as clean_runtime:
            clean_runtime.reset()
            clean_runtime.step(Control())
            capture = clean_runtime.sensor_view("front")
        expected = composite_objects(
            capture.data, capture.camera, capture.cam_from_world,
            (PlacedObject("sign", asset, pose),), scene_depth=capture.depth,
            occluders=capture.actors, occlusion_tolerance_m=inserted.occlusion_tolerance_m)
        np.testing.assert_array_equal(inserted.depth, expected.depth)


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
            assert view.object_masks.keys() == view.objects.keys()
            assert view.object_masks["sign"].dtype == bool
            assert view.object_masks["sign"].shape == view.data.shape[:2]
            assert visibility[sensor].visible_pixels == np.count_nonzero(view.object_masks["sign"])
            assert view.objects["sign"].object_id == "sign"
            assert view.objects["sign"].asset.category == "stop_sign"
            np.testing.assert_array_equal(view.objects["sign"].world_from_object, world_from_object)
            assert visibility[sensor].visible_pixels == np.count_nonzero(
                np.any(view.data != view.composition.background, axis=2))
        assert visibility["front"].visible_pixels == visibility["second"].visible_pixels
        assert simulator_runtime.sensor_view("away").objects == {}
        simulator_runtime.remove_object("sign")
        # Visibility describes the latest capture, which still contains the removed object.
        assert simulator_runtime.object_visibility("sign") == visibility
        simulator_runtime.step(Control())
        with pytest.raises(KeyError, match="sign"):
            simulator_runtime.object_visibility("sign")

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
        assert not simulator_runtime.sensor_view("front").object_masks["sign"].any()


def test_observe_recorded_at(make_object_in_front):
    dataset = MemoryDataset()
    start = dataset.scenario_start("scene", dataset.frames("scene")[3])
    with MemorySimulator().create_runtime() as runtime:
        with pytest.raises(NotImplementedError, match="recorded acquisition history"):
            runtime.observe_recorded_at(-100000)
        runtime.reset(start)
        asset, world_from_object = make_object_in_front(runtime.sensor_view("front"))
        clean_capture = runtime.observe_recorded_at(-100000)
        clean = clean_capture.observation
        pose_capture = runtime.observe_recorded_at(-100000, ())
        pose_only = pose_capture.observation
        assert pose_only.sensor_data == {} and pose_only.calibration == {}
        assert pose_only.vehicle_state == clean.vehicle_state
        assert pose_capture.sensor_geometry == {} and pose_capture.object_visibility == {}
        assert clean_capture.sensor_geometry.keys() == {"front"}
        assert clean_capture.physical_state.t == clean.t
        hooks = HookSet([Hook("simulator.setup", lambda value, context: context.simulator_runtime.insert_object(
            "sign", asset, world_from_object)), Hook("sensors.post", lambda value, context: value)])
        with hooks.session(runtime, ConstantAVStack()).activate():
            runtime.reset(start)
            state, current = runtime.state(), runtime.observe()
            captures = tuple(runtime.observe_recorded_at(time) for time in (-300000, -200000, -100000))
            past = tuple(item.observation for item in captures)
            assert captures[-1].physical_state.objects.keys() == {"sign"}
            assert captures[-1].object_visibility["sign"].keys() == {"front"}
            assert [item.frame for item in past] == [-3, -2, -1]
            assert [item.t for item in past] == [-.3, -.2, -.1]
            expected = composite_objects(clean.sensor_data["front"], CAMERA,
                captures[-1].sensor_geometry["front"].cam_from_world,
                (PlacedObject("sign", asset, world_from_object),))
            np.testing.assert_array_equal(past[-1].sensor_data["front"], expected.image)
            pristine = past[-1].sensor_data["front"].copy()
            past[-1].sensor_data["front"][:] = 255
            np.testing.assert_array_equal(runtime.observe_recorded_at(-100000).observation.sensor_data["front"], pristine)
            np.testing.assert_array_equal(runtime.observe().sensor_data["front"], current.sensor_data["front"])
            np.testing.assert_array_equal(runtime.state().ego.pose, state.ego.pose)
            assert runtime.clock.t == state.t
            with pytest.raises(ValueError, match="precedes available source history"):
                runtime.observe_recorded_at(-400000)
    geometries = {sensor: SensorGeometry(sensor, CAMERA, CAM_FROM_WORLD)
                  for sensor in ("front", "right")}
    with MemorySimulator(camera_geometries=geometries).create_runtime() as runtime:
        runtime.reset(start)
        capture = runtime.observe_recorded_at(-100000, ("right", "front"))
        assert list(capture.observation.sensor_data) == ["front", "right"]
        assert list(capture.sensor_geometry) == ["right", "front"]
    with MemorySimulator(can_observe_past=False).create_runtime() as runtime:
        runtime.reset(start)
        with pytest.raises(NotImplementedError, match="recorded acquisition history"):
            runtime.observe_recorded_at(-100000)


@pytest.mark.parametrize(("constructor", "options"), [
    (NuRecSimulator, {"renderer": NuRecRenderer(cameras=["front"]), "time_step": .1,
                      "depth": "none", "physics_update_mode": "none"}),
    (RecordedReplaySimulator, {}),
])
def test_camera_alias_validation(constructor, options):
    configured = constructor(**options, camera_aliases={"front": "forward"})
    assert configured.camera_aliases == {"front": "forward"}
    with pytest.raises(ValueError, match="nonempty native camera names"):
        constructor(**options, camera_aliases={"front": ""})
    with pytest.raises(ValueError, match="unique output names"):
        constructor(**options, camera_aliases={"front": "same", "side": "same"})
