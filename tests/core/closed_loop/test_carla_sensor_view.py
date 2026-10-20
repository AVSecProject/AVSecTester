"""CARLA exposes same-frame optical geometry directly, without selection-layer adapters."""
from types import SimpleNamespace

import numpy as np
import pytest

from avsectester.closed_loop.datatypes import Observation, WorldSnapshot
from avsectester.closed_loop.simulators.carla import CarlaSimulator


def native_actor(actor_id, xyz, *, box_center=(0., 0., .5)):
    transform = np.eye(4)
    transform[:3, 3] = xyz
    return SimpleNamespace(
        id=actor_id, type_id="vehicle.test",
        get_transform=lambda: SimpleNamespace(get_matrix=lambda: transform.copy()),
        get_velocity=lambda: SimpleNamespace(x=1., y=2., z=0.),
        bounding_box=SimpleNamespace(
            extent=SimpleNamespace(x=2., y=1., z=.7),
            location=SimpleNamespace(x=box_center[0], y=box_center[1], z=box_center[2]),
        ),
    )


def runtime_with_camera(depth=False):
    runtime = CarlaSimulator({}, depth_camera="front" if depth else None).create_runtime()
    camera_inverse = np.eye(4)
    camera_inverse[:3, 3] = [-1., 0., -2.]
    rgb = SimpleNamespace(
        source_identifier="front-0", frame0=10, imsize=(4, 6),
        P=np.array([[3., 0., 3., 0.], [0., 3., 2., 0.], [0., 0., 1., 0.]]),
        object=SimpleNamespace(type_id="sensor.camera.rgb", get_transform=lambda: SimpleNamespace(
            get_inverse_matrix=lambda: camera_inverse.copy())),
    )
    ego = native_actor(1, (5., 2., 0.))
    host = native_actor(2, (10., 3., 0.))
    runtime.ego = SimpleNamespace(actor=ego, sensors={"front": rgb})
    runtime.npcs = [SimpleNamespace(actor=host)]
    runtime._capture = Observation(2., 12, sensor_data={"front-0": SimpleNamespace(
        rgb_image=np.full((4, 6, 3), 60, np.uint8), frame=2)})
    runtime._world_initialized = True
    runtime._state = lambda: WorldSnapshot(2., 12)
    if depth:
        runtime.depth_frame = 12
        runtime.depth = np.full((4, 6), 9.)
    return runtime


def test_sensor_view_uses_native_camera_pose_canonical_world_and_box_centres():
    runtime = runtime_with_camera(depth=True)
    view = runtime.sensor_view("front")
    # World (11,-2,2) corresponds to CARLA (11,2,2); relative native camera
    # coordinates are (10,2,0), hence optical coordinates (2,0,10).
    np.testing.assert_allclose(view.cam_from_world @ [11., -2., 2., 1.], [2., 0., 10., 1.])
    np.testing.assert_allclose(view.camera.project(np.array([[2., 0., 10.]])), [[3.6, 2.]])
    np.testing.assert_allclose(view.actors["npc:0"].transform[:3, 3], [10., -3., .5])
    np.testing.assert_allclose(view.ego.transform[:3, 3], [5., -2., .5])
    assert view.ego.extent == (4., 2., 1.4)
    assert view.frame == 12 and view.time == 2.
    assert view.sensor == "front-0" and view.data.shape == (4, 6, 3)
    view.data[:] = 0
    view.depth[:] = 0
    assert runtime.sensor_view("front-0").data.min() == 60
    assert runtime.depth.min() == 9.


def test_sensor_view_rejects_misaligned_rgb_and_depth():
    runtime = runtime_with_camera(depth=True)
    runtime._capture.sensor_data["front-0"].frame = 1
    with pytest.raises(ValueError, match="same frame"):
        runtime.sensor_view("front")
    runtime._capture.sensor_data["front-0"].frame = 2
    runtime.depth_frame = 11
    with pytest.raises(RuntimeError, match="Depth and RGB"):
        runtime.sensor_view("front")


def test_unknown_camera_name_does_not_select_the_only_camera():
    simulator_runtime = runtime_with_camera()
    with pytest.raises(ValueError, match='specific RGB camera'):
        simulator_runtime.sensor_view('typo')


def test_camera_geometry_does_not_read_or_copy_sensor_buffers():
    simulator_runtime = runtime_with_camera()
    simulator_runtime.observe = lambda: pytest.fail('geometry must not copy an observation')
    simulator_runtime.copy_observation = lambda *_: pytest.fail('geometry must not copy an observation')
    geometry = simulator_runtime.sensor_geometry('front')
    assert geometry.sensor == 'front-0'
    np.testing.assert_allclose(geometry.cam_from_world[3], [0, 0, 0, 1])


def test_actor_extent_and_rotated_offset_box_pose_are_canonical():
    from avsectester.closed_loop.simulators.carla import _actor_state
    from avsectester.geometry.poses import rotation_matrix

    actor = native_actor(1, (30, 40, 2), box_center=(.3, .2, .8))
    actor.bounding_box.rotation = SimpleNamespace(roll=5., pitch=12., yaw=20.)
    world = np.eye(4)
    world[:3, :3] = rotation_matrix((10, -20, 35))
    world[:3, 3] = (30, 40, 2)
    actor.get_transform = lambda: SimpleNamespace(get_matrix=lambda: world)
    state = _actor_state(actor, actor.type_id)
    local = np.eye(4)
    local[:3, :3] = rotation_matrix((-5., -12., 20.))
    local[:3, 3] = (.3, .2, .8)
    reflection = np.diag([1., -1., 1., 1.])
    np.testing.assert_allclose(state.box_pose().transform, reflection @ world @ local @ reflection)
    assert state.extent == (4., 2., 1.4)


def depth_image(value, frame=1):
    code = int(round(value / 1000 * (2**24 - 1)))
    pixels = np.array([[[code >> 16, (code >> 8) & 255, code & 255, 255]]], np.uint8)
    return SimpleNamespace(raw_data=pixels.tobytes(), height=1, width=1, frame=frame)


@pytest.mark.parametrize('value', [0, 1, 17.25, 999, 1000])
def test_depth_decode_preserves_channel_order_and_metres(value):
    from avsectester.closed_loop.simulators.carla import decode_carla_depth

    assert decode_carla_depth(depth_image(value))[0, 0] == pytest.approx(value, abs=.00003)


def test_pinhole_registration_sets_lens_before_spawn_and_reuses_the_class(monkeypatch):
    import sys
    from avsectester.closed_loop.simulators.carla import register_pinhole_camera

    class Registry:
        def __init__(self):
            self.module_dict = {}
        def register_module(self):
            def register(cls):
                self.module_dict[cls.__name__] = cls
                return cls
            return register
    class RgbSensor:
        def __init__(self, *, do_spawn):
            assert not do_spawn
            self.lens = {}
            self.bp = SimpleNamespace(set_attribute=lambda key, value: self.lens.update({key: value}))
            self.spawned = False
        def spawn(self):
            assert self.lens == {'lens_k': '0', 'lens_kcube': '0', 'lens_circle_multiplier': '0'}
            self.spawned = True
    registry = Registry()
    monkeypatch.setitem(sys.modules, 'avcarla.config', SimpleNamespace(CARLA=registry))
    monkeypatch.setitem(sys.modules, 'avcarla.sensors', SimpleNamespace(CarlaRgbCamera=RgbSensor))
    camera_type = register_pinhole_camera()
    assert register_pinhole_camera() is camera_type
    assert camera_type().spawned
    assert not camera_type(do_spawn=False).spawned


def test_depth_capture_discards_older_frames_and_rejects_future_frames(monkeypatch):
    from queue import Queue
    from avsectester.closed_loop.simulators.carla import CarlaSimulatorRuntime

    monkeypatch.setattr(CarlaSimulatorRuntime, '_capture_sensors', lambda _: Observation(1., 12))
    simulator_runtime = CarlaSimulator({}).create_runtime()
    simulator_runtime._depth_queue = Queue()
    simulator_runtime._depth_queue.put(depth_image(3, frame=11))
    simulator_runtime._depth_queue.put(depth_image(8, frame=12))
    observation = simulator_runtime._observe()
    assert observation.frame == simulator_runtime.depth_frame == 12
    assert simulator_runtime.depth[0, 0] == pytest.approx(8, abs=.00003)
    simulator_runtime._depth_queue.put(depth_image(10, frame=13))
    with pytest.raises(RuntimeError, match='frames differ'):
        simulator_runtime._observe()
