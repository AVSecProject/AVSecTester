"""Datasets stay independent of simulator execution and preserve recorded boundaries."""

import subprocess
import sys
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from avsectester.datasets.base import ActorState, FrameReference, Unavailable
from avsectester.datasets.carla import CarlaDataset, CarlaRecipe, PrescribedMotion
from avsectester.datasets.nurec import NuRecDataset
from avsectester.datasets.nuscenes import NuScenesDataset
from avsectester.closed_loop.datatypes import Control


def recipe():
    return CarlaRecipe({
        "client": {"map_name": "Town03"},
        "ego": {"autopilot": False}, "npcs": [], "lead": {"gap": 9},
    }, decision_steps=3, time_step=0.05, motion={
        identity: PrescribedMotion(controls=tuple(Control(brake=1) for _ in range(3)))
        for identity in ("ego", "lead")
    })


def test_pure_sources_import_without_simulators_or_legacy_factories():
    subprocess.run([sys.executable, "-c", """
import sys
from avsectester.datasets.carla import CarlaDataset
from avsectester.datasets.nurec import NuRecDataset
from avsectester.datasets.nuscenes import NuScenesDataset
assert not any(name.startswith(('avsectester.closed_loop.simulators', 'carla', 'avcarla', 'torch'))
               for name in sys.modules)
for cls in (CarlaDataset, NuRecDataset, NuScenesDataset):
    assert not hasattr(cls, 'make_backend')
    assert not hasattr(cls, 'make_renderer')
"""], check=True)


def test_carla_exposes_recipe_only_and_detaches_every_read():
    original = recipe()
    dataset = CarlaDataset({"town": original})
    original.scenario["lead"]["gap"] = 100
    frames = dataset.frames("town")
    assert [frame.position for frame in frames] == [0, 1, 2, 3]
    assert frames[-1].timestamp == pytest.approx(0.15)
    assert dataset.frame_data("town", frames[1]) == Unavailable("recipe_only")
    source = dataset.scenario_start("town", frames[1])
    source.payload["recipe"].scenario["lead"]["gap"] = 50
    assert dataset.native("town").scenario["lead"]["gap"] == 9
    with pytest.raises(ValueError, match="scenario's frames"):
        dataset.scenario_start("town", replace(frames[1], timestamp=999))


def test_carla_requires_explicit_motion_and_rejects_traffic_manager():
    source = recipe()
    with pytest.raises(ValueError, match="exactly these actors"):
        replace(source, motion={"ego": source.motion["ego"]})
    scenario = deepcopy(source.scenario)
    scenario["ego"]["autopilot"] = True
    with pytest.raises(ValueError, match="Traffic Manager"):
        replace(source, scenario=scenario)
    scenario["ego"]["autopilot"] = False
    scenario["npcs"] = {"count": 6}
    with pytest.raises(ValueError, match="explicit spawn"):
        replace(source, scenario=scenario)


def test_decisions_need_terminal_reference_and_never_shrink():
    dataset = CarlaDataset({"town": recipe()})
    first = dataset.frames("town")[0]
    assert len(dataset.decision_window("town", first, 3)) == 4
    with pytest.raises(ValueError, match="terminal frame"):
        dataset.decision_window("town", first, 4)
    with pytest.raises(TypeError):
        dataset.decision_window("town", first)  # no implicit default segment length
    for bad in (True, -1, 1.5):
        with pytest.raises(ValueError):
            dataset.decision_window("town", first, bad)


def test_nurec_records_cover_entire_source_and_use_nurec_scene_ids(nurec_usdz_path):
    path = nurec_usdz_path
    dataset = NuRecDataset([path], time_step=0.25)
    identity = next(dataset.scenarios()).id
    assert identity == dataset.scene_id(path) == "test"
    assert next(dataset.scenarios()).payload["usdz_path"] == path
    frames = dataset.frames(identity)
    assert [frame.timestamp for frame in frames] == [0, 0.25, 0.5]
    reference = dataset.frame_data(identity, frames[1])
    np.testing.assert_allclose(reference.actors["host"].pose[:3, 3], [11, 0, 1])
    assert reference.actors["host"].category == "vehicle"
    assert reference.actors["host"].extent == (4.0, 2.0, 2.0)
    np.testing.assert_allclose(reference.actors["host"].velocity, [4, 0, 0])
    reference.actors["host"].pose[:] = 0
    native = dataset.native(identity)
    native["rig_trajectories.json"].clear()
    assert len(dataset.frame_data(identity, frames[1]).actors) == 2
    np.testing.assert_allclose(dataset.frame_data(identity, frames[1]).actors["host"].pose[:3, 3], [11, 0, 1])
    assert dataset.native(identity)["rig_trajectories.json"]["custom_annotation"]
    source = dataset.scenario_start(identity, frames[1])
    assert source.payload["terminal_timestamp"] == 0.5
    assert source.payload["usdz_path"] == path
    assert source.payload["scene_id"] == identity
    assert source.payload["camera_id"] == dataset.camera_id
    assert source.payload["camera_alias"] == dataset.camera_alias
    assert "scene" not in source.payload
    np.testing.assert_allclose(source.payload["ego"].pose, np.eye(4))
    with pytest.raises(ValueError, match="terminal frame"):
        dataset.decision_window(identity, frames[1], 2)


@pytest.fixture
def nuscenes_fixture():
    samples = {
        "a": {"token": "a", "scene_token": "s", "next": "b", "prev": "",
              "data": {"CAM_FRONT": "ca"}, "anns": ["car", "pedestrian", "barrier"]},
        "b": {"token": "b", "scene_token": "s", "next": "", "prev": "a",
              "data": {"CAM_FRONT": "cb"}, "anns": []},
    }
    scene = {"token": "s", "name": "street", "first_sample_token": "a"}
    annotations = {
        token: {"instance_token": f"stable-{token}", "category_name": category,
                "translation": translation, "rotation": [1, 0, 0, 0], "size": [1, 2, 3]}
        for token, category, translation in (
            ("car", "vehicle.car", [10, 0, 0]),
            ("pedestrian", "human.pedestrian.adult", [-4, 0, 0]),
            ("barrier", "movable_object.barrier", [0, 20, 0]),
        )
    }
    records = {
        "scene": {"s": scene}, "sample": samples,
        "sample_data": {"c" + token: {
            "timestamp": 1000000 + i * 500000, "ego_pose_token": "e" + token,
            "calibrated_sensor_token": "camera", "filename": token + ".jpg", "width": 64, "height": 48,
        } for i, token in enumerate(samples)},
        "ego_pose": {"e" + token: {"rotation": [1, 0, 0, 0], "translation": [i * 2, 0, 0]}
                     for i, token in enumerate(samples)},
        "calibrated_sensor": {"camera": {
            "rotation": [1, 0, 0, 0], "translation": [1, 0, 1], "camera_intrinsic": np.eye(3),
        }},
        "sample_annotation": annotations,
    }
    dataset = NuScenesDataset("/source")
    dataset._nusc = SimpleNamespace(
        scene=[scene], get=lambda table, token: records[table][token],
        box_velocity=lambda token: [float("nan"), 0, 0] if token == "barrier" else [1, 2, 0],
    )
    return dataset, records


def test_nuscenes_frames_follow_scene_links_and_preserve_native_tokens(nuscenes_fixture):
    dataset, _ = nuscenes_fixture
    frames = dataset.frames("s")
    assert frames == (FrameReference("s", 0, "a", 1), FrameReference("s", 1, "b", 1.5))
    assert dataset.frame_data("s", frames[1]).sensor_data["front"].uri == "/source/b.jpg"
    expected = [dataset.frame_data("s", frame) for frame in frames]
    frame_calls = []
    original_frames = dataset.frames
    def counted_frames(scenario):
        frame_calls.append(scenario)
        return original_frames(scenario)
    dataset.frames = counted_frames
    references = dataset.scenario_start("s", frames[0]).payload["frame_data"]
    assert frame_calls == ["s"]
    for actual, per_frame in zip(references, expected):
        assert actual.frame == per_frame.frame
        assert actual.sensor_data["front"].uri == per_frame.sensor_data["front"].uri
        np.testing.assert_array_equal(actual.ego.pose, per_frame.ego.pose)
        assert actual.ego.velocity == (4.0, 0.0, 0.0)
    dataset.native("s")["samples"].clear()
    assert len(dataset.frames("s")) == 2


def test_nuscenes_actor_states_include_annotations_outside_selected_camera(nuscenes_fixture):
    dataset, _ = nuscenes_fixture
    data = dataset.frame_data("s", dataset.frames("s")[0])
    assert set(data.actors) == {"stable-car", "stable-pedestrian", "stable-barrier"}
    assert data.actors["stable-pedestrian"].category == "human.pedestrian.adult"
    assert data.actors["stable-car"].extent == (2.0, 1.0, 3.0)
    assert data.actors["stable-barrier"].velocity is None
    np.testing.assert_array_equal(data.actors["stable-pedestrian"].pose[:3, 3], [-4, 0, 0])
    data.actors["stable-car"].pose[:] = 0
    reread = dataset.frame_data("s", dataset.frames("s")[0])
    np.testing.assert_array_equal(reread.actors["stable-car"].pose[:3, 3], [10, 0, 0])
    assert not hasattr(data, "scene")
    calibration = data.sensor_data["front"].metadata
    assert calibration["camera"].width == 64
    np.testing.assert_array_equal(calibration["ego_from_sensor"][:3, 3], [1, 0, 1])
    calibration["camera"].K[:] = 0
    assert dataset.frame_data("s", dataset.frames("s")[0]).sensor_data["front"].metadata["camera"].K[0, 0] == 1


def test_actor_extent_is_optional_and_rejects_invalid_dimensions():
    assert ActorState(np.eye(4)).extent is None
    for extent in ((1, 2), (1, -2, 3), (1, float("nan"), 3)):
        with pytest.raises(ValueError, match="extent"):
            ActorState(np.eye(4), extent=extent)


def test_decision_window_accepts_recorded_timestamp_jitter(nurec_usdz_path):
    path = nurec_usdz_path
    dataset = NuRecDataset([path])
    scene = dataset.scene_id(path)
    frames = dataset.frames(scene)
    jittered = (*frames[:2], replace(frames[2], timestamp=0.7))
    dataset.frames = lambda _: jittered
    assert dataset.decision_window(scene, frames[0], 2) == jittered


@pytest.mark.parametrize("key,value", [
    ("connect_ip", "127.0.0.1"), ("connect_port", 2000), ("traffic_manager_port", 8000),
    ("rate", 20), ("synchronous", True), ("seed", 0), ("traffic_manager_seed", 0),
])
def test_recipe_rejects_simulator_runtime_settings(key, value):
    original = recipe()
    scenario = deepcopy(original.scenario)
    scenario["client"][key] = value
    with pytest.raises(ValueError, match="runtime settings"):
        replace(original, scenario=scenario)


def test_constant_control_expands_into_independent_commands():
    original = recipe()
    control = Control(throttle=.3)
    result = replace(original, motion={
        actor: PrescribedMotion(constant_control=control) for actor in original.motion
    })
    controls = result.motion["ego"].controls
    assert len(controls) == original.decision_steps
    assert all(value == Control(throttle=.3) for value in controls)
    assert result.motion["ego"].constant_control is None
    control.throttle = .8
    controls[0].throttle = .9
    assert controls[1].throttle == .3
    assert result.motion["lead"].controls[0].throttle == .3


@pytest.mark.parametrize("motion", [
    PrescribedMotion(),
    PrescribedMotion(controls=(Control(),), constant_control=Control()),
    PrescribedMotion(trajectory=(ActorState(np.eye(4)),), constant_control=Control()),
])
def test_recipe_requires_exactly_one_motion_form(motion):
    original = recipe()
    with pytest.raises(ValueError, match="exactly one"):
        replace(original, motion={actor: motion for actor in original.motion})


def test_actor_bbox_center_is_local_and_validated():
    state = ActorState(np.eye(4), extent=(4, 2, 1), box_center=(1, 0, .5))
    assert state.box_center == (1., 0., .5)
    for center in ((1, 2), (0, float("nan"), 0)):
        with pytest.raises(ValueError, match="center"):
            ActorState(np.eye(4), box_center=center)


def test_recipe_npcs_require_explicit_spawns():
    original = recipe()
    scenario = deepcopy(original.scenario)
    scenario["npcs"] = [{"vehicle": "vehicle.tesla.model3"}]
    with pytest.raises(ValueError, match="explicit spawn"):
        replace(original, scenario=scenario)


def test_nurec_camera_geometry_and_video_times_are_independent(nurec_usdz_path):
    dataset = NuRecDataset([nurec_usdz_path], time_step=.1)
    lens, transform = dataset.camera_geometry(nurec_usdz_path)
    assert (lens.width, lens.height) == (640, 480)
    assert lens.angle_to_pixeldist == (0., 200.)
    np.testing.assert_allclose(dataset.camera_timestamps(nurec_usdz_path), [0, .25, .5])
    transform[:] = 0
    lens.cx = -1
    pristine, pose = dataset.camera_geometry(nurec_usdz_path)
    assert pristine.cx == 320
    assert pose[3, 3] == 1
    timeline = dataset.frames(dataset.scene_id(nurec_usdz_path))
    assert timeline[1].timestamp == .1  # video frame 1 is .25s, not decision 1


def test_actor_box_transform_and_pose_preserve_local_offset_and_rotation():
    from avsectester.geometry.poses import rotation_matrix

    world = np.eye(4)
    world[:3, :3] = rotation_matrix((10, 20, 30))
    state = ActorState(world, extent=(4, 2, 1), box_center=(1, 0, .5), box_rotation=(5, -3, 20))
    local = np.eye(4)
    local[:3, :3] = rotation_matrix((5, -3, 20))
    local[:3, 3] = (1, 0, .5)
    np.testing.assert_allclose(state.box_transform(), world @ local)
    np.testing.assert_allclose(state.box_pose().transform, world @ local)
    missing = ActorState(world, box_center=(1, 0, .5))
    np.testing.assert_allclose(missing.box_transform()[:3, 3], (world @ [1, 0, .5, 1])[:3])
    assert missing.box_pose().extent is None
