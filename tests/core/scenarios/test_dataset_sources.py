"""Source providers stay independent of simulator execution and preserve source boundaries."""

import subprocess
import sys
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import test_nurec_metadata

from avsectester.datasets.base import FrameRef, Unavailable
from avsectester.datasets.carla import CarlaDataset, CarlaRecipe, PrescribedMotion
from avsectester.datasets.nurec import NuRecDataset
from avsectester.datasets.nuscenes import NuScenesDataset
from avsectester.plane import Control
from avsectester.scenarios.scene import EgoState, SceneGT

recorded_dataset = test_nurec_metadata.recorded_dataset


def recipe():
    return CarlaRecipe({
        "client": {"synchronous": True, "rate": 20, "seed": 7},
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
assert not any(name.startswith(('avsectester.simulators', 'carla', 'avcarla', 'torch'))
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
    assert dataset.reference("town", frames[1]) == Unavailable("recipe_only")
    source = dataset.source_data("town", frames[1])
    source.payload["recipe"].scenario["lead"]["gap"] = 50
    assert dataset.native("town").scenario["lead"]["gap"] == 9
    assert dataset.sensor_data("town", frames[1], "front") is None
    with pytest.raises(ValueError, match="source sequence"):
        dataset.source_data("town", replace(frames[1], timestamp=999))


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
    with pytest.raises(ValueError, match="terminal reference"):
        dataset.decision_window("town", first, 4)
    with pytest.raises(TypeError):
        dataset.decision_window("town", first)  # no implicit default segment length
    for bad in (True, -1, 1.5):
        with pytest.raises(ValueError):
            dataset.decision_window("town", first, bad)


def test_nurec_records_cover_entire_source_independently_of_legacy_keyframe(recorded_dataset):
    _, path = recorded_dataset
    dataset = NuRecDataset([path], keyframe=1, time_step=0.25)
    identity = next(dataset.scenarios()).id
    frames = dataset.frames(identity)
    assert [frame.timestamp for frame in frames] == [0, 0.25, 0.5]
    reference = dataset.reference(identity, frames[1])
    np.testing.assert_allclose(reference.actors["host"].pose[:3, 3], [11, 0, 1])
    reference.scene.objects.clear()
    native = dataset.native(identity)
    native["rig_trajectories.json"].clear()
    assert len(dataset.reference(identity, frames[1]).scene.objects) == 2
    assert dataset.native(identity)["rig_trajectories.json"]["custom_annotation"]
    source = dataset.source_data(identity, frames[1])
    assert source.payload["terminal_timestamp"] == 0.5
    assert source.payload["scene"].source["timestamp_us"] == 250000
    with pytest.raises(ValueError, match="terminal reference"):
        dataset.decision_window(identity, frames[1], 2)


def test_nuscenes_frames_follow_scene_links_and_preserve_native_tokens():
    samples = {
        "a": {"token": "a", "scene_token": "s", "next": "b", "data": {"CAM_FRONT": "ca"}},
        "b": {"token": "b", "scene_token": "s", "next": "", "data": {"CAM_FRONT": "cb"}},
    }
    scene = {"token": "s", "name": "street", "first_sample_token": "a"}
    records = {"scene": {"s": scene}, "sample": samples,
               "sample_data": {"ca": {"timestamp": 1000000}, "cb": {"timestamp": 1500000}}}
    dataset = NuScenesDataset("/source")
    dataset._nusc = SimpleNamespace(scene=[scene], get=lambda table, token: records[table][token])
    dataset._sample_scene = lambda sample, index: SceneGT(
        index, 1 + index * 0.5, EgoState(pose=np.eye(4)), {}, [],
        source={"image_path": f"/{sample['token']}.jpg", "sample_token": sample["token"]},
    )
    frames = dataset.frames("s")
    assert frames == (FrameRef("s", 0, "a", 1), FrameRef("s", 1, "b", 1.5))
    assert dataset.sensor_data("s", frames[1], "front").uri == "/b.jpg"
    expected = [dataset.reference("s", frame) for frame in frames]
    frame_calls = []
    original_frames = dataset.frames
    def counted_frames(scenario):
        frame_calls.append(scenario)
        return original_frames(scenario)
    dataset.frames = counted_frames
    references = dataset.source_data("s", frames[0]).payload["references"]
    assert frame_calls == ["s"]
    for actual, per_frame in zip(references, expected):
        assert actual.frame == per_frame.frame and actual.scene.source == per_frame.scene.source
        np.testing.assert_array_equal(actual.actors["ego"].pose, per_frame.actors["ego"].pose)
    dataset.native("s")["samples"].clear()
    assert len(dataset.frames("s")) == 2
    assert not hasattr(dataset, "scenes")  # the source interface has no legacy keyframe iteration


def test_legacy_nuscenes_keeps_global_sample_range_order_and_annotation_ids():
    from avsectester.scenarios.datasets.nuscenes import NuScenesDataset as LegacyNuScenes
    from avsectester.scenarios.scene import ObjectGT

    # Two scenes; the SDK's global sample list interleaves them, as nusc.sample may.
    samples = [
        {"token": "a0", "scene_token": "s", "next": "a1", "data": {"CAM_FRONT": "ca0"}},
        {"token": "b0", "scene_token": "t", "next": "", "data": {"CAM_FRONT": "cb0"}},
        {"token": "a1", "scene_token": "s", "next": "", "data": {"CAM_FRONT": "ca1"}},
    ]
    scenes = [{"token": "s", "name": "one", "first_sample_token": "a0"},
              {"token": "t", "name": "two", "first_sample_token": "b0"}]
    records = {
        "scene": {scene["token"]: scene for scene in scenes},
        "sample": {sample["token"]: sample for sample in samples},
        "sample_data": {f"c{s['token']}": {"timestamp": 1000000 * (i + 1)} for i, s in enumerate(samples)},
        "sample_annotation": {f"ann-{s['token']}": {"instance_token": f"inst-{s['token']}"}
                              for s in samples},
    }
    nusc = SimpleNamespace(scene=scenes, sample=samples, box_velocity=lambda token: (0, 0, 0),
                           get=lambda table, token: records[table][token])

    def sample_scene(sample, index):
        obj = ObjectGT(track_id=f"ann-{sample['token']}", category="vehicle", center=(5, 0, 0),
                       extent=(4, 2, 1.5), yaw=0, pose=np.eye(4))
        return SceneGT(index, float(index), EgoState(pose=np.eye(4)), {}, [obj],
                       source={"image_path": f"/{sample['token']}.jpg", "sample_token": sample["token"]})

    legacy = LegacyNuScenes("/source", max_samples=2)
    legacy._nusc, legacy._sample_scene = nusc, sample_scene
    yielded = list(legacy.scenes())
    # The original global range and order, global frame indices and annotation-token IDs.
    assert [scene.source["sample_token"] for scene in yielded] == ["a0", "b0"]
    assert [scene.frame for scene in yielded] == [0, 1]
    assert [scene.objects[0].track_id for scene in yielded] == ["ann-a0", "ann-b0"]
    # One recorded state per legacy keyframe, so DatasetFilter's origin check holds.
    assert legacy.initial_sequence(yielded[1], 3) == (yielded[1],)
    assert not hasattr(legacy, "make_backend")

    # max_samples never truncates the source interface, which uses stable instance IDs.
    source = NuScenesDataset("/source")
    source._nusc, source._sample_scene = nusc, sample_scene
    frames = source.frames("s")
    assert [frame.frame_id for frame in frames] == ["a0", "a1"]
    assert legacy.frames("s") == frames
    assert "inst-a1" in source.reference("s", frames[1]).actors


def test_decision_window_accepts_recorded_timestamp_jitter(recorded_dataset):
    _, path = recorded_dataset
    dataset = NuRecDataset([path])
    frames = dataset.frames(path)
    jittered = (*frames[:2], replace(frames[2], timestamp=0.7))
    dataset.frames = lambda _: jittered
    assert dataset.decision_window(path, frames[0], 2) == jittered
