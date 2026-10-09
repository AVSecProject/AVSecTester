"""Recorded NuRec states used for selection and host attachment, without a renderer."""

import json
import math
import zipfile

import numpy as np
import pytest

from avsectester.scenarios.datasets.nurec import NuRecDataset


@pytest.fixture
def recorded_dataset(tmp_path):
    camera = "camera_front_wide_120fov@test"
    optical_to_rig = [[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 1], [0, 0, 0, 1]]
    timestamps = [[0, 1], [250000, 250001], [500000, 500001]]
    identity = np.eye(4).tolist()
    rig = {
        "camera_calibrations": {
            camera: {
                "camera_model": {
                    "parameters": {
                        "resolution": [640, 480],
                        "principal_point": [320, 240],
                        "angle_to_pixeldist_poly": [0, 200],
                        "max_angle": 1.5,
                    }
                },
                "T_sensor_rig": optical_to_rig,
            },
        },
        "rig_trajectories": [
            {
                "sequence_id": "test",
                "cameras_frame_timestamps_us": {camera: timestamps},
                "cameras_frame_T_rig_worlds": {camera: [[identity, identity]] * 3},
            }
        ],
        "custom_annotation": {"weather": "cloudy"},
    }
    tracks = {
        "test": {
            "tracks_data": {
                "tracks_id": ["host", "obstacle"],
                "tracks_label_class": ["automobile", "protruding_object"],
                "tracks_timestamps_us": [[0, 500000], [0, 500000]],
                "tracks_poses": [
                    [[10, 0, 1, 0, 0, 0, 1], [12, 0, 1, 0, 0, 1, 0]],
                    [[15, 1, 1, 0, 0, 0, 1], [15, 1, 1, 0, 0, 0, -1]],
                ],
                "tracks_flags": ["CONTROLLABLE", "NONE"],
            },
            "cuboidtracks_data": {"cuboids_dims": [[4, 2, 2], [1, 1, 1]]},
        }
    }
    path = tmp_path / "scene.usdz"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("rig_trajectories.json", json.dumps(rig))
        archive.writestr("sequence_tracks.json", json.dumps(tracks))
        archive.writestr("extra.json", '{"user_data": true}')
    return NuRecDataset([str(path)], keyframe=0), str(path)


def test_interpolates_full_host_pose_and_preserves_occluder_classes(recorded_dataset):
    dataset, path = recorded_dataset
    actors = dataset.actor_poses(path, 250000)
    host = actors["host"]
    np.testing.assert_allclose(host.transform[:3, 3], [11, 0, 1])
    np.testing.assert_allclose(host.transform[:3, 0], [0, 1, 0], atol=1e-12)
    np.testing.assert_allclose(actors["obstacle"].transform[:3, :3], np.eye(3))
    assert host.extent == (4, 2, 2)
    assert dataset.actor_poses(path, 500001) == {}

    scene = dataset.scene_at_frame(path, 1)
    assert {obj.category for obj in scene.objects} == {"vehicle", "protruding_object"}
    target = next(obj for obj in scene.objects if obj.track_id == "host")
    assert target.yaw == pytest.approx(math.pi / 2)
    np.testing.assert_allclose(target.pose, host.transform)


def test_initial_window_preserves_start_and_does_not_repeat_short_tail(recorded_dataset):
    dataset, path = recorded_dataset
    start = dataset.scene_at_frame(path, 1)
    window = dataset.initial_sequence(start, 3)
    assert [scene.frame for scene in window] == [1, 2]
    assert [scene.t for scene in window] == [0.25, 0.5]
    assert start.frame == 1
    with pytest.raises(ValueError, match="positive"):
        dataset.initial_sequence(start, 0)


def test_context_exposes_unmodified_metadata_archive_and_lazy_renderer(recorded_dataset):
    dataset, path = recorded_dataset
    context = dataset.context(next(dataset.scenes()))
    assert context.metadata["rig_trajectories.json"]["custom_annotation"] == {"weather": "cloudy"}
    assert context.renderer._stub is None
    assert context.dataset is dataset
    with context.native["open_archive"]() as archive:
        assert json.loads(archive.read("extra.json")) == {"user_data": True}
    assert "obstacle" in context.native["actor_poses"](250000)
    context.close()


def test_dataset_selection_releases_lazy_renderer(recorded_dataset, monkeypatch):
    from unittest.mock import Mock
    from avsectester.scenarios import Constraint, DatasetFilter, FilterResult, ScenarioRequirement

    dataset, _ = recorded_dataset
    renderer = Mock()
    monkeypatch.setattr(dataset, "make_renderer", lambda scene: renderer)

    class Render(Constraint):
        def evaluate(self, context):
            context.renderer.load_scene()
            return FilterResult("pass", "renderer used")

    cases = DatasetFilter(dataset).scenarios(ScenarioRequirement("render", constraints=[Render()]))
    next(cases)
    renderer.load_scene.assert_called_once_with()
    renderer.close.assert_called_once_with()


def test_runtime_adapter_uses_scene_time_stable_alias_and_current_ego(recorded_dataset):
    from types import SimpleNamespace
    from avsectester.insertion import AttachedPlacement, Insertion, Orientation, PlaneAsset
    from avsectester.plane import Observation
    from avsectester.rendering.cameras import PinholeCamera, planar_rig_pose
    from avsectester.simulators.nurec import EgoPose

    dataset, _path = recorded_dataset
    scene = next(dataset.scenes())
    camera = PinholeCamera(np.array([[200, 0, 320], [0, 200, 240], [0, 0, 1]]), 640, 480)

    class Renderer:
        service = object()

        def camera_model(self, _):
            return camera

        def timestamp_us(self, pose):
            return round(pose.t * 1e6)

        def rig_transform(self, pose):
            return planar_rig_pose(pose.x, pose.y, pose.yaw)

        def cam_from_world(self, pose, _):
            return np.linalg.inv(self.rig_transform(pose))

    backend = SimpleNamespace(renderer=Renderer())
    item = Insertion(
        "patch",
        PlaneAsset(np.full((4, 4, 4), 255, np.uint8), 1, 1),
        AttachedPlacement("hosts[0]", "rear_center"),
        Orientation("follow_host", (0, 0, 180)),
    )
    adapter = dataset.insertion_renderer(scene, backend, [item], bindings={"hosts": ("host",)})
    obs = Observation(0.25, 1, vehicle_state=EgoPose(2, 3, 0.2, t=0.25))
    actors, victim, transform = adapter.geometry(obs)
    assert actors["hosts[0]"] is actors["host"]
    np.testing.assert_allclose(actors["hosts"].transform[:3, 3], [11, 0, 1])
    np.testing.assert_allclose(victim.transform[:3, 3], [2, 3, 0])
    np.testing.assert_allclose(transform @ victim.transform, np.eye(4), atol=1e-12)


def test_insertion_pipeline_survives_backend_reset_and_drives_from_modified_input(recorded_dataset):
    from avsectester.backend import AVStack, run
    from avsectester.insertion import AttachedPlacement, Insertion, Orientation, PlaneAsset
    from avsectester.plane import Control
    from avsectester.rendering.cameras import PinholeCamera, planar_rig_pose
    from avsectester.simulators.nurec import NuRecBackend, StubRenderer
    from avsectester.simulators.patch_insertion import frame_perturbation

    dataset, _ = recorded_dataset
    scene = next(dataset.scenes())
    camera_to_rig = np.asarray(scene.cameras[dataset.camera].cam_to_ego)

    class RecordedRenderer(StubRenderer):
        service = object()
        loads = 0

        def load_scene(self, _):
            self.loads += 1

        def camera_model(self, _):
            return PinholeCamera(np.array([[200, 0, 320], [0, 200, 240], [0, 0, 1]]), 640, 480)

        def timestamp_us(self, pose):
            return round(pose.t * 1e6)

        def rig_transform(self, pose):
            return planar_rig_pose(pose.x, pose.y, pose.yaw)

        def cam_from_world(self, pose, _):
            return np.linalg.inv(self.rig_transform(pose) @ camera_to_rig)

    class ImageSensitiveStack(AVStack):
        received = []

        def __call__(self, obs):
            image = obs.sensor_data[dataset.camera]
            self.received.append(image.copy())
            return Control(throttle=float(image.any()))

    renderer = RecordedRenderer(cameras=[dataset.sensor])
    backend = NuRecBackend(
        {"dt": 0.1, "camera_aliases": {dataset.sensor: dataset.camera}}, renderer=renderer
    )
    clean = backend.reset()
    patch = Insertion(
        "patch",
        PlaneAsset(np.full((16, 16, 4), [240, 20, 30, 255], np.uint8), 1, 1),
        AttachedPlacement("lead", "rear_center", (-0.1, 0, 0)),
        Orientation("follow_host", (0, 0, 180)),
    )
    adapter = dataset.insertion_renderer(scene, backend, [patch], bindings={"lead": ("host",)})
    stack = ImageSensitiveStack()
    trace = run(
        backend, stack, frames=2, perturb=frame_perturbation(adapter, camera=dataset.camera)
    )
    assert renderer.loads == 2
    assert all(record.throttle == 1 for record in trace.records)
    assert len(stack.received) == 2 and all(image.any() for image in stack.received)
    assert not clean.sensor_data[dataset.camera].any()
    assert adapter.resolved[0].host_id == "lead"
    assert adapter.evidence["patch"].visibility.source == "cuboid_estimate"


def test_demo_source_and_rear_attachment_use_the_explicit_recorded_host(recorded_dataset, monkeypatch):
    from pathlib import Path
    from types import SimpleNamespace
    from avsectester.insertion import ActorPose, resolve_insertion

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "scripts"))
    from demo_common import nurec_rear_insertion, nurec_source

    _, path = recorded_dataset
    dataset, scene = nurec_source(path, "test")
    assert scene.source["timestamp_us"] == 0
    with pytest.raises(ValueError, match="does not match"):
        nurec_source(path, "another-scene")
    payload = SimpleNamespace(vehicle_texture=np.full((16, 16, 4), 255, np.uint8))
    item = nurec_rear_insertion(payload, scene, "host")
    initial = resolve_insertion(item, dataset.actor_poses(path, 0), ActorPose(np.eye(4)))
    later = resolve_insertion(item, dataset.actor_poses(path, 250000), ActorPose(np.eye(4)))
    assert initial.host_id == later.host_id == "host"
    np.testing.assert_allclose(initial.pose[:3, 3], [7.97, 0, 1.1])
    np.testing.assert_allclose(later.pose[:3, 3], [11, -2.03, 1.1])
    assert item.asset.width_m == pytest.approx(.8)
    assert item.asset.height_m == pytest.approx(.8)
    for invalid in ("missing", "obstacle"):
        with pytest.raises(ValueError, match="must be a vehicle"):
            nurec_rear_insertion(payload, scene, invalid)
