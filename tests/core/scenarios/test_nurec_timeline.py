"""NuRec decisions have their own fixed timeline over recorded support."""
import json
import zipfile
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import test_nurec_metadata
from avsectester.datasets.nurec import NuRecDataset, _interpolate_pose, _quat_to_rot
from avsectester.closed_loop import RecordingPolicy
from avsectester.driving_adapters import alpamayo_driving_adapter
from avsectester.services import ClosedLoopServices, CandidateRequest, CandidateOrigin
from avsectester.simulators.nurec import NuRecBackend, StubRenderer, TrajectoryFollower
from avsectester.stacks.alpamayo import AlpamayoAVStack

recorded_dataset = test_nurec_metadata.recorded_dataset


@pytest.fixture
def timeline(recorded_dataset, tmp_path):
    original, path = recorded_dataset
    raw = original.raw_metadata(path)
    rig = raw["rig_trajectories.json"]
    traj = rig["rig_trajectories"][0]
    key = next(iter(rig["camera_calibrations"]))
    times = [0, 33333, 66667, 100000, 133333, 166667, 200000, 233333]
    poses, matrices = [], []
    for t in times:
        angle = t/1e6
        q = [0, 0, np.sin(angle/2), np.cos(angle/2)]
        matrix = np.eye(4)
        matrix[:3, :3] = _quat_to_rot(q)
        matrix[0, 3] = angle*3
        poses.append([angle*3, 0, 0, *q])
        matrices.append(matrix.tolist())
    traj["cameras_frame_timestamps_us"][key] = [[t, t+1] for t in times]
    traj["cameras_frame_T_rig_worlds"][key] = [[m, m] for m in matrices]
    tracks = raw["sequence_tracks.json"]["test"]["tracks_data"]
    tracks["tracks_timestamps_us"] = [times, [33333, 133333]]
    tracks["tracks_poses"] = [poses, [poses[1], poses[4]]]
    target = str(tmp_path / "30hz.usdz")
    with zipfile.ZipFile(target, "w") as archive:
        for name, value in raw.items():
            archive.writestr(name, json.dumps(value))
    return NuRecDataset([target], time_step=.1), target, times, poses


def test_fixed_timeline_and_terminal_support(timeline):
    data, path, _, _ = timeline
    frames = data.frames(path)
    assert [f.timestamp for f in frames] == [0, .1, .2]
    np.testing.assert_allclose(np.diff([f.timestamp for f in frames]), .1, rtol=0, atol=0)
    assert len(data.decision_window(path, frames[0], 2)) == 3
    with pytest.raises(ValueError, match="terminal reference"):
        data.decision_window(path, frames[0], 3)
    source = data.source_data(path, frames[1])
    assert source.payload["time_step"] == .1
    assert source.payload["interpolation"] == "linear_translation_slerp_rotation"
    assert frames[-1].timestamp <= .233333


def test_rig_and_actors_match_recorded_and_interpolated_poses(timeline):
    data, path, times, poses = timeline
    for dt, position, timestamp in ((.1, 1, 100000), (.05, 1, 50000)):
        data = NuRecDataset([path], time_step=dt)
        frame = data.frames(path)[position]
        reference = data.reference(path, frame)
        expected = _interpolate_pose(times, poses, timestamp)
        np.testing.assert_allclose(reference.actors["ego"].pose, expected, atol=1e-12)
        np.testing.assert_allclose(reference.actors["host"].pose, expected, atol=1e-12)
        np.testing.assert_allclose(reference.scene.ego.pose, expected, atol=1e-12)
    data = NuRecDataset([path], time_step=.1)
    assert "obstacle" not in data.reference(path, data.frames(path)[0]).actors
    assert "obstacle" in data.reference(path, data.frames(path)[1]).actors
    assert "obstacle" not in data.reference(path, data.frames(path)[2]).actors


def test_internal_reads_do_not_copy_metadata_and_public_values_remain_independent(timeline, monkeypatch):
    data, path, _, _ = timeline
    pristine = deepcopy(data.raw_metadata(path))
    public = data.raw_metadata(path)
    public["sequence_tracks.json"].clear()
    assert data.raw_metadata(path) == pristine
    monkeypatch.setattr(data, "raw_metadata", lambda _: pytest.fail("internal full metadata copy"))
    pose = data.actor_poses(path, 100000)["host"]
    pose.transform.setflags(write=True)
    pose.transform[:] = -1
    _, _, _, camera, timestamps, transforms = data._camera_data(path)
    timestamps[:] = -1
    transforms[:] = -1
    camera.principal_point[:] = -1
    assert data.actor_poses(path, 100000)["host"].transform[0, 3] == pytest.approx(.3)
    assert data._camera_data(path)[-2][0, 0] == 0
    assert data._camera_data(path)[3].principal_point[0] == 320


def test_timestep_mismatch_fails_before_world_or_model_start(timeline):
    data, path, _, _ = timeline
    class Renderer(StubRenderer):
        def load_scene(self, scene):
            pytest.fail("world started despite timestep mismatch")
    def factory(source, seed):
        return NuRecBackend({"dt": .05}, Renderer(height=2, width=2))
    services = ClosedLoopServices(data, factory, backend_type=NuRecBackend)
    f = data.frames(path)
    with services.open_scope(0) as scope, pytest.raises(ValueError, match="time_step"):
        scope.rollout(CandidateRequest(CandidateOrigin(path, f[0], f[1]), 0, 1, {}, RecordingPolicy()))


def test_ten_hz_rollout_accepts_alpamayo_and_trajectory_follower(timeline, monkeypatch):
    import sys
    from types import ModuleType
    names = ("alpasim_driver", "alpasim_driver.models", "alpasim_driver.models.base", "alpasim_grpc",
             "alpasim_grpc.v0", "alpasim_grpc.v0.common_pb2")
    modules = {name: ModuleType(name) for name in names}
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
        if "." in name:
            parent, attr = name.rsplit(".", 1)
            setattr(modules[parent], attr, module)
    base = modules["alpasim_driver.models.base"]
    base.CameraFrame = base.PredictionInput = SimpleNamespace
    base.DriveCommand = SimpleNamespace(STRAIGHT=0)
    for name in ("PoseAtTime", "Pose", "Vec3", "Quat"):
        setattr(modules["alpasim_grpc.v0.common_pb2"], name, SimpleNamespace)
    data, path, _, _ = timeline
    def backend(source, seed):
        return NuRecBackend({"dt": .1}, StubRenderer(cameras=["front"], height=2, width=2), TrajectoryFollower())
    def driver(seed):
        model = SimpleNamespace(predict=lambda _: SimpleNamespace(
            candidate_positions=np.array([[[.1, 0, 0], [.2, 0, 0]]]), selected_plan=object()))
        return AlpamayoAVStack(model=model, camera_ids=["front"])
    services = ClosedLoopServices(data, backend, alpamayo_driving_adapter(driver), backend_type=NuRecBackend)
    f = data.frames(path)
    with services.open_scope(0) as scope:
        log = scope.rollout(CandidateRequest(CandidateOrigin(path, f[0], f[2]), 0, 2, {}, RecordingPolicy()))
    assert [s.physical_post.t for s in log.steps] == [.1, .2]
    assert log.metadata["dataset_time_step"] == .1
