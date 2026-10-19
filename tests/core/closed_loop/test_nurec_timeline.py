"""NuRec decisions have their own fixed timeline over recorded support."""
import json
import zipfile
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from avsectester.datasets.nurec import NuRecDataset, _interpolate_pose, _quat_to_rot
from avsectester.closed_loop.runner import RecordingPolicy
from avsectester.closed_loop.driving_adapters import alpamayo_driving_adapter
from avsectester.attack import AttackSegment
from avsectester.closed_loop.services import ClosedLoopServices, CandidateRequest
from avsectester.closed_loop.simulators.nurec import NuRecSimulator, StubRenderer, TrajectoryFollower
from avsectester.closed_loop.avstacks.alpamayo import AlpamayoAVStack


@pytest.fixture
def timeline(nurec_usdz_path, tmp_path):
    raw = NuRecDataset([nurec_usdz_path]).raw_metadata(nurec_usdz_path)
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
    scene = data.scene_id(path)
    frames = data.frames(scene)
    assert [f.timestamp for f in frames] == [0, .1, .2]
    np.testing.assert_allclose(np.diff([f.timestamp for f in frames]), .1, rtol=0, atol=0)
    assert len(data.decision_window(scene, frames[0], 2)) == 3
    with pytest.raises(ValueError, match="terminal frame"):
        data.decision_window(scene, frames[0], 3)
    source = data.scenario_start(scene, frames[1])
    assert source.payload["time_step"] == .1
    assert source.payload["interpolation"] == "linear_translation_slerp_rotation"
    assert frames[-1].timestamp <= .233333


def test_rig_and_actors_match_recorded_and_interpolated_poses(timeline):
    data, path, times, poses = timeline
    scene = data.scene_id(path)
    for dt, position, timestamp in ((.1, 1, 100000), (.05, 1, 50000)):
        data = NuRecDataset([path], time_step=dt)
        scene = data.scene_id(path)
        frame = data.frames(scene)[position]
        reference = data.frame_data(scene, frame)
        expected = _interpolate_pose(times, poses, timestamp)
        np.testing.assert_allclose(reference.actors["ego"].pose, expected, atol=1e-12)
        np.testing.assert_allclose(reference.actors["host"].pose, expected, atol=1e-12)
        np.testing.assert_allclose(reference.scene.ego.pose, expected, atol=1e-12)
    data = NuRecDataset([path], time_step=.1)
    scene = data.scene_id(path)
    assert "obstacle" not in data.frame_data(scene, data.frames(scene)[0]).actors
    assert "obstacle" in data.frame_data(scene, data.frames(scene)[1]).actors
    assert "obstacle" not in data.frame_data(scene, data.frames(scene)[2]).actors


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
    scene = data.scene_id(path)
    class Renderer(StubRenderer):
        def load_scene(self, scene):
            pytest.fail("world started despite timestep mismatch")
    services = ClosedLoopServices(data, NuRecSimulator({"dt": .05}, Renderer(height=2, width=2)))
    f = data.frames(scene)
    with services.session(0) as scope, pytest.raises(ValueError, match="time_step"):
        scope.rollout(CandidateRequest(AttackSegment(scene, f[0], f[1]), 0, 1, {}, RecordingPolicy()))


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
    scene = data.scene_id(path)
    simulator = NuRecSimulator({"dt": .1}, StubRenderer(cameras=["front"], height=2, width=2),
                               TrajectoryFollower())
    def driver(seed):
        model = SimpleNamespace(predict=lambda _: SimpleNamespace(
            candidate_positions=np.array([[[.1, 0, 0], [.2, 0, 0]]]), selected_plan=object()))
        return AlpamayoAVStack(model=model, camera_ids=["front"])
    services = ClosedLoopServices(data, simulator, alpamayo_driving_adapter(driver))
    f = data.frames(scene)
    with services.session(0) as scope:
        log = scope.rollout(CandidateRequest(AttackSegment(scene, f[0], f[2]), 0, 2, {}, RecordingPolicy()))
    assert [s.physical_post.t for s in log.steps] == [.1, .2]
    assert log.metadata["dataset_time_step"] == .1


def test_simulator_sensor_view_places_recorded_actors_at_the_render_time(timeline):
    from avsectester.geometry.cameras import PinholeCamera, planar_rig_pose

    data, path, _, _ = timeline
    scene = data.scene_id(path)
    source = data.scenario_start(scene, data.frames(scene)[1])
    lens = PinholeCamera(np.array([[200, 0, 320], [0, 200, 240], [0, 0, 1]]), 640, 480)

    class Renderer(StubRenderer):
        def camera_model(self, camera):
            return lens

        def timestamp_us(self, pose):
            return self.origin_timestamp_us + round(pose.t * 1e6)

        def rig_transform(self, pose):
            return planar_rig_pose(pose.x, pose.y, pose.yaw)

        def cam_from_world(self, pose, camera):
            return np.linalg.inv(self.rig_transform(pose))

    simulator = NuRecSimulator({"dt": .1}, Renderer(cameras=[data.camera_id], height=480, width=640)).create_runtime()
    simulator.reset(source)
    view = simulator.sensor_view(data.camera_id)
    expected = data.actor_poses(path, 100000)
    np.testing.assert_allclose(view.actors["host"].transform, expected["host"].transform)
    assert view.camera is lens and view.data.shape == (480, 640, 3)
    assert view.native["timestamp_us"] == 100000 and view.depth is None
    from avsectester.closed_loop.datatypes import Control

    simulator.step(Control())
    assert simulator.sensor_view(data.camera_id).native["timestamp_us"] == 200000


def test_simulator_state_includes_recorded_actors_and_rejects_actor_control(timeline):
    from avsectester.datasets.base import ActorState
    from avsectester.closed_loop.datatypes import Control

    data, path, _, _ = timeline
    scene = data.scene_id(path)
    simulator = NuRecSimulator({"dt": .1}, StubRenderer(cameras=[data.camera_id], height=2, width=2)).create_runtime()
    simulator.reset(data.scenario_start(scene, data.frames(scene)[1]))
    actors = simulator.state().actors
    np.testing.assert_allclose(actors["host"].pose, data.actor_poses(path, 100000)["host"].transform)
    assert actors["host"].velocity[0] == pytest.approx(3, rel=0.05)  # 0.3 m in 0.1 s
    assert actors["obstacle"].velocity is None  # a two-sample track one step away on neither side
    assert "ego" not in actors
    actors["host"].pose[:] = 0  # snapshots are independent copies
    assert simulator.state().actors["host"].pose[0, 0] != 0
    with pytest.raises(NotImplementedError, match="cannot control"):
        simulator.control_actor("host", Control(throttle=1))
    with pytest.raises(NotImplementedError, match="cannot set"):
        simulator.set_actor_state("host", ActorState(np.eye(4)))
