"""Real policy context and trajectory transport with only checkpoint weights replaced."""
from dataclasses import replace
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from avsectester.closed_loop.avstacks.implemented.alpamayo import AlpamayoAVStack, alpamayo_driving_adapter
from avsectester.geometry.poses import quaternion_to_rotation, rotation_matrix
from avsectester.runlog import Recorder, RunLog, StepRecord
from avsectester.execution.services import ClosedLoopServices
from tests.fakes import MemoryDataset, MemorySimulator


def test_driving_adapter(alpamayo_model_boundary, alpamayo_observations):
    adapter = alpamayo_driving_adapter(lambda seed: AlpamayoAVStack(
        model=alpamayo_model_boundary.create(), camera_ids=["front"], device="cpu"))
    assert adapter.past_steps == 15
    assert adapter.decision_time_step == .1
    observations = alpamayo_observations()
    with ClosedLoopServices(MemoryDataset(), MemorySimulator(), adapter) as services:
        with services.session(7) as session:
            component, = session.components()
            assert component.id == "policy" and set(component.inputs) == {"observation", "past"}
            result = session.evaluate("policy", {"observation": observations[-1], "past": observations[:-1]})
            assert result.value.trajectory == [
                ((0., 0., 0.), (1., 0., 0., 0.), 0),
                ((1., 0., 0.), (1., 0., 0., 0.), 100000),
                ((2., 0., 0.), (1., 0., 0., 0.), 200000),
            ]
            with pytest.raises(TypeError, match="observation and past"):
                session.evaluate("policy", observations[-1])


def test_trajectory_context(alpamayo_model_boundary, alpamayo_observations):
    model = alpamayo_model_boundary.create()
    avstack = AlpamayoAVStack(model=model, camera_ids=["front"], device="cpu", context_length=2)
    observations = alpamayo_observations(count=18, speed=2., acceleration=-1., yaw=np.pi / 2)
    avstack.reset(observations[15], past=observations[:15])
    command = avstack(observations[15])
    assert command.trajectory[0] == ((0., 0., 0.), (1., 0., 0., 0.), 0)
    model_input = model.last_input
    assert model_input.speed == pytest.approx(.5)
    assert model_input.acceleration == pytest.approx(-1.)
    assert model_input.previous_plan is None
    assert len(model_input.ego_pose_history) == 16
    np.testing.assert_allclose(
        [[pose.pose.vec.x, pose.pose.vec.y] for pose in model_input.ego_pose_history],
        [[observation.vehicle_state.x, observation.vehicle_state.y] for observation in observations[:16]])
    assert [pose.timestamp_us for pose in model_input.ego_pose_history] == [
        round(observation.t * 1e6) + avstack.EPOCH_US for observation in observations[:16]]
    assert [frame.image[0, 0, 0] for frame in model_input.camera_images["front"]] == [14, 15]
    avstack(observations[16])
    assert model.last_input.speed == pytest.approx(.4)
    assert model.last_input.acceleration == pytest.approx(-1.)
    assert model.last_input.previous_plan == "forward"
    np.testing.assert_allclose(
        [[pose.pose.vec.x, pose.pose.vec.y] for pose in model.last_input.ego_pose_history],
        [[observation.vehicle_state.x, observation.vehicle_state.y] for observation in observations[1:17]])
    log = avstack.component_log()
    assert log["history_pose_count"] == 16
    np.testing.assert_allclose(log["selected_image_times"]["front"], [0., .1])
    assert log["policy"].reasoning_text == "clear"
    avstack.reset(observations[15], past=observations[:15])
    avstack(observations[15])
    assert model.last_input.previous_plan is None
    avstack.close()

    model = alpamayo_model_boundary.create()
    avstack = AlpamayoAVStack(model=model, camera_ids=["front"], device="cpu", frame_subsample=3)
    observations = alpamayo_observations(count=17)
    avstack.reset(observations[15], past=observations[:15])
    avstack(observations[15])
    selected = model.last_input.camera_images["front"]
    assert [frame.image[0, 0, 0] for frame in selected] == [6, 9, 12, 15]
    np.testing.assert_allclose(avstack.component_log()["selected_image_times"]["front"], [-.9, -.6, -.3, 0.])
    avstack(observations[16])
    assert [frame.image[0, 0, 0] for frame in model.last_input.camera_images["front"]] == [7, 10, 13, 16]
    avstack.close()
    with pytest.raises(ValueError, match="frame_subsample must be a positive integer"):
        AlpamayoAVStack(model=model, frame_subsample=0)


def test_reset(alpamayo_model_boundary, alpamayo_observations):
    model = alpamayo_model_boundary.create()
    avstack = AlpamayoAVStack(model=model, camera_ids=["front"], device="cpu")
    observations = alpamayo_observations()
    with pytest.raises(ValueError, match="15 recorded past observations"):
        avstack.reset(observations[-1], past=observations[:14])
    with pytest.raises(ValueError, match="0.1 second decision interval"):
        avstack.reset(replace(observations[-1], t=.01), past=observations[:-1])
    with pytest.raises(ValueError, match="missing camera 'front'"):
        avstack.reset(replace(observations[-1], sensor_data={}), past=observations[:-1])
    with pytest.raises(ValueError, match="more than 15 past decisions"):
        AlpamayoAVStack(model=model, frame_subsample=6)
    with pytest.raises(RuntimeError, match="Reset AlpamayoAVStack"):
        avstack(observations[-1])
    avstack.reset(observations[-1], past=observations[:-1])
    avstack(observations[-1])
    with pytest.raises(ValueError, match="0.1 second decision interval"):
        avstack(replace(observations[-1], t=.05))
    avstack.close()


@pytest.mark.parametrize("route,command", [
    (np.array([[0., 0., 0.], [8., 3., 0.], [20., 8., 0.]]), "LEFT"),
    (np.array([[0., 0., 0.], [8., -3., 0.], [20., -8., 0.]]), "RIGHT"),
    (np.array([[0., 0., 0.], [8., 0., 0.], [20., 0., 0.]]), "STRAIGHT"),
])
def test_navigation(route, command, alpamayo_model_boundary, alpamayo_observations):
    model = alpamayo_model_boundary.create()
    observations = alpamayo_observations(route=route)
    avstack = AlpamayoAVStack(model=model, camera_ids=["front"], device="cpu")
    avstack.reset(observations[-1], past=observations[:-1])
    avstack(observations[-1])
    assert model.last_input.command.name == command
    assert model.last_input.route.timestamp_us == avstack.EPOCH_US
    np.testing.assert_array_equal([[point.x, point.y, point.z] for point in model.last_input.route.waypoints], route)
    np.testing.assert_array_equal(avstack.component_log()["route_waypoints"], route)
    assert avstack.component_log()["navigation_command"] == command
    assert avstack.component_log()["navigation_note"] == "Navigation command derived from observation.route"
    avstack.close()


def test_no_route(alpamayo_model_boundary, alpamayo_observations):
    model = alpamayo_model_boundary.create()
    observations = alpamayo_observations()
    avstack = AlpamayoAVStack(model=model, camera_ids=["front"], device="cpu")
    avstack.reset(observations[-1], past=observations[:-1])
    avstack(observations[-1])
    assert model.last_input.route is None and model.last_input.command.name == "STRAIGHT"
    log = avstack.component_log()
    assert log["route_waypoints"] is None and log["navigation_command"] == "STRAIGHT"
    assert log["navigation_note"] == "No route supplied; using STRAIGHT"
    avstack.close()


def test_prediction_timestamps(alpamayo_model_boundary, alpamayo_observations):
    model = alpamayo_model_boundary.create()
    observations = alpamayo_observations(start_time=-1.4999994, route=np.array([[0., 0., 0.], [8., 0., 0.]]))
    avstack = AlpamayoAVStack(model=model, camera_ids=["front"], device="cpu")
    avstack.reset(observations[-1], past=observations[:-1])
    command = avstack(observations[-1])
    assert [point[-1] for point in command.trajectory] == [1, 100001, 200001]
    assert model.last_input.camera_images["front"][-1].timestamp_us == avstack.EPOCH_US + 1
    assert model.last_input.ego_pose_history[-1].timestamp_us == avstack.EPOCH_US + 1
    assert model.last_input.route.timestamp_us == avstack.EPOCH_US + 1
    model.prediction.model_t0_us = avstack.EPOCH_US + 1
    model.prediction.waypoint_timestamps_us = np.array([avstack.EPOCH_US + 100003, avstack.EPOCH_US + 200007])
    model.prediction.candidate_rotations[0, 0] = rotation_matrix((0, 0, 30))
    avstack.reset(observations[-1], past=observations[:-1])
    command = avstack(observations[-1])
    assert command.trajectory[0] == ((0., 0., 0.), (1., 0., 0., 0.), 1)
    assert [point[-1] for point in command.trajectory] == [1, 100003, 200007]
    np.testing.assert_allclose(quaternion_to_rotation(*command.trajectory[1][1]),
                               rotation_matrix((0, 0, 30)), atol=1e-12)
    avstack.close()


def test_close(alpamayo_model_boundary, alpamayo_observations):
    observations = alpamayo_observations()
    observation, past = observations[-1], observations[:-1]
    owned = AlpamayoAVStack(checkpoint_path="synthetic-checkpoint", device="cpu", camera_ids=["front"])
    owned.reset(observation, past=past)
    model = owned.models["policy"]
    owned(observation)
    owned.close()
    assert model.closed and owned.models == {} and owned.component_log() is None
    owned.close()
    with pytest.raises(RuntimeError, match="closed"):
        owned.reset(observation, past=past)
    borrowed_model = alpamayo_model_boundary.create()
    borrowed = AlpamayoAVStack(model=borrowed_model, camera_ids=["front"], device="cpu")
    borrowed.reset(observation, past=past)
    borrowed.close()
    assert not borrowed_model.closed and borrowed.models == {}
    with pytest.raises(ValueError, match="checkpoint_path"):
        AlpamayoAVStack(checkpoint_path="", device="cpu", camera_ids=["front"]).reset(observation, past=past)


def test_cublas_workspace_config():
    environment = dict(os.environ)
    environment.pop("CUBLAS_WORKSPACE_CONFIG", None)
    process = subprocess.run(
        [sys.executable, "-c",
         "import os; import avsectester.closed_loop.avstacks.implemented.alpamayo; "
         "print(os.environ['CUBLAS_WORKSPACE_CONFIG'])"],
        cwd=Path(__file__).resolve().parents[5], env=environment,
        capture_output=True, text=True, check=True)
    assert process.stdout.strip() == ":4096:8"


def test_seed(alpamayo_model_boundary, alpamayo_observations):
    observations = alpamayo_observations(count=18)
    observation, past = observations[15], observations[:15]
    derived = {seed: [int(np.random.SeedSequence((seed, count)).generate_state(1)[0]) for count in (0, 1)]
               for seed in (7, 8)}
    assert not set(derived[7]) & set(derived[8])
    for seed, determinism, expected in ((7, True, derived[7]), (None, False, [0, 1])):
        avstack = AlpamayoAVStack(checkpoint_path="synthetic-checkpoint", device="cpu",
                                  camera_ids=["front"], seed=seed)
        for _ in range(2):
            avstack.reset(observation, past=past)
            model = avstack.models["policy"]
            seeds = []
            for current in observations[15:17]:
                avstack(current)
                seeds.append(model.last_input.inference_seed)
            assert seeds == expected
        assert model.options["model_cfg"].force_determinism is determinism
        avstack.close()
    with pytest.raises(ValueError, match="seed"):
        AlpamayoAVStack(camera_ids=["front"], seed=-1)
    # The default matches AlpaSim's Alpamayo 1.5 driver cameras.
    assert set(AlpamayoAVStack(device="cpu").camera_ids) == {
        "camera_front_wide_120fov", "camera_front_tele_30fov",
        "camera_cross_left_120fov", "camera_cross_right_120fov"}


def test_native_records(alpamayo_model_boundary, alpamayo_observations, tmp_path):
    avstack = AlpamayoAVStack(model=alpamayo_model_boundary.create(), camera_ids=["front"], device="cpu")
    observations = alpamayo_observations()
    avstack.reset(observations[-1], past=observations[:-1])
    command = avstack(observations[-1])
    components = avstack.component_log()
    with Recorder(tmp_path) as recorder:
        recorder.record_step(StepRecord(0, components=components, produced_command=command))
    saved = RunLog.read(tmp_path).steps[0]
    assert saved.components["policy"]["selected_index"] == 0
    assert saved.components["policy"]["reasoning_text"] == "clear"
    np.testing.assert_array_equal(saved.components["policy"]["candidate_positions"],
                                  components["policy"].candidate_positions)
    assert saved.produced_command.trajectory == command.trajectory
    assert saved.components["history_pose_count"] == 16
    assert saved.components["navigation_note"] == "No route supplied; using STRAIGHT"
    avstack.close()
