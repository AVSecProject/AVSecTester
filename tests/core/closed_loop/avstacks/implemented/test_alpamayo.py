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
    assert adapter.history.duration_us == 1_500_000
    assert adapter.decision_time_step == .1
    observations = alpamayo_observations()
    with ClosedLoopServices(MemoryDataset(), MemorySimulator(), adapter) as services:
        with services.session(7) as session:
            component, = session.components()
            assert component.id == "policy" and set(component.inputs) == {"observation"}
            result = session.evaluate("policy", {"observation": observations[-1]})
            assert result.value.trajectory == [
                ((0., 0., 0.), (1., 0., 0., 0.), 0),
                ((1., 0., 0.), (1., 0., 0., 0.), 100000),
                ((2., 0., 0.), (1., 0., 0., 0.), 200000),
            ]
            with pytest.raises(TypeError, match="require observation"):
                session.evaluate("policy", observations[-1])


def test_trajectory_context(alpamayo_model_boundary, alpamayo_observations):
    model = alpamayo_model_boundary.create()
    avstack = AlpamayoAVStack(model=model, camera_ids=["front"], device="cpu", context_length=2)
    observations = alpamayo_observations(count=18, speed=2., acceleration=-1., yaw=np.pi / 2)
    avstack.reset(observations[15])
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
    assert model_input.camera_images["front"][0].image is observations[14].sensor_data["front"]
    assert model_input.camera_images["front"][1].image is observations[15].sensor_data["front"]
    assert avstack.component_log()["selected_capture_ids"]["front"] == [-1, 0]
    cached = tuple(avstack.native["camera_history"]["front"])
    delivered = observations[15].delivered("front")
    assert len(cached) == avstack.history.samples
    assert all(item is source for item, source in zip(cached, delivered[-avstack.history.samples:]))
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
    avstack.reset(observations[15])
    before = len(model.inputs)
    avstack.ingest(observations[15])
    assert len(model.inputs) == before
    avstack(observations[16])
    assert model.last_input.previous_plan is None
    assert [frame.image[0, 0, 0] for frame in model.last_input.camera_images["front"]] == [15, 16]
    with pytest.raises(ValueError, match="missing camera 'front'"):
        avstack(replace(observations[17], sensor_data={}, deliveries={"front": ()}))
    avstack.close()

    model = alpamayo_model_boundary.create()
    avstack = AlpamayoAVStack(model=model, camera_ids=["front"], device="cpu", frame_subsample=3)
    observations = alpamayo_observations(count=17)
    avstack.reset(observations[15])
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
    with pytest.raises(ValueError, match="1.5 seconds of recorded pose history"):
        avstack.reset(replace(observations[-1], state_estimates=observations[-1].state_estimates[1:]))
    estimates = observations[-1].state_estimates
    with pytest.raises(ValueError, match="state estimate times must increase"):
        avstack.reset(replace(observations[-1], state_estimates=(
            estimates[0], replace(estimates[1], time_us=estimates[0].time_us), *estimates[2:])))
    with pytest.raises(ValueError, match="missing camera 'front'"):
        avstack.reset(replace(observations[-1], sensor_data={}))
    with pytest.raises(RuntimeError, match="Reset AlpamayoAVStack"):
        avstack(observations[-1])
    avstack.reset(observations[-1])
    avstack(observations[-1])
    # Nonuniform capture spacing is supported. Timestamped history replaces a fixed rate.
    avstack(replace(observations[-1], t=.05, deliveries={}, state_estimates=()))
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
    avstack.reset(observations[-1])
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
    avstack.reset(observations[-1])
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
    avstack.reset(observations[-1])
    command = avstack(observations[-1])
    assert [point[-1] for point in command.trajectory] == [1, 100001, 200001]
    assert model.last_input.camera_images["front"][-1].timestamp_us == avstack.EPOCH_US + 1
    assert model.last_input.ego_pose_history[-1].timestamp_us == avstack.EPOCH_US + 1
    assert model.last_input.route.timestamp_us == avstack.EPOCH_US + 1
    model.prediction.model_t0_us = avstack.EPOCH_US + 1
    model.prediction.waypoint_timestamps_us = np.array([avstack.EPOCH_US + 100003, avstack.EPOCH_US + 200007])
    model.prediction.candidate_rotations[0, 0] = rotation_matrix((0, 0, 30))
    avstack.reset(observations[-1])
    command = avstack(observations[-1])
    assert command.trajectory[0] == ((0., 0., 0.), (1., 0., 0., 0.), 1)
    assert [point[-1] for point in command.trajectory] == [1, 100003, 200007]
    np.testing.assert_allclose(quaternion_to_rotation(*command.trajectory[1][1]),
                               rotation_matrix((0, 0, 30)), atol=1e-12)
    model.prediction.model_t0_us = avstack.EPOCH_US - 99999
    avstack.reset(observations[-1])
    command = avstack(observations[-1])
    assert [point[-1] for point in command.trajectory] == [-99999, 100003, 200007]
    np.testing.assert_array_equal([point[0][0] for point in command.trajectory], [0., 1., 2.])
    avstack.close()


def test_close(alpamayo_model_boundary, alpamayo_observations):
    observations = alpamayo_observations()
    observation = observations[-1]
    owned = AlpamayoAVStack(checkpoint_path="synthetic-checkpoint", device="cpu", camera_ids=["front"])
    owned.reset(observation)
    model = owned.models["policy"]
    owned(observation)
    owned.close()
    assert model.closed and owned.models == {} and owned.component_log() is None
    owned.close()
    with pytest.raises(RuntimeError, match="closed"):
        owned.reset(observation)
    borrowed_model = alpamayo_model_boundary.create()
    borrowed = AlpamayoAVStack(model=borrowed_model, camera_ids=["front"], device="cpu")
    borrowed.reset(observation)
    borrowed.close()
    assert not borrowed_model.closed and borrowed.models == {}
    with pytest.raises(ValueError, match="checkpoint_path"):
        AlpamayoAVStack(checkpoint_path="", device="cpu", camera_ids=["front"]).reset(observation)


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
    observation = observations[15]
    derived = {seed: [int(np.random.SeedSequence((seed, count)).generate_state(1)[0]) for count in (0, 1)]
               for seed in (7, 8)}
    assert not set(derived[7]) & set(derived[8])
    for seed, determinism, expected in ((7, True, derived[7]), (None, False, [0, 1])):
        avstack = AlpamayoAVStack(checkpoint_path="synthetic-checkpoint", device="cpu",
                                  camera_ids=["front"], seed=seed)
        for _ in range(2):
            avstack.reset(observation)
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
    from alpasim_utils.geometry import Trajectory

    model = alpamayo_model_boundary.create()
    times = np.array([AlpamayoAVStack.EPOCH_US + 100000,
                      AlpamayoAVStack.EPOCH_US + 200000], dtype=np.uint64)
    model.prediction.waypoint_timestamps_us = times.copy()
    model.prediction.selected_plan = Trajectory(
        times, model.prediction.candidate_positions[0], np.tile([0., 0., 0., 1.], (2, 1)))
    avstack = AlpamayoAVStack(model=model, camera_ids=["front"], device="cpu")
    observations = alpamayo_observations()
    avstack.reset(observations[-1])
    command = avstack(observations[-1])
    components = avstack.component_log()
    with Recorder(tmp_path / "full", retain_sensor_data=False) as recorder:
        recorder.record_step(StepRecord(0, components=components, produced_command=command))
    saved = RunLog.read(tmp_path / "full").steps[0]
    assert saved.components["policy"]["selected_index"] == 0
    assert saved.components["policy"]["reasoning_text"] == "clear"
    np.testing.assert_array_equal(saved.components["policy"]["candidate_positions"],
                                  components["policy"].candidate_positions)
    assert saved.produced_command.trajectory == command.trajectory
    assert saved.components["history_pose_count"] == 16
    assert saved.components["navigation_note"] == "No route supplied; using STRAIGHT"
    np.testing.assert_array_equal(saved.components["policy"]["waypoint_timestamps_us"], times)
    np.testing.assert_array_equal(saved.components["policy"]["selected_plan"]["timestamps_us"], times)
    components["policy"].waypoint_timestamps_us = None
    with Recorder(tmp_path / "without_repeated_times", retain_sensor_data=False) as recorder:
        recorder.record_step(StepRecord(0, components=components, produced_command=command))
    assert sum(path.stat().st_size for path in (tmp_path / "full/payloads").glob("*.bin")) == sum(
        path.stat().st_size for path in (tmp_path / "without_repeated_times/payloads").glob("*.bin"))
    avstack.close()


def test_acquisition_history(alpamayo_model_boundary, nurec_longitudinal_dataset):
    from alpasim_driver.frame_cache import FrameCache
    from avsectester.closed_loop.simulators.ego_dynamics import TrajectoryFollower
    from avsectester.closed_loop.simulators.nurec import NuRecSimulator
    from avsectester.closed_loop.timing import SensorTiming, TimingConfiguration
    from avsectester.execution.runner import ClosedLoopRunner, RunRequest
    from tests.fakes import CAMERA, NuRecImageBoundary

    dataset = nurec_longitudinal_dataset
    start = dataset.scenario_start("scene", dataset.frames("scene")[16])
    model = alpamayo_model_boundary.create()
    driver = AlpamayoAVStack(model=model, camera_ids=["front", "tele"], frame_subsample=3, device="cpu")
    boundary = NuRecImageBoundary(CAMERA)
    boundary.cameras = ["front", "tele"]
    simulator = NuRecSimulator(boundary, TrajectoryFollower(), time_step=.1,
                               physics_update_mode="none", depth="none", route_generator="none")
    with simulator.create_runtime() as runtime:
        log = ClosedLoopRunner().run(runtime, driver, RunRequest(3, start=start,
            timing=TimingConfiguration(planner_delay_us=100000, sensors={
                "front": SensorTiming(33000), "tele": SensorTiming(100000)})))
    cache = FrameCache(context_length=4, subsample_factor=3)
    latest_us = 198000
    for capture in log.captures:
        if capture.time_us <= latest_us and "front" in capture.sensor_ids:
            cache.add_image(capture.time_us + driver.EPOCH_US, np.zeros((1, 1, 3), np.uint8))
    expected = [frame.timestamp_us for frame in cache.latest_frame_entries(4)]
    actual = [frame.timestamp_us for frame in model.last_input.camera_images["front"]]
    assert actual == expected
    first_input = model.inputs[0]
    image_origin_us = max(frame.timestamp_us for frames in first_input.camera_images.values()
                          for frame in frames)
    assert image_origin_us == driver.EPOCH_US
    assert first_input.camera_images["tele"][-1].timestamp_us == driver.EPOCH_US
    assert first_input.ego_pose_history[0].timestamp_us == image_origin_us - driver.history.duration_us
    assert first_input.ego_pose_history[-1].timestamp_us == driver.EPOCH_US
    for inputs in model.inputs:
        origin_us = max(frame.timestamp_us for frames in inputs.camera_images.values() for frame in frames)
        assert inputs.ego_pose_history[0].timestamp_us <= origin_us - driver.history.duration_us
        assert inputs.ego_pose_history[-1].timestamp_us >= origin_us
    assert [step.executed_command_issue_time_us for step in log.steps] == [0, 0, 100000]
    assert len(log.steps) == 3 and log.status == "completed"
    assert log.steps[-1].components["selected_image_times"]["front"] == [
        (timestamp - driver.EPOCH_US) / 1e6 for timestamp in expected]
    driver.close()
