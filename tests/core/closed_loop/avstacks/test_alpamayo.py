"""Our Alpamayo adapter, using schema stand-ins and a model with prescribed predictions.

These tests exercise buffering, clocks and trajectory conversion. They do not validate the
external AlpaSim schema or real model inference.
"""

import math
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from avsectester.closed_loop.datatypes import Observation
from avsectester.closed_loop.avstacks.alpamayo import AlpamayoAVStack


@pytest.mark.parametrize("explicit,environment,expected", [
    ("/models/explicit", "/models/environment", "/models/explicit"),
    (None, "/models/environment", "/models/environment"),
    (None, None, None),
])
def test_checkpoint_location_is_user_configured(monkeypatch, explicit, environment, expected):
    if environment is None:
        monkeypatch.delenv("AVSECTESTER_ALPAMAYO_CHECKPOINT", raising=False)
    else:
        monkeypatch.setenv("AVSECTESTER_ALPAMAYO_CHECKPOINT", environment)
    avstack = AlpamayoAVStack(checkpoint_path=explicit)
    assert avstack.checkpoint_path == expected
    if expected is None:
        with pytest.raises(ValueError, match="Provide checkpoint_path"):
            avstack._load()


@pytest.fixture
def driver_schema(monkeypatch):
    # Supply only the external records used at the adapter boundary. No checkpoint is loaded.
    names = (
        "alpasim_driver",
        "alpasim_driver.models",
        "alpasim_driver.models.base",
        "alpasim_grpc",
        "alpasim_grpc.v0",
        "alpasim_grpc.v0.common_pb2",
    )
    modules = {name: ModuleType(name) for name in names}
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
        if "." in name:
            parent, attr = name.rsplit(".", 1)
            setattr(modules[parent], attr, module)
    base = modules["alpasim_driver.models.base"]
    base.CameraFrame = SimpleNamespace
    base.PredictionInput = SimpleNamespace
    base.DriveCommand = SimpleNamespace(STRAIGHT=object())
    proto = modules["alpasim_grpc.v0.common_pb2"]
    for name in ("PoseAtTime", "Pose", "Vec3", "Quat"):
        setattr(proto, name, SimpleNamespace)
    return base


def test_camera_context_pads_startup_then_keeps_recent_frames(driver_schema):
    avstack = AlpamayoAVStack(camera_ids=["front", "left"], context_length=3)
    images = []
    expected_history = [[0, 0, 0], [0, 0, 1], [0, 1, 2], [1, 2, 3]]
    for frame, expected in enumerate(expected_history):
        cameras = {
            cam: np.full((2, 3, 3), frame + offset, np.uint8)
            for cam, offset in (("front", 0), ("left", 10))
        }
        images.append(cameras)
        obs = Observation(t=frame * 0.1, frame=frame, sensor_data=cameras)
        inputs = avstack._prediction_input(obs)
        for cam in cameras:
            assert len(inputs.camera_images[cam]) == 3
            for camera_frame, index in zip(inputs.camera_images[cam], expected):
                assert camera_frame.image is images[index][cam]
                assert camera_frame.timestamp_us == avstack.EPOCH_US + index * 100_000
    assert inputs.command is driver_schema.DriveCommand.STRAIGHT


def test_injected_model_keeps_independent_adapter_history(driver_schema):
    prediction = SimpleNamespace(candidate_positions=np.array([[[1, 0, 0]]]), selected_plan=object())
    model = SimpleNamespace(predict=Mock(return_value=prediction))
    first, second = [AlpamayoAVStack(model=model, camera_ids=["front"]) for _ in range(2)]
    observations = [Observation(0.0, 0, sensor_data={"front": np.full((2, 3, 3), i, np.uint8)})
                    for i in (0, 1)]
    first.reset(observations[0])
    first(observations[0])
    second.reset(observations[1])
    second(observations[1])
    inputs = model.predict.call_args.args[0]
    assert inputs.inference_seed == 0 and inputs.previous_plan is None
    assert all(frame.image is observations[1].sensor_data["front"]
               for frame in inputs.camera_images["front"])
    assert first.models["policy"] is second.models["policy"] is model
    assert first._frames is not second._frames


def test_ego_history_is_past_to_present_in_the_model_clock(driver_schema):
    avstack = AlpamayoAVStack()
    obs = Observation(
        t=0.0, frame=0, ego_speed=2.0, vehicle_state=SimpleNamespace(x=4.0, y=-2.0, yaw=math.pi / 2)
    )

    inputs = avstack._prediction_input(obs)
    history = inputs.ego_pose_history
    times = [p.timestamp_us for p in history]

    assert inputs.speed == 2.0
    assert times == sorted(set(times))
    assert times[0] > 0
    assert times[-1] == avstack.EPOCH_US
    assert times[-1] - times[0] >= 1_500_000
    for timed_pose in history:
        seconds_ago = (times[-1] - timed_pose.timestamp_us) / 1e6
        assert timed_pose.pose.vec.x == pytest.approx(4.0)
        assert timed_pose.pose.vec.y == pytest.approx(-2.0 - 2.0 * seconds_ago)
        assert timed_pose.pose.quat.w == pytest.approx(math.cos(math.pi / 4))
        assert timed_pose.pose.quat.z == pytest.approx(math.sin(math.pi / 4))


@pytest.mark.parametrize("selected_index,frequency", [(None, 10), (1, 5)])
def test_selected_trajectory_uses_the_simulator_runtime_clock(selected_index, frequency):
    avstack = AlpamayoAVStack(output_frequency_hz=frequency)
    candidates = np.array([[[1, 0, 0], [2, 0, 0]], [[3, 1, 0], [6, 2, 0]]])
    prediction = SimpleNamespace(candidate_positions=candidates)
    if selected_index is not None:
        prediction.selected_index = selected_index

    waypoints = avstack._to_waypoints(prediction, Observation(t=7.25, frame=10))

    expected_positions = (
        [(1.0, 0.0, 0.0), (2.0, 0.0, 0.0)]
        if selected_index is None
        else [(3.0, 1.0, 0.0), (6.0, 2.0, 0.0)]
    )
    assert [xyz for xyz, _, _ in waypoints] == expected_positions
    assert [t for _, _, t in waypoints] == [
        7_250_000 + 1_000_000 // frequency,
        7_250_000 + 2_000_000 // frequency,
    ]


def test_prediction_chains_plans_and_reset_starts_a_fresh_experiment(driver_schema, monkeypatch):
    avstack = AlpamayoAVStack(camera_ids=["front"], context_length=2)
    prediction = SimpleNamespace(
        candidate_positions=np.array([[[1, 0, 0], [2, 0, 0]]]), selected_plan=object()
    )
    model = SimpleNamespace(predict=Mock(return_value=prediction))
    load = Mock(side_effect=lambda: setattr(avstack, "_model", model))
    monkeypatch.setattr(avstack, "_load", load)
    first = Observation(t=0.0, frame=0, sensor_data={"front": np.zeros((2, 3, 3), np.uint8)})
    second = Observation(t=0.1, frame=1, sensor_data={"front": np.ones((2, 3, 3), np.uint8)})

    assert avstack.component_log() is None
    avstack.reset(first)
    control = avstack(first)
    recorded = avstack.component_log()
    assert recorded["policy"] is not prediction
    np.testing.assert_array_equal(recorded["policy"].candidate_positions, prediction.candidate_positions)
    assert recorded["action"] == control and recorded["action"] is not control
    next_control = avstack(second)
    assert avstack.component_log()["action"] == next_control
    assert avstack.component_log()["action"] is not next_control
    initial_input, next_input = [c.args[0] for c in model.predict.call_args_list]
    assert initial_input.previous_plan is None and initial_input.inference_seed == 0
    assert next_input.previous_plan is prediction.selected_plan and next_input.inference_seed == 1
    assert control.trajectory == [
        ((1.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0), 100_000),
        ((2.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0), 200_000),
    ]

    avstack.reset(first)
    assert avstack.component_log() is None
    assert avstack.last_reasoning is None
    avstack(first)
    restarted_input = model.predict.call_args.args[0]
    assert restarted_input.previous_plan is None and restarted_input.inference_seed == 0
    assert all(
        f.image is first.sensor_data["front"] for f in restarted_input.camera_images["front"]
    )
    load.assert_called_once()


def test_policy_input_and_output_replacements_reach_model_and_future_context(driver_schema):
    avstack = AlpamayoAVStack(camera_ids=["front"], context_length=2)
    model_prediction = SimpleNamespace(
        candidate_positions=np.array([[[1, 0, 0]]]), selected_plan="original", reasoning_text="original"
    )
    replacement = SimpleNamespace(
        candidate_positions=np.array([[[4, 2, 0]]]), selected_plan="replacement", reasoning_text="modified"
    )
    avstack._model = SimpleNamespace(predict=Mock(return_value=model_prediction))
    obs = Observation(t=0.0, frame=0, sensor_data={"front": np.zeros((2, 3, 3), np.uint8)})
    changed_inputs = []
    calls = []

    class Session:
        def emit(self, stage, value, sensor=None):
            calls.append(stage)
            if stage == "policy.pre":
                # These are actual model fields, including context beyond the observation.
                value.command = "turn-left"
                value.route = ["route-point"]
                value.acceleration = -2.0
                value.ego_pose_history = ["custom-pose"]
                value.camera_images = {"front": ["custom-camera-history"]}
                changed_inputs.append(value)
                return value
            assert value is model_prediction
            return replacement

    avstack.reset(obs)
    with avstack.bind_hooks(Session()):
        control = avstack(obs)
        next_control = avstack(obs)

    assert calls == ["policy.pre", "policy.post", "policy.pre", "policy.post"]
    assert [call.args[0] for call in avstack._model.predict.call_args_list] == changed_inputs
    assert changed_inputs[0].previous_plan is None
    assert changed_inputs[1].previous_plan == "replacement"
    assert control.trajectory[0][0] == (4.0, 2.0, 0.0)
    assert next_control.trajectory == control.trajectory
    assert avstack.last_reasoning == "modified"
    recorded = avstack.component_log()["policy"]
    assert recorded is not replacement and recorded.selected_plan == replacement.selected_plan
    np.testing.assert_array_equal(recorded.candidate_positions, replacement.candidate_positions)
    assert avstack.models["policy"] is avstack._model
    assert avstack.native["model"] is avstack._model
    assert avstack.supported_stages == {"policy.pre", "policy.post"}

    # The binding is removed, so ordinary inference again consumes the original prediction.
    assert avstack(obs).trajectory[0][0] == (1.0, 0.0, 0.0)


def test_component_snapshots_survive_prediction_reuse_and_command_mutation(driver_schema):
    prediction = SimpleNamespace(candidate_positions=np.array([[[1.0, 2.0, 3.0]]]))
    avstack = AlpamayoAVStack(model=SimpleNamespace(predict=lambda _: prediction), camera_ids=["front"])
    observation = Observation(0.0, 0, sensor_data={"front": np.zeros((2, 3, 3), np.uint8)})
    avstack.reset(observation)
    command = avstack(observation)
    # A command handler is allowed to change the emitted command in place.
    command.trajectory.clear()
    command.brake = 1.0
    # Models can reuse their output buffers at the next inference.
    prediction.candidate_positions[:] = 99
    recorded = avstack.component_log()
    np.testing.assert_array_equal(recorded["policy"].candidate_positions, [[[1.0, 2.0, 3.0]]])
    assert recorded["action"].brake == 0.0
    assert recorded["action"].trajectory == [((1.0, 2.0, 3.0), (1.0, 0.0, 0.0, 0.0), 100000)]


def test_close_releases_history_outputs_and_borrowed_model_without_closing_it(monkeypatch):
    import sys
    import weakref
    from unittest.mock import Mock

    class Image:
        pass

    model = SimpleNamespace(close=Mock())
    avstack = AlpamayoAVStack(model=model, camera_ids=["front"])
    image = Image()
    reference = weakref.ref(image)
    avstack._frames["front"].append(image)
    avstack._prev_plan = image
    avstack._last_log = {"policy": image}
    avstack.last_reasoning = "reasoning"
    del image
    cuda = SimpleNamespace(is_initialized=Mock(side_effect=AssertionError("borrowed CUDA touched")))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))

    avstack.close()
    avstack.close()
    assert reference() is None
    assert avstack._model is None and avstack.models == {}
    assert not avstack._frames["front"]
    assert avstack._prev_plan is None and avstack.component_log() is None
    assert avstack.last_reasoning is None
    model.close.assert_not_called()
    cuda.is_initialized.assert_not_called()
    with pytest.raises(RuntimeError, match="closed"):
        avstack(Observation(0., 0))
    with pytest.raises(RuntimeError, match="closed"):
        avstack.reset(Observation(0., 0))


def test_close_owned_model_releases_before_configured_cuda_cache(monkeypatch):
    import sys
    import weakref
    from contextlib import contextmanager

    calls = []
    class Model:
        def close(self):
            calls.append("model.close")

    @contextmanager
    def device(value):
        calls.append(("device", value))
        yield
        calls.append("device.exit")

    def empty_cache():
        assert reference() is None
        calls.append("empty_cache")

    avstack = AlpamayoAVStack(device="cuda:1")
    model = Model()
    reference = weakref.ref(model)
    avstack._model = model  # stand-in for the model built by _load, without loading weights
    del model
    cuda = SimpleNamespace(is_initialized=lambda: True, device=device, empty_cache=empty_cache)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))
    avstack.close()
    avstack.close()
    assert calls == ["model.close", ("device", "cuda:1"), "empty_cache", "device.exit"]


@pytest.mark.parametrize("device,loaded,initialized", [
    ("cpu", True, True), ("cuda:0", False, True), ("cuda:0", True, False),
])
def test_close_does_not_initialize_or_flush_unowned_cuda(monkeypatch, device, loaded, initialized):
    import sys
    from unittest.mock import Mock

    avstack = AlpamayoAVStack(device=device)
    if loaded:
        avstack._model = object()  # SDK models need not provide close
    cuda = SimpleNamespace(is_initialized=Mock(return_value=initialized), device=Mock(), empty_cache=Mock())
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))
    avstack.close()
    cuda.device.assert_not_called()
    cuda.empty_cache.assert_not_called()
    if device == "cpu" or not loaded:
        cuda.is_initialized.assert_not_called()


def test_close_owned_model_failure_still_clears_wrapper_state():
    from unittest.mock import Mock

    model = SimpleNamespace(close=Mock(side_effect=ValueError("close failed")))
    avstack = AlpamayoAVStack(device="cpu")
    avstack._model = model
    avstack._frames[avstack.camera_ids[0]].append(object())
    avstack._last_log = {"policy": object()}
    with pytest.raises(ValueError, match="close failed"):
        avstack.close()
    assert avstack._model is None and avstack._last_log is None
    assert not any(avstack._frames.values())
    avstack.close()
    model.close.assert_called_once()
