"""Attack and defense execution contracts without simulator or model dependencies."""

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from avsectester.backend import AVStack, WorldBackend, run
from avsectester.plane import Control, Observation, StateEstimate
from avsectester.runtime import Hook, Plugin, Runtime


class MemoryBackend(WorldBackend):
    """An intentionally mutable physical state, with a simulator clock offset."""

    def __init__(self):
        self.reset_calls = 0
        self.controls = []
        self.pose = SimpleNamespace(x=0.0, speed=0.0)
        self.frame = 41
        self.t = 7.0

    def reset(self):
        self.reset_calls += 1
        self.controls.clear()
        self.pose = SimpleNamespace(x=0.0, speed=0.0)
        self.frame, self.t = 41, 7.0
        return self.observe()

    def observe(self):
        return Observation(
            t=self.t, frame=self.frame, vehicle_state=self.pose, ego_speed=self.pose.speed,
            sensor_data={"position": self.pose.x, "camera": np.zeros((2, 3, 3), np.uint8)},
            calibration={"camera": {"focal_length": 10.0}},
        )

    def step(self, control):
        self.controls.append(deepcopy(control))
        self.pose.speed = max(0.0, self.pose.speed + control.throttle - control.brake)
        self.pose.x += self.pose.speed
        self.frame += 1
        self.t += 0.1
        return self.observe()


class RecordingStack(AVStack):
    def __init__(self, policy=None):
        self.policy = policy or (lambda observation: Control(throttle=1.0))
        self.seen = []

    def reset(self, observation):
        self.initial = observation
        self.seen.clear()

    def __call__(self, observation):
        self.seen.append(observation)
        return self.policy(observation)


def test_shared_plugin_lifecycle_and_seed_restart_for_each_run():
    class StatefulPlugin(Plugin):
        def __init__(self):
            self.resets = self.closes = 0
            self.episodes = []

        def reset(self, context):
            self.resets += 1
            self.history = []
            self.feedback_steps = []

        def transform(self, value, context):
            self.history.append((context.stage, context.step, context.rng.random()))
            return value

        def after_step(self, context):
            self.feedback_steps.append(context.step)

        def close(self):
            self.closes += 1
            self.episodes.append((self.history[:], self.feedback_steps[:]))

    plugin = StatefulPlugin()
    runtime = Runtime([Hook("observation", plugin), Hook("command", plugin)], seed=73)
    for _ in range(2):
        run(MemoryBackend(), RecordingStack(), frames=2, runtime=runtime)

    assert plugin.resets == plugin.closes == 2
    assert plugin.episodes[0] == plugin.episodes[1]
    history, feedback = plugin.episodes[0]
    assert [(stage, step) for stage, step, _ in history] == [
        ("observation", 0), ("command", 0), ("observation", 1), ("command", 1),
    ]
    assert feedback == [0, 1]


@pytest.mark.parametrize("failure_stage", ["reset", "transform"])
def test_failure_cleans_resources_and_preserves_original_exception(failure_stage):
    failure = RuntimeError("attack failed")
    closed = []

    class BrokenPlugin(Plugin):
        def reset(self, context):
            if failure_stage == "reset":
                raise failure

        def transform(self, value, context):
            raise failure

        def close(self):
            closed.append(True)
            raise ValueError("cleanup failed")

    backend, stack = MemoryBackend(), RecordingStack()
    with pytest.raises(RuntimeError, match="attack failed") as caught:
        run(backend, stack, frames=1, runtime=Runtime([Hook("observation", BrokenPlugin())]))

    assert caught.value is failure
    assert any("cleanup failed" in note for note in caught.value.__notes__)
    assert closed == [True]
    assert backend.reset_calls == (0 if failure_stage == "reset" else 1)
    assert backend.controls == []
    assert backend._runtime is None and stack._runtime is None


@pytest.mark.parametrize("stage", ["tracking.post", "localization.pre"])
def test_unavailable_stage_is_rejected_before_world_reset(stage):
    backend = MemoryBackend()
    runtime = Runtime([Hook(stage, lambda value, context: value)])
    with pytest.raises(ValueError, match="Unsupported runtime stages"):
        run(backend, RecordingStack(), frames=1, runtime=runtime)
    assert backend.reset_calls == 0


def test_initial_input_is_transformed_once_and_shared_by_reset_and_first_decision():
    calls = []

    def attack(observation, context):
        calls.append((context.step, observation.frame))
        return replace(observation, ego_speed=99.0)

    stack = RecordingStack()
    run(MemoryBackend(), stack, frames=3, runtime=Runtime([Hook("observation", attack)]))

    assert calls == [(0, 41), (1, 42), (2, 43)]
    assert stack.initial is stack.seen[0]
    assert [observation.ego_speed for observation in stack.seen] == [99.0] * 3


def test_observation_context_tracks_localization_and_prior_handler_replacements():
    received = []

    def localize(observation, context):
        return StateEstimate(SimpleNamespace(x=3.0), 4.0)

    def attack(observation, context):
        assert context.observation is observation
        assert context.control is None
        received.append(context.observation.ego_speed)
        return replace(observation, ego_speed=8.0)

    def defense(observation, context):
        assert context.observation is observation
        received.append(context.observation.ego_speed)
        return observation

    run(
        MemoryBackend(), RecordingStack(), frames=2,
        perturb=lambda observation: replace(observation, ego_speed=5.0),
        runtime=Runtime(
            [Hook("observation", attack), Hook("observation", defense)], localizer=localize,
        ),
    )
    assert received == [5.0, 8.0, 5.0, 8.0]


@pytest.mark.parametrize("configured_runtime", [False, True])
def test_empty_processing_chain_preserves_input_and_physical_response(configured_runtime):
    backend, stack = MemoryBackend(), RecordingStack()
    trace = run(backend, stack, frames=2, runtime=Runtime() if configured_runtime else None)

    assert [record.speed for record in trace.records] == [1.0, 2.0]
    assert [observation.sensor_data["position"] for observation in stack.seen] == [0.0, 1.0]
    assert stack.initial.calibration == {"camera": {"focal_length": 10.0}}
    # Model-visible objects never alias the backend's mutable physical state.
    assert stack.initial.vehicle_state.x == 0.0
    assert backend.pose.x == 3.0


def test_attack_and_defense_execute_in_registration_order():
    def attack(value, context):
        return replace(value, ego_speed=value.ego_speed + 10.0)

    def defense(value, context):
        return replace(value, ego_speed=value.ego_speed * 0.5)

    observed = []
    for handlers in ((attack, defense), (defense, attack)):
        stack = RecordingStack()
        run(
            MemoryBackend(), stack, frames=1,
            runtime=Runtime([Hook("observation", handler) for handler in handlers]),
        )
        observed.append(stack.seen[0].ego_speed)

    assert observed == [5.0, 10.0]


def test_sensor_and_localization_replacements_change_drive_without_moving_true_state():
    order = []

    def sensor_attack(observation, context):
        order.append("sensor")
        observation.sensor_data["position"] = 20.0
        observation.sensor_data["camera"][:] = 200
        observation.calibration["camera"]["focal_length"] = 100.0
        return observation

    def localization_pre(observation, context):
        order.append("pre")
        observation.sensor_data["position"] += 2.0
        return observation

    def localize(observation, context):
        order.append("localizer")
        return StateEstimate(SimpleNamespace(x=observation.sensor_data["position"]), 8.0)

    def localization_post(estimate, context):
        order.append("post")
        estimate.vehicle_state.x += 1.0
        return estimate

    backend = MemoryBackend()
    stack = RecordingStack(lambda obs: Control(brake=1.0) if obs.vehicle_state.x > 10 else Control(throttle=1.0))
    runtime = Runtime(
        [Hook("sensors.post", sensor_attack), Hook("localization.pre", localization_pre),
         Hook("localization.post", localization_post)],
        localizer=localize,
    )
    trace = run(backend, stack, frames=1, runtime=runtime)

    assert order == ["sensor", "pre", "localizer", "post"]
    assert stack.seen[0].vehicle_state.x == 23.0
    assert stack.seen[0].ego_speed == 8.0
    assert stack.seen[0].calibration["camera"]["focal_length"] == 100.0
    assert np.all(stack.seen[0].sensor_data["camera"] == 200)
    assert backend.pose.x == backend.pose.speed == 0.0
    assert backend.ground_truth().vehicle_state.x == 0.0
    assert trace.records[0].brake == 1.0 and trace.records[0].speed == 0.0


@pytest.mark.parametrize("state_source", ["estimator", "backend"])
def test_localizer_cached_state_is_detached_before_downstream_estimate_edits(state_source):
    backend = MemoryBackend()
    cache = SimpleNamespace(x=2.0)

    def localize(observation, context):
        state = cache if state_source == "estimator" else context.backend.pose
        return StateEstimate(state, 0.0)

    def attack(estimate, context):
        estimate.vehicle_state.x = 50.0
        return estimate

    stack = RecordingStack()
    run(
        backend, stack, frames=1,
        runtime=Runtime([Hook("localization.post", attack)], localizer=localize),
    )
    assert stack.initial.vehicle_state.x == 50.0
    assert cache.x == 2.0
    assert backend.pose.x == 1.0


def test_nested_adapter_binding_rejects_without_unbinding_active_session():
    backend, stack = MemoryBackend(), RecordingStack()
    outer = Runtime().session(backend, stack)
    inner = Runtime().session(backend, stack)

    with outer.activate():
        with pytest.raises(RuntimeError, match="already bound"):
            with inner.activate():
                pytest.fail("Nested session must not become active")
        assert backend._runtime is outer
        assert stack._runtime is outer
    assert backend._runtime is None and stack._runtime is None


def test_world_setup_changes_scene_before_initial_render():
    from avsectester.simulators.nurec import NuRecBackend, StubRenderer

    rendered_positions = []

    class Renderer(StubRenderer):
        def render(self, pose, camera):
            rendered_positions.append(pose.x)
            return super().render(pose, camera)

    def setup(value, context):
        assert value is None
        context.backend.pose.x = 12.0

    backend = NuRecBackend({"dt": 0.1}, renderer=Renderer(height=2, width=3))
    stack = RecordingStack()
    run(backend, stack, frames=1, runtime=Runtime([Hook("world.setup", setup)]))

    assert rendered_positions[0] == 12.0
    assert stack.initial.vehicle_state.x == 12.0
    assert backend.pose.x > 12.0


def test_command_replacement_is_actuated_and_recorded_despite_observer_mutation():
    observed = []

    def command_attack(command, context):
        return Control(throttle=0.25, steer=0.4)

    def observer(index, observation, command):
        observed.append((index, command.throttle, command.steer, observation.vehicle_state.x))
        command.throttle = 100.0
        observation.vehicle_state.x = 100.0
        observation.sensor_data["camera"][:] = 255

    backend, stack = MemoryBackend(), RecordingStack()
    trace = run(
        backend, stack, frames=2, on_step=observer,
        runtime=Runtime([Hook("command", command_attack)]),
    )

    assert observed == [(0, 0.25, 0.4, 0.0), (1, 0.25, 0.4, 0.25)]
    assert [control.throttle for control in backend.controls] == [0.25, 0.25]
    assert [record.throttle for record in trace.records] == [0.25, 0.25]
    assert [record.speed for record in trace.records] == [0.25, 0.5]
    assert backend.pose.x == 0.75
    assert all(np.all(observation.sensor_data["camera"] == 0) for observation in stack.seen)


def test_feedback_receives_next_unattacked_state_and_true_clock():
    feedback = []

    class SpoofedClock(Plugin):
        def transform(self, observation, context):
            assert context.sim_time == pytest.approx(7.0 + context.step * 0.1)
            return replace(observation, t=-100.0, frame=-99, ego_speed=50.0)

        def after_step(self, context):
            feedback.append(context)
            # A saved or mutated context snapshot must not affect later physics.
            context.ground_truth.vehicle_state.x = 100.0
            context.observation.vehicle_state.x = 100.0

    backend = MemoryBackend()
    run(backend, RecordingStack(), frames=2, runtime=Runtime([Hook("observation", SpoofedClock())]))

    assert [context.step for context in feedback] == [0, 1]
    assert [context.observation.frame for context in feedback] == [42, 43]
    assert [context.observation.ego_speed for context in feedback] == [1.0, 2.0]
    assert [context.sim_time for context in feedback] == pytest.approx([7.1, 7.2])
    assert [context.ground_truth.frame for context in feedback] == [42, 43]
    assert backend.pose.x == 3.0


def test_feedback_plugins_receive_independent_raw_observations_and_executed_commands():
    received = []

    class Feedback(Plugin):
        def transform(self, value, context):
            return value

        def after_step(self, context):
            received.append((context.observation.vehicle_state.x, context.control.throttle))
            context.observation.vehicle_state.x = 99.0
            context.control.throttle = 99.0

    backend = MemoryBackend()
    run(
        backend, RecordingStack(), frames=2,
        runtime=Runtime([Hook("observation", Feedback()), Hook("command", Feedback())]),
    )
    assert received == [(1.0, 1.0), (1.0, 1.0), (3.0, 1.0), (3.0, 1.0)]
    assert backend.pose.x == 3.0


def test_world_post_context_cannot_rewrite_the_executed_command():
    from avsectester.simulators.nurec import NuRecBackend, StubRenderer

    def world_post(value, context):
        assert context.control.throttle == 1.0
        context.control.throttle = 0.0
        return None

    backend = NuRecBackend({"dt": 0.1}, renderer=StubRenderer(height=2, width=3))
    trace = run(
        backend, RecordingStack(), frames=2,
        runtime=Runtime([Hook("world.step.post", world_post)]),
    )
    assert [record.throttle for record in trace.records] == [1.0, 1.0]
    assert trace.final_speed == pytest.approx(0.6)


def test_zero_frames_skips_model_input_transforms_and_actuation():
    calls = []

    class Attack(Plugin):
        def reset(self, context):
            calls.append("reset")

        def transform(self, value, context):
            calls.append("transform")
            return value

        def after_step(self, context):
            calls.append("after_step")

        def close(self):
            calls.append("close")

    backend, stack = MemoryBackend(), RecordingStack()
    trace = run(backend, stack, frames=0, runtime=Runtime([Hook("observation", Attack())]))

    assert calls == ["reset", "close"]
    assert trace.records == [] and backend.controls == [] and stack.seen == []
    assert stack.initial.frame == 41


def test_invalid_stage_output_stops_before_next_handler_or_model():
    later_calls = []
    runtime = Runtime([
        Hook("observation", lambda observation, context: None),
        Hook("observation", lambda value, context: later_calls.append(value)),
    ])
    backend, stack = MemoryBackend(), RecordingStack()

    with pytest.raises(TypeError, match="observation must return Observation"):
        run(backend, stack, frames=1, runtime=runtime)

    assert later_calls == []
    assert stack.seen == [] and backend.controls == []
