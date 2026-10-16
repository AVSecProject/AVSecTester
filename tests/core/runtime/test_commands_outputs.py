"""Command compatibility and intervention records across the real offline run loop."""

from dataclasses import replace

import pytest

from avsectester.backend import AVStack, run
from avsectester.plane import Control
from avsectester.runner import run_pair
from avsectester.runtime import Hook, Plugin, Runtime
from avsectester.simulators.carla import CarlaBackend
from avsectester.simulators.nurec import (
    EgoPose, KinematicBicycle, NuRecBackend, StubRenderer, TrajectoryFollower,
)
from avsectester.stacks.alpamayo import AlpamayoAVStack


PLAN = [((1.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0), 100_000)]


class CommandStack(AVStack):
    def __init__(self, modes=None, command=None):
        self.modes = modes
        self.command = command if command is not None else Control()

    @property
    def command_modes(self):
        return self.modes

    def __call__(self, observation):
        return replace(self.command)


class CountingBackend(NuRecBackend):
    def __init__(self, dynamics=None):
        super().__init__({"dt": 0.1}, StubRenderer(height=2, width=3), dynamics)
        self.prepares = self.resets = 0

    def prepare_clean_attack_pair(self):
        self.prepares += 1

    def _initialize_world(self, source, initialization, environment):
        self.resets += 1
        return super()._initialize_world(source, initialization, environment)


@pytest.mark.parametrize("dynamics,mode", [
    (KinematicBicycle(), "trajectory"), (TrajectoryFollower(), "actuator"),
])
def test_declared_incompatible_pair_fails_before_preparation(dynamics, mode):
    backend = CountingBackend(dynamics)
    with pytest.raises(ValueError, match="commands unsupported"):
        run_pair(
            backend, lambda: CommandStack(frozenset({mode})), frames=1,
            clean_runtime=Runtime(), attacked_runtime=Runtime(),
        )
    assert backend.prepares == backend.resets == 0


def test_command_conversion_preserves_custom_runtime_flexibility():
    backend = CountingBackend()
    stack = CommandStack(frozenset({"trajectory"}), Control(trajectory=PLAN))
    runtime = Runtime([Hook("command", lambda command, context: Control(throttle=1.0))])
    trace = run(backend, stack, frames=1, runtime=runtime)
    assert backend.pose.x > 0
    assert trace.records[0].throttle == 1.0


def test_final_command_is_checked_after_attack_and_defense_handlers():
    backend = CountingBackend(TrajectoryFollower())
    stack = CommandStack(frozenset({"trajectory"}), Control(trajectory=PLAN))
    attack = Hook("command", lambda command, context: replace(command, brake=1.0))
    with pytest.raises(ValueError, match="actuator command fields.*brake"):
        run(backend, stack, frames=1, runtime=Runtime([attack]))
    assert backend.frame == 0

    defense = Hook("command", lambda command, context: replace(command, brake=0.0))
    trace = run(backend, stack, frames=1, runtime=Runtime([attack, defense]))
    assert len(trace.records) == 1 and backend.pose.x == pytest.approx(1.0)


@pytest.mark.parametrize("backend,command", [
    (CarlaBackend({}), Control(trajectory=PLAN)),
    (NuRecBackend({}), Control(trajectory=PLAN)),
    (NuRecBackend({}, dynamics=TrajectoryFollower()), Control(steer=0.2)),
])
def test_direct_backend_calls_reject_ignored_fields_before_world_changes(backend, command):
    with pytest.raises(ValueError, match="does not support"):
        backend.step(command)
    if isinstance(backend, NuRecBackend):
        assert backend.pose == EgoPose() and backend.frame == 0


def test_trajectory_policy_mismatch_fails_without_loading_weights_or_world():
    backend = CountingBackend()
    stack = AlpamayoAVStack(checkpoint_path="unused-for-preflight")
    with pytest.raises(ValueError, match="AlpamayoAVStack.*trajectory.*unsupported"):
        run(backend, stack, frames=1)
    assert backend.resets == 0 and stack.models == {}


def test_direct_dynamics_enforce_modes_and_accept_neutral_trajectory_commands():
    follower = TrajectoryFollower()
    assert follower.step(EgoPose(speed=5), Control(), 0.1).speed == pytest.approx(4.6)
    assert follower.step(EgoPose(speed=5), Control(trajectory=[]), 0.1).speed == pytest.approx(4.6)
    with pytest.raises(ValueError, match="trajectory"):
        KinematicBicycle().step(EgoPose(), Control(trajectory=PLAN), 0.1)
    with pytest.raises(ValueError, match="actuator"):
        follower.step(EgoPose(), Control(brake=1), 0.1)


def test_custom_dynamics_can_keep_undeclared_command_semantics():
    class CustomDynamics:
        def step(self, pose, control, dt):
            return replace(pose, x=pose.x + control.brake, t=pose.t + dt)

    backend = CountingBackend(CustomDynamics())
    stack = CommandStack(frozenset({"trajectory"}), Control(brake=1, trajectory=PLAN))
    assert backend.command_modes is None
    run(backend, stack, frames=1)
    assert backend.pose.x == 1


def test_intervention_outputs_capture_timing_snapshots_and_independent_runs():
    class Recorder(Plugin):
        def reset(self, context):
            self.history = []

        def transform(self, value, context):
            self.history.append(context.stage)
            context.record_output("decision", self.history)
            self.last_context = context
            return value

        def after_step(self, context):
            self.history.append("feedback")
            context.record_output("feedback", self.history)

    recorder = Recorder()
    runtime = Runtime([Hook("observation", recorder), Hook("command", recorder)])
    backend = CountingBackend()
    first = run(backend, CommandStack(command=Control(throttle=1)), frames=2, runtime=runtime)
    second = run(backend, CommandStack(command=Control(throttle=1)), frames=2, runtime=runtime)
    expected = [
        (0, "observation", 0, 0), (0, "command", 0, 0), (0, "step.end", 1, 0.1),
        (1, "observation", 1, 0.1), (1, "command", 1, 0.1), (1, "step.end", 2, 0.2),
    ]
    assert [(v.step, v.stage, v.frame, v.t) for v in first.outputs] == expected
    assert first.outputs == second.outputs and first.outputs is not second.outputs
    assert first.outputs[0].value == ["observation"]
    recorder.history.append("later")
    assert first.outputs[-1].value[-1] == "feedback"
    first.outputs[0].value.append("review")
    assert second.outputs[0].value == ["observation"]
    with pytest.raises(RuntimeError, match="active runtime callback"):
        recorder.last_context.record_output("late", {})


def test_render_outputs_include_sensor_and_zero_step_run_setup():
    def render(image, context):
        context.record_output("sensor_summary", {"shape": image.shape})
        return image

    def setup(value, context):
        context.record_output("setup", {"x": context.ground_truth.vehicle_state.x})
        with pytest.raises(ValueError, match="nonempty"):
            context.record_output(" ", {})

    backend = CountingBackend()
    trace = run(backend, CommandStack(), frames=0, runtime=Runtime([
        Hook("world.setup", setup), Hook("render.post", render),
    ]))
    assert not trace.records
    assert [(v.name, v.stage, v.sensor) for v in trace.outputs] == [
        ("setup", "world.setup", None), ("sensor_summary", "render.post", "camera_front"),
    ]


def test_world_conditions_are_explicit_detached_and_backend_specific():
    backend = CarlaBackend({"weather": "ClearNoon"})
    backend._pair_scenario = {"weather": "ClearNoon"}
    backend.replay_scenario = {"weather": "ClearNoon"}
    settings = {"weather": {"fog_density": 50}}
    backend.configure_conditions(settings)
    settings["weather"]["fog_density"] = 100
    assert backend.scenario["weather"] == {"fog_density": 50}
    assert backend._pair_scenario["weather"] == backend.replay_scenario["weather"] == {"fog_density": 50}
    with pytest.raises(ValueError, match="Unsupported CARLA.*traffic"):
        backend.configure_conditions({"traffic": 10})
    nurec = CountingBackend()
    nurec.configure_conditions({})
    with pytest.raises(ValueError, match="does not support world conditions"):
        nurec.configure_conditions({"weather": "HardRainNoon"})
