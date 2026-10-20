"""Command compatibility and output capture through the unified closed-loop runner."""

from dataclasses import replace

import pytest

from avsectester.closed_loop.avstack import AVStack
from avsectester.closed_loop.runner import ClosedLoopRunner, RunRequest
from avsectester.closed_loop.datatypes import Control
from avsectester.closed_loop.hooks import Hook, Plugin, HookSet
from avsectester.closed_loop.simulators.carla import CarlaSimulator
from avsectester.closed_loop.simulators.nurec import (
    NuRecSimulatorRuntime,
    EgoPose, KinematicBicycle, NuRecSimulator, StubRenderer, TrajectoryFollower,
)
from avsectester.closed_loop.avstacks.alpamayo import AlpamayoAVStack


def drive(simulator_runtime, avstack, decision_steps, hooks=None):
    return ClosedLoopRunner().run(simulator_runtime, avstack, RunRequest(decision_steps, hooks=hooks))


def outputs(log):
    return [output for event in log.events for output in event.data.get("outputs", ())] + [
        output for step in log.steps for output in step.outputs]


PLAN = [((1.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0), 100_000)]


class CommandAVStack(AVStack):
    def __init__(self, modes=None, command=None):
        self.modes = modes
        self.command = command if command is not None else Control()

    @property
    def command_modes(self):
        return self.modes

    def __call__(self, observation):
        return replace(self.command)


class CountingSimulatorRuntime(NuRecSimulatorRuntime):
    def __init__(self, dynamics=None):
        super().__init__(NuRecSimulator({"dt": 0.1}, StubRenderer(height=2, width=3), dynamics))
        self.resets = 0

    def _initialize_world(self, source, initialization, environment):
        self.resets += 1
        return super()._initialize_world(source, initialization, environment)


@pytest.mark.parametrize("dynamics,mode", [
    (KinematicBicycle(), "trajectory"), (TrajectoryFollower(), "actuator"),
])
def test_declared_incompatible_pair_fails_before_world_reset(dynamics, mode):
    simulator_runtime = CountingSimulatorRuntime(dynamics)
    with pytest.raises(ValueError, match="commands unsupported"):
        drive(simulator_runtime, CommandAVStack(frozenset({mode})), decision_steps=1)
    assert simulator_runtime.resets == 0


def test_command_conversion_preserves_custom_runtime_flexibility():
    simulator_runtime = CountingSimulatorRuntime()
    avstack = CommandAVStack(frozenset({"trajectory"}), Control(trajectory=PLAN))
    hooks = HookSet([Hook("command", lambda command, context: Control(throttle=1.0))])
    run_log = drive(simulator_runtime, avstack, decision_steps=1, hooks=hooks)
    assert simulator_runtime.pose.x > 0
    assert run_log.steps[0].executed_command.throttle == 1.0


def test_final_command_is_checked_after_all_ordered_handlers():
    simulator_runtime = CountingSimulatorRuntime(TrajectoryFollower())
    avstack = CommandAVStack(frozenset({"trajectory"}), Control(trajectory=PLAN))
    attack = Hook("command", lambda command, context: replace(command, brake=1.0))
    with pytest.raises(ValueError, match="actuator command fields.*brake"):
        drive(simulator_runtime, avstack, decision_steps=1, hooks=HookSet([attack]))
    assert simulator_runtime.frame == 0

    correction = Hook("command", lambda command, context: replace(command, brake=0.0))
    run_log = drive(simulator_runtime, avstack, decision_steps=1, hooks=HookSet([attack, correction]))
    assert len(run_log.steps) == 1 and simulator_runtime.pose.x == pytest.approx(1.0)


@pytest.mark.parametrize("simulator_runtime,command", [
    (CarlaSimulator({}).create_runtime(), Control(trajectory=PLAN)),
    (NuRecSimulator({}).create_runtime(), Control(trajectory=PLAN)),
    (NuRecSimulator({}, dynamics=TrajectoryFollower()).create_runtime(), Control(steer=0.2)),
])
def test_direct_simulator_runtime_calls_reject_ignored_fields_before_world_changes(simulator_runtime, command):
    with pytest.raises(ValueError, match="does not support"):
        simulator_runtime.step(command)
    if isinstance(simulator_runtime, NuRecSimulatorRuntime):
        assert simulator_runtime.pose == EgoPose() and simulator_runtime.frame == 0


def test_trajectory_policy_mismatch_fails_without_loading_weights_or_world():
    simulator_runtime = CountingSimulatorRuntime()
    avstack = AlpamayoAVStack(checkpoint_path="unused-for-preflight")
    with pytest.raises(ValueError, match="AlpamayoAVStack.*trajectory.*unsupported"):
        drive(simulator_runtime, avstack, decision_steps=1)
    assert simulator_runtime.resets == 0 and avstack.models == {}


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

    simulator_runtime = CountingSimulatorRuntime(CustomDynamics())
    avstack = CommandAVStack(frozenset({"trajectory"}), Control(brake=1, trajectory=PLAN))
    assert simulator_runtime.command_modes is None
    drive(simulator_runtime, avstack, decision_steps=1)
    assert simulator_runtime.pose.x == 1


def test_hook_outputs_capture_timing_snapshots_and_independent_runs():
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
    hooks = HookSet([Hook("observation", recorder), Hook("command", recorder)])
    simulator_runtime = CountingSimulatorRuntime()
    first = drive(simulator_runtime, CommandAVStack(command=Control(throttle=1)), decision_steps=2, hooks=hooks)
    second = drive(simulator_runtime, CommandAVStack(command=Control(throttle=1)), decision_steps=2, hooks=hooks)
    expected = [
        (0, "observation", 0, 0), (0, "command", 0, 0), (0, "step.end", 1, 0.1),
        (1, "observation", 1, 0.1), (1, "command", 1, 0.1), (1, "step.end", 2, 0.2),
    ]
    assert [(v.step, v.stage, v.frame, v.t) for v in outputs(first)] == expected
    assert outputs(first) == outputs(second) and outputs(first)[0] is not outputs(second)[0]
    assert outputs(first)[0].value == ["observation"]
    recorder.history.append("later")
    assert outputs(first)[-1].value[-1] == "feedback"
    outputs(first)[0].value.append("review")
    assert outputs(second)[0].value == ["observation"]
    with pytest.raises(RuntimeError, match="active hook callback"):
        recorder.last_context.record_output("late", {})


def test_render_outputs_include_sensor_and_zero_step_run_setup():
    def render(image, context):
        context.record_output("sensor_summary", {"shape": image.shape})
        return image

    def setup(value, context):
        context.record_output("setup", {"x": context.physical_state.vehicle_state.x})
        with pytest.raises(ValueError, match="nonempty"):
            context.record_output(" ", {})

    simulator_runtime = CountingSimulatorRuntime()
    run_log = drive(simulator_runtime, CommandAVStack(), decision_steps=0, hooks=HookSet([
        Hook("simulator.setup", setup), Hook("render.post", render),
    ]))
    assert not run_log.steps
    assert [(v.name, v.stage, v.sensor) for v in outputs(run_log)] == [
        ("setup", "simulator.setup", None), ("sensor_summary", "render.post", "camera_front"),
    ]
