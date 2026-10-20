"""Run the actual modular pipeline through simulator, hook and attack lifecycles."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from avsectester.attack.runtime import AttackRuntime, AVStackAttackEffect, WorldAttackEffect
from avsectester.closed_loop.avstacks.modular import ModularAVStack
from avsectester.closed_loop.avstacks.native_hook import NativeHook
from avsectester.closed_loop.datatypes import Control, Observation, WorldSnapshot
from avsectester.closed_loop.hooks import Hook, HookSet
from avsectester.closed_loop.runner import ClosedLoopRunner, RunRequest
from avsectester.closed_loop.simulator import Simulator, SimulatorRuntime
from avsectester.datasets import ActorState
from avsectester.runlog import Recorder


@pytest.fixture
def simulation(pipeline_config, make_detections, make_ego):
    calls = []
    class World(Simulator):
        command_modes = frozenset({"actuator"})

        def create_runtime(self, seed=None):
            return WorldRuntime(self)

    class WorldRuntime(SimulatorRuntime):
        def _initialize_world(self, start, initialization, world):
            self.frame = 0
            self.speed = 0.
            self.fail = None
            calls.append("initialize")

        def _observe(self):
            from avstack.geometry import ReferenceFrame

            calls.append(("capture", self.frame))
            ego = make_ego(speed=self.speed, t=self.frame * .05)
            sensor_reference = ReferenceFrame(np.zeros(3), np.quaternion(1), ego.as_reference())
            return Observation(self.frame * .05, self.frame, ego_speed=self.speed,
                               vehicle_state=ego,
                               sensor_data={"lidar": make_detections([(30., 8., 0.)], frame=self.frame,
                                                                    reference=sensor_reference)})

        def _state(self):
            return WorldSnapshot(self.frame * .05, self.frame, ego_speed=self.speed,
                                 ego=ActorState(np.eye(4), (self.speed, 0., 0.), "vehicle", (4., 2., 1.5)))

        def _advance_native(self, control):
            calls.append(("control", control))
            if self.fail is not None:
                raise self.fail
            self.frame += 1
            self.speed += .5

        def close(self):
            self._capture = None
            self._world_initialized = False
            calls.append("close")

    return SimpleNamespace(calls=calls, runtime=World().create_runtime(),
                           avstack=ModularAVStack(pipeline_config))


def _run(simulation, decisions, *, hooks=None, attacks=(), recorder=None):
    try:
        return ClosedLoopRunner().run(simulation.runtime, simulation.avstack,
                                      RunRequest(decisions, hooks=hooks, attacks=attacks), recorder=recorder)
    finally:
        simulation.runtime.close()


def test_runner_records_each_decision_and_canonical_outputs(simulation):
    simulation.avstack.instrument()
    log = _run(simulation, 3)
    assert log.status == "completed"
    assert [step.step for step in log.steps] == [0, 1, 2]
    assert [step.physical_post.frame for step in log.steps] == [1, 2, 3]
    assert [step.physical_post.t for step in log.steps] == pytest.approx([.05, .10, .15])
    assert [step.physical_post.ego_speed for step in log.steps] == [.5, 1., 1.5]
    assert [len(step.detections) for step in log.steps] == [1, 1, 1]
    assert all("n_detections" not in step.telemetry for step in log.steps)
    assert len([call for call in simulation.calls if isinstance(call, tuple) and call[0] == "control"]) == 3
    assert simulation.calls[0] == "initialize" and simulation.calls[-1] == "close"


def test_runner_consumes_component_and_command_replacements(simulation):
    from avsectester.attack.implemented_attacks.phantom_injection_attack.attack import PhantomInjectionAttack

    simulation.avstack.instrument(("perception",))
    attack = PhantomInjectionAttack(decision_steps=2).create_runtime(None, None)
    hooks = HookSet([Hook("command", lambda value, context: Control(throttle=.3, steer=.1))])
    log = _run(simulation, 2, hooks=hooks, attacks=(attack,))
    assert [len(step.components["perception"]) for step in log.steps] == [2, 2]
    assert [len(step.detections) for step in log.steps] == [2, 2]
    assert [step.executed_command for step in log.steps] == [Control(throttle=.3, steer=.1)] * 2
    assert all(call[1].throttle == .3 for call in simulation.calls
               if isinstance(call, tuple) and call[0] == "control")


@pytest.mark.parametrize("append_first", [False, True])
def test_explicit_native_hook_and_framework_hook_preserve_order(simulation, monkeypatch, append_first):
    from avstack.config import HOOKS
    from avstack.datastructs import DataContainer

    class DoubleOutput:
        def __call__(self, output):
            return (DataContainer(output.frame, output.timestamp, list(output) * 2,
                                  output.source_identifier, source_reference=output.source_reference),)

    monkeypatch.setitem(HOOKS.module_dict, "DoubleOutputForTest", DoubleOutput)
    observed = []
    def count(output, context):
        observed.append(len(output))
        return output
    hooks = [Hook("perception.post", NativeHook({"type": "DoubleOutputForTest"})),
             Hook("perception.post", count)]
    log = _run(simulation, 2, hooks=HookSet(hooks if append_first else reversed(hooks)))
    assert observed == [2 if append_first else 1] * 2
    assert [len(step.detections) for step in log.steps] == [2, 2]
    assert len(simulation.avstack.pipeline.perception.post_hooks) == 1


def test_explicit_native_pre_hook_replaces_component_input(simulation, monkeypatch):
    from avstack.config import HOOKS
    from avstack.datastructs import DataContainer

    class RemoveInput:
        def __call__(self, output, **kwargs):
            assert len(output) == 1
            empty = DataContainer(output.frame, output.timestamp, [], output.source_identifier,
                                  source_reference=output.source_reference)
            return (empty,), kwargs

    monkeypatch.setitem(HOOKS.module_dict, "RemoveInputForTest", RemoveInput)
    hooks = HookSet([Hook("perception.pre", NativeHook({"type": "RemoveInputForTest"}, phase="pre"))])
    log = _run(simulation, 2, hooks=hooks)
    assert [len(step.detections) for step in log.steps] == [0, 0]


def test_runner_failure_records_error_and_releases_native_hooks(simulation):
    recorder = Recorder()
    def fail(control, context):
        raise RuntimeError("control failed")
    with pytest.raises(RuntimeError, match="control failed"):
        _run(simulation, 3, hooks=HookSet([Hook("command", fail)]), recorder=recorder)
    assert recorder.log.status == "failed"
    failure = next(event for event in recorder.log.events if event.kind == "failure")
    assert failure.data["message"] == "control failed"
    assert simulation.calls[-1] == "close"
    assert all(not module.pre_hooks for module in simulation.avstack.native.values()
               if hasattr(module, "pre_hooks"))


def test_world_setup_effect_visible_in_first_observation_and_cleaned(simulation):
    cleanup = []
    class Speed(WorldAttackEffect):
        def apply(self, simulator_runtime, resources):
            simulator_runtime.speed = 9.
            resources.callback(cleanup.append, "removed")
    class Attack(AttackRuntime):
        def setup(self, context):
            return (Speed(),)
        def step(self, context):
            return (AVStackAttackEffect("command", lambda control, context: replace(control, brake=.8, throttle=0)),)
    log = _run(simulation, 1, attacks=(Attack(),))
    assert log.steps[0].raw_observation.ego_speed == 9.
    assert log.steps[0].physical_pre.ego_speed == 9.
    assert log.steps[0].executed_command.brake == .8
    assert cleanup == ["removed"]
