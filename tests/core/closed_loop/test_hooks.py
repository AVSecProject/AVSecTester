"""Ordered hooks and lifecycle plugins through HookSet's public sessions."""

from dataclasses import replace

import pytest

from avsectester.data_types import Control, StateEstimate
from avsectester.closed_loop.avstacks.base import ObservationPipeline
from avsectester.closed_loop.hooks import Hook, HookSet, Plugin, UnsupportedStageError
from tests.fakes import ConstantAVStack, MemorySimulator


def test_with_hooks():
    base = HookSet([Hook("observation", lambda value, context: replace(
        value, ego_speed=value.ego_speed + 2.))])
    added = [Hook("observation", lambda value, context: replace(value, ego_speed=value.ego_speed * 3.))]
    outputs = []
    for hooks in (base, base.with_hooks(added), base.with_hooks(added, prepend=True)):
        with MemorySimulator().create_runtime() as simulator_runtime:
            with hooks.session(simulator_runtime, ConstantAVStack()).activate() as session:
                simulator_runtime.reset()
                outputs.append(session.observe(simulator_runtime.observe()).ego_speed)
    assert outputs[1] == outputs[0] * 3.
    assert outputs[2] == (outputs[0] - 2.) * 3. + 2.


def test_validate_stages():
    simulator_runtime = MemorySimulator().create_runtime()
    avstack = ConstantAVStack()
    avstack.observation_pipeline = ObservationPipeline(
        lambda value, context: StateEstimate(value.vehicle_state, value.ego_speed))
    available = HookSet([Hook("observation", lambda value, context: value)]).validate_stages(
        simulator_runtime, avstack)
    assert {"observation", "command", "simulator.setup", "localization.pre"} <= available
    with pytest.raises(UnsupportedStageError, match="Unsupported hook stages"):
        HookSet([Hook("not.a.stage", lambda value, context: value)]).validate_stages(
            simulator_runtime, avstack)
    with pytest.raises(TypeError, match="Hook handler"):
        Hook("command", object())


def test_session():
    events = []
    feedback = []

    class Capture(Plugin):
        def reset(self, context):
            events.append("reset")

        def transform(self, value, context):
            events.append("transform")
            context.record_output("reported_speed", value.ego_speed + 5.)
            return replace(value, ego_speed=value.ego_speed + 5.)

        def after_step(self, context):
            feedback.append((context.sim_time, context.observation.frame,
                             context.observation.ego_speed))

        def close(self):
            events.append("close")

    plugin = Capture()
    with MemorySimulator().create_runtime() as simulator_runtime:
        hooks = HookSet([Hook("observation", plugin),
                         Hook("command", lambda value, context: replace(value, brake=.4))])
        with hooks.session(simulator_runtime, ConstantAVStack()).activate() as session:
            simulator_runtime.reset()
            raw = simulator_runtime.observe()
            seen = session.observe(raw)
            assert seen.ego_speed == raw.ego_speed + 5.
            command = session.command(Control(throttle=.2))
            assert command == Control(throttle=.2, brake=.4)
            simulator_runtime.step(command)
            next_raw = simulator_runtime.observe()
            session.after_step(next_raw)
            assert feedback == [(simulator_runtime.clock.t, next_raw.frame, next_raw.ego_speed)]
            assert session.outputs[0].value == seen.ego_speed
        assert events == ["reset", "transform", "close"]
        # Invalid replacements stop the chain before downstream consumers see a wrong type.
        broken = HookSet([Hook("command", lambda value, context: 1),
                          Hook("command", lambda value, context: pytest.fail("invalid value escaped"))])
        with broken.session(simulator_runtime, ConstantAVStack()).activate() as session:
            with pytest.raises(TypeError, match="command must return Control"):
                session.command(Control())
        with HookSet([Hook("simulator.step.pre", lambda value, context: Control())]).session(
            simulator_runtime, ConstantAVStack()).activate():
            with pytest.raises(TypeError, match="must return None"):
                simulator_runtime.step(Control())
