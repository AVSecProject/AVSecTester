"""Ordered hooks and lifecycle plugins through HookSet's public sessions."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Event

import pytest

from avsectester.data_types import Control, RosMessage, StateEstimate
from avsectester.closed_loop.avstacks.base import ObservationPipeline
from avsectester.closed_loop.hooks import Hook, HookSet, Plugin, UnsupportedStageError
from avsectester.closed_loop.avstacks.implemented.external_control import ExternalAVStack
from avsectester.closed_loop.simulators.messages import MessageStage
from tests.fakes import (
    ConstantAVStack, ExternallyControlledMemorySimulator, MemoryMessageSimulator, MemorySimulator,
)


def test_hook():
    callback = lambda value, context: value
    hook = Hook("observation", callback)
    assert hook.stage == "observation" and hook.handler is callback
    with pytest.raises(ValueError, match="stage"):
        Hook("", callback)
    with pytest.raises(TypeError, match="Hook handler"):
        Hook("command", object())


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
    with MemorySimulator().create_runtime() as simulator_runtime:
        avstack = ConstantAVStack()
        avstack.observation_pipeline = ObservationPipeline(
            lambda value, context: StateEstimate(value.vehicle_state, value.ego_speed))
        available = HookSet([Hook("observation", lambda value, context: value)]).validate_stages(
            simulator_runtime, avstack)
        assert {"observation", "command", "simulator.setup", "localization.pre"} <= available
        with pytest.raises(UnsupportedStageError, match="Unsupported hook stages"):
            HookSet([Hook("not.a.stage", lambda value, context: value)]).validate_stages(
                simulator_runtime, avstack)
    with ExternallyControlledMemorySimulator().create_runtime() as runtime:
        driver = ExternalAVStack()
        hooks = HookSet([Hook("simulator.step.pre", lambda value, context: value)])
        assert "simulator.step.pre" in hooks.validate_stages(runtime, driver)
        with pytest.raises(UnsupportedStageError, match="do not reach the external"):
            HookSet([Hook("observation", lambda value, context: value)]).validate_stages(runtime, driver)


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
            with pytest.raises(RuntimeError, match="already active"):
                with session.activate():
                    pytest.fail("an active hook session was entered twice")
            with pytest.raises(ValueError, match="unsupported stage"):
                session.emit("unavailable.stage", None)
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
            context = session.stage_context("observation")
        with pytest.raises(RuntimeError, match="active hook callback"):
            context.record_output("late", 1.)
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


def test_message_hooks():
    stages = {"/input": MessageStage("imu"), "/observed": MessageStage(modifiable=False)}
    value = RosMessage("std_msgs/msg/Int32", {"count": 3})

    def replace_message(message, context):
        assert context.sensor == "imu"
        context.record_output("source_count", message.fields["count"])
        return replace(message, fields={"count": 9})

    def observe_message(message, context):
        message.fields["count"] = 11
        return message

    hooks = HookSet((Hook("/input", replace_message), Hook("/observed", observe_message)))
    with MemoryMessageSimulator(stages).create_runtime() as runtime:
        runtime.reset()
        state = runtime.state()
        outputs = []
        with hooks.session(runtime, ExternalAVStack()).activate() as session:
            modified = session.emit_message("/input", value, state, step=0, outputs=outputs)
            assert modified == RosMessage(value.type, {"count": 9})
            assert [item.value for item in outputs] == [3]
            assert session.emit_message("/observed", value, state, step=0, outputs=outputs) is value
            assert value.fields == {"count": 3}
        for stage, invalid, error in (
            ("/input", lambda message, _: replace(message, type="std_msgs/msg/String"),
             "handlers must return RosMessage"),
            ("/observed", lambda message, _: replace(message), "is read-only"),
        ):
            with HookSet((Hook(stage, invalid),)).session(runtime, ExternalAVStack()).activate() as session:
                with pytest.raises(TypeError, match=error):
                    session.emit_message(stage, value, state, step=0, outputs=outputs)


def test_message_and_world_callbacks_are_serialized():
    world_entered, receive_started = Event(), Event()
    callbacks = []

    def world(value, context):
        world_entered.set()
        assert receive_started.wait(2.)
        callbacks.append("world")
        return value

    def message(value, context):
        callbacks.append("message")
        return value

    with MemoryMessageSimulator({"/input": MessageStage()}).create_runtime() as runtime:
        runtime.reset()
        hooks = HookSet((Hook("simulator.step.pre", world), Hook("/input", message)))
        dispatcher = hooks.session(runtime, ExternalAVStack())
        state = runtime.state()

        def receive():
            assert world_entered.wait(2.)
            receive_started.set()
            return dispatcher.emit_message(
                "/input", RosMessage("std_msgs/msg/Bool", {"data": True}), state, step=0, outputs=[])

        with dispatcher.activate(), ThreadPoolExecutor(max_workers=1) as executor:
            reply = executor.submit(receive)
            dispatcher.emit("simulator.step.pre", None)
            assert reply.result(timeout=2.).fields == {"data": True}
    assert callbacks == ["world", "message"]
