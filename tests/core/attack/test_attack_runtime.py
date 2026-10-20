"""Online phase: one runtime owns its state; effects apply at their points of action."""

from contextlib import ExitStack
from dataclasses import FrozenInstanceError, dataclass, replace
from types import SimpleNamespace

import numpy as np
import pytest

from avsectester.attack.runtime import (
    OnlineContext, AttackDispatcher, AttackRuntime, AVStackAttackEffect, ResultSubscription,
    SensorAttackEffect, WorldAttackEffect,
)
from avsectester.closed_loop.datatypes import Control, Observation, WorldSnapshot
from avsectester.closed_loop.hooks import HookDispatcher, UnsupportedStageError
from avsectester.closed_loop.simulator import SensorView, Simulator, SimulatorRuntime


AVAILABLE = HookDispatcher.CORE_STAGES | {
    "perception.post", "tracking.post", "simulator.setup", "simulator.step.pre",
}


def data(step=0, stage="step"):
    return OnlineContext(
        step, stage, 7.0 + step * 0.1, 41 + step,
        observation=Observation(7.0 + step * 0.1, 41 + step, sensor_data={"camera": [1, 2]}),
        physical_state=WorldSnapshot(7.0 + step * 0.1, 41 + step, ego_speed=10.0),
    )


class WorldSimulator(Simulator):
    def create_runtime(self, seed=None):
        return World()


class World(SimulatorRuntime):
    """A simulator runtime double with native access and a deployment tick counter."""

    def __init__(self):
        super().__init__(WorldSimulator())
        self._native = {"objects": []}
        self.ticks = 0
        self.frame = 41
        # Tests drive dispatcher.setup directly, as if inside reset's simulator.setup.
        self._resetting = True

    @property
    def native(self):
        return self._native

    def setup_tick(self):
        self.ticks += 1
        self.frame += 1

    def _state(self):
        return WorldSnapshot(7.0, self.frame)

    def sensor_view(self, sensor):
        return SensorView(sensor, np.zeros((2, 2, 3), np.uint8), None, np.eye(4), {}, None, 7.0, 41)

    def _initialize_world(self, start, initialization, environment):
        pass

    def _advance_native(self, control):
        pass

    def _observe(self):
        return Observation(7.0, self.frame)


def emit(dispatcher, stage, value, context, simulator_runtime=None):
    native = SimpleNamespace(
        step=context.step, stage=stage, sim_time=context.sim_time,
        physical_state=context.physical_state or WorldSnapshot(context.sim_time, context.frame),
        simulator_runtime=simulator_runtime,
    )
    for hook in dispatcher.hooks:
        if hook.stage == stage:
            value = hook.handler(value, native)
    return value


def started(attacks, available=AVAILABLE, world=None, **kwargs):
    dispatcher = AttackDispatcher(attacks, **kwargs)
    dispatcher.validate(available)
    dispatcher.setup(data(0, "setup"), world)
    return dispatcher


@dataclass
class Panel(WorldAttackEffect):
    name: str
    setup_ticks: int = 1

    def apply(self, simulator, resources: ExitStack):
        for _ in range(self.setup_ticks):
            simulator.setup_tick()
        simulator.native["objects"].append(self.name)
        resources.callback(simulator.native["objects"].remove, self.name)


@dataclass
class Fill(SensorAttackEffect):
    sensor: str
    value: int

    def apply(self, data, view):
        assert (view.data == 0).all()
        changed = data.copy()
        changed[0, 0] = changed[0, 0] + self.value
        return changed


def test_ttc_component_effect_is_a_bound_method_sharing_runtime_state():
    class TTC(AttackRuntime):
        subscriptions = (ResultSubscription("physical_state"),)

        def __init__(self):
            self.history = []
            self.stopped = False

        def step(self, context):
            self.speed = context.physical_state.ego_speed
            return (AVStackAttackEffect("perception.post", self.inject),
                    AVStackAttackEffect("command", lambda value, context: Control(brake=1.0)))

        def inject(self, detections, context):
            assert context.stage == "perception.post"
            self.history.append(detections[0]["distance"] / self.speed)
            return [{"distance": 0.1}]

        def stop(self, context):
            self.stopped = True

    attack = TTC()
    dispatcher = started([attack])
    for index in range(2):
        context = data(index)
        dispatcher.step(context)
        replaced = emit(dispatcher, "perception.post", [{"distance": 15}], context)
        command = emit(dispatcher, "command", Control(throttle=1.0), context)
        dispatcher.finish_decision()
        assert replaced == [{"distance": 0.1}]
        assert command.brake == 1 and command.throttle == 0
    dispatcher.stop()
    assert attack.history == [1.5, 1.5] and attack.stopped
    assert [request["effect_id"] for request in dispatcher.requests] == [0, 1, 2, 3]
    assert [receipt["effect_id"] for receipt in dispatcher.receipts] == [0, 1, 2, 3]
    assert [receipt["stage"] for receipt in dispatcher.receipts] == [
        "perception.post", "command", "perception.post", "command",
    ]
    assert {request["category"] for request in dispatcher.requests} == {"avstack"}
    assert all(receipt["outcome"] == "applied" for receipt in dispatcher.receipts)


def test_context_is_filtered_and_nested_data_cannot_change_another_attack():
    received = []

    class Reader(AttackRuntime):
        subscriptions = (ResultSubscription("observation"),)

        def step(self, context):
            assert context.physical_state is None
            assert not hasattr(context, "backend") and not hasattr(context, "stack")
            with pytest.raises(FrozenInstanceError):
                context.sim_time = 999
            received.append(context.observation.sensor_data["camera"][:])
            context.observation.sensor_data["camera"].append(99)
            return ()

    dispatcher = started([Reader(), Reader()])
    context = data()
    dispatcher.step(context)
    dispatcher.stop()
    assert received == [[1, 2], [1, 2]]
    assert context.observation.sensor_data["camera"] == [1, 2]


def test_component_results_are_delivered_to_effects_not_subscriptions():
    with pytest.raises(ValueError, match="Unknown attack result kind"):
        ResultSubscription("component")


def test_setup_applies_world_effects_with_ticks_and_cleans_them_up_at_stop():
    class Physical(AttackRuntime):
        setup_ticks = 2

        def setup(self, context):
            assert context.stage == "setup" and context.observation is None
            return (Panel("a"), Panel("b"))

    world = World()
    dispatcher = started([Physical()], world=world)
    assert dispatcher.setup_ticks == 2
    assert world.native["objects"] == ["a", "b"] and world.ticks == 2
    assert [receipt["stage"] for receipt in dispatcher.receipts] == ["simulator.setup"] * 2
    assert [receipt["frame"] for receipt in dispatcher.receipts] == [42, 43]
    dispatcher.stop()
    assert world.native["objects"] == []


@pytest.mark.parametrize("effect", [
    Fill("cam", 1), AVStackAttackEffect("command", lambda value, context: value),
])
def test_setup_returns_only_world_effects(effect):
    class Bad(AttackRuntime):
        def setup(self, context):
            return (effect,)

    dispatcher = AttackDispatcher([Bad()])
    dispatcher.validate(AVAILABLE)
    with pytest.raises(TypeError, match="only WorldAttackEffect"):
        dispatcher.setup(data(0, "setup"), World())
    dispatcher.stop()


def test_world_deployment_uses_the_unified_simulator_setup_hook():
    class Physical(AttackRuntime):
        def setup(self, context):
            return (Panel("a", 0),)

    simulator_runtime = World()
    dispatcher = AttackDispatcher([Physical()])
    dispatcher.validate(AVAILABLE)
    assert [hook.stage for hook in dispatcher.hooks].count("simulator.setup") == 1
    emit(dispatcher, "simulator.setup", None, data(0, "setup"), simulator_runtime)
    assert simulator_runtime.native["objects"] == ["a"]
    assert [receipt["stage"] for receipt in dispatcher.receipts] == ["simulator.setup"]
    dispatcher.stop()
    assert simulator_runtime.native["objects"] == []


@pytest.mark.parametrize("effect", [
    AVStackAttackEffect("policy.post", lambda value, context: value),
    AVStackAttackEffect("simulator.step.pre", lambda value, context: value),
])
def test_unsupported_component_stage_fails_before_any_effect_is_requested(effect):
    class Bad(AttackRuntime):
        def step(self, context):
            return (effect,)

    dispatcher = started([Bad()])
    with pytest.raises(UnsupportedStageError, match="Unsupported attack effect stage"):
        dispatcher.step(data())
    assert dispatcher.requests == []
    dispatcher.stop()


def test_world_effects_from_step_apply_before_the_world_step():
    class Mover(AttackRuntime):
        def step(self, context):
            return (Panel("moved", 0),)

    world = World()
    dispatcher = started([Mover()], world=world)
    dispatcher.step(data())
    dispatcher.finish_decision()
    assert world.native["objects"] == []
    emit(dispatcher, "simulator.step.pre", None, data(), simulator_runtime=world)
    assert world.native["objects"] == ["moved"]
    assert dispatcher.receipts[-1]["stage"] == "simulator.step.pre"
    dispatcher.stop()
    assert world.native["objects"] == []


def test_sensor_effects_chain_in_issue_order_on_the_capture():
    class Insert(AttackRuntime):
        def step(self, context):
            return (Fill("cam", 3), Fill("cam", 4))

    dispatcher = started([Insert()])
    dispatcher.step(data())
    raw = Observation(7.0, 41, sensor_data={"cam": np.zeros((2, 2, 3), np.uint8)})
    attacked = dispatcher.apply_sensor_effects(raw, World())
    assert attacked.sensor_data["cam"][0, 0].tolist() == [7, 7, 7]
    assert (raw.sensor_data["cam"] == 0).all()
    assert [receipt["stage"] for receipt in dispatcher.receipts] == ["capture", "capture"]
    dispatcher.stop()


def test_partial_setup_and_cleanup_failure_still_stop_every_started_runtime():
    stopped = []

    class Attack(AttackRuntime):
        def __init__(self, number):
            self.number = number

        def setup(self, context):
            if self.number == 2:
                raise RuntimeError("setup failed")
            return ()

        def stop(self, context):
            stopped.append(self.number)
            raise ValueError(f"cleanup {self.number}")

    dispatcher = AttackDispatcher([Attack(1), Attack(2), Attack(3)])
    dispatcher.validate(AVAILABLE)
    with pytest.raises(RuntimeError, match="setup failed"):
        dispatcher.setup(data(0, "setup"), World())
    assert dispatcher.started
    with pytest.raises(ValueError, match="cleanup 2") as caught:
        dispatcher.stop()
    assert stopped == [2, 1]
    assert any("cleanup 1" in note for note in caught.value.__notes__)
    assert not dispatcher.started
    dispatcher.stop()
    assert stopped == [2, 1]


def test_steps_are_once_per_decision_and_unapplied_effects_do_not_leak():
    class Pending(AttackRuntime):
        def step(self, context):
            return (AVStackAttackEffect("command", lambda value, context: Control(brake=1)),)

    dispatcher = started([Pending()])
    dispatcher.step(data())
    with pytest.raises(RuntimeError, match="not applied"):
        dispatcher.finish_decision()
    with pytest.raises(ValueError, match="exactly once"):
        dispatcher.step(data())
    with pytest.raises(RuntimeError, match="not applied"):
        dispatcher.step(data(1))
    dispatcher.stop()


def test_one_runtime_cannot_be_registered_twice():
    attack = AttackRuntime()
    with pytest.raises(ValueError, match="only once"):
        AttackDispatcher([attack, attack])


def test_observation_subscription_uses_adapter_copy_for_native_sensor_payloads():
    class NativeCapture:
        def __deepcopy__(self, memo):
            raise TypeError("SDK capture cannot be deep copied")

    captures = []

    class Reader(AttackRuntime):
        subscriptions = (ResultSubscription("observation"),)

        def step(self, context):
            captures.append(context.observation.sensor_data["camera"])
            return ()

    native = NativeCapture()
    context = data()
    context.observation.sensor_data["camera"] = native

    def adapter_copy(observation):
        # A native adapter owns the payload-copy contract; no simulator handle is exposed.
        return Observation(observation.t, observation.frame, sensor_data={"camera": [3, 4]})

    dispatcher = started([Reader()], observation_copy=adapter_copy)
    dispatcher.step(context)
    dispatcher.stop()
    assert captures == [[3, 4]]
    assert context.observation.sensor_data["camera"] is native


def test_unapplied_effect_has_a_not_applied_receipt_even_when_stop_fails():
    class Failure(AttackRuntime):
        def step(self, context):
            return (AVStackAttackEffect("command", lambda value, context: value),)

        def stop(self, context):
            raise RuntimeError("stop failed")

    dispatcher = started([Failure()])
    dispatcher.step(data())
    with pytest.raises(RuntimeError, match="stop failed"):
        dispatcher.stop()
    assert dispatcher.receipts[0]["outcome"] == "not_applied"
    assert dispatcher.receipts[0]["reason"] == "run_ended_before_point_of_action"
    assert dispatcher.receipts[0]["stage"] == "stop"


def test_explicit_record_drain_keeps_unique_ids_across_decisions():
    class Repeated(AttackRuntime):
        def setup(self, context):
            return (Panel("startup", 0),)

        def step(self, context):
            return (AVStackAttackEffect("command", lambda value, context: Control(brake=1)),)

    dispatcher = started([Repeated()], world=World())
    public_requests, public_receipts = dispatcher.requests, dispatcher.receipts
    initialization = dispatcher.drain_records()
    assert initialization["requests"][0]["effect_id"] == 0
    assert initialization["receipts"][0]["outcome"] == "applied"
    assert dispatcher.requests is public_requests and dispatcher.receipts is public_receipts
    assert public_requests == [] and public_receipts == []

    dispatcher.step(data())
    emit(dispatcher, "command", Control(), data())
    dispatcher.finish_decision()
    first_decision = dispatcher.drain_records()
    assert [request["effect_id"] for request in first_decision["requests"]] == [1]
    assert [receipt["effect_id"] for receipt in first_decision["receipts"]] == [1]

    dispatcher.step(data(1))
    request = dispatcher.drain_records()["requests"][0]
    assert request["effect_id"] == 2 and request["issuer"].endswith("Repeated[0].step")
    dispatcher.stop()
    cleanup = dispatcher.drain_records()
    assert cleanup["requests"] == []
    assert cleanup["receipts"][0]["effect_id"] == 2
    assert cleanup["receipts"][0]["outcome"] == "not_applied"
    assert dispatcher.drain_records() == {"requests": [], "receipts": []}


def test_stop_context_reuses_the_latest_clock():
    contexts = []

    class Recorder(AttackRuntime):
        subscriptions = (ResultSubscription("physical_state"),)

        def stop(self, context):
            contexts.append(context)

    dispatcher = started([Recorder()])
    dispatcher.step(data())
    dispatcher.record_end(replace(data(), stage="step.end", sim_time=7.05, frame=42))
    dispatcher.stop()
    assert contexts[0].stage == "stop" and contexts[0].sim_time == 7.05
    assert contexts[0].physical_state.ego_speed == 10
