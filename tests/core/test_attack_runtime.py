"""Online attack subscriptions keep state together and prevent hidden world access."""

from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace

import pytest

from avsectester.attack_runtime import (
    AttackContext, AttackDispatcher, AttackRuntime, ResultSubscription, StageAction,
)
from avsectester.plane import Control, Observation, WorldSnapshot
from avsectester.runtime import HookDispatcher, UnsupportedStageError


AVAILABLE = HookDispatcher.CORE_STAGES | {"perception.post", "tracking.post"}


def data(step=0, stage="step"):
    return AttackContext(
        step, stage, 7.0 + step * 0.1, 41 + step,
        observation=Observation(7.0 + step * 0.1, 41 + step, sensor_data={"camera": [1, 2]}),
        physical_state=WorldSnapshot(7.0 + step * 0.1, 41 + step, ego_speed=10.0),
    )


def emit(dispatcher, stage, value, context):
    native = SimpleNamespace(
        step=context.step, stage=stage, sim_time=context.sim_time, ground_truth=context.physical_state,
    )
    for hook in dispatcher.hooks:
        if hook.stage == stage:
            value = hook.handler(value, native)
    return value


def test_ttc_component_callback_shares_runtime_state_and_changes_downstream_command():
    class TTC(AttackRuntime):
        reads = (ResultSubscription("physical_state"),
                 ResultSubscription("component", "perception.post"))

        def start(self, context):
            self.history = []
            self.stopped = False
            return ()

        def step(self, context):
            self.speed = context.physical_state.ego_speed
            assert context.component_output is None
            return ()

        def on_component(self, context):
            ttc = context.component_output[0]["distance"] / self.speed
            self.history.append(ttc)
            return (StageAction("perception.post", [{"distance": 0.1}]),
                    StageAction("command", Control(brake=1.0)))

        def stop(self, context):
            self.stopped = True

    attack = TTC()
    dispatcher = AttackDispatcher([attack])
    dispatcher.validate(AVAILABLE)
    dispatcher.start(data(-1, "start"))
    for index in range(2):
        context = data(index)
        dispatcher.step(context)
        replaced = emit(dispatcher, "perception.post", [{"distance": 15}], context)
        command = emit(dispatcher, "command", Control(throttle=1.0), context)
        assert replaced == [{"distance": 0.1}]
        assert command.brake == 1 and command.throttle == 0
    dispatcher.stop(data(2, "stop"))
    assert attack.history == [1.5, 1.5] and attack.stopped
    assert len(dispatcher.actions) == 4
    assert [request["action_id"] for request in dispatcher.requests] == [0, 1, 2, 3]
    assert [receipt["action_id"] for receipt in dispatcher.receipts] == [0, 1, 2, 3]
    assert [receipt["stage"] for receipt in dispatcher.receipts] == [
        "perception.post", "command", "perception.post", "command",
    ]
    assert all(receipt["outcome"] == "applied" for receipt in dispatcher.receipts)


def test_context_is_filtered_and_nested_data_cannot_change_another_attack():
    received = []

    class Reader(AttackRuntime):
        reads = (ResultSubscription("observation"),)

        def step(self, context):
            assert context.physical_state is None and context.component_output is None
            assert not hasattr(context, "backend") and not hasattr(context, "stack")
            with pytest.raises(FrozenInstanceError):
                context.sim_time = 999
            received.append(context.observation.sensor_data["camera"][:])
            context.observation.sensor_data["camera"].append(99)
            return ()

    dispatcher = AttackDispatcher([Reader(), Reader()])
    dispatcher.validate(AVAILABLE)
    context = data()
    dispatcher.start(data(-1, "start"))
    dispatcher.step(context)
    dispatcher.stop(data(1, "stop"))
    assert received == [[1, 2], [1, 2]]
    assert context.observation.sensor_data["camera"] == [1, 2]


def test_unavailable_subscription_fails_without_starting_attack():
    class Reader(AttackRuntime):
        reads = (ResultSubscription("component", "policy.post"),)

        def start(self, context):
            pytest.fail("Validation must precede start")

    dispatcher = AttackDispatcher([Reader()])
    with pytest.raises(UnsupportedStageError, match="Unavailable component"):
        dispatcher.validate(AVAILABLE)


@pytest.mark.parametrize("stage", ["observation", "world.step.pre"])
def test_component_callback_cannot_change_the_past_or_request_unsupported_world_actions(stage):
    class Bad(AttackRuntime):
        reads = (ResultSubscription("component", "perception.post"),)

        def on_component(self, context):
            return (StageAction(stage, None),)

    dispatcher = AttackDispatcher([Bad()])
    dispatcher.validate(AVAILABLE)
    dispatcher.start(data(-1, "start"))
    dispatcher.step(data())
    with pytest.raises(ValueError, match="elapsed stage|Unsupported attack action"):
        emit(dispatcher, "perception.post", [], data())
    dispatcher.stop(data(1, "stop"))


def test_partial_start_and_cleanup_failure_still_stop_every_started_runtime():
    stopped = []

    class Attack(AttackRuntime):
        def __init__(self, number):
            self.number = number

        def start(self, context):
            if self.number == 2:
                raise RuntimeError("start failed")
            return ()

        def stop(self, context):
            stopped.append(self.number)
            raise ValueError(f"cleanup {self.number}")

    dispatcher = AttackDispatcher([Attack(1), Attack(2), Attack(3)])
    dispatcher.validate(AVAILABLE)
    with pytest.raises(RuntimeError, match="start failed"):
        dispatcher.start(data(-1, "start"))
    with pytest.raises(ValueError, match="cleanup 2") as caught:
        dispatcher.stop(data(-1, "stop"))
    assert stopped == [2, 1]
    assert any("cleanup 1" in note for note in caught.value.__notes__)
    dispatcher.stop(data(-1, "stop"))
    assert stopped == [2, 1]


def test_steps_are_once_per_decision_and_unused_actions_do_not_leak_into_next_decision():
    class Pending(AttackRuntime):
        def step(self, context):
            return (StageAction("command", Control(brake=1)),)

    dispatcher = AttackDispatcher([Pending()])
    dispatcher.validate(AVAILABLE)
    dispatcher.start(data(-1, "start"))
    dispatcher.step(data())
    with pytest.raises(RuntimeError, match="not consumed"):
        dispatcher.finish_decision()
    with pytest.raises(ValueError, match="exactly once"):
        dispatcher.step(data())
    with pytest.raises(RuntimeError, match="not consumed"):
        dispatcher.step(data(1))
    dispatcher.stop(data(1, "stop"))


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
        reads = (ResultSubscription("observation"),)

        def step(self, context):
            captures.append(context.observation.sensor_data["camera"])
            return ()

    native = NativeCapture()
    context = data()
    context.observation.sensor_data["camera"] = native

    def adapter_copy(observation):
        # A native adapter owns the payload-copy contract; no backend handle is exposed.
        return Observation(observation.t, observation.frame, sensor_data={"camera": [3, 4]})

    dispatcher = AttackDispatcher([Reader()], observation_copy=adapter_copy)
    dispatcher.validate(AVAILABLE)
    dispatcher.start(data(-1, "start"))
    dispatcher.step(context)
    dispatcher.stop(data(1, "stop"))
    assert captures == [[3, 4]]
    assert context.observation.sensor_data["camera"] is native


@pytest.mark.parametrize("decision_steps", [0, 1])
def test_startup_request_and_receipt_have_separate_timing_and_outcomes(decision_steps):
    class Startup(AttackRuntime):
        def start(self, context):
            return (StageAction("command", Control(brake=1)),)

    dispatcher = AttackDispatcher([Startup()])
    dispatcher.validate(AVAILABLE)
    dispatcher.start(data(-1, "start"))
    assert len(dispatcher.requests) == 1 and dispatcher.receipts == []
    request = dispatcher.requests[0]
    assert request["action_id"] == 0 and request["step"] == -1 and request["stage"] == "start"
    assert request["sim_time"] == pytest.approx(6.9) and request["frame"] == 40
    if decision_steps:
        dispatcher.step(data())
        command = emit(dispatcher, "command", Control(), data())
        dispatcher.finish_decision()
        assert command.brake == 1
    dispatcher.stop(data(decision_steps, "stop"))
    receipt = dispatcher.receipts[0]
    assert receipt["action_id"] == 0 and receipt["target_stage"] == "command"
    assert receipt["sim_time"] == 7.0 and receipt["frame"] == 41
    assert receipt["stage"] == ("command" if decision_steps else "stop")
    assert receipt["outcome"] == ("applied" if decision_steps else "not_applied")
    dispatcher.stop(data(decision_steps, "stop"))
    assert len(dispatcher.receipts) == 1


def test_cleanup_failure_still_records_unconsumed_action_as_not_applied():
    class Failure(AttackRuntime):
        def start(self, context):
            return (StageAction("command", Control()),)

        def stop(self, context):
            raise RuntimeError("stop failed")

    dispatcher = AttackDispatcher([Failure()])
    dispatcher.validate(AVAILABLE)
    dispatcher.start(data(-1, "start"))
    with pytest.raises(RuntimeError, match="stop failed"):
        dispatcher.stop(data(0, "stop"))
    assert dispatcher.receipts[0]["outcome"] == "not_applied"
    assert dispatcher.receipts[0]["reason"] == "run_ended_before_target_stage"


def test_observation_actions_also_use_native_adapter_copy_at_enqueue_and_apply():
    class NativeCapture:
        def __deepcopy__(self, memo):
            raise TypeError("SDK capture cannot be deep copied")

    class Spoof(AttackRuntime):
        reads = (ResultSubscription("observation"),)

        def step(self, context):
            return (StageAction("observation", replace(context.observation, ego_speed=99)),)

    native = NativeCapture()
    context = data()
    context.observation.sensor_data["camera"] = native

    def adapter_copy(observation):
        # CARLA adapters keep borrowed native captures while copying Python wrappers.
        return replace(observation, sensor_data=dict(observation.sensor_data))

    dispatcher = AttackDispatcher([Spoof()], observation_copy=adapter_copy)
    dispatcher.validate(AVAILABLE)
    dispatcher.start(data(-1, "start"))
    dispatcher.step(context)
    seen = emit(dispatcher, "observation", context.observation, context)
    assert seen.ego_speed == 99 and context.observation.ego_speed == 0
    assert seen.sensor_data["camera"] is native
    assert dispatcher.receipts[0]["outcome"] == "applied"
    dispatcher.stop(data(1, "stop"))


def test_explicit_record_drain_keeps_pending_actions_and_unique_ids_across_decisions():
    class Repeated(AttackRuntime):
        def start(self, context):
            return (StageAction("perception.post", ["startup"]),)

        def step(self, context):
            return (StageAction("command", Control(brake=1)),)

    dispatcher = AttackDispatcher([Repeated()])
    dispatcher.validate(AVAILABLE)
    dispatcher.start(data(-1, "start"))
    public_requests, public_receipts = dispatcher.requests, dispatcher.receipts
    initialization = dispatcher.drain_records()
    assert initialization["requests"][0]["action_id"] == 0
    assert initialization["receipts"] == []
    assert dispatcher.requests is public_requests and dispatcher.receipts is public_receipts
    assert public_requests == [] and public_receipts == [] and dispatcher.actions == []

    dispatcher.step(data())
    assert emit(dispatcher, "perception.post", [], data()) == ["startup"]
    emit(dispatcher, "command", Control(), data())
    dispatcher.finish_decision()
    first_decision = dispatcher.drain_records()
    assert [request["action_id"] for request in first_decision["requests"]] == [1]
    assert [receipt["action_id"] for receipt in first_decision["receipts"]] == [0, 1]
    assert dispatcher.requests == [] and dispatcher.receipts == [] and dispatcher.actions == []

    dispatcher.step(data(1))
    request = dispatcher.drain_records()["requests"][0]
    assert request["action_id"] == 2
    dispatcher.stop(data(1, "stop"))
    cleanup = dispatcher.drain_records()
    assert cleanup["requests"] == []
    assert cleanup["receipts"][0]["action_id"] == 2
    assert cleanup["receipts"][0]["outcome"] == "not_applied"
    assert dispatcher.drain_records() == {"requests": [], "receipts": []}
