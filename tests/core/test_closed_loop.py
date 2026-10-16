"""Integration contracts for the unified loop and its compatibility views."""

from dataclasses import replace

import numpy as np
import pytest

from avsectester.attack_runtime import AttackRuntime, ResultSubscription, StageAction
from avsectester.backend import AVStack, WorldBackend, run
from avsectester.closed_loop import ClosedLoopRunner, RecordingPolicy, RunRequest, trace_from_log
from avsectester.plane import Control, Observation
from avsectester.runlog import Recorder, RunLog
from avsectester.runtime import Hook, HookSet


class MemoryWorld(WorldBackend):
    def __init__(self, fail=False):
        self.fail = fail
        self.closed = False
        self.resets = 0
        self.commands = []
        self.pixel = np.zeros((2, 2, 3), dtype=np.uint8)

    def reset(self):
        self.resets += 1
        self.frame = 10
        self.t = 8.0
        self.speed = 0.0
        return self._observe()

    def _observe(self):
        return Observation(self.t, self.frame, {"front": self.pixel}, ego_speed=self.speed)

    def step(self, control):
        if self.fail:
            raise RuntimeError("world advance failed")
        self.commands.append(control)
        self.frame += 1
        self.t += 0.1
        self.speed += control.throttle
        return self._observe()

    def close(self):
        self.closed = True


class Policy(AVStack):
    supported_stages = frozenset({"perception.post"})
    command_modes = frozenset({"actuator"})

    def __init__(self):
        self.reset_inputs = []
        self.inputs = []
        self.outputs = {}
        self.boxes = []

    def reset(self, observation):
        self.reset_inputs.append(observation)

    def __call__(self, observation):
        self.inputs.append(observation)
        self.boxes[:] = [len(self.inputs)]
        result = self._emit("perception.post", self.boxes)
        self.outputs = {"perception": result}
        return Control(throttle=0.5)

    def component_log(self):
        return self.outputs


def test_log_separates_input_truth_and_both_commands_without_aliasing(tmp_path):
    world, policy = MemoryWorld(), Policy()
    calls = []

    def change_input(obs, ctx):
        calls.append(ctx.step)
        obs.sensor_data["front"][:] = 123
        return replace(obs, frame=-7, t=-8, ego_speed=-9)

    def change_command(command, ctx):
        command.throttle = 0.75
        return command

    log = ClosedLoopRunner().run(
        world, policy, RunRequest(
            2, hooks=HookSet([Hook("observation", change_input), Hook("command", change_command)]),
            output_path=tmp_path / "run",
            recording=RecordingPolicy(("front",), ("raw", "model_input")),
        ),
    )
    assert calls == [0, 1]
    assert policy.reset_inputs[0] is policy.inputs[0]
    assert not world.closed
    row = log.steps[0]
    assert row.physical_pre.t == 8.0 and row.physical_post.t == 8.1
    assert row.raw_observation.t == 8.0 and row.model_observation.t == -8
    assert row.produced_command.throttle == 0.5
    assert row.executed_command.throttle == 0.75
    assert row.components["perception"] == [1]
    np.testing.assert_array_equal(row.raw_observation.sensor_data["front"], 0)
    np.testing.assert_array_equal(row.model_observation.sensor_data["front"], 123)
    loaded = RunLog.read(tmp_path / "run")
    assert trace_from_log(loaded).final_speed == trace_from_log(log).final_speed
    assert loaded.steps[0].components["perception"] == [1]
    assert trace_from_log(loaded).mean_detections == trace_from_log(log).mean_detections == 1


def test_sampling_only_changes_sensor_payloads():
    log = ClosedLoopRunner().run(
        MemoryWorld(), Policy(),
        RunRequest(3, recording=RecordingPolicy(("front",), every_n_steps=2)),
    )
    assert len(log.steps) == 3
    assert [bool(s.model_observation.sensor_data) for s in log.steps] == [True, False, True]
    assert all(s.physical_post is not None for s in log.steps)
    assert all(s.raw_observation.sensor_data == {} for s in log.steps)


class Phantom(AttackRuntime):
    reads = (ResultSubscription("physical_state"), ResultSubscription("component", "perception.post"))

    def __init__(self):
        self.events = []

    def start(self, context):
        self.events.append("start")
        self.count = 0
        return ()

    def step(self, context):
        self.events.append(("step", context.step))
        assert context.observation is None
        self.speed = context.physical_state.ego_speed
        return ()

    def on_component(self, context):
        self.count += 1
        return (StageAction("perception.post", [*context.component_output, 999]),)

    def stop(self, context):
        self.events.append("stop")


def test_one_attack_state_owner_and_component_effect_reaches_log():
    attack = Phantom()
    log = ClosedLoopRunner().run(MemoryWorld(), Policy(), RunRequest(2, attacks=(attack,)))
    assert attack.events == ["start", ("step", 0), ("step", 1), "stop"]
    assert attack.count == 2
    assert log.steps[0].components["perception"] == [1, 999]
    assert log.steps[0].actions
    request, receipt = log.steps[0].actions
    assert request["kind"] == "request" and receipt["kind"] == "receipt"
    assert request["action_id"] == receipt["action_id"] == 0
    assert receipt["outcome"] == "applied"
    assert receipt["stage"] == "perception.post"
    assert receipt["sim_time"] == log.steps[0].physical_pre.t
    assert log.steps[1].actions[0]["action_id"] == 1


def test_failed_world_step_still_stops_attack_and_keeps_partial_log():
    attack = Phantom()
    world = MemoryWorld(fail=True)
    with pytest.raises(RuntimeError, match="world advance failed") as caught:
        ClosedLoopRunner().run(world, Policy(), RunRequest(1, attacks=(attack,)))
    assert attack.events[-1] == "stop"
    assert not world.closed
    assert caught.value.run_log.status == "failed"
    assert caught.value.run_log.steps == []
    cleanup = next(event for event in caught.value.run_log.events if event.kind == "attack_cleanup")
    assert cleanup.data["actions"][1]["outcome"] == "applied"


def test_zero_decisions_resets_model_without_attack_step_or_transforms():
    attack, policy = Phantom(), Policy()
    def unexpected(*args):
        raise AssertionError("zero-step input must not be transformed")
    log = ClosedLoopRunner().run(
        MemoryWorld(), policy,
        RunRequest(0, attacks=(attack,), hooks=HookSet([Hook("observation", unexpected)])),
    )
    assert log.steps == []
    assert len(policy.reset_inputs) == 1
    assert attack.events == ["start", "stop"]


def test_bad_subscription_fails_before_reset():
    attack = Phantom()
    attack.reads = (ResultSubscription("component", "tracking.post"),)
    world = MemoryWorld()
    with pytest.raises(ValueError, match="Unavailable"):
        ClosedLoopRunner().run(world, Policy(), RunRequest(1, attacks=(attack,)))
    assert world.resets == 0


def test_legacy_trace_comes_from_runlog():
    trace = run(MemoryWorld(), Policy(), 2, runtime=HookSet())
    assert trace.run_log is not None
    assert len(trace.run_log.steps) == len(trace.records) == 2
    assert trace.final_speed == trace.run_log.steps[-1].physical_post.ego_speed


def test_unnamed_legacy_sensor_container_is_not_recorded_by_default():
    obs = Observation(0.0, 0, sensor_data=["legacy-container"])
    assert RecordingPolicy().observation(obs, 0, "model_input").sensor_data == {}
    with pytest.raises(ValueError, match="adapter sensor_data mapping"):
        RecordingPolicy(("front",)).observation(obs, 0, "model_input")


def test_unapplied_start_action_has_a_cleanup_receipt(tmp_path):
    class Startup(AttackRuntime):
        def start(self, context):
            return (StageAction("command", Control(brake=1.0)),)

    path = tmp_path / "zero"
    ClosedLoopRunner().run(MemoryWorld(), Policy(), RunRequest(0, attacks=(Startup(),), output_path=path))
    log = RunLog.read(path)
    initialization = next(e for e in log.events if e.kind == "initialization")
    cleanup = next(e for e in log.events if e.kind == "attack_cleanup")
    request = initialization.data["actions"][0]
    receipt = cleanup.data["actions"][0]
    assert receipt["action_id"] == request["action_id"]
    assert receipt["outcome"] == "not_applied"
    assert receipt["reason"] == "run_ended_before_target_stage"


def test_disk_streaming_releases_consumed_callback_records(tmp_path):
    sizes = []
    def output(obs, ctx):
        ctx.record_output("size", np.ones(1000))
        sizes.append(len(ctx.backend._runtime.outputs))
        return obs

    path = tmp_path / "stream"
    log = ClosedLoopRunner().run(
        MemoryWorld(), Policy(),
        RunRequest(5, output_path=path, retain_steps=False,
                   hooks=HookSet([Hook("observation", output)]), attacks=(Phantom(),)),
    )
    assert sizes == [1] * 5
    assert log.steps == [] and log.events == []
    recovered = RunLog.read(path)
    assert len(recovered.steps) == 5
    assert [s.actions[0]["action_id"] for s in recovered.steps] == list(range(5))


def test_failed_run_retains_pending_outputs_in_disk_log(tmp_path):
    def output(obs, ctx):
        ctx.record_output("before_failure", 12)
        return obs

    path = tmp_path / "failed"
    recorder = Recorder(path, chunk_steps=1)
    with pytest.raises(RuntimeError, match="world advance failed"):
        ClosedLoopRunner().run(
            MemoryWorld(fail=True), Policy(),
            RunRequest(1, hooks=HookSet([Hook("observation", output)])), recorder=recorder,
        )
    log = RunLog.read(path)
    failure = next(event for event in log.events if event.kind == "failure")
    assert failure.data["outputs"][0].value == 12
    assert log.status == "failed"


def test_cleanup_logging_failure_preserves_original_stop_error():
    class BrokenStop(AttackRuntime):
        def stop(self, context):
            raise ValueError("original stop failure")

    class BrokenCleanupLog(Recorder):
        def record_event(self, kind, **data):
            if kind == "attack_cleanup":
                raise OSError("cleanup log failed")
            return super().record_event(kind, **data)

    with pytest.raises(ValueError, match="original stop failure") as caught:
        ClosedLoopRunner().run(
            MemoryWorld(), Policy(), RunRequest(1, attacks=(BrokenStop(),)),
            recorder=BrokenCleanupLog(),
        )
    assert "cleanup log failed" in caught.value.__notes__[0]
    failure = next(event for event in caught.value.run_log.events if event.kind == "failure")
    assert failure.data["error_type"] == "ValueError"
    assert "cleanup log failed" in failure.data["cleanup_errors"][0]


def test_feedback_failure_keeps_the_completed_physical_decision(tmp_path):
    class FeedbackFailure:
        def transform(self, observation, context):
            return observation

        def after_step(self, context):
            context.record_output("feedback", "before error")
            raise RuntimeError("feedback failure")

    world = MemoryWorld()
    path = tmp_path / "feedback"
    with pytest.raises(RuntimeError, match="feedback failure"):
        ClosedLoopRunner().run(
            world, Policy(), RunRequest(
                2, output_path=path, hooks=HookSet([Hook("observation", FeedbackFailure())]),
            ),
        )
    log = RunLog.read(path)
    assert len(log.steps) == len(world.commands) == 1
    assert log.steps[0].physical_post.frame == log.steps[0].physical_pre.frame + 1
    assert log.steps[0].executed_command.throttle == 0.5
    assert log.steps[0].outputs[0].value == "before error"
    assert log.status == "failed"


@pytest.mark.parametrize("conflict", ["two_attacks", "start_step", "one_batch"])
def test_conflicting_replacements_fail_before_stage_and_stop_started_attacks(conflict, tmp_path):
    class Replacement(AttackRuntime):
        def __init__(self, startup=False, duplicate=False):
            self.startup, self.duplicate = startup, duplicate
            self.stopped = False

        def start(self, context):
            return (StageAction("observation", Observation(0, 0)),) if self.startup else ()

        def step(self, context):
            return (StageAction("observation", Observation(1, 1)),) * (2 if self.duplicate else 1)

        def stop(self, context):
            self.stopped = True

    attacks = ([Replacement(), Replacement()] if conflict == "two_attacks" else
               [Replacement(startup=conflict == "start_step", duplicate=conflict == "one_batch")])
    world, policy = MemoryWorld(), Policy()
    visited = []
    def observation_stage(value, context):
        visited.append(context.step)
        return value

    with pytest.raises(ValueError, match="Conflicting pending replacements.*observation") as caught:
        ClosedLoopRunner().run(world, policy, RunRequest(
            1, attacks=attacks, output_path=tmp_path,
            hooks=HookSet([Hook("observation", observation_stage)]),
        ))
    message = str(caught.value)
    assert "Replacement[0]." in message
    if conflict == "two_attacks":
        assert "Replacement[0].step" in message and "Replacement[1].step" in message
    elif conflict == "start_step":
        # The initialization request was drained before decision 0, but its issuer remains known.
        assert "Replacement[0].start" in message and "Replacement[0].step" in message
    else:
        assert message.count("Replacement[0].step") == 2
    assert all(attack.stopped for attack in attacks)
    assert visited == [] and policy.inputs == [] and world.commands == []
    log = RunLog.read(tmp_path)
    assert log.status == "failed" and log.steps == []
    actions = [action for event in log.events for action in event.data.get("actions", [])]
    requests = [action for action in actions if action["kind"] == "request"]
    receipts = [action for action in actions if action["kind"] == "receipt"]
    assert len(requests) == len(receipts) == 1
    assert receipts[0]["action_id"] == requests[0]["action_id"]
    assert receipts[0]["outcome"] == "not_applied"


def test_two_component_attacks_chain_replacements_in_registration_order():
    seen = []
    class Append(AttackRuntime):
        reads = (ResultSubscription("component", "perception.post"),)

        def __init__(self, value):
            self.value = value
            self.stopped = False

        def on_component(self, context):
            seen.append((self.value, list(context.component_output)))
            return (StageAction("perception.post", [*context.component_output, self.value]),)

        def stop(self, context):
            self.stopped = True

    attacks = [Append(10), Append(20)]
    log = ClosedLoopRunner().run(MemoryWorld(), Policy(), RunRequest(1, attacks=attacks))
    assert seen == [(10, [1]), (20, [1, 10])]
    assert log.steps[0].components["perception"] == [1, 10, 20]
    receipts = [action for action in log.steps[0].actions if action["kind"] == "receipt"]
    assert [receipt["action_id"] for receipt in receipts] == [0, 1]
    assert [receipt["outcome"] for receipt in receipts] == ["applied", "applied"]
    assert all(attack.stopped for attack in attacks)
