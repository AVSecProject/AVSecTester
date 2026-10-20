"""Driving decisions, effects and failures are observed through ClosedLoopRunner.run."""

from dataclasses import replace

import numpy as np
import pytest

from avsectester.attack.runtime import (
    AttackRuntime, AVStackAttackEffect, SensorAttackEffect, WorldAttackEffect,
)
from avsectester.closed_loop.datatypes import Control, Detection
from avsectester.closed_loop.hooks import Hook, HookSet
from avsectester.closed_loop.runner import ClosedLoopRunner, RecordingPolicy, RunRequest
from tests.fakes import ConstantAVStack, MemoryDataset, MemorySimulator


RECORDING = RecordingPolicy(("front",), ("raw", "model_input"))


def test_run():
    data = MemoryDataset()
    start = data.scenario_start("scene", data.frames("scene")[2])
    with MemorySimulator().create_runtime() as simulator_runtime:
        log = ClosedLoopRunner().run(simulator_runtime, ConstantAVStack(Control(throttle=.3)),
                                     RunRequest(2, start=start, recording=RECORDING))
        assert log.status == "completed"
        assert len(log.steps) == 2
        for index, record in enumerate(log.steps):
            assert record.raw_observation.frame == index + 2
            assert record.model_observation.frame == record.physical_pre.frame
            np.testing.assert_array_equal(record.raw_observation.sensor_data["front"],
                                          record.model_observation.sensor_data["front"])
            assert record.produced_command == record.executed_command == Control(throttle=.3)
            assert record.physical_post.frame == record.physical_pre.frame + 1
            assert record.physical_post.t == pytest.approx(record.physical_pre.t + .1)
        empty = ClosedLoopRunner().run(simulator_runtime, ConstantAVStack(), RunRequest(0))
        assert empty.status == "completed" and not empty.steps


def test_run_with_hooks():
    hooks = HookSet([
        Hook("observation", lambda value, context: replace(value, ego_speed=value.ego_speed + 5.)),
        Hook("command", lambda value, context: replace(value, brake=.7)),
    ])
    with MemorySimulator().create_runtime() as simulator_runtime:
        log = ClosedLoopRunner().run(simulator_runtime, ConstantAVStack(Control(throttle=.2)),
                                     RunRequest(2, hooks=hooks, recording=RECORDING))
    for record in log.steps:
        assert record.model_observation.ego_speed == record.raw_observation.ego_speed + 5.
        assert record.produced_command == Control(throttle=.2)
        assert record.executed_command == Control(throttle=.2, brake=.7)
        assert record.physical_pre.ego_speed == record.raw_observation.ego_speed


def test_run_with_attacks(tmp_path):
    lifecycle = []
    resource = (tmp_path / "world-resource.txt").open("w")

    class MoveHost(WorldAttackEffect):
        def __init__(self, x, cleanup=False):
            self.x, self.cleanup = x, cleanup

        def apply(self, simulator_runtime, resources):
            actor = simulator_runtime.state().actors["lead"]
            pose = actor.pose.copy()
            pose[0, 3] = self.x
            simulator_runtime.set_actor_state("lead", replace(actor, pose=pose))
            if self.cleanup:
                resources.callback(resource.close)

    class Paint(SensorAttackEffect):
        sensor = "front"

        def apply(self, data, view):
            data[:] = 42
            return data

    class OnlineAttack(AttackRuntime):
        def setup(self, context):
            lifecycle.append("setup")
            return (MoveHost(12., cleanup=True),)

        def step(self, context):
            context.record_output("decision", context.step)
            return (Paint(), MoveHost(14.), AVStackAttackEffect("perception.post", self.inject))

        def inject(self, value, context):
            return [*value, Detection((6., 0., 0.), (4., 2., 1.5), 0., "car", .9)]

        def stop(self, context):
            lifecycle.append("stop")

    with MemorySimulator().create_runtime() as simulator_runtime:
        log = ClosedLoopRunner().run(simulator_runtime, ConstantAVStack(perception_output=[]),
                                     RunRequest(1, attacks=(OnlineAttack(),), recording=RECORDING))
    assert lifecycle == ["setup", "stop"]
    assert resource.closed
    record = log.steps[0]
    assert record.physical_pre.actors["lead"].pose[0, 3] == 12.
    assert record.physical_post.actors["lead"].pose[0, 3] == 14.
    assert np.all(record.model_observation.sensor_data["front"] == 42)
    assert np.any(record.raw_observation.sensor_data["front"] != 42)
    assert len(record.detections) == 1
    assert record.outputs[0].name == "decision" and record.outputs[0].value == 0
    receipts = [effect for event in log.events for effect in event.data.get("effects", [])
                if effect["kind"] == "receipt"]
    receipts += [effect for effect in record.effects if effect["kind"] == "receipt"]
    assert {effect["category"] for effect in receipts} == {"world", "sensor", "avstack"}
    assert all(effect["outcome"] == "applied" for effect in receipts)


def test_run_failure(tmp_path):
    stopped = []
    resource = (tmp_path / "run-resource.txt").open("w")

    class ActiveAttack(AttackRuntime):
        def stop(self, context):
            stopped.append(context.stage)

    with MemorySimulator().create_runtime() as simulator_runtime:
        with pytest.raises(RuntimeError) as failure:
            ClosedLoopRunner().run(simulator_runtime, ConstantAVStack(fail_on_position=1),
                                   RunRequest(3, attacks=(ActiveAttack(),)),
                                   release_owned_resources=resource.close)
        log = failure.value.run_log
        assert log.status == "failed"
        assert len(log.steps) == 1
        assert any(event.kind == "failure" for event in log.events)
        assert stopped == ["stop"] and resource.closed
        # A failed run relinquishes the simulator reservation and handler binding.
        again = ClosedLoopRunner().run(simulator_runtime, ConstantAVStack(), RunRequest(1))
        assert again.status == "completed"
        with pytest.raises(KeyboardInterrupt) as cancelled:
            ClosedLoopRunner().run(simulator_runtime, ConstantAVStack(interrupt_on_position=0), RunRequest(1))
        assert cancelled.value.run_log.status == "cancelled"


def test_run_rejects_invalid_request():
    with pytest.raises(ValueError, match="nonnegative"):
        RunRequest(-1)
    with MemorySimulator().create_runtime() as simulator_runtime:
        with pytest.raises(ValueError, match="unavailable sensors"):
            ClosedLoopRunner().run(simulator_runtime, ConstantAVStack(), RunRequest(
                1, recording=RecordingPolicy(("missing",))))
