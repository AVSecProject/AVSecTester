"""Component trace collection and processing without simulator or model dependencies."""

import numpy as np

from avsectester.backend import AVStack, WorldBackend
from avsectester.evaluation.component_log import ComponentTrace, StepLog, run_logged
from avsectester.plane import Control, Observation
from avsectester.runtime import Hook, Runtime


def test_component_trace_counts_and_clean_vs_attacked_degradation():
    clean = ComponentTrace(
        [StepLog(0, {"perception": ["a", "b", "c"]}), StepLog(1, {"perception": ["a", "b"]})]
    )
    attacked = ComponentTrace([StepLog(0, {"perception": ["a"]}), StepLog(1, {"perception": []})])
    assert clean.counts("perception") == [3, 2]
    assert attacked.counts("perception") == [1, 0]
    assert clean.degradation(attacked, "perception") == [2, 2]
    assert StepLog(0, {}).count("planning") == 0
    assert StepLog(0, {"planning": None}).count("planning") == 0
    assert StepLog(0, {"planning": object()}).count("planning") == 1


class _StubBackend(WorldBackend):
    def __init__(self):
        self.i = 0

    def _obs(self):
        return Observation(t=0.1 * self.i, frame=self.i, sensor_data={"front": 0})

    def reset(self):
        self.i = 0
        return self._obs()

    def step(self, control):
        self.i += 1
        return self._obs()


class _PlainStack(AVStack):
    def __call__(self, obs):
        return Control(throttle=1.0)


def test_run_logged_collects_named_outputs_alongside_driving_trace():
    class Stack(_PlainStack):
        def __call__(self, obs):
            self.output = list(range(obs.frame + 1))
            return super().__call__(obs)

        def component_log(self):
            return {"perception": self.output, "policy": {"control": "forward"}}

    trace, comp = run_logged(_StubBackend(), Stack(), frames=3)
    assert len(trace.records) == 3
    assert comp.stage_names == ["perception", "policy"]
    assert [step.frame for step in comp.steps] == [0, 1, 2]
    assert comp.counts("perception") == [1, 2, 3]
    assert comp.counts("policy") == [1, 1, 1]


def test_run_logged_without_component_interface_returns_only_driving_trace():
    trace, comp = run_logged(_StubBackend(), _PlainStack(), frames=2)
    assert len(trace.records) == 2
    assert comp.steps == []


def test_run_logged_collects_processed_inputs_and_records_final_command():
    from dataclasses import replace

    class Stack(_PlainStack):
        def __call__(self, observation):
            self.speed = observation.ego_speed
            return super().__call__(observation)

        def component_log(self):
            return {"policy": {"speed": self.speed}}

    runtime = Runtime([
        Hook("observation", lambda value, context: replace(value, ego_speed=5.0)),
        Hook("command", lambda value, context: Control(brake=1.0)),
    ])
    trace, component = run_logged(_StubBackend(), Stack(), frames=2, runtime=runtime)

    assert [step.stages["policy"]["speed"] for step in component.steps] == [5.0, 5.0]
    assert [record.brake for record in trace.records] == [1.0, 1.0]
    assert [record.speed for record in trace.records] == [0.0, 0.0]


def test_logged_outputs_are_snapshots_of_reused_mutable_buffers():
    class ReusedOutput(_PlainStack):
        def __init__(self):
            super().__init__()
            self.output = {"positions": np.zeros((1, 3)), "history": []}

        def __call__(self, obs):
            self.output["positions"][:] = obs.frame
            self.output["history"].append(obs.frame)
            return Control()

        def component_log(self):
            return {"policy": self.output}

    stack = ReusedOutput()
    _, comp = run_logged(_StubBackend(), stack, frames=3)
    stack.output["positions"][:] = 99
    for i, step in enumerate(comp.steps):
        np.testing.assert_array_equal(step.stages["policy"]["positions"], np.full((1, 3), i))
        assert step.stages["policy"]["history"] == list(range(i + 1))
