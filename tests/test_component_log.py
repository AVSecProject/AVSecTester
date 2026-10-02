"""Component-level logging: the generic capture hook, the modular stack's instrument/component_log
wiring, and the evaluation-side gatherer (run_logged + ComponentTrace).

Uses a fake pipeline (stages that replay the avstack post-hook contract) so the wiring is tested without
building a real avstack perception stack, and a stub closed loop for run_logged — mirroring the other
evaluation tests.
"""

from avsectester.backend import AVStack, WorldBackend
from avsectester.evaluation.component_log import (
    ComponentTrace,
    InstrumentedStack,
    StepLog,
    run_logged,
)
from avsectester.plane import Control, Observation
from avsectester.stacks.alpamayo import AlpamayoAVStack
from avsectester.stacks.modular import ModularAVStack, _StageCapture


def test_stage_capture_stores_output_and_passes_it_through():
    cap = _StageCapture()
    out = cap(["a", "b"])
    assert out == (["a", "b"],)                 # avstack post-hook contract: return a 1-tuple
    assert cap.last == ["a", "b"]               # and remember the output


class _FakeStage:
    """Replays avstack's post-hook application so instrument()/component_log() can be tested standalone."""

    def __init__(self):
        self.post_hooks = []

    def register_post_hook(self, hook):
        self.post_hooks.append(hook)

    def emit(self, output):                      # simulate the stage producing `output`
        for hook in self.post_hooks:
            output = hook(output)
            if isinstance(output, tuple) and len(output) == 1:
                output = output[0]
        return output


class _FakePipeline:
    def __init__(self):
        self.perception = _FakeStage()
        self.tracking = _FakeStage()
        self.planning = _FakeStage()
        self.control = _FakeStage()


def _bare_modular():
    stack = object.__new__(ModularAVStack)       # bypass the heavy avstack pipeline build
    stack.pipeline = _FakePipeline()
    stack._counter = None
    stack.detection_counts = []
    return stack


def test_instrument_captures_each_stage_and_is_passthrough():
    stack = _bare_modular()
    assert stack.component_log() is None          # not instrumented yet
    stack.instrument()
    passed = stack.pipeline.perception.emit(["d1", "d2", "d3"])
    stack.pipeline.tracking.emit(["t1", "t2"])
    stack.pipeline.planning.emit("plan")
    stack.pipeline.control.emit("ctrl")
    assert passed == ["d1", "d2", "d3"]           # capture does not alter the stage output
    log = stack.component_log()
    assert log["perception"] == ["d1", "d2", "d3"] and log["tracking"] == ["t1", "t2"]
    assert log["planning"] == "plan" and log["control"] == "ctrl"


def test_component_trace_counts_and_clean_vs_attacked_degradation():
    clean = ComponentTrace([StepLog(0, {"perception": ["a", "b", "c"]}),
                            StepLog(1, {"perception": ["a", "b"]})])
    attacked = ComponentTrace([StepLog(0, {"perception": ["a"]}),
                               StepLog(1, {"perception": []})])
    assert clean.counts("perception") == [3, 2]
    assert attacked.counts("perception") == [1, 0]
    assert clean.degradation(attacked, "perception") == [2, 2]   # attack removed 2 detections/frame
    assert StepLog(0, {"planning": object()}).count("planning") == 1  # non-collection output -> 1
    assert StepLog(0, {}).count("perception") == 0                    # missing/None -> 0


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


class _InstrumentedStub(AVStack):
    """Each frame 'detects' one more object, so the per-frame count is observable."""

    def __init__(self):
        self.n = 0

    def __call__(self, obs):
        self.n += 1
        return Control(throttle=1.0)

    def component_log(self):
        return {"perception": list(range(self.n))}


def test_run_logged_gathers_component_trace_alongside_the_drive():
    trace, comp = run_logged(_StubBackend(), _InstrumentedStub(), frames=3)
    assert len(trace.records) == 3                 # the ordinary driving Trace is still produced
    assert comp.counts("perception") == [1, 2, 3]  # component_log pulled once per frame


def test_run_logged_is_empty_when_stack_is_not_instrumented():
    class _Plain(AVStack):
        def __call__(self, obs):
            return Control()

    trace, comp = run_logged(_StubBackend(), _Plain(), frames=2)
    assert len(trace.records) == 2 and comp.steps == []


# --- the common interface across stack shapes (modular stages vs e2e policy+action) ----------------


class _E2EStub(AVStack):
    """An end-to-end stack: one key for the policy's candidate trajectories, one for the action."""

    def __call__(self, obs):
        return Control(trajectory=[(0.0, 0.0, 0.0)])

    def component_log(self):
        return {"policy": [1, 2, 3], "action": "ctrl"}   # (candidate trajectories), (emitted command)


def test_common_interface_collects_e2e_logs_the_same_way():
    trace, comp = run_logged(_StubBackend(), _E2EStub(), frames=2)
    assert len(trace.records) == 2 and len(comp.steps) == 2
    assert comp.stage_names == ["policy", "action"]       # e2e keys, not modular stages
    assert comp.counts("policy") == [3, 3]                # processed uniformly (len of the free output)


def test_alpamayo_component_log_emits_free_logs():
    stack = object.__new__(AlpamayoAVStack)               # bypass the heavy model load
    assert stack.component_log() is None                  # before any call
    stack._last_log = {"policy": "prediction", "action": "ctrl"}
    assert stack.component_log() == {"policy": "prediction", "action": "ctrl"}


def test_both_stack_shapes_satisfy_the_common_interface():
    assert isinstance(_E2EStub(), InstrumentedStack)      # e2e conforms
    assert isinstance(_bare_modular(), InstrumentedStack)  # modular conforms

    class _Plain(AVStack):
        def __call__(self, obs):
            return Control()

    assert not isinstance(_Plain(), InstrumentedStack)    # a stack without component_log does not
