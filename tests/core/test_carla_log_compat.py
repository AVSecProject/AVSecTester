"""CARLA compatibility telemetry is a view of the canonical run record."""

from avsectester.backend import AVStack
from avsectester.carla_runner import _logged_run
from avsectester.closed_loop import trace_from_log
from avsectester.plane import Control
from avsectester.runlog import Recorder, RunLog
from avsectester.runtime import HookSet
from avsectester.simulators.nurec import NuRecBackend, StubRenderer


def test_carla_compatibility_detection_metric_survives_runlog_round_trip(tmp_path):
    class DetectedCruise(AVStack):
        def __call__(self, observation):
            self.detections = [{"id": index} for index in range(observation.frame + 1)]
            return Control(throttle=0.1)

        def component_log(self):
            return {"perception": self.detections}

    # _logged_run needs no CARLA SDK: it consumes an instrumented stack and replay metadata.
    backend = NuRecBackend({}, renderer=StubRenderer(height=2, width=3))
    backend.replay_scenario = {"seed": 7}
    trace = _logged_run(backend, DetectedCruise(), 3, runtime=HookSet())
    assert [record.n_detections for record in trace.records] == [1, 2, 3]
    assert trace_from_log(trace.run_log).mean_detections == trace.mean_detections == 2.0
    assert trace.replay_scenario == {"seed": 7}

    with Recorder(tmp_path) as recorder:
        for record in trace.run_log.steps:
            recorder.record_step(record)
    restored = trace_from_log(RunLog.read(tmp_path))
    assert restored.mean_detections == trace.mean_detections
