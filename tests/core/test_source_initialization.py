"""Source-only reset, bounded replay and initialization failures in the unified RunLog."""

from dataclasses import replace

import numpy as np
import pytest

from avsectester.backend import AVStack
from avsectester.closed_loop import ClosedLoopRunner, RunRequest
from avsectester.datasets.base import ActorState, FrameRef, ReferenceFrame, SourceData
from avsectester.plane import Control
from avsectester.scenarios.scene import EgoState, SceneGT
from avsectester.simulator_runtime import SimulatorRuntime
from avsectester.simulators.nurec import NuRecBackend, NuRecRenderer, StubRenderer
from avsectester.simulators.recorded import RecordedSequenceBackend


class SourceRenderer(StubRenderer):
    start_transform = None
    start_timestamp_us = None


class Driver(AVStack):
    def __call__(self, observation):
        return Control(throttle=1)


def nurec_source():
    pose = np.eye(4)
    pose[:3, 3] = [12, 3, 2]
    scene = SceneGT(5, 100.5, EgoState(4, pose), {}, [], {"scene_id": "recording"})
    return SourceData("nurec", "clip", FrameRef("clip", 5, 42, 100.5), {
        "scene": scene, "first_timestamp": 100, "terminal_timestamp": 100.6,
    })


def test_nurec_reset_uses_selected_source_and_rejects_extrapolation():
    backend = NuRecBackend({"dt": 0.05}, SourceRenderer(height=2, width=2))
    sim = SimulatorRuntime(backend)
    sim.reset(nurec_source())
    assert sim.observe().vehicle_state.x == 12
    assert sim.observe().ego_speed == 4
    assert backend.renderer.origin_timestamp_us == 100500000
    np.testing.assert_allclose(np.asarray(backend.renderer.origin_transform)[:3, 3], [12, 3, 2])
    sim.step(Control())
    sim.step(Control())
    assert sim.clock.t == pytest.approx(0.1)
    with pytest.raises(ValueError, match="terminal source"):
        sim.step(Control())
    assert backend.ground_truth().t == pytest.approx(0.1)


def test_nurec_render_rejects_out_of_source_time_before_requesting_rpc():
    renderer = NuRecRenderer(source_time_bounds=(100, 101))
    with pytest.raises(ValueError, match="timestamp exceeds"):
        renderer.render_at(np.eye(4), 102000000, "front")


def test_source_runtime_does_not_accept_selection_metadata():
    backend = NuRecBackend({}, SourceRenderer())
    with pytest.raises(TypeError, match="SourceData"):
        SimulatorRuntime(backend).reset({"scenario_id": "clip", "start": 5, "roles": {}})
    assert not backend._world_initialized
    with pytest.raises(ValueError, match="nurec SourceData"):
        SimulatorRuntime(backend).reset(replace(nurec_source(), kind="carla"))


def test_recorded_replay_advances_real_frames_and_never_repeats_terminal():
    references = tuple(ReferenceFrame(
        FrameRef("drive", i, f"sample-{i}", 3 + i * 0.5),
        SceneGT(i, 3 + i * 0.5, EgoState(i), {}, []),
        {"ego": ActorState(np.eye(4), (i, 0, 0))},
    ) for i in range(3))
    source = SourceData("nuscenes", "drive", references[0].frame, {"references": references})
    backend = RecordedSequenceBackend()
    log = ClosedLoopRunner().run(backend, Driver(), RunRequest(2, source=source))
    assert [step.physical_post.t for step in log.steps] == [3.5, 4]
    assert [step.physical_post.ego_speed for step in log.steps] == [1, 2]
    with pytest.raises(ValueError, match="terminal source") as error:
        ClosedLoopRunner().run(backend, Driver(), RunRequest(3, source=source))
    assert len(error.value.run_log.steps) == 2
    assert error.value.run_log.status == "failed"
    # Recorded replay is open loop, so irregular recorded intervals are replayed as recorded.
    irregular = (*references[:2], replace(references[2], frame=replace(references[2].frame, timestamp=5)))
    sim = SimulatorRuntime(RecordedSequenceBackend())
    sim.reset(replace(source, payload={"references": irregular}))
    sim.step(Control())
    sim.step(Control())
    assert sim.clock.t == 5


class InitializationFailure(ValueError):
    """Any initialization error may carry a report; the runner logs it before decision zero."""

    def __init__(self, report):
        self.report = report
        super().__init__("initialization failed")


def test_initialization_failure_is_logged_before_any_decision():
    report = {"failures": ["missing aligned capture: ['front']"]}

    class FailedReplay(NuRecBackend):
        def _initialize_world(self, *args, **kwargs):
            raise InitializationFailure(report)

    with pytest.raises(InitializationFailure) as error:
        ClosedLoopRunner().run(FailedReplay({}), Driver(), RunRequest(1, source=nurec_source()))
    log = error.value.run_log
    assert log.steps == [] and log.status == "failed"
    failure = next(event for event in log.events if event.kind == "failure")
    assert failure.data["initialization_report"] == report
