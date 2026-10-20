"""Reset from a ScenarioStart alone, bounded replay and initialization failures in the RunLog."""

from dataclasses import replace

import numpy as np
import pytest

from avsectester.closed_loop.avstack import AVStack
from avsectester.closed_loop.runner import ClosedLoopRunner, RunRequest
from avsectester.datasets.base import ActorState, FrameReference, FrameData, ScenarioStart
from avsectester.closed_loop.datatypes import Control
from avsectester.closed_loop.simulators.nurec import NuRecSimulatorRuntime, NuRecSimulator, NuRecRenderer, StubRenderer
from avsectester.closed_loop.simulators.recorded import RecordedReplaySimulator


class SourceRenderer(StubRenderer):
    pass


class Driver(AVStack):
    def __call__(self, observation):
        return Control(throttle=1)


def nurec_source():
    pose = np.eye(4)
    pose[:3, 3] = [12, 3, 2]
    ego = ActorState(pose, (4., 0., 0.), "vehicle", (4.5, 2., 1.6))
    return ScenarioStart("NuRecDataset", "clip", FrameReference("clip", 5, 42, 100.5), {
        "ego": ego, "scene_id": "recording", "first_timestamp": 100, "terminal_timestamp": 100.6,
    })


def test_nurec_reset_uses_selected_source_and_rejects_extrapolation():
    simulator_runtime = NuRecSimulator({"dt": 0.05}, SourceRenderer(height=2, width=2)).create_runtime()
    simulator_runtime.reset(nurec_source())
    assert simulator_runtime.observe().vehicle_state.x == 12
    assert simulator_runtime.observe().ego_speed == 4
    assert simulator_runtime.state().ego.extent == (4.5, 2., 1.6)
    np.testing.assert_allclose(simulator_runtime.state().ego.pose[:3, 3], [12, 3, 2])
    assert simulator_runtime.renderer.origin_timestamp_us == 100500000
    np.testing.assert_allclose(np.asarray(simulator_runtime.renderer.origin_transform)[:3, 3], [12, 3, 2])
    simulator_runtime.step(Control())
    simulator_runtime.step(Control())
    assert simulator_runtime.clock.t == pytest.approx(0.1)
    with pytest.raises(ValueError, match="terminal recorded"):
        simulator_runtime.step(Control())
    assert simulator_runtime._state().t == pytest.approx(0.1)


def test_nurec_render_rejects_out_of_source_time_before_requesting_rpc():
    renderer = NuRecRenderer()
    renderer.reset_origin(nurec_source())
    with pytest.raises(ValueError, match="timestamp exceeds"):
        renderer.render_at(np.eye(4), 102000000, "front")


def test_source_runtime_does_not_accept_selection_metadata():
    simulator_runtime = NuRecSimulator({}, SourceRenderer()).create_runtime()
    with pytest.raises(TypeError, match="ScenarioStart"):
        simulator_runtime.reset({"scenario_id": "clip", "start": 5, "roles": {}})
    assert not simulator_runtime._world_initialized
    with pytest.raises(ValueError, match="NuRecDataset ScenarioStart"):
        simulator_runtime.reset(replace(nurec_source(), kind="CarlaDataset"))


def test_recorded_replay_advances_real_frames_and_never_repeats_terminal():
    references = tuple(FrameData(
        FrameReference("drive", i, f"sample-{i}", 3 + i * 0.5),
        {"car": ActorState(np.eye(4), (2 * i, 0, 0))},
        ego=ActorState(np.eye(4), (i, 0, 0)),
    ) for i in range(3))
    source = ScenarioStart("NuScenesDataset", "drive", references[0].frame, {"frame_data": references})
    simulator_runtime = RecordedReplaySimulator().create_runtime()
    log = ClosedLoopRunner().run(simulator_runtime, Driver(), RunRequest(2, start=source))
    assert [step.physical_post.t for step in log.steps] == [3.5, 4]
    assert [step.physical_post.ego_speed for step in log.steps] == [1, 2]
    assert [step.physical_post.actors["car"].velocity[0] for step in log.steps] == [2, 4]
    assert all("ego" not in step.physical_post.actors for step in log.steps)
    with pytest.raises(ValueError, match="terminal recorded") as error:
        ClosedLoopRunner().run(simulator_runtime, Driver(), RunRequest(3, start=source))
    assert len(error.value.run_log.steps) == 2
    assert error.value.run_log.status == "failed"
    # Recorded replay is open loop, so irregular recorded intervals are replayed as recorded.
    irregular = (*references[:2], replace(references[2], frame=replace(references[2].frame, timestamp=5)))
    sim = RecordedReplaySimulator().create_runtime()
    sim.reset(replace(source, payload={"frame_data": irregular}))
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

    class FailedReplay(NuRecSimulatorRuntime):
        def __init__(self, *args, **kwargs):
            super().__init__(NuRecSimulator(*args, **kwargs))

        def _initialize_world(self, *args, **kwargs):
            raise InitializationFailure(report)

    with pytest.raises(InitializationFailure) as error:
        ClosedLoopRunner().run(FailedReplay({}), Driver(), RunRequest(1, start=nurec_source()))
    log = error.value.run_log
    assert log.steps == [] and log.status == "failed"
    failure = next(event for event in log.events if event.kind == "failure")
    assert failure.data["initialization_report"] == report


def test_frame_data_generation_is_shared_and_validates_frames():
    references = tuple(FrameData(
        FrameReference("drive", i, f"sample-{i}", 3 + i * 0.5),
        {}, ego=ActorState(np.eye(4), (i, 0, 0)),
    ) for i in range(3))
    start = ScenarioStart("NuScenesDataset", "drive", references[0].frame, {"frame_data": references})
    frames = tuple(record.frame for record in references)
    generated = RecordedReplaySimulator().create_runtime().generate_frame_data(start, frames)
    # Replaying recorded motion reproduces the recorded frame data.
    assert [record.frame for record in generated.frames] == list(frames)
    assert [record.ego.velocity for record in generated.frames] == [
        record.ego.velocity for record in references]
    assert generated.provenance["simulator"]["simulator"] == "RecordedReplaySimulator"
    with pytest.raises(ValueError, match="begin at the start frame"):
        RecordedReplaySimulator().create_runtime().generate_frame_data(start, frames[1:])
    with pytest.raises(ValueError, match="consecutive frames"):
        RecordedReplaySimulator().create_runtime().generate_frame_data(start, (frames[0], frames[2]))
    with pytest.raises(ValueError, match="terminal recorded"):
        RecordedReplaySimulator().create_runtime().generate_frame_data(
            replace(start, payload={"frame_data": references[:2]}), frames)
    with pytest.raises(NotImplementedError, match="cannot generate frame data"):
        NuRecSimulator({}, SourceRenderer()).create_runtime().generate_frame_data(nurec_source(), (nurec_source().frame,))
