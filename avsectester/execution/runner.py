"""One closed-loop scheduler with independent RunLog captures."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from contextlib import ExitStack
from dataclasses import asdict, dataclass, field, fields, is_dataclass, replace
import math
from pathlib import Path
import struct
from typing import Any, TYPE_CHECKING

from avsectester.validation import require_nonnegative_integer, require_positive_integer
from avsectester.closed_loop.environment import (
    SensorEnvironment, normalize_environment, validate_environment,
)
from avsectester.closed_loop.avstacks.base import AVStack
from avsectester.data_types import (
    Control, Observation, RecordedOutput, WorldSnapshot, without_sensor_payloads,
)
from avsectester.runlog import CaptureRecord, Recorder, RunLog, StepRecord
from avsectester.closed_loop.hooks import HookSet
from avsectester.closed_loop.simulators.base import DeploymentPlan, SimulatorRuntime
from avsectester.closed_loop.timing import (
    TimingConfiguration, DelayBuffer, validate_decision_period,
    reporting_timestamps, is_force_gt, is_handoff,
)
from avsectester.execution.scheduler import EventQueue
from avsectester.execution.acquisitions import AcquisitionBuffer

if TYPE_CHECKING:
    from avsectester.attack.runtime import OnlineContext


@dataclass(frozen=True)
class RecordingPolicy:
    """Choose sensor payloads without sampling commands or physical state."""

    sensor_ids: tuple[str, ...] = ()
    representations: tuple[str, ...] = ("model_input",)
    every_n_captures: int = 1

    def __post_init__(self) -> None:
        require_positive_integer(self.every_n_captures, "every_n_captures")
        if not set(self.representations) <= {"raw", "model_input"}:
            raise ValueError("representations must contain raw and/or model_input")
        if len(set(self.sensor_ids)) != len(self.sensor_ids):
            raise ValueError("sensor_ids must be unique")

    def observation(self, value: Observation, record_index: int, representation: str) -> Observation:
        """Retain the selected representation's payloads at the requested recording index."""
        return _record_observation(
            self, value, record_index, representation,
            capture_indices={sensor: record_index for sensor in self.sensor_ids},
            available_sensors=frozenset(value.sensor_data))


@dataclass(frozen=True)
class RunRequest:
    """One caller-owned simulator/driving-system run, independent of experiment orchestration.

    ``terminate_when`` receives each recorded decision and ends a successful run when true.
    ``retain_steps`` and ``retain_sensor_data`` choose what the returned log keeps in memory,
    as in ``Recorder``. Setting either to False requires ``output_path``.
    """

    decision_steps: int
    hooks: HookSet | None = None
    attacks: Sequence[Any] = ()
    recording: RecordingPolicy = field(default_factory=RecordingPolicy)
    output_path: str | Path | None = None
    retain_steps: bool = True
    retain_sensor_data: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)
    start: Any = None
    initialization: Any = None
    deployment: Any = None
    environment: Mapping[str, Any] = field(default_factory=dict)
    terminate_when: Callable[[StepRecord], bool] | None = None
    timing: TimingConfiguration = field(default_factory=TimingConfiguration)

    def __post_init__(self) -> None:
        require_nonnegative_integer(self.decision_steps, "decision_steps")
        if not isinstance(self.timing, TimingConfiguration):
            raise TypeError("timing must be TimingConfiguration")
        object.__setattr__(self, "environment", normalize_environment(self.environment))
        if not self.retain_steps and self.output_path is None:
            raise ValueError("retain_steps=False requires an output_path")
        if not self.retain_sensor_data and self.output_path is None:
            raise ValueError("retain_sensor_data=False requires an output_path")


def _record_observation(policy: RecordingPolicy, value: Observation, record_index: int,
                        representation: str, *, capture_indices: Mapping[str, int],
                        available_sensors: frozenset[str], check_available: bool = True) -> Observation:
    requested = set(policy.sensor_ids) if representation in policy.representations else set()
    eligible = {sensor for sensor in requested
                if capture_indices.get(sensor, record_index) % policy.every_n_captures == 0}
    due = (eligible & capture_indices.keys()) | (eligible - available_sensors)
    unknown = due - value.sensor_data.keys()
    if check_available and unknown:
        raise ValueError(f"Recording requested unavailable sensors: {sorted(unknown)}")
    return without_sensor_payloads(value, keep_sensor_ids=eligible & value.sensor_data.keys())


def _same_payload(a: Any, b: Any) -> bool:
    """Compare complete payload values without changing either input."""
    import numpy as np

    if type(a) is not type(b):
        return False
    if isinstance(a, np.ndarray):
        return (type(a) is np.ndarray and a.dtype == b.dtype and a.shape == b.shape
                and a.tobytes() == b.tobytes())
    if isinstance(a, bytes):
        return a == b
    if isinstance(a, np.generic):
        return a.tobytes() == b.tobytes()
    if isinstance(a, float):
        return not (math.isnan(a) or math.isnan(b)) and struct.pack("d", a) == struct.pack("d", b)
    if a is None or np.isscalar(a):
        return bool(a == b)
    if is_dataclass(a) and not isinstance(a, type):
        return all(_same_payload(getattr(a, item.name), getattr(b, item.name)) for item in fields(a))
    if isinstance(a, Mapping):
        return list(a) == list(b) and all(_same_payload(a[key], b[key]) for key in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_same_payload(left, right) for left, right in zip(a, b))
    if hasattr(a, "__dict__"):
        return _same_payload(vars(a), vars(b))
    return False


def _freeze_payload(value: Any) -> None:
    """Make array leaves in a newly created payload snapshot read-only."""
    import numpy as np

    if isinstance(value, np.ndarray):
        value.flags.writeable = False
    elif is_dataclass(value) and not isinstance(value, type):
        for item in fields(value):
            _freeze_payload(getattr(value, item.name))
    elif isinstance(value, Mapping):
        for item in value.values():
            _freeze_payload(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _freeze_payload(item)
    elif hasattr(value, "__dict__"):
        _freeze_payload(vars(value))


class _CaptureRecording:
    """Acquisition recording drafts and physical evidence for the event loop."""
    def __init__(self, recorder: Recorder, policy: RecordingPolicy, sensors: frozenset[str]) -> None:
        self.recorder, self.policy, self.sensors = recorder, policy, sensors
        self.past_metadata: list[dict[str, Any]] = []
        self.metadata: dict[str, CaptureRecord] = {}
        self.drafts: dict[int, tuple[CaptureRecord, Mapping[str, int]]] = {}
        self.last_payload: dict[tuple[str, str], Any] = {}

    def _payload(self, sensor: str, value: Any, representation: str, same_capture_raw: Any = None,
                 *, include_previous_raw: bool = False) -> Any:
        result = self.recorder.snapshot(value)
        candidates: tuple[Any, ...] = (
            same_capture_raw, self.last_payload.get((sensor, representation)))
        if include_previous_raw:
            candidates += (self.last_payload.get((sensor, "raw")),)
        for candidate in candidates:
            if _same_payload(result, candidate):
                result = candidate
                break
        else:
            _freeze_payload(result)
        self.last_payload[sensor, representation] = result
        return result

    def snapshot_raw_observation(self, observation: Observation) -> Observation:
        result = self.recorder.snapshot(without_sensor_payloads(observation))
        result.sensor_data = {sensor: self._payload(sensor, payload, "raw")
                              for sensor, payload in observation.sensor_data.items()}
        return result

    def indices(self, identity: int | None) -> Mapping[str, int]:
        return {} if identity is None else self.drafts[identity][1]

    def capture_record(self, identity: int, time_us: int, sensors: Sequence[str], raw: Observation,
                       truth: Any, geometry: Mapping[str, Any], visibility: Mapping[str, Any],
                       indices: Mapping[str, int], ranges: Mapping[str, range]) -> None:
        draft = CaptureRecord(
            identity, time_us, tuple(sensors), raw, physical_state=truth,
            sensor_geometry=geometry, object_visibility=visibility,
            frame_ranges_us={sensor: (span.start, span.stop) for sensor, span in ranges.items()})
        self.drafts[identity] = draft, dict(indices)
        for sensor in sensors:
            self.metadata[sensor] = replace(
                draft, raw_observation=None,
                sensor_geometry={sensor: geometry[sensor]} if sensor in geometry else {})

    def finalize_captures(self, observation: Observation | None = None
                          ) -> tuple[dict[str, tuple[Any, Any]], Mapping[str, Any]]:
        delivered: dict[int, dict[str, Any]] = {}
        model_payloads: dict[str, tuple[Any, Any]] = {}
        decision_raw: Mapping[str, Any] = {}
        decision_time = None if observation is None else observation.estimated_states()[-1].time_us
        if observation is not None:
            for sensor in observation.deliveries:
                if sensor not in observation.sensor_data:
                    continue
                for sample in observation.delivered(sensor):
                    if sample.capture_id is not None and sample.capture_id in self.drafts:
                        delivered.setdefault(sample.capture_id, {})[sensor] = sample
        for identity, (draft, indices) in tuple(self.drafts.items()):
            if draft.time_us == decision_time:
                decision_raw = draft.raw_observation.sensor_data
            model = None
            samples = delivered.get(identity, {})
            if samples:
                sample = next(iter(samples.values()))
                model = replace(
                    draft.raw_observation, t=sample.t, frame=sample.frame,
                    sensor_data={sensor: item.data for sensor, item in samples.items()},
                    calibration={sensor: item.calibration for sensor, item in samples.items()
                                 if item.calibration is not None},
                    vehicle_state=sample.state.vehicle_state, ego_speed=sample.state.ego_speed,
                    deliveries={}, state_estimates=())
                if observation is not None and draft.time_us == decision_time:
                    model.route = observation.route
                model = _record_observation(
                    self.policy, model, identity, "model_input", capture_indices=indices,
                    available_sensors=self.sensors, check_available=False)
            if model is not None:
                selected = model.sensor_data
                model = self.recorder.snapshot(without_sensor_payloads(model))
                payloads = {
                    sensor: self._payload(sensor, value, "model",
                                          draft.raw_observation.sensor_data.get(sensor))
                    for sensor, value in selected.items()
                }
                model.sensor_data = payloads
                model_payloads.update({sensor: (selected[sensor], payload)
                                       for sensor, payload in payloads.items()})
            retained = set(draft.raw_observation.sensor_data)
            if model is not None:
                retained.update(model.sensor_data)
            geometry = {sensor: value for sensor, value in draft.sensor_geometry.items()
                        if sensor in retained}
            del self.drafts[identity]
            record = self.recorder.record_capture(replace(draft, model_observation=model,
                                                         sensor_geometry=geometry))
            for sensor in draft.sensor_ids:
                if self.metadata[sensor].capture_id == identity:
                    self.metadata[sensor] = replace(
                        record, raw_observation=None, model_observation=None,
                        sensor_geometry={sensor: geometry[sensor]} if sensor in geometry else {})
        return model_payloads, decision_raw

    def decision_records(self, observation: Observation, model: Observation, raw: Observation
                         ) -> tuple[Observation, Observation]:
        """Finalize delivered captures, then snapshot the decision's model and raw records."""
        model_payloads, capture_raw = self.finalize_captures(observation)
        model_record = self.recorder.snapshot(without_sensor_payloads(model))
        for sensor, value in model.sensor_data.items():
            captured = model_payloads.get(sensor)
            model_record.sensor_data[sensor] = (
                captured[1] if captured is not None and captured[0] is value
                else self._payload(sensor, value, "model", include_previous_raw=True))
        raw_record = self.recorder.snapshot(without_sensor_payloads(raw))
        raw_record.sensor_data = {
            sensor: self._payload(sensor, value, "raw", capture_raw.get(sensor))
            for sensor, value in raw.sensor_data.items()}
        return model_record, raw_record

    def physical_evidence(self) -> dict[str, Any]:
        geometry = {
            sensor: record.sensor_geometry[sensor]
            for sensor in self.policy.sensor_ids
            if sensor in self.metadata
            and sensor in (record := self.metadata[sensor]).sensor_geometry
        }
        return {
            "sensor_geometry": geometry,
            "physical_captures": {sensor: record.capture_id for sensor, record in self.metadata.items()},
            "object_visibility": {
                identity: {
                    sensor: record.object_visibility[identity][sensor]
                    for sensor, record in sorted(self.metadata.items())
                    if identity in record.object_visibility
                    and sensor in record.object_visibility[identity]
                }
                for identity in dict.fromkeys(identity for record in self.metadata.values()
                                               for identity in record.object_visibility)
            },
        }


class _ClosedLoopRun:
    """Explicit state and scheduled operations for one closed-loop run."""
    def __init__(self, simulator_runtime: SimulatorRuntime, avstack: AVStack, request: RunRequest,
                 recorder: Recorder | None, release_owned_resources: Callable[[], None] | None) -> None:
        self.sim = simulator_runtime
        self.avstack = avstack
        self.request = request
        self._supplied_recorder = recorder
        self.recorder: Recorder
        self.release_owned_resources = release_owned_resources
        self.resources_released = False
        self.current_step = -1
        self.terminated_by_condition = False
        self.flush_capture_records: Callable[[], Any] | None = None
        self.origin_us: int
        self.period: int
        self.decision_period_us: int | None
        self.buffer: AcquisitionBuffer
        self.recording: _CaptureRecording
        self.events: EventQueue
        self.command_buffer: DelayBuffer
        self.pending: StepRecord | None
        self.raw: Observation | None
        self.captured: Observation | None
        self.localized: Observation | None
        self.pre_state: WorldSnapshot
        self.executed: Control | None
        self.force_gt: bool
        self.terminal: bool
        self.active_time: int | None
        self.time_capture_id: int | None
        self.capture_id: int
        self.latest_capture_ids: dict[str, int | None]
        self.expected_delivery_indices: dict[str, int]
        self.timing: TimingConfiguration

    def _record_attack_output(self, context: "OnlineContext", name: str, value: Any) -> None:
        # Attack outputs join the hook outputs, so both are logged with their decision.
        self.session.outputs.append(RecordedOutput(
            name=name, value=deepcopy(value), step=context.step, stage=context.stage,
            frame=context.frame, t=context.sim_time))

    def _prepare_participants(self) -> None:
        from avsectester.execution.attack_dispatch import AttackDispatcher

        if not isinstance(self.sim, SimulatorRuntime):
            raise TypeError("ClosedLoopRunner requires SimulatorRuntime")
        self.history = self.avstack.history
        if (self.history.duration_us or self.history.samples > 1) and not self.sim.can_observe_past:
            raise NotImplementedError(
                f"{type(self.sim).__name__} cannot provide recorded acquisition history "
                f"observations required by {type(self.avstack).__name__}")
        validate_environment(self.request.environment, self.sim.simulator)
        self.hooks = self.request.hooks or HookSet()
        self.sensor_environment = SensorEnvironment(self.request.environment, seed=self.hooks.seed or 0)
        self.attack_dispatch = AttackDispatcher(
            self.request.attacks, observation_copy=self.sim.copy_observation,
            output_recorder=self._record_attack_output)
        # Validate before world reset or handlers acquire native resources.
        available = self.hooks.validate_stages(self.sim, self.avstack)
        self.attack_dispatch.validate(available)
        if self.request.attacks:
            self.hooks = self.hooks.with_hooks(self.attack_dispatch.hooks)
        self.session = self.hooks.session(self.sim, self.avstack)
        self.deployment = self.request.deployment or DeploymentPlan(self.hooks.setup_ticks)
        if self.deployment.ticks < self.hooks.setup_ticks:
            raise ValueError("Deployment plan is shorter than the participants require")
        self.recorder = self._supplied_recorder or Recorder(
            path=self.request.output_path, retain_steps=self.request.retain_steps,
            retain_sensor_data=self.request.retain_sensor_data,
            metadata={
                **dict(self.request.metadata),
                "avstack_type": type(self.avstack).__name__,
                "environment": deepcopy(dict(self.request.environment)),
                "decision_steps": self.request.decision_steps,
                "timing": asdict(self.request.timing),
                "recording_policy": {
                    "sensor_ids": list(self.request.recording.sensor_ids),
                    "representations": list(self.request.recording.representations),
                    "every_n_captures": self.request.recording.every_n_captures,
                },
            },
        )

    def _actions(self) -> list[dict[str, Any]]:
        records = self.attack_dispatch.records()
        return [
            {"kind": kind, **item}
            for kind, values in (("request", records["requests"]), ("receipt", records["receipts"]))
            for item in values
        ]

    def _drain(self) -> None:
        self.session.drain_outputs()
        self.attack_dispatch.drain_records()

    def _cleanup_attacks(self) -> None:
        failure: BaseException | None = None
        try:
            # Cleanup reuses the latest context: a failed reset or step may invalidate the state.
            self.attack_dispatch.stop()
        except BaseException as error:
            failure = error
        try:
            # Failed decisions and unapplied actions still have evidence.
            self.recorder.record_event(
                "attack_cleanup", step=max(self.current_step, 0),
                actions=self._actions(), outputs=self.session.outputs)
            self._drain()
        except BaseException as error:
            if failure is None:
                raise
            failure.add_note(f"Attack cleanup logging also failed: {error!r}")
        if failure is not None:
            raise failure

    def _context(self, stage: str, observation: Observation | None = None) -> OnlineContext:
        from avsectester.attack.runtime import OnlineContext

        clock = self.sim.clock
        step = max(self.current_step, 0)
        period = self.decision_period_us
        decision_time = self.origin_us + step * (period or 0)
        return OnlineContext(
            step=step, stage=stage, sim_time=clock.t, frame=clock.frame, observation=observation,
            physical_state=self.sim.state(), decision_time_us=decision_time,
            decision_period_us=period,
            force_gt=period is not None and is_force_gt(
                self.request.timing, decision_time, self.origin_us, period))

    def _release(self) -> None:
        if self.release_owned_resources is not None and not self.resources_released:
            self.resources_released = True
            self.release_owned_resources()

    def _environment_indices(self, time_us: int, sensors: Sequence[str]
                             ) -> tuple[int, dict[str, int], dict[str, int]]:
        offset = time_us - self.origin_us
        grid, remainder = divmod(offset, self.period)
        indices = {sensor: self.buffer.index_at(sensor, time_us) for sensor in sensors}
        seed_indices = {
            sensor: 0 if remainder == 0 else indices[sensor]
            - self.buffer.timings[sensor].latest_frame_index(grid * self.period, self.period)
            for sensor in sensors}
        return self.start_position + grid, indices, seed_indices

    def _prepare(self) -> None:
        # AttackRuntime.setup runs inside the reset, at the simulator.setup stage.
        self.sim.configure_timing(self.request.timing, history=self.history)
        if self.request.decision_steps:
            self.sim.simulator.validate_startup(
                self.request.start, self.request.timing.force_gt_duration_us)
        self.sim.reset(
            self.request.start, self.request.initialization, deployment=self.deployment,
            world=self.request.environment["world"])
        self.sensor_environment.bind(self.sim.observe())
        self.initial_state = self.sim.state()
        try:
            self.decision_period_us = self.sim.decision_period_us
        except ValueError:
            if self.request.decision_steps or self.history.duration_us or self.history.samples > 1:
                raise
            # A zero-decision run acquires only the origin. No period is used.
            self.decision_period_us = None
        self.timing = self.request.timing
        if self.decision_period_us is not None:
            validate_decision_period(self.timing, self.decision_period_us)
        unknown_sensors = self.timing.sensors.keys() - self.sim.sensors
        if unknown_sensors:
            raise ValueError(f"Unknown timing sensors: {sorted(unknown_sensors)}")
        if self.timing.decision_period_us not in (None, self.decision_period_us):
            raise ValueError("timing.decision_period_us must equal the dataset time grid")
        self.origin_us = self.sim.clock.time_us
        # Zero-decision runs without a declared grid acquire only the origin.
        self.period = 1 if self.decision_period_us is None else self.decision_period_us
        self.buffer = AcquisitionBuffer(
            self.sim.sensors, self.period, self.origin_us, self.timing, self.history,
            interpolate_vehicle_state=self.sim.interpolate_vehicle_state)
        self.start_position = self.request.start.frame.position if self.request.start is not None else 0
        self.recording = _CaptureRecording(self.recorder, self.request.recording, self.sim.sensors)

    def _read_history(self) -> None:
        past_times, past_pose_times = self.buffer.past_times()
        self.flush_capture_records = self.recording.finalize_captures
        for index, (offset, sensors) in enumerate(past_times.items()):
            acquisition = self.sim.observe_recorded_at(offset, sensors)
            previous = acquisition.observation
            grid, indices, seed_indices = self._environment_indices(self.origin_us + offset, sensors)
            previous = self.sensor_environment.apply(
                previous, grid, interval_capture_index=seed_indices, available_sensors=self.sim.sensors)
            identity = index - len(past_times)
            ranges = {sensor: self.sim.frame_range_us(sensor, self.origin_us + offset)
                      for sensor in sensors}
            self.buffer.append(
                previous, self.origin_us + offset, identity, sensors, frame_ranges=ranges)
            if offset in past_pose_times:
                self.buffer.append_state(previous, self.origin_us + offset)
            truth = acquisition.physical_state
            geometry = acquisition.sensor_geometry
            past_raw = self.recording.snapshot_raw_observation(_record_observation(
                self.request.recording, previous, identity, "raw", capture_indices=indices,
                available_sensors=self.sim.sensors, check_available=False))
            self.recording.capture_record(
                identity, self.origin_us + offset, sensors, past_raw, truth, geometry,
                acquisition.object_visibility, indices, ranges)
            self.recording.past_metadata.append(
                {"t": previous.t, "vehicle_state": self.recorder.snapshot(previous.vehicle_state)})
        if past_times:
            del previous

    def _initialize_events(self) -> None:
        self.recorder.record_event(
            "initialization", physical_state=self.initial_state, clock=self.sim.clock,
            setup_ticks=self.deployment.ticks, decision_period_us=self.decision_period_us,
            initialization_report=self.sim.initialization_report, outputs=self.session.outputs,
            actions=self._actions(),
            object_visibility={identity: self.sim.object_visibility(identity)
                               for identity in self.initial_state.objects},
            past_captures=len(self.recording.past_metadata),
            past_observations=self.recording.past_metadata)
        self._drain()
        self.events = EventQueue(self.origin_us)
        self.events.schedule(self.origin_us, ("frame", tuple(sorted(self.sim.sensors))), priority=10)
        self.events.schedule(self.origin_us, ("pose", ()), priority=15)
        self.events.schedule(self.origin_us, ("decision", ()), priority=20)
        if self.request.decision_steps:
            self.events.schedule(self.origin_us, ("advance", ()), priority=40)
        self.pending = None
        self.command_buffer = DelayBuffer(self.timing.planner_delay_us)
        self.capture_id = 0
        self.active_time = None
        self.time_capture_id = None
        self.latest_capture_ids = {}
        self.expected_delivery_indices = {}
        self.raw = self.captured = self.localized = None
        self.pre_state = self.initial_state
        self.executed = None
        self.force_gt = False
        self.terminal = False

    def _begin_time(self, time_us: int, sensors: Sequence[str]) -> None:
        if self.active_time == time_us:
            return
        self.active_time = time_us
        self.time_capture_id = None
        self.raw = self.sim.capture_at(time_us, sensors)
        self.captured = self.localized = None
        self.pre_state = self.sim.state()
        offset = time_us - self.origin_us
        decision, remainder = divmod(offset, self.period)
        if remainder == 0 and self.pending is not None:
            self._finish_step()
        self.terminal = 0 < self.request.decision_steps <= decision or self.terminated_by_condition
        if remainder == 0 and not self.terminal:
            self.current_step = decision if self.request.decision_steps else -1
            self.session.step, self.session.control = decision, None
            if self.request.attacks and self.request.decision_steps:
                self.attack_dispatch.step(self._context("step", self.raw))
                self.attack_dispatch.discard_missing_sensor_actions(self.raw, self.sim)

    def _finish_step(self) -> None:
        assert self.raw is not None
        assert self.pending is not None
        if self.request.attacks:
            self.attack_dispatch.record_end(self._context("step.end", self.raw))
        feedback_error = None
        try:
            self.session.after_step(self.raw)
        except BaseException as error:
            feedback_error = error
        try:
            record = self.recorder.record_step(replace(
                self.pending, physical_post=self.recorder.snapshot(self.pre_state),
                actions=self._actions(), outputs=self.session.outputs))
        except BaseException as error:
            if feedback_error is not None:
                feedback_error.add_note(f"Completed decision logging also failed: {error!r}")
                raise feedback_error
            raise
        self._drain()
        self.pending = None
        if feedback_error is not None:
            raise feedback_error
        if self.request.terminate_when is not None and self.request.terminate_when(record):
            self.terminated_by_condition = True

    def _on_frame(self, time_us: int, sensors: Sequence[str]) -> None:
        assert self.raw is not None
        self.time_capture_id = self.capture_id
        ranges = {sensor: self.sim.frame_range_us(sensor, time_us) for sensor in sensors}
        geometry = {sensor: self.sim.sensor_geometry(sensor)
                    for sensor in sensors if sensor in self.sim.camera_sensors}
        visibility = self.recorder.snapshot(
            {identity: self.sim.object_visibility(identity) for identity in self.pre_state.objects})
        grid, indices, seed_indices = self._environment_indices(time_us, sensors)
        self.expected_delivery_indices.update(indices)
        raw_record = self.recording.snapshot_raw_observation(_record_observation(
            self.request.recording, self.raw, self.capture_id, "raw", capture_indices=indices,
            available_sensors=self.sim.sensors, check_available=False))
        self.recording.capture_record(
            self.capture_id, time_us, sensors, raw_record, self.pre_state, geometry,
            visibility, indices, ranges)
        if self.terminal:
            self.recording.finalize_captures()
            self.capture_id += 1
            return
        attacked = (self.attack_dispatch.apply_sensor_actions(self.raw, self.sim)
                    if self.request.attacks and self.request.decision_steps else self.raw)
        conditioned = self.sensor_environment.apply(
            attacked, grid, interval_capture_index=seed_indices, available_sensors=self.sim.sensors)
        self.captured = self.session.observation_pipeline.capture(conditioned, self.session)
        self.buffer.append(self.captured, time_us, self.capture_id, sensors, frame_ranges=ranges)
        self.capture_id += 1

    def _on_pose(self, time_us: int) -> None:
        assert self.raw is not None
        if self.terminal and self.time_capture_id is None:
            self.recording.capture_record(self.capture_id, time_us, (), self.raw,
                                          self.pre_state, {}, {}, {}, {})
            self.recording.finalize_captures()
            self.capture_id += 1
        if not self.terminal:
            self.localized = self.session.observation_pipeline.localize(
                self.captured if self.captured is not None else self.sim.copy_observation(self.raw),
                self.session)
            if self.time_capture_id is not None:
                self.buffer.update_capture(self.localized, time_us, self.time_capture_id)
            self.buffer.append_state(self.localized, time_us)

    def _on_decision(self, time_us: int) -> bool:
        assert self.raw is not None
        decision = (time_us - self.origin_us) // self.period
        if self.terminal:
            return False
        assert self.localized is not None
        seen = self.buffer.take_ready(self.localized, time_us)
        delivered_indices = dict(self.expected_delivery_indices)
        self.expected_delivery_indices.clear()
        seen = self.session.observation_pipeline.decision(seen, self.session)
        seen.deliveries = {sensor: values if sensor in seen.sensor_data else ()
                           for sensor, values in seen.deliveries.items()}
        raw_selected = _record_observation(
            self.request.recording, self.raw, decision, "raw",
            capture_indices=self.recording.indices(self.time_capture_id),
            available_sensors=self.sim.sensors)
        model_selected = _record_observation(
            self.request.recording, seen, decision, "model_input",
            capture_indices=delivered_indices, available_sensors=self.sim.sensors)
        model_record, raw_record = self.recording.decision_records(
            seen, model_selected, raw_selected)
        if decision == 0:
            self.avstack.reset(seen)
        if self.request.decision_steps == 0:
            return False
        self.force_gt = is_force_gt(self.timing, time_us, self.origin_us, self.period)
        skip_driver = self.force_gt and self.timing.skip_driver_during_force_gt
        if skip_driver:
            self.avstack.ingest(seen)
            produced = command = detections = produced_record = None
        else:
            produced = self.avstack(seen)
            detections = self.recorder.snapshot(self.avstack.detections())
            produced_record = self.recorder.snapshot(produced)
            command = self.session.command(produced)
        trajectory_reference = "trajectory" in (self.sim.command_modes or ())
        buffered = (self.sim.recorded_reference(time_us, time_us + self.period)
                    if self.force_gt and trajectory_reference else command)
        self.command_buffer.add(buffered, time_us)
        production_time, self.executed = self.command_buffer.item_at(time_us)
        if self.force_gt and not trajectory_reference:
            production_time, self.executed = None, None
        self.attack_dispatch.finish_decision(
            skip_driver=skip_driver, every_sensor_due=self.sim.sensors <= seen.deliveries.keys())
        detection_sensor = self.avstack.detection_sensor
        for sensor, values in seen.deliveries.items():
            self.latest_capture_ids[sensor] = values[-1].capture_id if values else None
        detection_capture_id = (None if detection_sensor is None
                                else self.latest_capture_ids.get(detection_sensor))
        evidence = self.recording.physical_evidence()
        self.pending = StepRecord(
            step=decision, raw_observation=raw_record, model_observation=model_record,
            physical_pre=self.recorder.snapshot(self.pre_state), produced_command=produced_record,
            executed_command=self.recorder.snapshot(self.executed),
            executed_command_issue_time_us=production_time, force_gt=self.force_gt,
            delivered_captures={sensor: tuple(item.capture_id for item in values)
                                for sensor, values in seen.deliveries.items()},
            components={} if skip_driver else self.recorder.snapshot(
                self.avstack.component_log() or {}),
            detections=detections, detection_capture_id=detection_capture_id,
            detection_sensor=detection_sensor,
            telemetry={"sensor_payloads": {
                "raw": {key: "recorded" if key in raw_record.sensor_data else "not_recorded"
                        for key in self.raw.sensor_data},
                "model_input": {key: "recorded" if key in model_record.sensor_data
                                else "not_recorded" for key in seen.sensor_data},
            }},
            **evidence)
        return True

    def _on_advance(self, time_us: int) -> None:
        decision = (time_us - self.origin_us) // self.period
        end = time_us + self.period
        frame_events = self.buffer.due_between(time_us, end)
        for timestamp, due in frame_events.items():
            self.events.schedule(timestamp, ("frame", due), priority=10)
        for timestamp in reporting_timestamps(time_us, end, self.timing.pose_reporting_interval_us):
            self.events.schedule(timestamp, ("pose", ()), priority=15)
        self.events.schedule(end, ("pose", ()), priority=15)
        self.events.schedule(end, ("decision", ()), priority=20)
        if decision + 1 < self.request.decision_steps:
            self.events.schedule(end, ("advance", ()), priority=40)
        self.session.control = self.executed
        self.sim.advance_interval(
            end, self.executed,
            report_times_us=reporting_timestamps(time_us, end, self.timing.pose_reporting_interval_us),
            force_gt=self.force_gt,
            coerce_dynamic_state=self.force_gt or is_handoff(self.timing, time_us, self.origin_us))

    def _loop(self) -> None:
        for event in self.events:
            kind, sensors = event.value
            self._begin_time(event.time_us, sensors)
            assert self.raw is not None
            if kind == "frame":
                self._on_frame(event.time_us, sensors)
            elif kind == "pose":
                self._on_pose(event.time_us)
            elif kind == "decision":
                if not self._on_decision(event.time_us):
                    break
            elif kind == "advance":
                self._on_advance(event.time_us)


class ClosedLoopRunner:
    """Schedule one driving loop. Never close caller-owned simulator/driving objects."""
    def run(self, simulator_runtime: SimulatorRuntime, avstack: AVStack, request: RunRequest, *,
            recorder: Recorder | None = None,
            release_owned_resources: Callable[[], None] | None = None) -> RunLog:
        """Execute the requested treatment, record its decisions and close all run resources."""
        from avsectester.attack.runtime import OnlineContext

        run = _ClosedLoopRun(simulator_runtime, avstack, request, recorder, release_owned_resources)
        try:
            run._prepare_participants()
        except BaseException as error:
            origin = request.start.frame if request.start is not None else None
            # Ownership transfers at entry, before capability checks or recorder allocation.
            # These runtimes have not entered AttackDispatcher.setup yet.
            stopped: set[int] = set()
            for attack in reversed(request.attacks):
                if id(attack) in stopped:
                    continue
                stopped.add(id(attack))
                stop = getattr(attack, "stop", None)
                if not callable(stop):
                    continue
                try:
                    stop(OnlineContext(0, "stop", origin.timestamp if origin else 0.0,
                                       origin.position if origin else 0))
                except BaseException as cleanup:
                    error.add_note(f"Attack cleanup also failed: {cleanup!r}")
            if release_owned_resources is not None:
                try:
                    release_owned_resources()
                except BaseException as cleanup:
                    error.add_note(f"Owned resource cleanup also failed: {cleanup!r}")
            raise
        try:
            with run.sim.exclusive(), ExitStack() as run_resources:
                run_resources.callback(run._release)
                run_resources.enter_context(run.session.activate())
                try:
                    run._prepare()
                    run._read_history()
                    run._initialize_events()
                    run._loop()
                except BaseException as error:
                    if run.attack_dispatch.started:
                        try:
                            run._cleanup_attacks()
                        except BaseException as cleanup:
                            error.add_note(f"Attack cleanup also failed: {cleanup!r}")
                    raise
                finally:
                    if run.attack_dispatch.started:
                        run._cleanup_attacks()
            run._release()
            termination_details = dict(run.sim.termination_details)
            if run.terminated_by_condition:
                termination_details["terminated_by_condition"] = True
            if run.flush_capture_records is not None:
                run.flush_capture_records()
            return run.recorder.finish(status="completed", **termination_details)
        except BaseException as error:
            if run.flush_capture_records is not None:
                try:
                    run.flush_capture_records()
                except BaseException as cleanup:
                    error.add_note(f"Capture finalization also failed: {cleanup!r}")
            try:
                run._release()
            except BaseException as cleanup:
                error.add_note(f"Owned resource cleanup also failed: {cleanup!r}")
            try:
                step = run.current_step if run.current_step >= 0 else None
                run.recorder.record_event(
                    "failure", step=step, error_type=type(error).__name__, message=str(error),
                    initialization_report=getattr(error, "report", {}),
                    cleanup_errors=list(getattr(error, "__notes__", ())),
                    actions=run._actions(), outputs=run.session.outputs)
                run._drain()
                run.recorder.finish(
                    status="cancelled" if isinstance(error, (KeyboardInterrupt, SystemExit))
                    else "failed",
                    incomplete_step=step, **run.sim.termination_details)
            except BaseException as cleanup:
                error.add_note(f"Log finalization also failed: {cleanup!r}")
                try:
                    run.recorder.abort()
                except BaseException as abort_error:
                    error.add_note(f"Log resource cleanup also failed: {abort_error!r}")
            # Preserve the exception identity while making its in-memory partial log inspectable.
            try:
                setattr(error, "run_log", run.recorder.log)
            except (AttributeError, TypeError):
                pass
            raise


__all__ = ["RecordingPolicy", "RunRequest", "ClosedLoopRunner"]
