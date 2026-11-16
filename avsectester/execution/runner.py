"""One closed-loop scheduler with independent RunLog captures."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from contextlib import ExitStack
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from avsectester.validation import require_nonnegative_integer, require_positive_integer
from avsectester.closed_loop.environment import SensorEnvironment, normalize_environment, validate_environment
from avsectester.closed_loop.avstacks.base import AVStack
from avsectester.data_types import Observation, RecordedOutput, without_sensor_payloads
from avsectester.runlog import CaptureRecord, Recorder, RunLog, StepRecord
from avsectester.closed_loop.hooks import HookSet
from avsectester.closed_loop.simulators.base import DeploymentPlan, SimulatorRuntime
from avsectester.closed_loop.timing import TimingConfiguration, DelayBuffer
from avsectester.execution.scheduler import EventQueue
from avsectester.execution.acquisitions import AcquisitionBuffer


@dataclass(frozen=True)
class RecordingPolicy:
    """Choose sensor payloads without sampling commands or physical state."""

    sensor_ids: tuple[str, ...] = ()
    representations: tuple[str, ...] = ("model_input",)
    every_n_captures: int = 1
    sensor_every_n_captures: Mapping[str, int] = field(default_factory=dict, hash=False)

    def __post_init__(self) -> None:
        require_positive_integer(self.every_n_captures, "every_n_captures")
        for name, interval in self.sensor_every_n_captures.items():
            if name not in self.sensor_ids:
                raise ValueError("Per-sensor recording intervals must name selected sensor_ids")
            require_positive_integer(interval, f"sensor_every_n_captures.{name}")
        if not set(self.representations) <= {"raw", "model_input"}:
            raise ValueError("representations must contain raw and/or model_input")
        object.__setattr__(self, "sensor_every_n_captures", dict(self.sensor_every_n_captures))
        if len(set(self.sensor_ids)) != len(self.sensor_ids):
            raise ValueError("sensor_ids must be unique")

    def observation(self, value: Observation, record_index: int, representation: str, *,
                    capture_indices: Mapping[str, int] | None = None,
                    available_sensors: frozenset[str] | None = None,
                    check_available: bool = True) -> Observation:
        """Select due payloads by acquisition index, preserving capture metadata.

        Available sensor IDs distinguish an unknown name from a sensor not due at this time.
        """
        requested = set(self.sensor_ids) if representation in self.representations else set()
        eligible = {sensor for sensor in requested
                    if (capture_indices or {}).get(sensor, record_index)
                    % self.sensor_every_n_captures.get(sensor, self.every_n_captures) == 0}
        due = eligible if capture_indices is None else eligible & capture_indices.keys()
        if available_sensors is not None:
            due |= eligible - available_sensors
        unknown = due - value.sensor_data.keys()
        if check_available and unknown:
            raise ValueError(f"Recording requested unavailable sensors: {sorted(unknown)}")
        retained = eligible & value.sensor_data.keys()
        return without_sensor_payloads(value, keep_sensor_ids=retained)



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


class ClosedLoopRunner:
    """Schedule one driving loop. Never close caller-owned simulator/driving objects."""

    def run(
        self,
        simulator_runtime: SimulatorRuntime,
        avstack: AVStack,
        request: RunRequest,
        *,
        recorder: Recorder | None = None,
        release_owned_resources: Callable[[], None] | None = None,
    ) -> RunLog:
        """Execute the requested treatment, record its decisions and close all run resources."""
        from avsectester.attack.runtime import OnlineContext
        from avsectester.execution.attack_dispatch import AttackDispatcher

        try:
            if not isinstance(simulator_runtime, SimulatorRuntime):
                raise TypeError("ClosedLoopRunner requires SimulatorRuntime")
            sim = simulator_runtime
            history = avstack.history
            if (history.duration_us or history.samples > 1) and not sim.can_observe_past:
                raise NotImplementedError(
                    f"{type(sim).__name__} cannot provide recorded acquisition history "
                    f"observations required by {type(avstack).__name__}"
                )
            validate_environment(request.environment, simulator_runtime.simulator)
            hooks = request.hooks or HookSet()
            sensor_environment = SensorEnvironment(request.environment, seed=hooks.seed or 0)

            def record_attack_output(context: OnlineContext, name: str, value: Any) -> None:
                # Attack outputs join the hook outputs, so both are logged with their decision.
                session.outputs.append(RecordedOutput(
                    name=name, value=deepcopy(value), step=context.step, stage=context.stage,
                    frame=context.frame, t=context.sim_time,
                ))

            attack_dispatch = AttackDispatcher(
                request.attacks, observation_copy=simulator_runtime.copy_observation,
                output_recorder=record_attack_output,
            )
            # Validate before world reset or handlers acquire native resources.
            available = hooks.validate_stages(simulator_runtime, avstack)
            attack_dispatch.validate(available)
            if request.attacks:
                hooks = hooks.with_hooks(attack_dispatch.hooks)
            session = hooks.session(simulator_runtime, avstack)
            deployment = request.deployment or DeploymentPlan(hooks.setup_ticks)
            if deployment.ticks < hooks.setup_ticks:
                raise ValueError("Deployment plan is shorter than the participants require")
            recorder = recorder or Recorder(path=request.output_path,
                                            retain_steps=request.retain_steps,
                                            retain_sensor_data=request.retain_sensor_data, metadata={
                **dict(request.metadata),
                "avstack_type": type(avstack).__name__,
                "environment": deepcopy(dict(request.environment)),
                "decision_steps": request.decision_steps,
                "timing": asdict(request.timing),
                "recording_policy": {
                    "sensor_ids": list(request.recording.sensor_ids),
                    "representations": list(request.recording.representations),
                    "every_n_captures": request.recording.every_n_captures,
                    "sensor_every_n_captures": dict(request.recording.sensor_every_n_captures),
                },
            })
        except BaseException as error:
            # Ownership transfers at entry, before capability checks or recorder allocation.
            # These runtimes have not entered AttackDispatcher.setup yet.
            origin = request.start.frame if request.start is not None else None
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
        current_step = -1
        resources_released = False
        terminated_by_condition = False
        flush_capture_records: Callable[[], None] | None = None

        def release() -> None:
            nonlocal resources_released
            if release_owned_resources is not None and not resources_released:
                resources_released = True
                release_owned_resources()

        def actions() -> list[dict[str, Any]]:
            records = attack_dispatch.records()
            return [
                {"kind": kind, **item}
                for kind, values in (("request", records["requests"]),
                                     ("receipt", records["receipts"]))
                for item in values
            ]

        def drain() -> None:
            session.drain_outputs()
            attack_dispatch.drain_records()

        def cleanup_attacks() -> None:
            failure: BaseException | None = None
            try:
                attack_dispatch.stop()
            except BaseException as error:
                failure = error
            # Failed decisions and unapplied actions still have evidence.
            try:
                recorder.record_event(
                    "attack_cleanup", step=max(current_step, 0),
                    actions=actions(), outputs=session.outputs,
                )
                drain()
            except BaseException as error:
                if failure is None:
                    raise
                failure.add_note(f"Attack cleanup logging also failed: {error!r}")
            if failure is not None:
                raise failure

        def context(stage: str, observation: Observation | None = None) -> OnlineContext:
            # Cleanup reuses the latest context: a failed reset or step may invalidate the state.
            clock = sim.clock
            return OnlineContext(
                step=max(current_step, 0), stage=stage, sim_time=clock.t, frame=clock.frame,
                observation=observation, physical_state=sim.state(),
                decision_time_us=(origin_us if decision_period_us is None else
                                  origin_us + max(current_step, 0) * decision_period_us),
                capture_time_us=clock.time_us, decision_period_us=decision_period_us,
                force_gt=(False if decision_period_us is None else request.timing.is_force_gt(
                    origin_us + max(current_step, 0) * decision_period_us, origin_us, decision_period_us)),
            )

        try:
            with sim.exclusive(), ExitStack() as run_resources:
                run_resources.callback(release)
                run_resources.enter_context(session.activate())
                try:
                    # AttackRuntime.setup runs inside the reset, at the simulator.setup stage.
                    sim.configure_timing(request.timing, history=history)
                    if request.decision_steps:
                        sim.simulator.validate_startup(request.start, request.timing.force_gt_duration_us)
                    sim.reset(request.start, request.initialization,
                              deployment=deployment, world=request.environment["world"])
                    sensor_environment.bind(sim.observe())
                    initial_state = sim.state()
                    try:
                        decision_period_us = sim.decision_period_us
                    except ValueError:
                        if request.decision_steps or history.duration_us or history.samples > 1:
                            raise
                        # A zero-decision run acquires only the origin. No period is used.
                        decision_period_us = None
                    timing = request.timing
                    if decision_period_us is not None:
                        timing.validate_period(decision_period_us)
                    unknown_sensors = timing.sensors.keys() - sim.sensors
                    if unknown_sensors:
                        raise ValueError(f"Unknown timing sensors: {sorted(unknown_sensors)}")
                    if timing.decision_period_us not in (None, decision_period_us):
                        raise ValueError("timing.decision_period_us must equal the dataset time grid")
                    if (avstack.required_sensors is not None
                            and avstack.required_sensors - sim.sensors):
                        raise ValueError(f"Missing required sensors: {sorted(avstack.required_sensors - sim.sensors)}")
                    origin_us = sim.clock.time_us
                    period = decision_period_us
                    if period is None:
                        # Zero-decision runs without a declared grid acquire only the origin.
                        period = 1
                    buffer = AcquisitionBuffer(sim.sensors, period, origin_us, timing, history)
                    start_position = request.start.frame.position if request.start is not None else 0
                    past_metadata = []
                    capture_metadata: dict[str, CaptureRecord] = {}
                    capture_drafts: dict[int, tuple[CaptureRecord, Mapping[str, int]]] = {}
                    past_times, past_pose_times = buffer.past_times()

                    def environment_indices(time_us: int, sensors: Sequence[str]) -> tuple[int, dict[str, int], dict[str, int]]:
                        offset = time_us - origin_us
                        grid, remainder = divmod(offset, period)
                        indices = {sensor: buffer.index_at(sensor, time_us) for sensor in sensors}
                        seed_indices = {sensor: (0 if remainder == 0 else indices[sensor] -
                            buffer.timings[sensor].latest_index_at_or_before(grid * period, period))
                            for sensor in sensors}
                        return start_position + grid, indices, seed_indices

                    def capture_record(identity: int, time_us: int, sensors: Sequence[str], raw: Observation,
                                       truth: Any, geometry: Mapping[str, Any], visibility: Mapping[str, Any],
                                       indices: Mapping[str, int], ranges: Mapping[str, range]) -> None:
                        draft = CaptureRecord(identity, time_us, tuple(sensors), raw,
                            physical_state=truth, sensor_geometry=geometry, object_visibility=visibility,
                            frame_ranges_us={sensor: (span.start, span.stop) for sensor, span in ranges.items()})
                        capture_drafts[identity] = draft, dict(indices)
                        for sensor in sensors:
                            capture_metadata[sensor] = replace(draft, raw_observation=None,
                                sensor_geometry={sensor: geometry[sensor]} if sensor in geometry else {})

                    def finalize_captures(observation: Observation | None = None) -> None:
                        delivered: dict[int, dict[str, Any]] = {}
                        decision_time = None if observation is None else observation.estimated_states()[-1].time_us
                        if observation is not None:
                            for sensor in observation.deliveries:
                                if sensor not in observation.sensor_data:
                                    continue
                                for sample in observation.delivered(sensor):
                                    if sample.capture_id is not None and sample.capture_id in capture_drafts:
                                        delivered.setdefault(sample.capture_id, {})[sensor] = sample
                        for identity, (draft, indices) in tuple(capture_drafts.items()):
                            model = None
                            samples = delivered.get(identity, {})
                            if samples:
                                sample = next(iter(samples.values()))
                                model = replace(draft.raw_observation,
                                    t=sample.t, frame=sample.frame,
                                    sensor_data={sensor: item.data for sensor, item in samples.items()},
                                    calibration={sensor: item.calibration for sensor, item in samples.items()
                                                 if item.calibration is not None},
                                    vehicle_state=sample.state.vehicle_state, ego_speed=sample.state.ego_speed,
                                    deliveries={}, state_estimates=())
                                if observation is not None and draft.time_us == decision_time:
                                    model.route = observation.route
                                model = request.recording.observation(model, identity, "model_input",
                                    capture_indices=indices, available_sensors=sim.sensors, check_available=False)
                            retained = set(draft.raw_observation.sensor_data)
                            if model is not None:
                                retained.update(model.sensor_data)
                            geometry = {sensor: value for sensor, value in draft.sensor_geometry.items()
                                        if sensor in retained}
                            del capture_drafts[identity]
                            record = recorder.record_capture(replace(draft,
                                model_observation=model, sensor_geometry=geometry))
                            for sensor in draft.sensor_ids:
                                if capture_metadata[sensor].capture_id == identity:
                                    capture_metadata[sensor] = replace(record,
                                        raw_observation=None, model_observation=None,
                                        sensor_geometry=({sensor: geometry[sensor]} if sensor in geometry else {}))

                    flush_capture_records = finalize_captures

                    for index, (offset, sensors) in enumerate(past_times.items()):
                        previous = sim.observe_recorded_at(offset, sensors)
                        grid, indices, seed_indices = environment_indices(origin_us + offset, sensors)
                        previous = sensor_environment.apply(previous, grid,
                            interval_capture_index=seed_indices, available_sensors=sim.sensors)
                        identity = index - len(past_times)
                        recorder.begin_capture(identity)
                        ranges = {sensor: sim.frame_range_us(sensor, origin_us + offset) for sensor in sensors}
                        buffer.append(previous, origin_us + offset, identity, sensors,
                            frame_ranges=ranges)
                        if offset in past_pose_times:
                            buffer.append_state(previous, origin_us + offset)
                        truth = sim.recorded_capture_state
                        geometry = {sensor: sim.recorded_sensor_geometry_at(offset, sensor)
                                    for sensor in sensors if sensor in sim.camera_sensors}
                        past_raw = recorder.snapshot(request.recording.observation(
                            previous, identity, "raw", capture_indices=indices, available_sensors=sim.sensors,
                            check_available=False))
                        capture_record(identity, origin_us + offset, sensors, past_raw, truth,
                                       geometry, sim.recorded_capture_visibility, indices, ranges)
                        past_metadata.append({"t": previous.t,
                            "vehicle_state": recorder.snapshot(previous.vehicle_state)})
                    if past_times:
                        del previous
                    recorder.begin_capture(0)
                    recorder.record_event(
                        "initialization", physical_state=initial_state, clock=sim.clock,
                        setup_ticks=deployment.ticks, decision_period_us=decision_period_us,
                        initialization_report=simulator_runtime.initialization_report,
                        outputs=session.outputs, actions=actions(),
                        object_visibility={identity: sim.object_visibility(identity) for identity in initial_state.objects},
                        past_captures=len(past_metadata), past_observations=past_metadata)
                    drain()
                    events = EventQueue(origin_us)
                    events.schedule(origin_us, ("frame", tuple(sorted(sim.sensors))), priority=10)
                    events.schedule(origin_us, ("pose", ()), priority=15)
                    events.schedule(origin_us, ("decision", ()), priority=20)
                    if request.decision_steps:
                        events.schedule(origin_us, ("advance", ()), priority=40)
                    pending: StepRecord | None = None
                    command_buffer = DelayBuffer(timing.planner_delay_us)
                    capture_id = 0
                    active_time = None
                    time_capture_id: int | None = None
                    latest_capture_ids: dict[str, int | None] = {}
                    expected_delivery_indices: dict[str, int] = {}
                    raw = captured = localized = None
                    pre_state = initial_state
                    executed = None
                    force_gt = False
                    terminal = False

                    def begin_time(time_us: int, sensors: Sequence[str]) -> None:
                        nonlocal active_time, raw, captured, localized, pre_state, pending
                        nonlocal current_step, terminated_by_condition, terminal, time_capture_id
                        if active_time == time_us:
                            return
                        active_time = time_us
                        time_capture_id = None
                        raw = sim.capture_at(time_us, sensors)
                        captured = localized = None
                        pre_state = sim.state()
                        recorder.begin_capture(capture_id)
                        offset = time_us - origin_us
                        decision, remainder = divmod(offset, period)
                        if remainder == 0 and pending is not None:
                            if request.attacks:
                                attack_dispatch.record_end(context("step.end", raw))
                            feedback_error = None
                            try:
                                session.after_step(raw)
                            except BaseException as error:
                                feedback_error = error
                            try:
                                record = recorder.record_step(replace(pending,
                                    physical_post=recorder.snapshot(pre_state), actions=actions(), outputs=session.outputs))
                            except BaseException as error:
                                if feedback_error is not None:
                                    feedback_error.add_note(f"Completed decision logging also failed: {error!r}")
                                    raise feedback_error
                                raise
                            recorder.release_captures({identity for ids in pending.delivered_captures.values()
                                                       for identity in ids if identity is not None})
                            drain()
                            pending = None
                            if feedback_error is not None:
                                raise feedback_error
                            if request.terminate_when is not None and request.terminate_when(record):
                                terminated_by_condition = True
                        terminal = (decision >= request.decision_steps and request.decision_steps > 0) or terminated_by_condition
                        if remainder == 0 and not terminal:
                            current_step = decision if request.decision_steps else -1
                            session.step, session.control = decision, None
                            if request.attacks and request.decision_steps:
                                attack_dispatch.step(context("step", raw))
                                attack_dispatch.discard_missing_sensor_actions(raw, sim)

                    for event in events:
                        kind, sensors = event.value
                        begin_time(event.time_us, sensors)
                        assert raw is not None
                        offset = event.time_us - origin_us
                        decision, remainder = divmod(offset, period)
                        if kind == "frame":
                            recorder.begin_capture(capture_id)
                            time_capture_id = capture_id
                            ranges = {sensor: sim.frame_range_us(sensor, event.time_us) for sensor in sensors}
                            geometry = {sensor: sim.sensor_geometry(sensor)
                                        for sensor in sensors if sensor in sim.camera_sensors}
                            visibility = recorder.snapshot({identity: sim.object_visibility(identity)
                                                            for identity in pre_state.objects})
                            grid, indices, seed_indices = environment_indices(event.time_us, sensors)
                            expected_delivery_indices.update(indices)
                            raw_record = recorder.snapshot(request.recording.observation(
                                raw, capture_id, "raw", capture_indices=indices, available_sensors=sim.sensors,
                                check_available=False))
                            capture_record(capture_id, event.time_us, sensors, raw_record,
                                           pre_state, geometry, visibility, indices, ranges)
                            if terminal:
                                finalize_captures()
                                capture_id += 1
                                continue
                            attacked = attack_dispatch.apply_sensor_actions(raw, sim) if request.attacks and request.decision_steps else raw
                            conditioned = sensor_environment.apply(attacked, grid,
                                interval_capture_index=seed_indices, available_sensors=sim.sensors)
                            captured = session.observation_pipeline.capture(conditioned, session)
                            buffer.append(captured, event.time_us, capture_id, sensors,
                                frame_ranges=ranges)
                            capture_id += 1
                        elif kind == "pose":
                            if terminal and time_capture_id is None:
                                recorder.begin_capture(capture_id)
                                capture_record(capture_id, event.time_us, (), raw, pre_state, {}, {}, {}, {})
                                finalize_captures()
                                capture_id += 1
                            if not terminal:
                                localized = session.observation_pipeline.localize(
                                    captured if captured is not None else sim.copy_observation(raw), session)
                                if time_capture_id is not None:
                                    buffer.update_capture(localized, event.time_us, time_capture_id)
                                buffer.append_state(localized, event.time_us)
                        elif kind == "decision":
                            if terminal:
                                break
                            assert localized is not None
                            seen = buffer.take_ready(localized, event.time_us)
                            delivered_indices = dict(expected_delivery_indices)
                            expected_delivery_indices.clear()
                            seen = session.observation_pipeline.decision(seen, session)
                            seen.deliveries = {sensor: values if sensor in seen.sensor_data else ()
                                               for sensor, values in seen.deliveries.items()}
                            decision_raw_record = recorder.snapshot(request.recording.observation(
                                raw, decision, "raw", capture_indices=(indices if time_capture_id is not None else {}),
                                available_sensors=sim.sensors))
                            model_record = recorder.snapshot(request.recording.observation(
                                seen, decision, "model_input", capture_indices=delivered_indices, available_sensors=sim.sensors))
                            finalize_captures(seen)
                            if decision == 0:
                                avstack.reset(seen)
                            if request.decision_steps == 0:
                                break
                            force_gt = timing.is_force_gt(event.time_us, origin_us, period)
                            skip_driver = force_gt and timing.skip_driver_during_force_gt
                            if skip_driver:
                                avstack.ingest(seen)
                                produced = command = None
                                detections = None
                            else:
                                produced = avstack(seen)
                                detections = recorder.snapshot(avstack.detections())
                                produced_record = recorder.snapshot(produced)
                                command = session.command(produced)
                            if skip_driver:
                                produced_record = None
                            buffered = (sim.recorded_reference(event.time_us, event.time_us + period)
                                        if force_gt and "trajectory" in (sim.command_modes or ()) else command)
                            command_buffer.add(buffered, event.time_us)
                            production_time, executed = command_buffer.item_at(event.time_us)
                            if force_gt and "trajectory" not in (sim.command_modes or ()):
                                production_time, executed = None, None
                            attack_dispatch.finish_decision(skip_driver=skip_driver,
                                every_sensor_due=sim.sensors <= seen.deliveries.keys())
                            metadata = capture_metadata
                            retained_geometry = {sensor: record.sensor_geometry[sensor]
                                for sensor in request.recording.sensor_ids
                                if sensor in metadata and sensor in (record := metadata[sensor]).sensor_geometry
                                }
                            detection_sensor = avstack.detection_sensor
                            for sensor, values in seen.deliveries.items():
                                latest_capture_ids[sensor] = values[-1].capture_id if values else None
                            pending = StepRecord(step=decision,
                                raw_observation=decision_raw_record,
                                model_observation=model_record, physical_pre=recorder.snapshot(pre_state),
                                produced_command=produced_record, executed_command=recorder.snapshot(executed),
                                executed_command_issue_time_us=production_time, force_gt=force_gt,
                                delivered_captures={sensor: tuple(item.capture_id for item in values)
                                                    for sensor, values in seen.deliveries.items()},
                                components={} if skip_driver else recorder.snapshot(avstack.component_log() or {}), detections=detections,
                                sensor_geometry=retained_geometry,
                                physical_captures={sensor: record.capture_id for sensor, record in metadata.items()},
                                detection_capture_id=(latest_capture_ids.get(detection_sensor)
                                    if detection_sensor is not None else None), detection_sensor=detection_sensor,
                                object_visibility={identity: {sensor: record.object_visibility[identity][sensor]
                                    for sensor, record in sorted(metadata.items())
                                    if identity in record.object_visibility and sensor in record.object_visibility[identity]}
                                    for identity in dict.fromkeys(identity for record in metadata.values()
                                        for identity in record.object_visibility)},
                                telemetry={"sensor_payloads": {
                                    "raw": {key: "recorded" if key in decision_raw_record.sensor_data else "not_recorded" for key in raw.sensor_data},
                                    "model_input": {key: "recorded" if key in model_record.sensor_data else "not_recorded" for key in seen.sensor_data}}})
                        elif kind == "advance":
                            end = event.time_us + period
                            frame_events = buffer.due_between(event.time_us, end)
                            for timestamp, due in frame_events.items():
                                events.schedule(timestamp, ("frame", due), priority=10)
                            for timestamp in timing.reporting_timestamps(event.time_us, end):
                                events.schedule(timestamp, ("pose", ()), priority=15)
                            events.schedule(end, ("pose", ()), priority=15)
                            events.schedule(end, ("decision", ()), priority=20)
                            if decision + 1 < request.decision_steps:
                                events.schedule(end, ("advance", ()), priority=40)
                            session.control = executed
                            sim.advance_interval(end, executed,
                                report_times_us=timing.reporting_timestamps(event.time_us, end),
                                force_gt=force_gt, coerce_dynamic_state=force_gt or timing.is_handoff(event.time_us, origin_us))
                except BaseException as error:
                    if attack_dispatch.started:
                        try:
                            cleanup_attacks()
                        except BaseException as cleanup:
                            error.add_note(f"Attack cleanup also failed: {cleanup!r}")
                    raise
                finally:
                    if attack_dispatch.started:
                        cleanup_attacks()
            release()
            termination_details = dict(sim.termination_details)
            if terminated_by_condition:
                termination_details["terminated_by_condition"] = True
            if flush_capture_records is not None:
                flush_capture_records()
            return recorder.finish(status="completed", **termination_details)
        except BaseException as error:
            if flush_capture_records is not None:
                try:
                    flush_capture_records()
                except BaseException as cleanup:
                    error.add_note(f"Capture finalization also failed: {cleanup!r}")
            try:
                release()
            except BaseException as cleanup:
                error.add_note(f"Owned resource cleanup also failed: {cleanup!r}")
            try:
                recorder.record_event(
                    "failure", step=current_step if current_step >= 0 else None,
                    error_type=type(error).__name__, message=str(error),
                    initialization_report=getattr(error, "report", {}),
                    cleanup_errors=list(getattr(error, "__notes__", ())),
                    actions=actions(), outputs=session.outputs,
                )
                drain()
                recorder.finish(
                    status="cancelled" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed",
                    incomplete_step=current_step if current_step >= 0 else None,
                    **sim.termination_details,
                )
            except BaseException as cleanup:
                error.add_note(f"Log finalization also failed: {cleanup!r}")
                try:
                    recorder.abort()
                except BaseException as abort_error:
                    error.add_note(f"Log resource cleanup also failed: {abort_error!r}")
            # Preserve the exception identity while making its in-memory partial log inspectable.
            try:
                setattr(error, "run_log", recorder.log)
            except (AttributeError, TypeError):
                pass
            raise


__all__ = [
    'RecordingPolicy',
    'RunRequest',
    'ClosedLoopRunner',
]
