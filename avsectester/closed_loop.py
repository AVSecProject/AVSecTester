"""One closed-loop scheduler, with RunLog as the source of compatibility traces."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from .backend import AVStack, WorldBackend
from .plane import Control, FrameRecord, Observation, Trace
from .runlog import Recorder, RunLog, StepRecord
from .runtime import HookSet
from .simulator_runtime import SimulatorRuntime


@dataclass(frozen=True)
class RecordingPolicy:
    """Choose sensor payloads without sampling commands or physical state."""

    sensor_ids: tuple[str, ...] = ()
    representations: tuple[str, ...] = ("model_input",)
    every_n_steps: int = 1

    def __post_init__(self) -> None:
        if (not isinstance(self.every_n_steps, int) or isinstance(self.every_n_steps, bool)
                or self.every_n_steps < 1):
            raise ValueError("every_n_steps must be a positive integer")
        if not set(self.representations) <= {"raw", "model_input"}:
            raise ValueError("representations must contain raw and/or model_input")
        if len(set(self.sensor_ids)) != len(self.sensor_ids):
            raise ValueError("sensor_ids must be unique")

    def observation(self, value: Observation, step: int, representation: str) -> Observation:
        retained = (
            set(self.sensor_ids)
            if representation in self.representations and step % self.every_n_steps == 0
            else set()
        )
        if not retained:
            # Legacy adapters may expose an unnamed native sensor container.
            return replace(value, sensor_data={})
        if not isinstance(value.sensor_data, Mapping):
            raise ValueError("Named sensor recording requires an adapter sensor_data mapping")
        unknown = retained - value.sensor_data.keys()
        if unknown:
            raise ValueError(f"Recording requested unavailable sensors: {sorted(unknown)}")
        return replace(value, sensor_data={
            name: payload for name, payload in value.sensor_data.items() if name in retained
        })


@dataclass(frozen=True)
class RunRequest:
    """One caller-owned simulator/driving-system run, independent of experiment orchestration."""

    decision_steps: int
    hooks: HookSet | None = None
    attacks: Sequence[Any] = ()
    recording: RecordingPolicy = field(default_factory=RecordingPolicy)
    output_path: str | Path | None = None
    retain_steps: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)
    source: Any = None
    initialization: Any = None
    deployment: Any = None
    environment: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (not isinstance(self.decision_steps, int) or isinstance(self.decision_steps, bool)
                or self.decision_steps < 0):
            raise ValueError("decision_steps must be a nonnegative integer")
        if not self.retain_steps and self.output_path is None:
            raise ValueError("retain_steps=False requires an output_path")


def _field(value: Any, name: str, default: Any = None) -> Any:
    return value.get(name, default) if isinstance(value, Mapping) else getattr(value, name, default)


def trace_from_log(log: RunLog) -> Trace:
    """Build the legacy scalar view from captured post-state and executed commands."""
    records = []
    outputs = [output for event in log.events if event.kind == "initialization"
               for output in event.data.get("outputs", [])]
    for record in log.steps:
        state, command = record.physical_post, record.executed_command
        telemetry = record.telemetry
        records.append(FrameRecord(
            frame=record.step, t=_field(state, "t", 0.0),
            speed=_field(state, "ego_speed", 0.0),
            throttle=_field(command, "throttle", 0.0),
            brake=_field(command, "brake", 0.0), steer=_field(command, "steer", 0.0),
            n_detections=telemetry.get("n_detections", 0),
        ))
        outputs.extend(record.outputs)
    trace = Trace(records=records, outputs=outputs)
    trace.run_log = log
    return trace


class ClosedLoopRunner:
    """Schedule one driving loop; never close caller-owned simulator/driving objects."""

    def run(
        self,
        simulator: SimulatorRuntime | WorldBackend,
        driving_system: AVStack,
        request: RunRequest,
        *,
        recorder: Recorder | None = None,
        perturb: Callable[[Observation], Observation] | None = None,
        on_step: Callable[[int, Observation, Control], None] | None = None,
        record_views: Mapping[str, Callable[[Observation], Any]] | None = None,
        view_sink: Callable[[int, str, Any], Any] | None = None,
        release_owned_resources: Callable[[], None] | None = None,
    ) -> RunLog:
        from .attack_runtime import AttackContext, AttackDispatcher
        from .log_codecs import default_codecs

        sim = simulator if isinstance(simulator, SimulatorRuntime) else SimulatorRuntime(simulator)
        backend = sim.backend
        hooks = request.hooks or HookSet()
        attack_dispatch = AttackDispatcher(
            request.attacks, observation_copy=backend.copy_observation,
        )
        # Validate before world reset or handlers acquire native resources.
        available = hooks.validate_stages(backend, driving_system)
        attack_dispatch.validate(available)
        if request.attacks:
            hooks = hooks.with_hooks(attack_dispatch.hooks)
        session = hooks.session(backend, driving_system, perturb)
        recorder = recorder or Recorder(path=request.output_path, codecs=default_codecs(),
                                        retain_steps=request.retain_steps, metadata={
            **dict(request.metadata),
            "decision_steps": request.decision_steps,
            "recording_policy": {
                "sensor_ids": list(request.recording.sensor_ids),
                "representations": list(request.recording.representations),
                "every_n_steps": request.recording.every_n_steps,
            },
        })
        started = False
        current_step = -1
        last_observation: Observation | None = None
        last_attack_context: AttackContext | None = None
        resources_released = False

        def release() -> None:
            nonlocal resources_released
            if release_owned_resources is not None and not resources_released:
                resources_released = True
                release_owned_resources()

        def actions() -> list[dict[str, Any]]:
            return [
                {"kind": kind, **item}
                for kind, values in (("request", attack_dispatch.requests),
                                     ("receipt", attack_dispatch.receipts))
                for item in values
            ]

        def drain() -> None:
            session.drain_outputs()
            attack_dispatch.drain_records()

        def cleanup_attacks() -> None:
            nonlocal started
            started = False
            failure: BaseException | None = None
            try:
                attack_dispatch.stop(stop_context())
            except BaseException as error:
                failure = error
            # Failed decisions and unconsumed start actions still have evidence.
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

        def context(stage: str, observation: Observation | None = None) -> AttackContext:
            nonlocal last_attack_context
            clock = sim.clock
            last_attack_context = AttackContext(
                step=max(current_step, 0), stage=stage, sim_time=clock.t, frame=clock.frame,
                observation=observation, physical_state=sim.state(),
            )
            return last_attack_context

        def stop_context() -> AttackContext:
            # Failed reset/step may invalidate simulator state. Cleanup must not query it.
            assert last_attack_context is not None
            return replace(last_attack_context, stage="stop")

        def start_attacks() -> None:
            nonlocal started
            started = True  # stop also handles a partially failed start
            attack_dispatch.start(context("start"))

        try:
            with sim.exclusive(), session.activate():
                try:
                    sim.reset(request.source, request.initialization,
                              deployment=request.deployment, environment=request.environment)
                    last_observation = sim.observe()
                    if request.attacks:
                        start_attacks()
                    recorder.record_event(
                        "initialization", physical_state=sim.state(),
                        clock=sim.clock, setup_ticks=backend.setup_ticks,
                        source_initialization=getattr(backend, "initialization_report", {}),
                        outputs=session.outputs, actions=actions(),
                    )
                    drain()
                    if request.decision_steps == 0:
                        # Historical callers still receive the initial raw reset input.
                        driving_system.reset(backend.copy_observation(last_observation))
                    for current_step in range(request.decision_steps):
                        session.step = current_step
                        raw = sim.observe()
                        last_observation = raw
                        raw_record = recorder.capture(request.recording.observation(raw, current_step, "raw"))
                        pre = recorder.capture(sim.state())
                        if request.attacks:
                            attack_dispatch.step(context("step", raw))
                        seen = session.observe(raw)
                        model_record = recorder.capture(
                            request.recording.observation(seen, current_step, "model_input")
                        )
                        if current_step == 0:
                            driving_system.reset(seen)
                        produced = driving_system(seen)
                        produced_record = recorder.capture(produced)
                        control = session.command(produced)
                        executed_record = recorder.capture(control)
                        attack_dispatch.finish_decision()
                        components = {}
                        component_log = getattr(driving_system, "component_log", None)
                        if callable(component_log):
                            components = recorder.capture(component_log() or {})
                        telemetry: dict[str, Any] = {
                            "sensor_payloads": {
                                "raw": {k: "recorded" if k in raw_record.sensor_data else "not_recorded"
                                        for k in (raw.sensor_data if isinstance(raw.sensor_data, Mapping)
                                                  else ("unnamed",))},
                                "model_input": {
                                    k: "recorded" if k in model_record.sensor_data else "not_recorded"
                                    for k in (seen.sensor_data if isinstance(seen.sensor_data, Mapping)
                                              else ("unnamed",))
                                },
                            },
                        }
                        if "perception" in components:
                            from .evaluation.component_log import _count

                            telemetry["n_detections"] = _count(components["perception"])
                        if record_views:
                            telemetry["views"] = {}
                            for name, view in record_views.items():
                                captured_view = recorder.capture(view(backend.copy_observation(seen)))
                                telemetry["views"][name] = (
                                    view_sink(current_step, name, captured_view)
                                    if view_sink is not None else captured_view
                                )
                        if on_step is not None:
                            on_step(current_step, backend.copy_observation(seen),
                                    recorder.capture(control))
                        sim.step(control)
                        post = recorder.capture(sim.state())
                        last_observation = sim.observe()
                        if request.attacks:
                            context("step.end", last_observation)
                        feedback_error: BaseException | None = None
                        try:
                            session.after_step(last_observation)
                        except BaseException as error:
                            feedback_error = error
                        try:
                            recorder.record_step(StepRecord(
                                step=current_step, raw_observation=raw_record,
                                model_observation=model_record, physical_pre=pre, physical_post=post,
                                produced_command=produced_record, executed_command=executed_record,
                                components=components,
                                actions=actions(), outputs=session.outputs,
                                telemetry=telemetry,
                            ))
                        except BaseException as error:
                            if feedback_error is None:
                                raise
                            feedback_error.add_note(f"Completed decision logging also failed: {error!r}")
                            raise feedback_error
                        drain()
                        if feedback_error is not None:
                            raise feedback_error
                except BaseException as error:
                    if started:
                        try:
                            cleanup_attacks()
                        except BaseException as cleanup:
                            error.add_note(f"Attack cleanup also failed: {cleanup!r}")
                    raise
                finally:
                    if started:
                        cleanup_attacks()
            release()
            return recorder.finish(status="completed")
        except BaseException as error:
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


def run_trace(
    backend: WorldBackend,
    stack: AVStack,
    frames: int,
    perturb: Callable[[Observation], Observation] | None = None,
    on_step: Callable[[int, Observation, Control], None] | None = None,
    *,
    runtime: HookSet | None = None,
) -> Trace:
    """Compatibility entry point backed by the single RunLog capture."""
    if not isinstance(frames, int) or isinstance(frames, bool) or frames < 0:
        raise ValueError("frames must be a nonnegative integer")
    log = ClosedLoopRunner().run(
        backend, stack, RunRequest(frames, hooks=runtime), perturb=perturb, on_step=on_step,
    )
    return trace_from_log(log)
