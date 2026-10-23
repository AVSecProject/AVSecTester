"""One closed-loop scheduler with independent RunLog captures."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from contextlib import ExitStack
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from avsectester.validation import require_nonnegative_integer, require_positive_integer
from avsectester.closed_loop.environment import SensorEnvironment, normalize_environment, validate_environment
from avsectester.closed_loop.avstacks.base import AVStack
from avsectester.data_types import Observation, RecordedOutput
from avsectester.runlog import Recorder, RunLog, StepRecord
from avsectester.closed_loop.hooks import HookSet
from avsectester.closed_loop.simulators.base import DeploymentPlan, SimulatorRuntime


@dataclass(frozen=True)
class RecordingPolicy:
    """Choose sensor payloads without sampling commands or physical state."""

    sensor_ids: tuple[str, ...] = ()
    representations: tuple[str, ...] = ("model_input",)
    every_n_steps: int = 1

    def __post_init__(self) -> None:
        require_positive_integer(self.every_n_steps, "every_n_steps")
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
    start: Any = None
    initialization: Any = None
    deployment: Any = None
    environment: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        require_nonnegative_integer(self.decision_steps, "decision_steps")
        object.__setattr__(self, "environment", normalize_environment(self.environment))
        if not self.retain_steps and self.output_path is None:
            raise ValueError("retain_steps=False requires an output_path")


class ClosedLoopRunner:
    """Schedule one driving loop; never close caller-owned simulator/driving objects."""

    def run(
        self,
        simulator_runtime: SimulatorRuntime,
        avstack: AVStack,
        request: RunRequest,
        *,
        recorder: Recorder | None = None,
        release_owned_resources: Callable[[], None] | None = None,
    ) -> RunLog:
        from avsectester.attack.runtime import OnlineContext
        from avsectester.execution.attack_dispatch import AttackDispatcher

        try:
            if not isinstance(simulator_runtime, SimulatorRuntime):
                raise TypeError("ClosedLoopRunner requires SimulatorRuntime")
            sim = simulator_runtime
            past_steps = avstack.past_steps
            require_nonnegative_integer(past_steps, "past_steps")
            if past_steps and not sim.can_observe_past:
                raise NotImplementedError(
                    f"{type(sim).__name__} cannot provide the {past_steps} recorded past "
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
                                            retain_steps=request.retain_steps, metadata={
                **dict(request.metadata),
                "avstack_type": type(avstack).__name__,
                "environment": deepcopy(dict(request.environment)),
                "decision_steps": request.decision_steps,
                "recording_policy": {
                    "sensor_ids": list(request.recording.sensor_ids),
                    "representations": list(request.recording.representations),
                    "every_n_steps": request.recording.every_n_steps,
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
        last_observation: Observation | None = None
        resources_released = False

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
            )

        try:
            with sim.exclusive(), ExitStack() as run_resources:
                run_resources.callback(release)
                run_resources.enter_context(session.activate())
                try:
                    # AttackRuntime.setup runs inside the reset, at the simulator.setup stage.
                    sim.reset(request.start, request.initialization,
                              deployment=deployment, world=request.environment["world"])
                    last_observation = sim.observe()
                    initial_state = sim.state()
                    start_position = request.start.frame.position if request.start is not None else 0
                    past = tuple(
                        sensor_environment.apply(observation, start_position - past_steps + index)
                        for index, observation in enumerate(sim.observe_past(past_steps))
                    ) if past_steps else ()
                    recorder.record_event(
                        "initialization", physical_state=initial_state,
                        clock=sim.clock, setup_ticks=deployment.ticks,
                        initialization_report=simulator_runtime.initialization_report,
                        outputs=session.outputs, actions=actions(),
                        object_visibility={object_id: sim.object_visibility(object_id)
                                           for object_id in initial_state.objects},
                        past_steps=len(past),
                        past_observations=[{"t": observation.t,
                                            "vehicle_state": recorder.snapshot(observation.vehicle_state)}
                                           for observation in past],
                    )
                    drain()
                    if request.decision_steps == 0:
                        conditioned = sensor_environment.apply(last_observation, start_position)
                        avstack.reset(session.observe(conditioned), past=past)
                    for current_step in range(request.decision_steps):
                        session.step = current_step
                        raw = sim.observe()
                        last_observation = raw
                        pre_state = sim.state()
                        object_visibility = recorder.snapshot({
                            object_id: sim.object_visibility(object_id)
                            for object_id in pre_state.objects
                        })
                        raw_record = recorder.snapshot(request.recording.observation(raw, current_step, "raw"))
                        pre = recorder.snapshot(pre_state)
                        sensor_geometry = {}
                        retained_sensors = (request.recording.sensor_ids
                            if current_step % request.recording.every_n_steps == 0
                            and request.recording.representations else ())
                        for sensor in retained_sensors:
                            if sensor in sim.camera_sensors:
                                sensor_geometry[sensor] = recorder.snapshot(sim.sensor_geometry(sensor))
                        attacked = raw
                        if request.attacks:
                            attack_dispatch.step(context("step", raw))
                            attacked = attack_dispatch.apply_sensor_actions(raw, sim)
                        position = (request.start.frame.position if request.start is not None else 0) + current_step
                        conditioned = sensor_environment.apply(attacked, position)
                        seen = session.observe(conditioned)
                        model_record = recorder.snapshot(
                            request.recording.observation(seen, current_step, "model_input")
                        )
                        if current_step == 0:
                            avstack.reset(seen, past=past)
                        produced = avstack(seen)
                        detections = recorder.snapshot(avstack.detections())
                        produced_record = recorder.snapshot(produced)
                        command = session.command(produced)
                        executed_record = recorder.snapshot(command)
                        attack_dispatch.finish_decision()
                        components = recorder.snapshot(avstack.component_log() or {})
                        telemetry: dict[str, Any] = {
                            "sensor_payloads": {
                                "raw": {k: "recorded" if k in raw_record.sensor_data else "not_recorded"
                                        for k in raw.sensor_data},
                                "model_input": {
                                    k: "recorded" if k in model_record.sensor_data else "not_recorded"
                                    for k in seen.sensor_data
                                },
                            },
                        }
                        sim.step(command)
                        post = recorder.snapshot(sim.state())
                        last_observation = sim.observe()
                        if request.attacks:
                            attack_dispatch.record_end(context("step.end", last_observation))
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
                                components=components, detections=detections, sensor_geometry=sensor_geometry,
                                object_visibility=object_visibility,
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
            return recorder.finish(status="completed", **sim.termination_details)
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
