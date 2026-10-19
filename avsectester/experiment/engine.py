"""Prepare each attack once, then run independent treatments at its selected starts."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from avsectester.attack import AttackSegment, OfflineContext
from avsectester.datasets import FrameData, ScenarioStart
from avsectester.datasets.base import nonnegative
from avsectester.attack.runtime import AttackRuntime
from avsectester.closed_loop.avstack import AVStack
from avsectester.closed_loop.environment import merge_environments, validate_environment
from avsectester.closed_loop.frame_data import FrameDataCache
from avsectester.closed_loop.hooks import HookSet
from avsectester.closed_loop.runner import ClosedLoopRunner, RunRequest
from avsectester.closed_loop.services import ClosedLoopServices, ensure_fresh
from avsectester.closed_loop.simulator import DeploymentPlan, SimulatorRuntime
from avsectester.runlog import Recorder, RunLog

from .config import ExperimentConfiguration, TreatmentConfiguration, load_configuration
from .models import ExperimentResult, PreparedExperiment, SelectionDecision, TreatmentResult


def _hooks(spec: Any, seed: int) -> HookSet:
    value = spec(seed) if callable(spec) else spec
    if value is None:
        return HookSet(seed=seed)
    if not isinstance(value, HookSet):
        raise TypeError("A hook factory must return HookSet")
    # The experiment seed controls hooks and sensor environments together. Copy only the
    # configuration shell: caller-supplied handlers retain their documented reset lifecycle.
    result = value.with_hooks(())
    result.seed = seed
    return result


def validate_configuration_environment(config: ExperimentConfiguration) -> None:
    """Check every enabled environment before preparation can create a world."""
    validate_environment(config.environment, config.simulator)
    for name, attack in config.attacks.items():
        if not isinstance(attack.environment, Mapping):
            raise TypeError(f"Attack {name!r} environment must be a mapping")
        validate_environment(merge_environments(config.environment, attack.environment), config.simulator)
    for treatment in config.treatments:
        attack = config.attacks[treatment.attack_name] if treatment.attack_name is not None else None
        validate_environment(merge_environments(
            config.environment, attack.environment if attack is not None else {}, treatment.environment,
        ), config.simulator)


def _segments(attack: Any, view: Any, config: ExperimentConfiguration) -> tuple[AttackSegment, ...]:
    selected = tuple(attack.build_metadata(view))
    for segment in selected:
        if not isinstance(segment, AttackSegment):
            raise TypeError("build_metadata must return AttackSegment records")
        if segment.scenario_id != view.scenario_id:
            raise ValueError("Selected segment belongs to another scenario")
        config.dataset.checked_frame(segment.scenario_id, segment.start)
        config.dataset.checked_frame(segment.scenario_id, segment.stop)
        config.dataset.decision_window(segment.scenario_id, segment.start, segment.decision_steps)
        if segment.roles:
            available = view.frame_data(segment.start).actors
            missing = {identity for identities in segment.roles.values() for identity in identities} - available.keys()
            if missing:
                raise ValueError(f"Selected roles contain actors absent at the start: {sorted(missing)}")
    return selected


class Experiment:
    """The fixed build_metadata → run_offline → online experiment workflow.

    Configuration loading creates descriptions and factories, not a driving run. Preparation
    owns its services and closes them. Execution creates and closes fresh simulator and AVStack
    instances for every named treatment, including clean.
    """

    @staticmethod
    def prepare(config: ExperimentConfiguration | Mapping[str, Any] | str | Path) -> PreparedExperiment:
        configuration = load_configuration(config)
        if not configuration.attacks:
            raise ValueError("Experiment.prepare requires an attack to select segments; "
                             "use ClosedLoopRunner for an individual clean run")
        validate_configuration_environment(configuration)
        metadata: dict[str, tuple[AttackSegment, ...]] = {}
        artifacts: dict[str, Any] = {}
        selections: list[SelectionDecision] = []
        cache = FrameDataCache()
        seed = configuration.seeds[0]
        for name, attack in configuration.attacks.items():
            selected: list[AttackSegment] = []
            for scenario in configuration.dataset.scenarios():
                view = None
                try:
                    # A builder may open sessions itself. Closing services closes even sessions
                    # a user forgot to close, before advancing to the next scenario.
                    with ClosedLoopServices(
                        configuration.dataset, configuration.simulator,
                        configuration.avstack, cache,
                    ) as services:
                        view = services.scenario(scenario.id, seed=seed)
                        values = _segments(attack, view, configuration)
                    selected.extend(values)
                    selections.append(SelectionDecision(
                        name, scenario.id, "selected" if values else "not_selected",
                        tuple(view.reasons),
                    ))
                except Exception as error:
                    selections.append(SelectionDecision(
                        name, scenario.id, "error", tuple(view.reasons) if view is not None else (),
                        f"{type(error).__name__}: {error}",
                    ))
                    if configuration.strict:
                        raise
            metadata[name] = tuple(selected)
            with ClosedLoopServices(
                configuration.dataset, configuration.simulator,
                configuration.avstack, cache,
            ) as services:
                artifacts[name] = attack.run_offline(OfflineContext(services, metadata[name]))
        prepared = PreparedExperiment(configuration, metadata, artifacts, tuple(selections))
        if configuration.output is not None:
            prepared.save(configuration.output)
        return prepared

    @staticmethod
    def execute(prepared: PreparedExperiment) -> ExperimentResult:
        """Execute all comparison groups with one execution-owned frame-data cache."""
        validate_configuration_environment(prepared.configuration)
        return _Execution(prepared).run()


@dataclass(frozen=True)
class _Group:
    attack_name: str
    index: int
    segment: AttackSegment
    seed: int
    decision_steps: int


@dataclass(frozen=True)
class _Treatment:
    configuration: TreatmentConfiguration
    hooks: HookSet
    attack_runtime: AttackRuntime | None
    setup_ticks: int


def _validate_setup_ticks(value: int) -> None:
    """Reject malformed participant declarations before computing a shared window."""
    nonnegative(value, "setup_ticks")


class _Execution:
    """Resources and failure accounting for one execute call, never retained by the config."""

    def __init__(self, prepared: PreparedExperiment) -> None:
        self.prepared = prepared
        self.configuration = prepared.configuration
        self.cache = FrameDataCache()
        self.runs: list[TreatmentResult] = []

    def _description(self, group: _Group, treatment: TreatmentConfiguration,
                     deployment: DeploymentPlan | None = None) -> tuple[dict[str, Any], Path | None]:
        config = self.configuration
        attack = config.attacks[group.attack_name]
        environment = merge_environments(
            config.environment, attack.environment if treatment.attack_name is not None else {},
            treatment.environment,
        )
        destination = None
        if config.output is not None:
            destination = (config.output / "runs" / group.attack_name / f"segment-{group.index:05d}"
                           / f"seed-{group.seed}" / treatment.name)
        metadata: dict[str, Any] = {
            "attack_name": group.attack_name, "segment_index": group.index,
            "segment": asdict(group.segment), "treatment": treatment.name, "seed": group.seed,
            "environment": environment, "decision_steps": group.decision_steps,
            "selected_decision_steps": group.segment.decision_steps,
        }
        if deployment is not None:
            metadata["deployment_ticks"] = deployment.ticks
        return metadata, destination

    def _failed(self, group: _Group, treatment: TreatmentConfiguration, error: Exception,
                deployment: DeploymentPlan | None = None) -> None:
        metadata, destination = self._description(group, treatment, deployment)
        log = getattr(error, "run_log", None)
        if log is None:
            recorder = None
            try:
                recorder = Recorder(path=destination, metadata=metadata)
                recorder.record_event("failure", error_type=type(error).__name__, message=str(error))
                log = recorder.finish(status="failed")
            except Exception as storage_error:
                error.add_note(f"Failure log could not be saved: {storage_error!r}")
                if recorder is not None:
                    try:
                        recorder.abort()
                    except Exception as cleanup_error:
                        error.add_note(f"Failure log cleanup also failed: {cleanup_error!r}")
                log = RunLog(metadata=metadata, status="failed")
        self.runs.append(TreatmentResult(
            group.attack_name, group.index, group.segment, treatment.name, group.seed, log,
            f"{type(error).__name__}: {error}",
        ))
        if self.configuration.strict:
            raise error

    def _treatments(self, group: _Group) -> tuple[_Treatment, ...]:
        config = self.configuration
        result = []
        for treatment in config.treatments:
            if treatment.attack_name is not None and treatment.attack_name != group.attack_name:
                continue
            try:
                hooks = _hooks(config.clean_hooks if treatment.attack_name is None else config.hooks,
                               group.seed)
                runtime = None
                if treatment.attack_name is not None:
                    runtime = config.attacks[group.attack_name].create_runtime(
                        group.segment, self.prepared.artifacts[group.attack_name])
                    if not isinstance(runtime, AttackRuntime):
                        raise TypeError("Attack.create_runtime must return AttackRuntime")
                    ensure_fresh(runtime)
                    _validate_setup_ticks(runtime.setup_ticks)
                _validate_setup_ticks(hooks.setup_ticks)
                ticks = hooks.setup_ticks + (runtime.setup_ticks if runtime is not None else 0)
                result.append(_Treatment(treatment, hooks, runtime, ticks))
            except Exception as error:
                self._failed(group, treatment, error)
        return tuple(result)

    def _initialize_group(self, group: _Group,
                          deployment: DeploymentPlan) -> tuple[ScenarioStart, Any]:
        config = self.configuration
        segment = group.segment
        start = config.dataset.scenario_start(segment.scenario_id, segment.start)
        initialization = None
        if start.kind == "CarlaDataset":
            # Resolve this seed's actual spawns once. The cache key includes the recipe,
            # simulator, seed, base environment and the shared deployment window.
            with ClosedLoopServices(
                config.dataset, config.simulator, frame_data_cache=self.cache, deployment=deployment,
            ) as services, services.session(group.seed) as session:
                initialization = session.initialization(start, config.environment)
        if segment.roles:
            record = (initialization.expected if initialization is not None
                      else config.dataset.frame_data(segment.scenario_id, segment.start))
            if not isinstance(record, FrameData):
                raise ValueError("Selected roles require actor data at the execution start")
            required = {identity for identities in segment.roles.values() for identity in identities}
            missing = required - record.actors.keys()
            if missing:
                raise ValueError(f"Selected role actors absent at start for seed {group.seed}: "
                                 f"{sorted(missing)}")
        return start, initialization

    def _run_treatment(self, group: _Group, treatment: _Treatment,
                       start: ScenarioStart, initialization: Any, deployment: DeploymentPlan) -> None:
        config = self.configuration
        metadata, destination = self._description(group, treatment.configuration, deployment)
        try:
            with ExitStack() as resources:
                simulator_runtime = config.simulator.create_runtime(group.seed)
                if not isinstance(simulator_runtime, SimulatorRuntime):
                    raise TypeError("Simulator.create_runtime must return SimulatorRuntime")
                ensure_fresh(simulator_runtime)
                resources.callback(simulator_runtime.close)
                avstack = config.avstack.factory(group.seed)
                if not isinstance(avstack, AVStack):
                    raise TypeError("DrivingAdapter.factory must return AVStack")
                ensure_fresh(avstack)
                close = getattr(avstack, "close", None)
                if callable(close):
                    resources.callback(close)
                log = ClosedLoopRunner().run(simulator_runtime, avstack, RunRequest(
                    group.decision_steps, hooks=treatment.hooks,
                    attacks=(treatment.attack_runtime,) if treatment.attack_runtime is not None else (),
                    recording=config.recording, output_path=destination, metadata=metadata,
                    start=deepcopy(start), initialization=deepcopy(initialization),
                    deployment=deployment, environment=metadata["environment"],
                ), release_owned_resources=resources.close)
            self.runs.append(TreatmentResult(
                group.attack_name, group.index, group.segment, treatment.configuration.name,
                group.seed, log,
            ))
        except Exception as error:
            self._failed(group, treatment.configuration, error, deployment)

    def _run_group(self, group: _Group) -> None:
        treatments = self._treatments(group)
        if not treatments:
            return
        deployment = DeploymentPlan(max(t.setup_ticks for t in treatments))
        try:
            start, initialization = self._initialize_group(group, deployment)
        except Exception as error:
            for treatment in treatments:
                self._failed(group, treatment.configuration, error, deployment)
            return
        for treatment in treatments:
            self._run_treatment(group, treatment, start, initialization, deployment)

    def run(self) -> ExperimentResult:
        for attack_name, segments in self.prepared.metadata.items():
            for index, segment in enumerate(segments):
                steps = segment.decision_steps
                if self.configuration.decision_steps is not None:
                    steps = min(steps, self.configuration.decision_steps)
                for seed in self.configuration.seeds:
                    self._run_group(_Group(attack_name, index, segment, seed, steps))
        return ExperimentResult(tuple(self.runs), self.prepared.selections)
