"""Prepare each attack once, then run independent treatments at its selected starts."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import shutil
from pathlib import Path
from typing import Any

from avsectester.attack import AttackSegment, OfflineContext, OfflineReport
from avsectester.datasets import ScenarioStart
from avsectester.data_types import Unavailable
from avsectester.validation import require_nonnegative_integer
from avsectester.attack.runtime import AttackRuntime
from avsectester.closed_loop.environment import merge_environments, validate_environment
from avsectester.execution.frame_data_cache import FrameDataCache
from avsectester.closed_loop.hooks import HookSet
from avsectester.execution.runner import ClosedLoopRunner, RunRequest
from avsectester.execution.services import (
    ClosedLoopServices, create_avstack, create_simulator_runtime, ensure_fresh, stop_unstarted_attacks,
)
from avsectester.closed_loop.simulators.base import DeploymentPlan, UnsupportedStartError
from avsectester.runlog import Recorder, RunLog
from avsectester.maps import ScenarioMap

from .config import ExperimentConfiguration, TreatmentConfiguration, load_configuration
from .configuration_summary import type_name
from .metadata_file import _read_document, save_metadata
from .models import (
    ExperimentResult, PreparedExperiment, SelectionDecision, TreatmentResult,
    preparation_record, treatment_applies, build_units, segment_key,
)
from .output_directory import (
    run_directory, load_preparation_progress, save_preparation_progress, save_completed_artifact,
    recorded_data_equal,
)


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
    validate_environment(config.runs.environment, config.simulator)
    for name, attack in config.attacks.items():
        if not isinstance(attack.environment, Mapping):
            raise TypeError(f"Attack {name!r} environment must be a mapping")
        validate_environment(merge_environments(config.runs.environment, attack.environment), config.simulator)
    for treatment in config.runs.treatments:
        attack = config.attacks[treatment.attack_name] if treatment.attack_name is not None else None
        validate_environment(merge_environments(
            config.runs.environment, attack.environment if attack is not None else {}, treatment.environment,
        ), config.simulator)


def _segments(attack_name: str, attack: Any, view: Any,
              config: ExperimentConfiguration) -> tuple[AttackSegment, ...]:
    selected = tuple(attack.build_metadata(view))
    for segment in selected:
        if not isinstance(segment, AttackSegment):
            raise TypeError("build_metadata must return AttackSegment records")
        if segment.scenario_id != view.scenario_id:
            raise ValueError("Selected segment belongs to another scenario")
        config.dataset.checked_frame(segment.scenario_id, segment.start)
        config.dataset.checked_frame(segment.scenario_id, segment.stop)
        config.dataset.decision_window(segment.scenario_id, segment.start, segment.decision_steps)
        attack.validate_segment(segment)
    deployment = _deployment_plan(config, attack_name, selected)
    accepted = []
    with ClosedLoopServices(config.dataset, config.simulator, config.avstack,
                            view.services.frame_data_cache, deployment=deployment) as services:
        with services.session(view.seed) as session:
            for segment in selected:
                if not session.can_start_at(segment.start.position):
                    view.record_reason(
                        f"Segment at position {segment.start.position} cannot start: "
                        "insufficient recorded history or deployment window")
                    continue
                start = config.dataset.scenario_start(segment.scenario_id, segment.start)
                try:
                    config.simulator.validate_start(start)
                except UnsupportedStartError as error:
                    view.record_reason(f"Segment at position {segment.start.position} cannot start: {error}")
                    continue
                accepted.append(segment)
    return tuple(accepted)


@dataclass(frozen=True)
class _ExperimentSegments:
    metadata: Mapping[str, tuple[AttackSegment, ...]]
    deployment: DeploymentPlan
    selections: tuple[SelectionDecision, ...]
    source: Mapping[str, str] | None


def _selected_metadata(configuration: ExperimentConfiguration, cache: FrameDataCache,
                       seed: int, scenarios: tuple[Any, ...]) -> _ExperimentSegments:
    """Select each attack and recheck its starts against the shared deployment budget."""
    metadata: dict[str, tuple[AttackSegment, ...]] = {}
    selections: list[SelectionDecision] = []
    for name, attack in configuration.attacks.items():
        selected: list[AttackSegment] = []
        for scenario in scenarios:
            view = None
            try:
                with ClosedLoopServices(
                    configuration.dataset, configuration.simulator,
                    configuration.avstack, cache,
                ) as services:
                    view = services.scenario(scenario.id, seed=seed)
                    values = _segments(name, attack, view, configuration)
                selected.extend(values)
                selections.append(SelectionDecision(
                    name, scenario.id, "selected" if values else "not_selected", tuple(view.reasons),
                ))
            except Exception as error:
                selections.append(SelectionDecision(
                    name, scenario.id, "error", tuple(view.reasons) if view is not None else (),
                    f"{type(error).__name__}: {error}",
                ))
                if configuration.runs.strict:
                    raise
        metadata[name] = tuple(selected)
    plans = {name: _deployment_plan(configuration, name, segments)
             for name, segments in metadata.items()}
    deployment = DeploymentPlan(max((plan.ticks for plan in plans.values()), default=0))
    for name, segments in metadata.items():
        # Selection may have used smaller per-scenario budgets within this attack.
        # Recheck every selected start against the final experiment-wide budget.
        rejected: dict[str, list[str]] = {}
        accepted = []
        with ClosedLoopServices(
            configuration.dataset, configuration.simulator, configuration.avstack,
            cache, deployment=deployment,
        ) as services, services.session(seed) as session:
            for segment in segments:
                if session.can_start_at(segment.start.position):
                    accepted.append(segment)
                else:
                    rejected.setdefault(segment.scenario_id, []).append(
                        f"Segment at position {segment.start.position} cannot start with shared "
                        f"deployment budget {deployment.ticks}: insufficient recorded history "
                        "or deployment window")
        metadata[name] = tuple(accepted)
        for index, decision in enumerate(selections):
            if decision.attack_name == name and decision.scenario_id in rejected:
                remains = any(x.scenario_id == decision.scenario_id for x in accepted)
                selections[index] = replace(
                    decision, status="selected" if remains else "not_selected",
                    reasons=decision.reasons + tuple(rejected[decision.scenario_id]),
                )
    return _ExperimentSegments(metadata, deployment, tuple(selections), None)


def _validated_metadata(configuration: ExperimentConfiguration,
                        scenarios: tuple[Any, ...]) -> _ExperimentSegments:
    """Strictly validate a user-provided segment list without running selection."""
    path = configuration.metadata.load
    assert path is not None
    document = _read_document(path)
    dataset_type = type_name(configuration.dataset)
    if document.dataset_type != dataset_type:
        raise ValueError(f"Metadata dataset_type {document.dataset_type!r} does not match "
                         f"the configured dataset type {dataset_type!r}")
    segments = document.segments
    source = {"load": str(path.resolve()), "sha256": document.sha256}
    scenario_ids = {scenario.id for scenario in scenarios}
    for index, segment in enumerate(segments):
        try:
            if segment.scenario_id not in scenario_ids:
                raise ValueError(f"Unknown scenario {segment.scenario_id!r}")
            configuration.dataset.checked_frame(segment.scenario_id, segment.start)
            configuration.dataset.checked_frame(segment.scenario_id, segment.stop)
            configuration.dataset.decision_window(segment.scenario_id, segment.start, segment.decision_steps)
            for name, attack in configuration.attacks.items():
                try:
                    attack.validate_segment(segment)
                except (TypeError, ValueError) as error:
                    raise ValueError(f"Attack {name!r} rejected the segment: {error}") from error
        except Exception as error:
            raise ValueError(f"Metadata segment {index} ({segment.scenario_id}, "
                             f"{segment.start.position}–{segment.stop.position}): {error}") from error
    metadata = {name: segments for name in configuration.attacks}
    deployment = DeploymentPlan(max(
        (_deployment_plan(configuration, name, segments).ticks for name in configuration.attacks), default=0,
    ))
    with ClosedLoopServices(
        configuration.dataset, configuration.simulator, configuration.avstack,
        deployment=deployment,
    ) as services, services.session(configuration.runs.seeds[0]) as session:
        for index, segment in enumerate(segments):
            try:
                if not session.can_start_at(segment.start.position):
                    raise ValueError(f"Cannot start with shared deployment budget {deployment.ticks}: "
                                     "insufficient recorded history or deployment window")
                configuration.simulator.validate_start(
                    configuration.dataset.scenario_start(segment.scenario_id, segment.start))
            except Exception as error:
                raise ValueError(f"Metadata segment {index} ({segment.scenario_id}, "
                                 f"{segment.start.position}–{segment.stop.position}): {error}") from error
    return _ExperimentSegments(metadata, deployment, (), source)


@dataclass(frozen=True)
class _Validation:
    segments: _ExperimentSegments | None = None
    scenarios: tuple[Any, ...] = ()
    prepared: PreparedExperiment | None = None
    progress: tuple[PreparedExperiment, tuple[str, ...]] | None = None


def _validate(configuration: ExperimentConfiguration, resume: bool) -> _Validation:
    """Validate once and retain loaded preparation inputs for the preparation workflow."""
    validate_configuration_environment(configuration)
    output = configuration.runs.output
    if output is not None:
        if output.exists() and not output.is_dir():
            raise ValueError(f"Preparation output is not a directory: {output}")
        if resume and (output / "prepared.json").is_file():
            return _Validation(prepared=PreparedExperiment.load(configuration, output))
        elif resume and (output / "preparation-progress.json").is_file():
            prepared, completed = load_preparation_progress(configuration, output)
            return _Validation(progress=(prepared, completed))
        elif not resume and any(
            (output / name).exists() for name in
            ("prepared.json", "preparation-progress.json", "attacks", "runs", "offline",
             "offline_report.html", "evaluation_report.html", "offline-incomplete")
        ):
            raise ValueError(f"Preparation output already exists: {output}")
    if configuration.metadata.load is not None:
        scenarios = tuple(configuration.dataset.scenarios())
        segments = _validated_metadata(configuration, scenarios)
        build_units(configuration, segments.metadata, scenario_ids=tuple(x.id for x in scenarios))
        return _Validation(segments=segments, scenarios=scenarios)
    else:
        # Construction checks adapter compatibility; sessions and native resources stay unopened.
        with ClosedLoopServices(configuration.dataset, configuration.simulator, configuration.avstack):
            pass
    return _Validation()


class Experiment:
    """The fixed build_metadata → run_offline → online experiment workflow.

    Configuration loading creates descriptions and factories, not a driving run. Preparation
    owns its services and closes them. Execution creates and closes fresh simulator and AVStack
    instances for every named treatment, including clean.
    """

    @staticmethod
    def validate(config: ExperimentConfiguration | Mapping[str, Any] | str | Path,
                 resume: bool = False) -> None:
        """Validate preparation inputs without creating a runtime or connecting to services.

        Loaded metadata is checked against the dataset, attacks, shared deployment budget and
        simulator start requirements. Resuming also checks the saved preparation configuration.
        """
        _validate(load_configuration(config), resume)

    @staticmethod
    def prepare(config: ExperimentConfiguration | Mapping[str, Any] | str | Path,
                resume: bool = False) -> PreparedExperiment:
        """Select all attacks, partition units and checkpoint each completed offline stage."""
        configuration = load_configuration(config)
        validated = _validate(configuration, resume)
        output = configuration.runs.output
        if validated.prepared is not None:
            return validated.prepared
        cache = FrameDataCache()
        seed = configuration.runs.seeds[0]
        if validated.progress is not None:
            prepared, completed_names = validated.progress
            completed = list(completed_names)
            deployment = prepared.deployment
            if deployment is None:
                raise ValueError("Preparation progress requires a deployment budget")
        else:
            preparation = preparation_record(configuration)
            scenarios = validated.scenarios or tuple(configuration.dataset.scenarios())
            segments = (validated.segments if validated.segments is not None else
                        _selected_metadata(configuration, cache, seed, scenarios))
            metadata, deployment = segments.metadata, segments.deployment
            units = build_units(configuration, metadata, scenario_ids=tuple(x.id for x in scenarios))
            prepared = PreparedExperiment(configuration, metadata, {}, segments.selections,
                                          preparation=preparation, units=units, deployment=deployment,
                                          metadata_source=segments.source)
            if configuration.metadata.save is not None:
                save_metadata(tuple(unit.segment for unit in units), configuration.metadata.save,
                              dataset_type=type_name(configuration.dataset))
            completed = []
            if output is not None:
                save_preparation_progress(prepared, output, completed)
        artifacts = dict(prepared.artifacts)
        reports = dict(prepared.offline_reports)
        prepared = replace(prepared, artifacts=artifacts, offline_reports=reports)
        for name, attack in configuration.attacks.items():
            if name in completed:
                continue
            if configuration.split is None:
                training = prepared.metadata[name]
                validation: tuple[AttackSegment, ...] = ()
            else:
                training_keys = {segment_key(unit.segment) for unit in prepared.units
                                 if name in unit.attack_names and unit.part == "train"}
                validation_keys = {segment_key(unit.segment) for unit in prepared.units
                                   if name in unit.attack_names and unit.part == "validation"}
                training = tuple(x for x in prepared.metadata[name] if segment_key(x) in training_keys)
                validation = tuple(x for x in prepared.metadata[name] if segment_key(x) in validation_keys)
            environment = merge_environments(configuration.runs.environment, attack.environment)
            report_root = None if output is None else output / "offline" / name
            if output is not None and report_root is not None and report_root.exists():
                archive_root = output / "offline-incomplete"
                archive_root.mkdir(parents=True, exist_ok=True)
                stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
                archive = archive_root / f"{name}-{stamp}"
                suffix = 1
                while archive.exists():
                    archive = archive_root / f"{name}-{stamp}-{suffix}"
                    suffix += 1
                report_root.rename(archive)
            report = OfflineReport(None if report_root is None else report_root / "files")
            reports[name] = report
            if report.directory is not None:
                report.directory.mkdir(parents=True, exist_ok=True)
            try:
                with ClosedLoopServices(
                    configuration.dataset, configuration.simulator, configuration.avstack,
                    cache, deployment=deployment,
                ) as services:
                    artifacts[name] = attack.run_offline(OfflineContext(
                        services, training, seed, environment, deployment, validation, report,
                    ))
            except BaseException as error:
                report.status = "failed"
                try:
                    if output is not None and report_root is not None:
                        report.save(report_root)
                        from .html_report import write_offline_html

                        write_offline_html(output)
                except BaseException as report_error:
                    error.add_note(
                        "Failed to save the partial offline report: "
                        f"{type(report_error).__name__}: {report_error}"
                    )
                raise
            report.status = "completed"
            completed.append(name)
            if output is not None:
                save_completed_artifact(prepared, name, output, completed)
        if output is not None:
            prepared.save(output)
        return prepared

    @staticmethod
    def execute(prepared: PreparedExperiment, resume: bool = False) -> ExperimentResult:
        """Execute unit/seed groups, retaining matching completed runs when resuming."""
        if not isinstance(prepared, PreparedExperiment):
            raise TypeError("Experiment.execute requires PreparedExperiment with loaded artifacts")
        if set(prepared.artifacts) != set(prepared.metadata):
            raise ValueError("PreparedExperiment requires artifacts for every attack before execution")
        output = prepared.configuration.runs.output
        if output is not None and not resume and (output / "runs").exists():
            raise ValueError(f"Run output already exists: {output / 'runs'}")
        validate_configuration_environment(prepared.configuration)
        return _Execution(prepared, resume=resume).run()


@dataclass(frozen=True)
class _Group:
    attack_names: tuple[str, ...]
    index: int
    segment: AttackSegment
    seed: int
    decision_steps: int
    part: str | None


@dataclass(frozen=True)
class _Treatment:
    configuration: TreatmentConfiguration
    hooks: HookSet
    attack_runtime: AttackRuntime | None
    setup_ticks: int


def _validate_setup_ticks(value: int) -> None:
    """Reject malformed participant declarations before computing a shared window."""
    require_nonnegative_integer(value, "setup_ticks")


def _deployment_plan(config: ExperimentConfiguration, attack_name: str,
                     segments: tuple[AttackSegment, ...]) -> DeploymentPlan:
    """Use one budget across preparation, segments, seeds and comparison treatments."""
    attack = config.attacks[attack_name]
    ticks = 0
    for treatment in config.runs.treatments:
        if not treatment_applies(treatment, attack_name):
            continue
        for seed in config.runs.seeds:
            hooks = _hooks(config.hooks, seed)
            _validate_setup_ticks(hooks.setup_ticks)
            for segment in segments:
                declared = attack.deployment_ticks(segment) if treatment.attack_name is not None else 0
                _validate_setup_ticks(declared)
                ticks = max(ticks, hooks.setup_ticks + declared)
    return DeploymentPlan(ticks)


class _Execution:
    """Resources and failure accounting for one execute call, never retained by the config."""

    def __init__(self, prepared: PreparedExperiment, *, resume: bool = False) -> None:
        self.prepared = prepared
        self.resume = resume
        self.configuration = prepared.configuration
        self.cache = FrameDataCache()
        self.runs: list[TreatmentResult] = []
        required_ticks = max((_deployment_plan(self.configuration, name, segments).ticks
                              for name, segments in prepared.metadata.items()), default=0)
        self.deployment = prepared.deployment or DeploymentPlan(required_ticks)
        if self.deployment.ticks < required_ticks:
            raise ValueError("Saved deployment budget is smaller than the configured requirement")

    def _description(self, group: _Group, treatment: TreatmentConfiguration,
                     deployment: DeploymentPlan | None = None) -> tuple[dict[str, Any], Path | None]:
        config = self.configuration
        attack = config.attacks[treatment.attack_name] if treatment.attack_name is not None else None
        environment = merge_environments(
            config.runs.environment, attack.environment if attack is not None else {},
            treatment.environment,
        )
        destination = None
        if config.runs.output is not None:
            destination = run_directory(config.runs.output, group.index,
                                        group.seed, treatment.name)
        metadata: dict[str, Any] = {
            "attack_name": treatment.attack_name, "segment_index": group.index, "part": group.part,
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
            treatment.attack_name, group.index, group.segment, treatment.name, group.seed, log,
            f"{type(error).__name__}: {error}", part=group.part,
        ))
        if self.configuration.runs.strict:
            raise error

    def _treatments(self, group: _Group,
                    configurations: tuple[TreatmentConfiguration, ...]) -> tuple[_Treatment, ...]:
        config = self.configuration
        result = []
        for treatment in configurations:
            runtime = None
            try:
                hooks = _hooks(config.hooks, group.seed)
                runtime = None
                if treatment.attack_name is not None:
                    runtime = config.attacks[treatment.attack_name].create_runtime(
                        group.segment, self.prepared.artifacts[treatment.attack_name])
                    if not isinstance(runtime, AttackRuntime):
                        raise TypeError("Attack.create_runtime must return AttackRuntime")
                    ensure_fresh(runtime)
                    _validate_setup_ticks(runtime.setup_ticks)
                _validate_setup_ticks(hooks.setup_ticks)
                declared_ticks = (config.attacks[treatment.attack_name].deployment_ticks(group.segment)
                                  if treatment.attack_name is not None else 0)
                _validate_setup_ticks(declared_ticks)
                if runtime is not None and runtime.setup_ticks > declared_ticks:
                    raise ValueError("AttackRuntime.setup_ticks exceeds Attack.deployment_ticks declaration")
                ticks = hooks.setup_ticks + (declared_ticks if runtime is not None else 0)
                result.append(_Treatment(treatment, hooks, runtime, ticks))
            except BaseException as error:
                stop_unstarted_attacks((runtime,), group.segment.start, error)
                if not isinstance(error, Exception):
                    for pending in result:
                        stop_unstarted_attacks((pending.attack_runtime,), group.segment.start, error)
                    raise
                if config.runs.strict:
                    for pending in result:
                        stop_unstarted_attacks((pending.attack_runtime,), group.segment.start, error)
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
                initialization = session.initialization(start, config.runs.environment)
        return start, initialization

    def _run_treatment(self, group: _Group, treatment: _Treatment,
                       start: ScenarioStart, initialization: Any, deployment: DeploymentPlan) -> None:
        config = self.configuration
        metadata, destination = self._description(group, treatment.configuration, deployment)
        runner_owns_attack = False
        try:
            with ExitStack() as resources:
                simulator_runtime = create_simulator_runtime(config.simulator, group.seed, resources)
                avstack = create_avstack(config.avstack, group.seed, resources)
                request = RunRequest(
                    group.decision_steps, hooks=treatment.hooks,
                    attacks=(treatment.attack_runtime,) if treatment.attack_runtime is not None else (),
                    recording=config.runs.recording, output_path=destination, metadata=metadata,
                    start=deepcopy(start), initialization=deepcopy(initialization),
                    deployment=deployment, environment=metadata["environment"],
                )
                runner_owns_attack = True
                log = ClosedLoopRunner().run(simulator_runtime, avstack, request,
                                             release_owned_resources=resources.close)
            self.runs.append(TreatmentResult(
                treatment.configuration.attack_name, group.index, group.segment, treatment.configuration.name,
                group.seed, log, part=group.part,
            ))
        except BaseException as error:
            if not runner_owns_attack:
                stop_unstarted_attacks((treatment.attack_runtime,), group.segment.start, error)
            if not isinstance(error, Exception):
                raise
            self._failed(group, treatment.configuration, error, deployment)

    def _run_group(self, group: _Group) -> None:
        configurations = tuple(t for t in self.configuration.runs.treatments
                               if t.attack_name is None or t.attack_name in group.attack_names)
        pending_configurations = []
        for specification in configurations:
            destination = None
            if self.configuration.runs.output is not None:
                destination = run_directory(self.configuration.runs.output, group.index,
                                            group.seed, specification.name)
            if self.resume and destination is not None and (destination.exists() or destination.is_symlink()):
                retained = None
                try:
                    log = RunLog.read(destination)
                    expected, _ = self._description(group, specification, self.deployment)
                    if log.status == "completed" and all(
                        key in log.metadata and recorded_data_equal(log.metadata[key], value)
                        for key, value in expected.items()
                    ):
                        retained = TreatmentResult(specification.attack_name, group.index, group.segment,
                                                   specification.name, group.seed, log, part=group.part)
                except Exception:
                    pass
                if retained is not None:
                    self.runs.append(retained)
                    continue
                if destination.is_symlink() or not destination.is_dir():
                    destination.unlink()
                else:
                    shutil.rmtree(destination)
            pending_configurations.append(specification)
        if not pending_configurations:
            return
        treatments = self._treatments(group, tuple(pending_configurations))
        if not treatments:
            return
        deployment = self.deployment
        try:
            if any(treatment.setup_ticks > deployment.ticks for treatment in treatments):
                raise ValueError("Treatment deployment exceeds the shared offline and execution budget")
            start, initialization = self._initialize_group(group, deployment)
        except BaseException as error:
            for treatment in treatments:
                stop_unstarted_attacks((treatment.attack_runtime,), group.segment.start, error)
            if not isinstance(error, Exception):
                raise
            for treatment in treatments:
                self._failed(group, treatment.configuration, error, deployment)
            return
        for index, treatment in enumerate(treatments):
            try:
                self._run_treatment(group, treatment, start, initialization, deployment)
            except BaseException as error:
                for pending in treatments[index + 1:]:
                    stop_unstarted_attacks((pending.attack_runtime,), group.segment.start, error)
                raise

    def run(self) -> ExperimentResult:
        scenario_maps: dict[str, ScenarioMap | Unavailable] = {}
        for segments in self.prepared.metadata.values():
            for segment in segments:
                if segment.scenario_id not in scenario_maps:
                    scenario_map = self.configuration.dataset.scenario_map(segment.scenario_id)
                    if not isinstance(scenario_map, (ScenarioMap, Unavailable)):
                        raise TypeError("Dataset.scenario_map must return ScenarioMap or Unavailable")
                    scenario_maps[segment.scenario_id] = scenario_map
        if self.configuration.runs.output is not None:
            from .output_directory import save_scenario_maps

            save_scenario_maps(scenario_maps, self.configuration.runs.output)
        for unit in self.prepared.units:
            if self.configuration.split is not None and unit.part not in self.configuration.split.execute:
                continue
            steps = unit.segment.decision_steps
            if self.configuration.runs.decision_steps is not None:
                steps = min(steps, self.configuration.runs.decision_steps)
            for seed in self.configuration.runs.seeds:
                self._run_group(_Group(unit.attack_names, unit.index, unit.segment, seed, steps, unit.part))
        return ExperimentResult(tuple(self.runs), self.prepared.selections, scenario_maps=scenario_maps)


__all__ = [
    'Experiment',
]
