"""Preparation and selection share one isolated closed-loop service boundary.

Plain factories construct fresh resources for each operation. This module orchestrates them; it
never implements another simulator clock, driving loop, scoring objective or selector.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any

from .backend import AVStack
from .capture import CaptureBackend, DeploymentPlan
from .closed_loop import ClosedLoopRunner, RecordingPolicy, RunRequest
from .datasets.base import Dataset, FrameRef, ReferenceFrame, SourceData, Unavailable, nonnegative
from .plane import Observation
from .reference import Initialization, ReferenceCache, ReferenceSequence
from .resources import backend_key, reserve
from .runlog import Recorder, RunLog
from .runtime import HookSet
from .simulator_runtime import SimulatorRuntime


@dataclass(frozen=True)
class CandidateOrigin:
    scenario_id: str
    start: FrameRef
    stop: FrameRef
    roles: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    placement: Any = None

    def __post_init__(self) -> None:
        if self.start.scenario_id != self.scenario_id or self.stop.scenario_id != self.scenario_id:
            raise ValueError("Candidate boundaries must belong to the candidate scenario")
        if self.stop.position <= self.start.position:
            raise ValueError("Candidate stop must follow start and include the terminal state")
        roles = {}
        for role, identities in self.roles.items():
            if not isinstance(role, str) or not role or isinstance(identities, str):
                raise ValueError("Roles map nonempty names to sequences of stable actor IDs")
            roles[role] = tuple(identities)
            if any(not isinstance(i, str) or not i for i in identities):
                raise ValueError("Role actor IDs must be nonempty strings")
        object.__setattr__(self, "roles", roles)
        object.__setattr__(self, "placement", deepcopy(self.placement))

    @property
    def decision_steps(self) -> int:
        return self.stop.position - self.start.position


@dataclass(frozen=True)
class CandidateRequest:
    origin: CandidateOrigin
    seed: int
    decision_steps: int
    environment: Mapping[str, Any]
    recording: RecordingPolicy
    attack: Any = None
    artifacts: Any = None
    output_path: str | Path | None = None

    def __post_init__(self) -> None:
        nonnegative(self.seed, "seed")
        nonnegative(self.decision_steps, "decision_steps")
        if self.decision_steps != self.origin.decision_steps:
            raise ValueError("Candidate decision_steps must equal its origin interval")
        if self.attack is None and self.artifacts is not None:
            raise ValueError("Candidate artifacts require an attack runtime factory")
        if self.attack is not None and not callable(getattr(self.attack, "runtime_factory", None)):
            raise TypeError("Candidate attack must provide runtime_factory(attack, origin, artifacts)")
        object.__setattr__(self, "environment", deepcopy(dict(self.environment)))


@dataclass(frozen=True)
class SensorRequest:
    scenario_id: str
    frame: FrameRef
    sensor_ids: tuple[str, ...]
    environment: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RenderRequest:
    scenario_id: str
    frame: FrameRef
    sensor_id: str
    insertions: Sequence[Any]
    roles: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    environment: Mapping[str, Any] = field(default_factory=dict)
    scene: Any = None
    compositor: Any = None


@dataclass(frozen=True)
class SensorSample:
    frame: FrameRef
    observation: Observation
    provenance: Mapping[str, Any]


@dataclass(frozen=True)
class RenderResult:
    frame: FrameRef
    observation: Observation
    provenance: Mapping[str, Any]
    evidence: Mapping[str, Any] = field(default_factory=dict)
    resolved: Sequence[Any] = ()


@dataclass(frozen=True)
class ComponentDescriptor:
    id: str
    inputs: Mapping[str, str]
    outputs: Mapping[str, str]
    codecs: Mapping[str, str]
    differentiable_inputs: tuple[str, ...] = ()


@dataclass(frozen=True)
class ScorerDescriptor(ComponentDescriptor):
    output_shape: tuple[int, ...] = ()
    direction: str = "minimize"

    def __post_init__(self) -> None:
        if self.direction not in {"minimize", "maximize"}:
            raise ValueError("Scorer direction must be minimize or maximize")


@dataclass(frozen=True)
class ComponentResult:
    value: Any
    descriptor: ComponentDescriptor
    provenance: Mapping[str, Any]


@dataclass(frozen=True)
class ScoreResult:
    value: Any
    descriptor: ScorerDescriptor
    provenance: Mapping[str, Any]


@dataclass(frozen=True)
class ClosedLoopCapabilities:
    source_kinds: tuple[str, ...]
    sensor_data: bool
    render: bool
    rollout: bool
    components: tuple[ComponentDescriptor, ...]
    scorers: tuple[ScorerDescriptor, ...]


class DrivingAdapter:
    """Explicit component/scorer bindings over a fresh AVStack factory.

    Callbacks receive (stack, inputs), or (stack, inputs, target) for scorers. Input
    schemas describe native argument conventions; adapters own conversion and warmup.
    No mutable pipeline instance is shared between operations. Sharing model weights
    behind separate stacks is permitted only when the factory guarantees isolation.
    """

    def __init__(
        self, factory: Callable[[int], AVStack], *,
        components: Sequence[tuple[ComponentDescriptor, Callable[..., Any]]] = (),
        scorers: Sequence[tuple[ScorerDescriptor, Callable[..., Any]]] = (),
    ) -> None:
        self.factory = factory
        self.component_bindings = self._bindings(components)
        self.scorer_bindings = self._bindings(scorers)

    @staticmethod
    def _bindings(entries: Sequence[Any]) -> dict[str, Any]:
        result = {}
        for descriptor, callback in entries:
            if not descriptor.id or descriptor.id in result or not callable(callback):
                raise ValueError("Component/scorer IDs must be unique and callbacks callable")
            if not descriptor.inputs or not descriptor.outputs or not descriptor.codecs:
                raise ValueError("Declare component input/output schemas and codecs")
            if set(descriptor.differentiable_inputs) - descriptor.inputs.keys():
                raise ValueError("Differentiable inputs must name declared inputs")
            result[descriptor.id] = (deepcopy(descriptor), callback)
        return result


def _fresh(resource: Any) -> Any:
    if getattr(resource, "_services_used", False) or getattr(resource, "_runtime", None) is not None:
        raise RuntimeError("Resource factory returned a reused or active instance")
    resource._services_used = True
    return resource


def _close(resource: Any) -> None:
    close = getattr(resource, "close", None)
    if callable(close):
        close()


class ClosedLoopServices:
    def __init__(
        self, dataset: Dataset, simulator_factory: Callable[[SourceData, int], CaptureBackend],
        driving_adapter: DrivingAdapter | None = None, reference_cache: ReferenceCache | None = None,
        *, backend_type: type[CaptureBackend],
        deployment: DeploymentPlan = DeploymentPlan(),
    ) -> None:
        self.dataset = dataset
        if not issubclass(backend_type, CaptureBackend):
            raise TypeError("Services require a CaptureBackend class and a lazy backend factory")
        self.simulator_factory = simulator_factory
        self.backend_type = backend_type
        self.deployment = deployment
        self.driving_adapter = driving_adapter
        self.reference_cache = reference_cache if reference_cache is not None else ReferenceCache()

    @property
    def capabilities(self) -> ClosedLoopCapabilities:
        driver = self.driving_adapter
        return ClosedLoopCapabilities(
            self.backend_type.source_kinds, True, self.backend_type.can_render,
            driver is not None,
            tuple(deepcopy(pair[0]) for pair in driver.component_bindings.values()) if driver else (),
            tuple(deepcopy(pair[0]) for pair in driver.scorer_bindings.values()) if driver else (),
        )

    def open_scope(self, seed: int, history: str = "fresh") -> ClosedLoopScope:
        nonnegative(seed, "seed")
        if history != "fresh":
            raise ValueError("Candidate scopes require fresh history")
        return ClosedLoopScope(self, seed)

    def scenario(self, scenario_id: str, *, seed: int) -> ScenarioView:
        self.dataset.frames(scenario_id)  # validate without materializing anything
        nonnegative(seed, "seed")
        return ScenarioView(scenario_id, self, seed)


class ClosedLoopScope:
    def __init__(self, services: ClosedLoopServices, seed: int) -> None:
        self.services, self.seed = services, seed
        self._closed = False
        self._busy = Lock()

    def __enter__(self) -> ClosedLoopScope:
        self._check()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def _check(self) -> None:
        if self._closed:
            raise RuntimeError("Closed-loop scope is closed")

    @contextmanager
    def _operation(self) -> Iterator[ExitStack]:
        self._check()
        if not self._busy.acquire(blocking=False):
            raise RuntimeError("A candidate operation is already active in this scope")
        resources = ExitStack()
        try:
            yield resources
        except BaseException as error:
            try:
                resources.close()
            except BaseException as cleanup:
                error.add_note(f"Candidate resource cleanup also failed: {cleanup!r}")
            raise
        finally:
            try:
                resources.close()
            finally:
                self._busy.release()

    def close(self) -> None:
        if not self._busy.acquire(blocking=False):
            raise RuntimeError("Cannot close a scope while its operation is active")
        try:
            if not self._closed:
                self._closed = True

        finally:
            self._busy.release()

    def _source(self, scenario: str, frame: FrameRef, environment: Mapping[str, Any]) -> SourceData:
        return self.services.dataset.source_data(scenario, frame)

    def _backend(self, resources: ExitStack, source: SourceData) -> tuple[CaptureBackend, SimulatorRuntime]:
        # Factories construct configuration-only objects. Native world creation happens at reset.
        backend = _fresh(self.services.simulator_factory(source, self.seed))
        if not isinstance(backend, self.services.backend_type):
            raise TypeError("Simulator factory returned a different backend type")
        # Callbacks run in reverse order: close the backend, then release its reservation, so no
        # other operation can reset this server while its actors are still being destroyed.
        reservation = ExitStack()
        resources.callback(reservation.close)
        resources.callback(_close, backend)
        backend.validate_source(source)
        reservation.enter_context(reserve(backend_key(backend), self))
        sim = SimulatorRuntime(backend, resource_owner=self)
        resources.callback(sim.close)
        return backend, sim

    def _sequence(self, scenario: str, environment: Mapping[str, Any]) -> ReferenceSequence:
        dataset = self.services.dataset
        frames = tuple(dataset.frames(scenario))
        source = self._source(scenario, frames[0], environment)
        with self._operation() as resources:
            backend, sim = self._backend(resources, source)
            key = ReferenceCache.key(source, {
                **backend.identity, "environment": dict(environment),
                "deployment_ticks": self.services.deployment.ticks,
            }, self.seed)

            def generate() -> ReferenceSequence:
                with sim.exclusive():
                    sequence = backend.materialize_reference(
                        source, frames, environment=environment, deployment=self.services.deployment)
                if tuple(record.frame for record in sequence.frames) != frames:
                    raise ValueError("Reference materializer must preserve every source boundary")
                return sequence

            return self.services.reference_cache.materialize(key, generate)

    def reference(self, scenario: str, frame: FrameRef,
                  environment: Mapping[str, Any] | None = None) -> ReferenceFrame:
        self._check()
        record = self.services.dataset.reference(scenario, frame)
        if isinstance(record, Unavailable):
            return self._sequence(scenario, environment or {}).at(frame)
        if environment:
            raise ValueError("Recorded references describe the source environment")
        return record

    def _initialization(self, source: SourceData, environment: Mapping[str, Any]) -> Initialization | None:
        if source.kind != "carla":
            return None
        sequence = self._sequence(source.scenario_id, environment)
        return Initialization(sequence.at(source.start), sequence.resolved_scenario,
                              provenance=sequence.provenance)

    def _provenance(self, source: SourceData, environment: Mapping[str, Any],
                    backend: CaptureBackend | None = None) -> dict[str, Any]:
        return {"source": asdict(source.start), "seed": self.seed,
                "environment": deepcopy(dict(environment)),
                "simulator": deepcopy(dict(backend.identity)) if backend is not None else {
                    "simulator": self.services.backend_type.__name__},
                "dataset_time_step": source.payload.get("time_step"),
                "interpolation": source.payload.get("interpolation")}

    def sensor_data(self, request: SensorRequest) -> SensorSample:
        source = self._source(request.scenario_id, request.frame, request.environment)
        initialization = self._initialization(source, request.environment)
        with self._operation() as resources:
            backend, sim = self._backend(resources, source)
            with sim.exclusive():
                sim.reset(source, initialization, environment=request.environment,
                          deployment=self.services.deployment)
                observation = sim.observe()
                missing = set(request.sensor_ids) - observation.sensor_data.keys()
                if missing:
                    raise ValueError(f"Unavailable sensors: {sorted(missing)}")
                observation.sensor_data = {key: observation.sensor_data[key] for key in request.sensor_ids}
                observation.calibration = {key: value for key, value in observation.calibration.items()
                                           if key in request.sensor_ids}
                return SensorSample(source.start, observation, self._provenance(source, request.environment, backend))

    def render(self, request: RenderRequest) -> RenderResult:
        if not self.services.backend_type.can_render:
            raise NotImplementedError("The simulator cannot render insertions")
        source = self._source(request.scenario_id, request.frame, request.environment)
        initialization = self._initialization(source, request.environment)
        with self._operation() as resources:
            backend, sim = self._backend(resources, source)
            with sim.exclusive():
                sim.reset(source, initialization, environment=request.environment,
                          deployment=self.services.deployment)
                result = backend.render(sim.observe(), request)
                return RenderResult(source.start, result.observation,
                                    self._provenance(source, request.environment, backend),
                                    result.evidence, result.resolved)

    def components(self) -> Sequence[ComponentDescriptor]:
        self._check()
        return self.services.capabilities.components

    def scorers(self) -> Sequence[ScorerDescriptor]:
        self._check()
        return self.services.capabilities.scorers

    def _driver(self, resources: ExitStack) -> AVStack:
        adapter = self.services.driving_adapter
        if adapter is None:
            raise ValueError("This service has no driving-system adapter")
        stack = _fresh(adapter.factory(self.seed))
        resources.callback(_close, stack)
        return stack

    def evaluate(self, component_id: str, inputs: Any) -> ComponentResult:
        adapter = self.services.driving_adapter
        if adapter is None or component_id not in adapter.component_bindings:
            raise ValueError(f"Unavailable component: {component_id!r}")
        descriptor, call = adapter.component_bindings[component_id]
        with self._operation() as resources:
            value = call(self._driver(resources), inputs)
            return ComponentResult(value, deepcopy(descriptor), {"seed": self.seed})

    def score(self, scorer_id: str, inputs: Any, target: Any, require_grad: bool = False) -> ScoreResult:
        adapter = self.services.driving_adapter
        if adapter is None or scorer_id not in adapter.scorer_bindings:
            raise ValueError(f"Unavailable scorer: {scorer_id!r}")
        descriptor, call = adapter.scorer_bindings[scorer_id]
        if require_grad and not descriptor.differentiable_inputs:
            raise ValueError(f"Scorer {scorer_id!r} does not support gradients")
        with self._operation() as resources:
            value = call(self._driver(resources), inputs, target)
            if require_grad and not getattr(value, "requires_grad", False):
                raise ValueError(f"Scorer {scorer_id!r} returned a detached/non-differentiable value")
            return ScoreResult(value, deepcopy(descriptor), {"seed": self.seed, "require_grad": require_grad})

    def rollout(self, request: CandidateRequest) -> RunLog:
        self._check()
        if request.seed != self.seed:
            raise ValueError("Candidate seed must match its scope")
        origin = request.origin
        dataset = self.services.dataset
        frames = dataset.decision_window(origin.scenario_id, origin.start, request.decision_steps)
        if frames[-1] != origin.stop:
            raise ValueError("Candidate stop does not match the requested terminal reference")
        source = self._source(origin.scenario_id, origin.start, request.environment)
        try:
            initialization = self._initialization(source, request.environment)
        except BaseException as error:
            # Materialization precedes runner construction, but failed reference evidence
            # still belongs to the requested run and must survive without a live simulator.
            recorder = None
            try:
                from .log_codecs import default_codecs

                recorder = Recorder(path=request.output_path, codecs=default_codecs(), metadata={
                    **self._provenance(source, request.environment), "candidate": asdict(origin),
                    "decision_steps": request.decision_steps,
                })
                recorder.record_event("failure", phase="reference_preparation",
                                      error_type=type(error).__name__, message=str(error),
                                      initialization_report=getattr(error, "report", {}),
                                      cleanup_errors=list(getattr(error, "__notes__", ())))
                recorder.finish("cancelled" if isinstance(error, (KeyboardInterrupt, SystemExit))
                                else "failed")
                setattr(error, "run_log", recorder.log)
            except BaseException as cleanup:
                error.add_note(f"Reference failure logging also failed: {cleanup!r}")
                if recorder is not None:
                    try:
                        recorder.abort()
                    except BaseException as abort_error:
                        error.add_note(f"Reference log cleanup also failed: {abort_error!r}")
            raise
        with self._operation() as resources:
            backend, sim = self._backend(resources, source)
            stack = self._driver(resources)
            attacks: tuple[Any, ...] = ()
            if request.attack is not None:
                runtime = _fresh(request.attack.runtime_factory(request.attack, origin, request.artifacts))
                attacks = (runtime,)
            return ClosedLoopRunner().run(sim, stack, RunRequest(
                request.decision_steps, hooks=HookSet(seed=self.seed),
                attacks=attacks, recording=request.recording,
                output_path=request.output_path, source=source, initialization=initialization,
                environment=request.environment, deployment=self.services.deployment,
                metadata={**self._provenance(source, request.environment, backend), "candidate": asdict(origin)},
            ), release_owned_resources=resources.close)


@dataclass(frozen=True)
class ScenarioView:
    scenario_id: str
    services: ClosedLoopServices
    seed: int

    @property
    def frames(self) -> Sequence[FrameRef]:
        return self.services.dataset.frames(self.scenario_id)

    def reference(self, frame: FrameRef) -> ReferenceFrame:
        """Resolve recorded data or a preparation-owned cache, without mutating Dataset."""
        with self.services.open_scope(self.seed) as scope:
            return scope.reference(self.scenario_id, frame)
