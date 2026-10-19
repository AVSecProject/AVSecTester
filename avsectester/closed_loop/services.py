"""The metabuilder and the offline phase share one isolated closed-loop service boundary.

Plain factories construct fresh resources for each operation. This module orchestrates them; it
never implements another simulator clock, driving loop, scoring objective or selector.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from threading import Lock
from typing import Any

from avsectester.attack import AttackSegment
from avsectester.attack.runtime import AttackEffect, SensorAttackEffect, WorldAttackEffect
from avsectester.closed_loop.avstack import AVStack
from avsectester.closed_loop.runner import ClosedLoopRunner, RecordingPolicy, RunRequest
from avsectester.datasets.base import Dataset, FrameReference, FrameData, ScenarioStart, Unavailable, nonnegative
from avsectester.perturbations.image import write_camera_rgb
from avsectester.closed_loop.datatypes import Observation
from avsectester.closed_loop.frame_data import Initialization, FrameDataCache, FrameDataSequence
from avsectester.closed_loop.resources import reserve, simulator_key
from avsectester.runlog import Recorder, RunLog
from avsectester.closed_loop.hooks import HookSet
from avsectester.closed_loop.simulator import DeploymentPlan, SensorView, Simulator, SimulatorRuntime


@dataclass(frozen=True)
class CandidateRequest:
    """One candidate rollout. ``attack`` creates the deployed runtime with its create_runtime."""

    segment: AttackSegment
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
        if self.decision_steps != self.segment.decision_steps:
            raise ValueError("Candidate decision_steps must equal its segment interval")
        if self.attack is None and self.artifacts is not None:
            raise ValueError("Candidate artifacts require an attack")
        if self.attack is not None and not callable(getattr(self.attack, "create_runtime", None)):
            raise TypeError("Candidate attack must provide create_runtime(segment, artifacts)")
        object.__setattr__(self, "environment", deepcopy(dict(self.environment)))


@dataclass(frozen=True)
class SensorRequest:
    scenario_id: str
    frame: FrameReference
    sensor_ids: tuple[str, ...]
    environment: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RenderRequest:
    """Capture ``sensor_id`` at ``frame`` with attack effects applied.

    World attack effects are applied at the simulator.setup stage, within the services'
    deployment plan, so they are visible in the capture. Sensor attack effects then change the
    captured data. Component attack effects need a rollout and are rejected.
    """

    scenario_id: str
    frame: FrameReference
    sensor_id: str
    effects: Sequence[AttackEffect] = ()
    environment: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        effects = tuple(self.effects)
        if any(not isinstance(e, (WorldAttackEffect, SensorAttackEffect)) for e in effects):
            raise TypeError("Render requests take world and sensor attack effects only")
        object.__setattr__(self, "effects", effects)


@dataclass(frozen=True)
class SensorSample:
    frame: FrameReference
    observation: Observation
    provenance: Mapping[str, Any]


@dataclass(frozen=True)
class RenderResult:
    """The changed capture and the requested sensor's view before its sensor effects.

    The view is detached: its ``native`` access is empty once the session operation ends.
    """

    frame: FrameReference
    observation: Observation
    view: SensorView
    provenance: Mapping[str, Any]


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
    dataset_kinds: tuple[str, ...]
    sensor_data: bool
    render: bool
    rollout: bool
    components: tuple[ComponentDescriptor, ...]
    scorers: tuple[ScorerDescriptor, ...]


class DrivingAdapter:
    """Explicit component/scorer bindings over a fresh AVStack factory.

    Callbacks receive (avstack, inputs), or (avstack, inputs, target) for scorers. Input
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


class _WorldSetup:
    """Apply world attack effects at simulator.setup when no driving run binds the runtime."""

    def __init__(self, simulator_runtime: SimulatorRuntime, effects: Sequence[WorldAttackEffect],
                 resources: ExitStack) -> None:
        self.simulator_runtime, self.effects, self.resources = simulator_runtime, tuple(effects), resources

    def emit(self, stage: str, value: Any, sensor: str | None = None) -> Any:
        if stage == "simulator.setup":
            for effect in self.effects:
                effect.apply(self.simulator_runtime, self.resources)
        return value


def _fresh(resource: Any) -> Any:
    if getattr(resource, "_services_used", False) or getattr(resource, "_dispatcher", None) is not None:
        raise RuntimeError("Resource factory returned a reused or active instance")
    resource._services_used = True
    return resource


def _close(resource: Any) -> None:
    close = getattr(resource, "close", None)
    if callable(close):
        close()


class ClosedLoopServices:
    def __init__(
        self, dataset: Dataset, simulator: Simulator,
        driving_adapter: DrivingAdapter | None = None, frame_data_cache: FrameDataCache | None = None,
        *, deployment: DeploymentPlan = DeploymentPlan(),
    ) -> None:
        """``simulator`` is a configured ``Simulator``; each operation runs in its own runtime."""
        self.dataset = dataset
        if not isinstance(simulator, Simulator):
            raise TypeError("Services require a configured Simulator")
        self.simulator = simulator
        self.deployment = deployment
        self.driving_adapter = driving_adapter
        self.frame_data_cache = frame_data_cache if frame_data_cache is not None else FrameDataCache()

    @property
    def capabilities(self) -> ClosedLoopCapabilities:
        driver = self.driving_adapter
        return ClosedLoopCapabilities(
            self.simulator.dataset_kinds, True, self.simulator.can_render,
            driver is not None,
            tuple(deepcopy(pair[0]) for pair in driver.component_bindings.values()) if driver else (),
            tuple(deepcopy(pair[0]) for pair in driver.scorer_bindings.values()) if driver else (),
        )

    def session(self, seed: int, history: str = "fresh") -> ClosedLoopSession:
        nonnegative(seed, "seed")
        if history != "fresh":
            raise ValueError("Candidate scopes require fresh history")
        return ClosedLoopSession(self, seed)

    def scenario(self, scenario_id: str, *, seed: int) -> ScenarioView:
        self.dataset.frames(scenario_id)  # validate without generating anything
        nonnegative(seed, "seed")
        return ScenarioView(scenario_id, self, seed)


class ClosedLoopSession:
    def __init__(self, services: ClosedLoopServices, seed: int) -> None:
        self.services, self.seed = services, seed
        self._closed = False
        self._busy = Lock()

    def __enter__(self) -> ClosedLoopSession:
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

    def _scenario_start(self, scenario: str, frame: FrameReference,
                        environment: Mapping[str, Any]) -> ScenarioStart:
        return self.services.dataset.scenario_start(scenario, frame)

    def _simulator_runtime(self, resources: ExitStack, start: ScenarioStart) -> SimulatorRuntime:
        # Every operation gets a new runtime. Native world creation happens at reset.
        self.services.simulator.validate_start(start)
        simulator_runtime = _fresh(self.services.simulator.create_runtime(self.seed))
        if not isinstance(simulator_runtime, SimulatorRuntime):
            raise TypeError("Simulator.create_runtime must return a SimulatorRuntime")
        # Callbacks run in reverse order: close the runtime, then release its reservation, so no
        # other operation can reset this server while its actors are still being destroyed.
        reservation = ExitStack()
        resources.callback(reservation.close)
        resources.callback(_close, simulator_runtime)
        reservation.enter_context(reserve(simulator_key(simulator_runtime), self))
        simulator_runtime.resource_owner = self
        return simulator_runtime

    def _sequence(self, scenario: str, environment: Mapping[str, Any]) -> FrameDataSequence:
        dataset = self.services.dataset
        frames = tuple(dataset.frames(scenario))
        start = self._scenario_start(scenario, frames[0], environment)
        with self._operation() as resources:
            simulator_runtime = self._simulator_runtime(resources, start)
            key = FrameDataCache.key(start, {
                **simulator_runtime.identity, "environment": dict(environment),
                "deployment_ticks": self.services.deployment.ticks,
            }, self.seed)

            def generate() -> FrameDataSequence:
                with simulator_runtime.exclusive():
                    sequence = simulator_runtime.generate_frame_data(
                        start, frames, environment=environment, deployment=self.services.deployment)
                if tuple(record.frame for record in sequence.frames) != frames:
                    raise ValueError("Frame data generation must preserve every dataset frame")
                return sequence

            return self.services.frame_data_cache.get_or_generate(key, generate)

    def frame_data(self, scenario: str, frame: FrameReference,
                  environment: Mapping[str, Any] | None = None) -> FrameData:
        self._check()
        record = self.services.dataset.frame_data(scenario, frame)
        if isinstance(record, Unavailable):
            return self._sequence(scenario, environment or {}).at(frame)
        if environment:
            raise ValueError("Recorded frame data describes the recorded environment")
        return record

    def _initialization(self, start: ScenarioStart, environment: Mapping[str, Any]) -> Initialization | None:
        if start.kind != "CarlaDataset":
            return None
        sequence = self._sequence(start.scenario_id, environment)
        return Initialization(sequence.at(start.frame), sequence.resolved_scenario,
                              provenance=sequence.provenance)

    def _provenance(self, start: ScenarioStart, environment: Mapping[str, Any],
                    simulator_runtime: SimulatorRuntime | None = None) -> dict[str, Any]:
        return {"start_frame": asdict(start.frame), "seed": self.seed,
                "environment": deepcopy(dict(environment)),
                "simulator": deepcopy(dict(simulator_runtime.identity)) if simulator_runtime is not None else {
                    "simulator": type(self.services.simulator).__name__},
                "dataset_time_step": start.payload.get("time_step"),
                "interpolation": start.payload.get("interpolation")}

    def sensor_data(self, request: SensorRequest) -> SensorSample:
        start = self._scenario_start(request.scenario_id, request.frame, request.environment)
        initialization = self._initialization(start, request.environment)
        with self._operation() as resources:
            simulator_runtime = self._simulator_runtime(resources, start)
            with simulator_runtime.exclusive():
                simulator_runtime.reset(start, initialization, environment=request.environment,
                                        deployment=self.services.deployment)
                observation = simulator_runtime.observe()
                missing = set(request.sensor_ids) - observation.sensor_data.keys()
                if missing:
                    raise ValueError(f"Unavailable sensors: {sorted(missing)}")
                observation.sensor_data = {key: observation.sensor_data[key] for key in request.sensor_ids}
                observation.calibration = {key: value for key, value in observation.calibration.items()
                                           if key in request.sensor_ids}
                return SensorSample(start.frame, observation,
                                    self._provenance(start, request.environment, simulator_runtime))

    def render(self, request: RenderRequest) -> RenderResult:
        if not self.services.simulator.can_render:
            raise NotImplementedError("The simulator does not provide sensor views")
        world = [e for e in request.effects if isinstance(e, WorldAttackEffect)]
        sensor = [e for e in request.effects if isinstance(e, SensorAttackEffect)]
        if sum(effect.setup_ticks for effect in world) > self.services.deployment.ticks:
            raise ValueError("World effects need more setup ticks than the deployment plan")
        start = self._scenario_start(request.scenario_id, request.frame, request.environment)
        initialization = self._initialization(start, request.environment)
        with self._operation() as resources:
            simulator_runtime = self._simulator_runtime(resources, start)
            with simulator_runtime.exclusive():
                # Effects register their removal on resources, which close before the runtime.
                with simulator_runtime.bind_hooks(_WorldSetup(simulator_runtime, world, resources)):
                    simulator_runtime.reset(start, initialization, environment=request.environment,
                                            deployment=self.services.deployment)
                observation = simulator_runtime.observe()
                views = {request.sensor_id: simulator_runtime.sensor_view(request.sensor_id)}
                data: dict[str, Any] = {}
                for effect in sensor:
                    if effect.sensor not in views:
                        views[effect.sensor] = simulator_runtime.sensor_view(effect.sensor)
                    view = views[effect.sensor]
                    data[effect.sensor] = effect.apply(data.get(effect.sensor, view.data), view)
                for name, value in data.items():
                    observation = write_camera_rgb(observation, views[name].sensor, value)
                return RenderResult(
                    start.frame, observation, replace(views[request.sensor_id], native={}),
                    self._provenance(start, request.environment, simulator_runtime))

    def components(self) -> Sequence[ComponentDescriptor]:
        self._check()
        return self.services.capabilities.components

    def scorers(self) -> Sequence[ScorerDescriptor]:
        self._check()
        return self.services.capabilities.scorers

    def _avstack(self, resources: ExitStack) -> AVStack:
        adapter = self.services.driving_adapter
        if adapter is None:
            raise ValueError("This service has no driving-system adapter")
        avstack = _fresh(adapter.factory(self.seed))
        resources.callback(_close, avstack)
        return avstack

    def evaluate(self, component_id: str, inputs: Any) -> ComponentResult:
        adapter = self.services.driving_adapter
        if adapter is None or component_id not in adapter.component_bindings:
            raise ValueError(f"Unavailable component: {component_id!r}")
        descriptor, call = adapter.component_bindings[component_id]
        with self._operation() as resources:
            value = call(self._avstack(resources), inputs)
            return ComponentResult(value, deepcopy(descriptor), {"seed": self.seed})

    def score(self, scorer_id: str, inputs: Any, target: Any, require_grad: bool = False) -> ScoreResult:
        adapter = self.services.driving_adapter
        if adapter is None or scorer_id not in adapter.scorer_bindings:
            raise ValueError(f"Unavailable scorer: {scorer_id!r}")
        descriptor, call = adapter.scorer_bindings[scorer_id]
        if require_grad and not descriptor.differentiable_inputs:
            raise ValueError(f"Scorer {scorer_id!r} does not support gradients")
        with self._operation() as resources:
            value = call(self._avstack(resources), inputs, target)
            if require_grad and not getattr(value, "requires_grad", False):
                raise ValueError(f"Scorer {scorer_id!r} returned a detached/non-differentiable value")
            return ScoreResult(value, deepcopy(descriptor), {"seed": self.seed, "require_grad": require_grad})

    def rollout(self, request: CandidateRequest) -> RunLog:
        self._check()
        if request.seed != self.seed:
            raise ValueError("Candidate seed must match its scope")
        segment = request.segment
        dataset = self.services.dataset
        frames = dataset.decision_window(segment.scenario_id, segment.start, request.decision_steps)
        if frames[-1] != segment.stop:
            raise ValueError("Candidate stop does not match the requested terminal frame")
        start = self._scenario_start(segment.scenario_id, segment.start, request.environment)
        try:
            initialization = self._initialization(start, request.environment)
        except BaseException as error:
            # Frame data generation precedes runner construction, but failed generation evidence
            # still belongs to the requested run and must survive without a live simulator runtime.
            recorder = None
            try:
                from avsectester.runlog.codecs import default_codecs

                recorder = Recorder(path=request.output_path, codecs=default_codecs(), metadata={
                    **self._provenance(start, request.environment), "candidate": asdict(segment),
                    "decision_steps": request.decision_steps,
                })
                recorder.record_event("failure", phase="frame_data_generation",
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
            simulator_runtime = self._simulator_runtime(resources, start)
            avstack = self._avstack(resources)
            attacks: tuple[Any, ...] = ()
            if request.attack is not None:
                attack_runtime = _fresh(request.attack.create_runtime(segment, request.artifacts))
                attacks = (attack_runtime,)
            return ClosedLoopRunner().run(simulator_runtime, avstack, RunRequest(
                request.decision_steps, hooks=HookSet(seed=self.seed),
                attacks=attacks, recording=request.recording,
                output_path=request.output_path, start=start, initialization=initialization,
                environment=request.environment, deployment=self.services.deployment,
                metadata={**self._provenance(start, request.environment, simulator_runtime), "candidate": asdict(segment)},
            ), release_owned_resources=resources.close)


@dataclass(frozen=True)
class ScenarioView:
    scenario_id: str
    services: ClosedLoopServices
    seed: int

    @property
    def frames(self) -> Sequence[FrameReference]:
        return self.services.dataset.frames(self.scenario_id)

    def frame_data(self, frame: FrameReference) -> FrameData:
        """Resolve recorded data or a preparation-owned cache, without mutating Dataset."""
        with self.services.session(self.seed) as scope:
            return scope.frame_data(self.scenario_id, frame)
