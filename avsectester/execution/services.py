"""The metabuilder and the offline phase share one isolated closed-loop service boundary.

Rendering and rollouts construct independent resources. Forward sensor scans without attack
actions share one runtime within a session. This module implements no separate simulator clock,
driving loop, scoring objective or selector.
"""
from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from threading import Lock
from typing import Any, TYPE_CHECKING

from avsectester.attack import AttackSegment
from avsectester.attack.runtime import AttackAction, AttackRuntime, OnlineContext, SensorAttackAction, WorldAttackAction
from avsectester.execution.attack_dispatch import chain_sensor_actions
from avsectester.closed_loop.avstacks.base import AVStack
from avsectester.closed_loop.avstacks import driving_adapter
from avsectester.execution.runner import ClosedLoopRunner, RecordingPolicy, RunRequest
from avsectester.datasets.base import Dataset, FrameReference, FrameData, ScenarioStart
from avsectester.data_types import Unavailable
from avsectester.validation import require_nonnegative_integer
from avsectester.closed_loop.environment import SensorEnvironment, normalize_environment, validate_environment
from avsectester.data_types import Observation
from avsectester.execution.frame_data_cache import FrameDataCache
from avsectester.closed_loop.simulators.base import Initialization, FrameDataSequence
from avsectester.runlog import Recorder, RunLog
from avsectester.closed_loop.hooks import HookSet
from avsectester.closed_loop.simulators.base import DeploymentPlan, SensorView, Simulator, SimulatorRuntime

if TYPE_CHECKING:
    from avsectester.maps import ScenarioMap
    from avsectester.rendering.compositing import ObjectVisibility


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
        require_nonnegative_integer(self.seed, "seed")
        require_nonnegative_integer(self.decision_steps, "decision_steps")
        if self.decision_steps != self.segment.decision_steps:
            raise ValueError("Candidate decision_steps must equal its segment interval")
        if self.attack is None and self.artifacts is not None:
            raise ValueError("Candidate artifacts require an attack")
        if self.attack is not None and not callable(getattr(self.attack, "create_runtime", None)):
            raise TypeError("Candidate attack must provide create_runtime(segment, artifacts)")
        object.__setattr__(self, "environment", normalize_environment(self.environment))


@dataclass(frozen=True)
class SensorRequest:
    """Recorded sensor request with an optional camera selection."""
    scenario_id: str
    frame: FrameReference
    sensor_ids: tuple[str, ...]
    environment: Mapping[str, Any] = field(default_factory=dict)
    views: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "environment", normalize_environment(self.environment))


@dataclass(frozen=True)
class RenderRequest:
    """Capture ``sensor_id`` at ``frame`` with attack actions applied.

    World attack actions are applied at the simulator.setup stage, within the services'
    deployment plan, so they are visible in the capture. Sensor attack actions then change the
    captured data. AVStack attack actions need a rollout and are rejected.
    """

    scenario_id: str
    frame: FrameReference
    sensor_id: str
    actions: Sequence[AttackAction] = ()
    environment: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        actions = tuple(self.actions)
        if any(not isinstance(e, (WorldAttackAction, SensorAttackAction)) for e in actions):
            raise TypeError("Render requests take world and sensor attack actions only")
        object.__setattr__(self, "actions", actions)
        object.__setattr__(self, "environment", normalize_environment(self.environment))


@dataclass(frozen=True)
class SensorSample:
    """An environment-processed observation and optional detached raw sensor views."""

    frame: FrameReference
    observation: Observation
    provenance: Mapping[str, Any]
    views: Mapping[str, SensorView] = field(default_factory=dict)


@dataclass(frozen=True)
class RenderResult:
    """The changed capture and the requested sensor's view before its sensor actions.

    The view is detached: its ``native`` access is empty once the session operation ends.
    ``object_visibility`` retains each world's object visibility by object and camera ID
    from the same capture, before cleanup removes the inserted objects.
    """

    frame: FrameReference
    observation: Observation
    view: SensorView
    provenance: Mapping[str, Any]
    object_visibility: Mapping[str, Mapping[str, ObjectVisibility]] = field(default_factory=dict)


@dataclass(frozen=True)
class ComponentResult:
    """Outputs returned by an explicitly evaluated driving-system component."""
    value: Any
    descriptor: driving_adapter.ComponentDescriptor
    provenance: Mapping[str, Any]


@dataclass(frozen=True)
class ScoreResult:
    """One score per input returned by a named batch scorer."""
    value: Any
    descriptor: driving_adapter.ScorerDescriptor
    provenance: Mapping[str, Any]


@dataclass(frozen=True)
class ClosedLoopCapabilities:
    """Capabilities exposed by the dataset, simulator and driving adapter combination."""
    dataset_kinds: tuple[str, ...]
    sensor_data: bool
    render: bool
    insert_objects: bool
    rollout: bool
    components: tuple[driving_adapter.ComponentDescriptor, ...]
    scorers: tuple[driving_adapter.ScorerDescriptor, ...]


class _WorldSetup:
    """Apply world attack actions at simulator.setup when no driving run binds the runtime."""

    def __init__(self, simulator_runtime: SimulatorRuntime, actions: Sequence[WorldAttackAction],
                 resources: ExitStack) -> None:
        self.simulator_runtime, self.actions, self.resources = simulator_runtime, tuple(actions), resources

    def emit(self, stage: str, value: Any, sensor: str | None = None) -> Any:
        if stage == "simulator.setup":
            for action in self.actions:
                action.apply(self.simulator_runtime, self.resources)
        return value


def ensure_fresh(resource: Any) -> Any:
    """Claim a newly constructed resource; reject reused or active mutable instances."""
    if getattr(resource, "_services_used", False) or getattr(resource, "_dispatcher", None) is not None:
        raise RuntimeError("Resource factory returned a reused or active instance")
    resource._services_used = True
    return resource


def create_simulator_runtime(simulator: Simulator, seed: int | None,
                             resources: ExitStack) -> SimulatorRuntime:
    """Create a fresh simulator runtime and register its close on ``resources``."""
    simulator_runtime = simulator.create_runtime(seed)
    if not isinstance(simulator_runtime, SimulatorRuntime):
        raise TypeError("Simulator.create_runtime must return a SimulatorRuntime")
    ensure_fresh(simulator_runtime)
    resources.callback(simulator_runtime.close)
    return simulator_runtime


def create_avstack(adapter: driving_adapter.DrivingAdapter, seed: int, resources: ExitStack) -> AVStack:
    """Create a fresh driving system and register its close on ``resources``."""
    avstack = adapter.factory(seed)
    if not isinstance(avstack, AVStack):
        raise TypeError("DrivingAdapter.factory must return an AVStack")
    ensure_fresh(avstack)
    resources.callback(avstack.close)
    return avstack


def stop_unstarted_attacks(runtimes: Sequence[Any], frame: FrameReference,
                           error: BaseException) -> None:
    """Stop attack runtimes the runner never started; cleanup failures become notes on ``error``."""
    for runtime in reversed([item for item in runtimes if isinstance(item, AttackRuntime)]):
        try:
            runtime.stop(OnlineContext(0, "stop", frame.timestamp, frame.position))
        except BaseException as cleanup:
            error.add_note(f"Attack runtime cleanup also failed: {cleanup!r}")


class ClosedLoopServices:
    """Factory for isolated scenario and closed-loop sessions using shared frame-data caching."""
    def __init__(
        self, dataset: Dataset, simulator: Simulator,
        driving_adapter: driving_adapter.DrivingAdapter | None = None, frame_data_cache: FrameDataCache | None = None,
        *, deployment: DeploymentPlan = DeploymentPlan(),
    ) -> None:
        """Own session resources; rendering and rollouts always use fresh simulator runtimes."""
        self.dataset = dataset
        if not isinstance(simulator, Simulator):
            raise TypeError("Services require a configured Simulator")
        if (driving_adapter is not None and driving_adapter.decision_time_step is not None
                and simulator.decision_time_step is not None
                and driving_adapter.decision_time_step != simulator.decision_time_step):
            raise ValueError("AVStack decision_time_step does not match simulator.decision_time_step")
        self.simulator = simulator
        self.deployment = deployment
        self.driving_adapter = driving_adapter
        self.frame_data_cache = frame_data_cache if frame_data_cache is not None else FrameDataCache()
        self._sessions: set[ClosedLoopSession] = set()
        self._closed = False

    def __enter__(self) -> ClosedLoopServices:
        if self._closed:
            raise RuntimeError("Closed-loop services are closed")
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def close(self) -> None:
        """Close all sessions, including sessions a metabuilder left open on error."""
        resources = ExitStack()
        for session in tuple(self._sessions):
            resources.callback(session.close)
        try:
            resources.close()
        finally:
            self._closed = True

    @property
    def capabilities(self) -> ClosedLoopCapabilities:
        """Capabilities available without constructing a simulator runtime."""
        driver = self.driving_adapter
        return ClosedLoopCapabilities(
            self.simulator.dataset_kinds, True, self.simulator.can_render,
            self.simulator.can_insert_objects,
            driver is not None,
            tuple(deepcopy(pair[0]) for pair in driver.component_bindings.values()) if driver else (),
            tuple(deepcopy(pair[0]) for pair in driver.scorer_bindings.values()) if driver else (),
        )

    def session(self, seed: int) -> ClosedLoopSession:
        """Create an isolated session with fresh world and driving history for ``seed``."""
        require_nonnegative_integer(seed, "seed")
        if self._closed:
            raise RuntimeError("Closed-loop services are closed")
        scope = ClosedLoopSession(self, seed)
        self._sessions.add(scope)
        return scope

    def scenario(self, scenario_id: str, *, seed: int) -> ScenarioView:
        """Create a selection view for the requested scenario and seed."""
        self.dataset.frames(scenario_id)  # validate without generating anything
        require_nonnegative_integer(seed, "seed")
        return ScenarioView(scenario_id, self, seed)


class ClosedLoopSession:
    """Owned simulator and driving-system services for selection, rendering, scoring and rollouts."""
    def __init__(self, services: ClosedLoopServices, seed: int) -> None:
        self.services, self.seed = services, seed
        self._closed = False
        self._busy = Lock()
        self._scan_resources = ExitStack()
        self._scan_runtime: SimulatorRuntime | None = None
        self._scan_key: tuple[str, Mapping[str, Any]] | None = None
        self._scan_position = -1
        self._scan_environment: SensorEnvironment | None = None
        self._scoring_resources = ExitStack()
        self._scoring_avstack: AVStack | None = None

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
        """Close session resources and release world ownership."""
        if not self._busy.acquire(blocking=False):
            raise RuntimeError("Cannot close a scope while its operation is active")
        try:
            if not self._closed:
                try:
                    cleanup = ExitStack()
                    cleanup.callback(self._scoring_resources.close)
                    cleanup.callback(self._discard_scan)
                    try:
                        cleanup.close()
                    finally:
                        self._scoring_avstack = None
                finally:
                    self._closed = True
                    self.services._sessions.discard(self)

        finally:
            self._busy.release()

    def _discard_scan(self) -> None:
        resources, self._scan_resources = self._scan_resources, ExitStack()
        self._scan_runtime, self._scan_key, self._scan_position = None, None, -1
        self._scan_environment = None
        resources.close()

    def _scenario_start(self, scenario: str, frame: FrameReference,
                        environment: Mapping[str, Any]) -> ScenarioStart:
        validate_environment(environment, self.services.simulator)
        return self.services.dataset.scenario_start(scenario, frame)

    def _simulator_runtime(self, resources: ExitStack, start: ScenarioStart, *,
                           scanning: bool = False) -> SimulatorRuntime:
        if not scanning:
            self._discard_scan()
        # Renders and rollouts always start fresh. Only an action-free frame scan is retained.
        self.services.simulator.validate_start(start)
        # Callbacks run in reverse order: close the runtime, then release its reservation, so no
        # other operation can reset this server while its actors are still being destroyed.
        reservation = ExitStack()
        resources.callback(reservation.close)
        simulator_runtime = create_simulator_runtime(self.services.simulator, self.seed, resources)
        reservation.enter_context(simulator_runtime.claim_world(self))
        return simulator_runtime

    def _sequence(self, scenario: str, environment: Mapping[str, Any],
                  deployment: DeploymentPlan | None = None) -> FrameDataSequence:
        deployment = self.services.deployment if deployment is None else deployment
        environment = normalize_environment(environment)
        dataset = self.services.dataset
        frames = tuple(dataset.frames(scenario))
        start = self._scenario_start(scenario, frames[0], environment)
        key = FrameDataCache.key(start, {
            **self.services.simulator.identity, "environment": dict(environment),
            "deployment_ticks": deployment.ticks,
        }, self.seed)

        def generate() -> FrameDataSequence:
            with self._operation() as resources:
                simulator_runtime = self._simulator_runtime(resources, start)
                with simulator_runtime.exclusive():
                    sequence = simulator_runtime.generate_frame_data(
                        start, frames, world=environment["world"], deployment=deployment)
                if tuple(record.frame for record in sequence.frames) != frames:
                    raise ValueError("Frame data generation must preserve every dataset frame")
                return sequence

        return self.services.frame_data_cache.get_or_generate(key, generate)

    def can_start_at(self, position: int) -> bool:
        """Check deployment and recorded-history requirements without starting a runtime."""
        from avsectester.closed_loop.simulators.base import can_deploy_at

        deployment_allowed = can_deploy_at(position, self.services.deployment,
                                            self.services.simulator.capture_ticks)
        adapter = self.services.driving_adapter
        required_history = max(self.services.simulator.past_steps,
                               adapter.past_steps if adapter is not None else 0)
        return deployment_allowed and position >= required_history

    def frame_data(self, scenario_id: str, frame: FrameReference,
                  environment: Mapping[str, Any] | None = None) -> FrameData:
        """Return recorded frame data through the shared cache."""
        self._check()
        record = self.services.dataset.frame_data(scenario_id, frame)
        if isinstance(record, Unavailable):
            return self._sequence(scenario_id, environment or {}).at(frame)
        if any(normalize_environment(environment).values()):
            raise ValueError("Recorded frame data describes the recorded environment")
        return record

    def initialization(self, start: ScenarioStart, environment: Mapping[str, Any],
                       deployment: DeploymentPlan | None = None) -> Initialization | None:
        """Resolve replay inputs for a selected start, using the preparation-owned cache."""
        self._check()
        if start.kind != "CarlaDataset":
            return None
        sequence = self._sequence(start.scenario_id, environment, deployment)
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

    def read_sensors(self, request: SensorRequest) -> SensorSample:
        """Read environment-processed sensor observations at the requested recorded frame.

        A different scenario/environment, a backwards frame request, render or rollout closes
        the scan. Repeated reads of a frame do not advance the simulator. Optional views retain
        the raw capture, before the sensor environment, and have no native runtime access.
        """
        if request.views and not self.services.simulator.can_render:
            raise NotImplementedError("The simulator does not provide sensor views")
        self._check()
        start = self._scenario_start(request.scenario_id, request.frame, request.environment)
        key = (start.scenario_id, request.environment)
        reusable = (self._scan_runtime is not None and self._scan_key == key
                    and request.frame.position >= self._scan_position
                    and self._scan_runtime.can_replay)
        if not reusable:
            self._discard_scan()
            initialization = self.initialization(start, request.environment)
        else:
            initialization = None
        with self._operation():
            try:
                if self._scan_runtime is None:
                    simulator_runtime = self._simulator_runtime(self._scan_resources, start, scanning=True)
                    with simulator_runtime.exclusive():
                        simulator_runtime.start_replay(start, initialization, world=request.environment["world"],
                                                      deployment=self.services.deployment)
                    self._scan_runtime = simulator_runtime
                    self._scan_key = key
                    self._scan_position = request.frame.position
                    self._scan_environment = SensorEnvironment(request.environment, self.seed)
                else:
                    simulator_runtime = self._scan_runtime
                    with simulator_runtime.exclusive():
                        simulator_runtime.advance_replay(request.frame.position - self._scan_position)
                    self._scan_position = request.frame.position
                observation = simulator_runtime.observe()
                assert self._scan_environment is not None
                observation = self._scan_environment.apply(observation, request.frame.position)
                missing = set(request.sensor_ids) - observation.sensor_data.keys()
                if missing:
                    raise ValueError(f"Unavailable sensors: {sorted(missing)}")
                observation.sensor_data = {key: observation.sensor_data[key] for key in request.sensor_ids}
                observation.calibration = {key: value for key, value in observation.calibration.items()
                                           if key in request.sensor_ids}
                views = {
                    sensor: deepcopy(replace(simulator_runtime.sensor_view(sensor), native={}))
                    for sensor in request.sensor_ids
                } if request.views else {}
                return SensorSample(start.frame, observation,
                                    self._provenance(start, request.environment, simulator_runtime),
                                    views)
            except BaseException as error:
                try:
                    self._discard_scan()
                except BaseException as cleanup:
                    error.add_note(f"Sensor scan cleanup also failed: {cleanup!r}")
                raise

    def render(self, request: RenderRequest) -> RenderResult:
        """Render the requested frame or viewpoint using this session."""
        if not self.services.simulator.can_render:
            raise NotImplementedError("The simulator does not provide sensor views")
        world = [e for e in request.actions if isinstance(e, WorldAttackAction)]
        sensor = [e for e in request.actions if isinstance(e, SensorAttackAction)]
        deployment = self.services.deployment
        if sum(action.setup_ticks for action in world) > deployment.ticks:
            raise ValueError("Render actions exceed the shared deployment tick budget")
        start = self._scenario_start(request.scenario_id, request.frame, request.environment)
        initialization = self.initialization(start, request.environment, deployment)
        with self._operation() as resources:
            simulator_runtime = self._simulator_runtime(resources, start)
            with simulator_runtime.exclusive():
                # Actions register their removal on resources, which close before the runtime.
                with simulator_runtime.bind_hooks(_WorldSetup(simulator_runtime, world, resources)):
                    simulator_runtime.reset(start, initialization, world=request.environment["world"],
                                            deployment=deployment)
                observation = simulator_runtime.observe()
                object_visibility = deepcopy({
                    object_id: simulator_runtime.object_visibility(object_id)
                    for object_id in simulator_runtime.state().objects
                })
                views = {request.sensor_id: simulator_runtime.sensor_view(request.sensor_id)}
                detached_view = deepcopy(replace(views[request.sensor_id], native={}))
                observation = chain_sensor_actions(observation, simulator_runtime, sensor, views=views)
                observation = SensorEnvironment(request.environment, self.seed).apply(
                    observation, request.frame.position)
                return RenderResult(
                    start.frame, observation, detached_view,
                    self._provenance(start, request.environment, simulator_runtime), object_visibility)

    def components(self) -> Sequence[driving_adapter.ComponentDescriptor]:
        """Return the available driving-system component descriptors."""
        self._check()
        return self.services.capabilities.components

    def scorers(self) -> Sequence[driving_adapter.ScorerDescriptor]:
        """Return the available driving-system scorer descriptors."""
        self._check()
        return self.services.capabilities.scorers

    def _avstack(self, resources: ExitStack) -> AVStack:
        adapter = self.services.driving_adapter
        if adapter is None:
            raise ValueError("This service has no driving-system adapter")
        return create_avstack(adapter, self.seed, resources)

    def _get_scoring_avstack(self) -> AVStack:
        """Retain the session's scoring models, never component or rollout history."""
        if self._scoring_avstack is None:
            self._scoring_avstack = self._avstack(self._scoring_resources)
        return self._scoring_avstack

    def evaluate(self, component_id: str, inputs: Any) -> ComponentResult:
        """Evaluate an explicitly bound driving-system component."""
        adapter = self.services.driving_adapter
        if adapter is None or component_id not in adapter.component_bindings:
            raise ValueError(f"Unavailable component: {component_id!r}")
        descriptor, call = adapter.component_bindings[component_id]
        with self._operation() as resources:
            value = call(self._avstack(resources), inputs)
            return ComponentResult(value, deepcopy(descriptor), {"seed": self.seed})

    def score(self, scorer_id: str, inputs: Any, targets: Any, require_grad: bool = False) -> ScoreResult:
        """Evaluate a batch, passing inputs and targets unchanged to the bound callback."""
        adapter = self.services.driving_adapter
        if adapter is None or scorer_id not in adapter.scorer_bindings:
            raise ValueError(f"Unavailable scorer: {scorer_id!r}")
        descriptor, call = adapter.scorer_bindings[scorer_id]
        if require_grad and not descriptor.differentiable_inputs:
            raise ValueError(f"Scorer {scorer_id!r} does not support gradients")
        with self._operation():
            value = call(self._get_scoring_avstack(), inputs, targets)
            if require_grad and not getattr(value, "requires_grad", False):
                raise ValueError(f"Scorer {scorer_id!r} returned a detached/non-differentiable value")
            return ScoreResult(value, deepcopy(descriptor), {"seed": self.seed, "require_grad": require_grad})

    def rollout(self, request: CandidateRequest) -> RunLog:
        """Execute one candidate rollout from its recorded start."""
        self._check()
        if request.seed != self.seed:
            raise ValueError("Candidate seed must match its scope")
        segment = request.segment
        dataset = self.services.dataset
        frames = dataset.decision_window(segment.scenario_id, segment.start, request.decision_steps)
        if frames[-1] != segment.stop:
            raise ValueError("Candidate stop does not match the requested terminal frame")
        start = self._scenario_start(segment.scenario_id, segment.start, request.environment)
        attacks: tuple[Any, ...] = ()
        if request.attack is not None:
            runtime = ensure_fresh(request.attack.create_runtime(segment, request.artifacts))
            attacks = (runtime,)
        deployment = self.services.deployment
        try:
            if sum(runtime.setup_ticks for runtime in attacks) > deployment.ticks:
                raise ValueError("Candidate attack exceeds the shared deployment tick budget")
            initialization = self.initialization(start, request.environment, deployment)
        except BaseException as error:
            stop_unstarted_attacks(attacks, start.frame, error)
            # Frame data generation precedes runner construction, but failed generation evidence
            # still belongs to the requested run and must survive without a live simulator runtime.
            recorder = None
            try:
                recorder = Recorder(path=request.output_path, metadata={
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
        runner_owns_attacks = False
        try:
            with self._operation() as resources:
                simulator_runtime = self._simulator_runtime(resources, start)
                avstack = self._avstack(resources)
                run_request = RunRequest(
                    request.decision_steps, hooks=HookSet(seed=self.seed),
                    attacks=attacks, recording=request.recording,
                    output_path=request.output_path, start=start, initialization=initialization,
                    environment=request.environment, deployment=deployment,
                    metadata={**self._provenance(start, request.environment, simulator_runtime), "candidate": asdict(segment)},
                )
                runner_owns_attacks = True
                return ClosedLoopRunner().run(simulator_runtime, avstack, run_request,
                                              release_owned_resources=resources.close)
        except BaseException as error:
            if not runner_owns_attacks:
                stop_unstarted_attacks(attacks, start.frame, error)
            raise


MAX_SELECTION_REASONS = 100
OTHER_SELECTION_REASONS = "Other selection reasons"


@dataclass(frozen=True)
class ScenarioView:
    """Recorded scenario and selection services, with candidate-rejection reasons."""
    scenario_id: str
    services: ClosedLoopServices
    seed: int
    _reasons: dict[str, int] = field(default_factory=dict, repr=False, compare=False)

    def record_reason(self, text: str) -> None:
        """Attach a user-supplied selection explanation to this scenario's preparation record."""
        if not isinstance(text, str) or not text.strip():
            raise ValueError("A selection reason must be nonempty text")
        if text not in self._reasons and len(self._reasons) >= MAX_SELECTION_REASONS:
            text = OTHER_SELECTION_REASONS
        self._reasons[text] = self._reasons.get(text, 0) + 1

    @property
    def reasons(self) -> tuple[str, ...]:
        """Bounded, insertion-ordered reason counts; repeated candidates add no entries."""
        return tuple(f"{reason} (count={count})" for reason, count in self._reasons.items())

    @property
    def frames(self) -> Sequence[FrameReference]:
        """Ordered decision-frame references available to this selection view."""
        return self.services.dataset.frames(self.scenario_id)

    def frame_data(self, frame: FrameReference) -> FrameData:
        """Resolve recorded data or a preparation-owned cache, without mutating Dataset."""
        with self.services.session(self.seed) as scope:
            return scope.frame_data(self.scenario_id, frame)

    def scenario_map(self) -> ScenarioMap | Unavailable:
        """Read this scenario's semantic map without starting a simulator."""
        return self.services.dataset.scenario_map(self.scenario_id)


__all__ = [
    'CandidateRequest',
    'SensorRequest',
    'RenderRequest',
    'SensorSample',
    'RenderResult',
    'ComponentResult',
    'ScoreResult',
    'ClosedLoopCapabilities',
    'ClosedLoopServices',
    'ClosedLoopSession',
    'ScenarioView',
]
