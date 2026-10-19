# Python API reference

> **Audience:** Users calling AVSecTester from Python.  
> **Outcome:** Look up public classes and functions by module, with their signatures and return values.  
> **Prerequisites:** [Architecture](../concepts/architecture.md).

This reference lists the public API by module. Signatures are copied from the code. A bare `*`
starts the keyword-only parameters. Defaults written as `{}` or `[]` in tables are per-instance
`field(default_factory=...)` values. Usage belongs to the linked guides. Current experiment
concepts are defined in
[Architecture](../concepts/architecture.md). The legacy intervention and prepared-scenario
interfaces are documented separately below.

The package root `avsectester` defines only `__version__` and has no `__all__`. Import every name
from its module. The packages `avsectester.scenarios`, `avsectester.rendering`,
`avsectester.evaluation`, `avsectester.closed_loop.simulators`, `avsectester.closed_loop.avstacks`
and `avsectester.attacks` re-export selected names through `__all__`. The sections below note them.

## Experiment preparation and execution

`avsectester.experiment` exports the experiment API below. Configuration fields and YAML
constructor specifications are described in
[Experiment configuration (Python API)](configuration.md#experiment-configuration-python-api).

```python
from avsectester.experiment import Experiment, PreparedExperiment

prepared = Experiment.prepare("experiment.yaml")
result = Experiment.execute(prepared)
prepared.save("results/prepared")

# Reuse saved metadata and artifacts without selection or optimization.
restored = PreparedExperiment.load("experiment.yaml", "results/prepared")
```

| Function or method | Returns or does |
|---|---|
| `load_configuration(config: ExperimentConfiguration \| Mapping[str, Any] \| str \| Path) -> ExperimentConfiguration` | Loads a Python configuration or YAML file, resolves constructor specifications and validates fields. Does not instantiate an AVStack or simulator runtime |
| `Experiment.prepare(config: ExperimentConfiguration \| Mapping[str, Any] \| str \| Path) -> PreparedExperiment` | Calls each attack's `build_metadata` once per scenario, validates its segments, then calls `run_offline` once per attack. Closes preparation services and saves preparation data when `output` is configured |
| `PreparedExperiment.load(config, directory: str \| Path) -> PreparedExperiment` | Class method. Validates the saved configuration summary against the supplied configuration, then loads metadata, decisions and artifacts without calling `build_metadata` or `run_offline` |
| `Experiment.execute(prepared: PreparedExperiment) -> ExperimentResult` | Runs named treatments from selected starts with fresh simulator, AVStack and attack instances. Reuses prepared artifacts and records each result as a `RunLog` |
| `PreparedExperiment.save(directory: str \| Path) -> Path` | Writes selected segments, selection decisions and each attack's artifacts. Does not serialize model instances, factories or live resources |

`ExperimentConfiguration` and `TreatmentConfiguration` are frozen dataclasses:

```python
# avsectester.experiment.config
class TreatmentConfiguration:
    name: str
    attack_name: str | None = None
    environment: Mapping[str, Any] = {}

class ExperimentConfiguration:
    dataset: Dataset
    simulator: Simulator
    avstack: DrivingAdapter
    attacks: Mapping[str, Attack] = {}
    treatments: tuple[TreatmentConfiguration, ...] = ()
    seeds: tuple[int, ...] = (0,)
    decision_steps: int | None = None
    environment: Mapping[str, Any] = {}
    recording: RecordingPolicy = RecordingPolicy()
    output: Path | None = None
    strict: bool = False
    hooks: HookSet | Callable[[int], HookSet] | None = None
    clean_hooks: HookSet | Callable[[int], HookSet] | None = None
```

Empty `treatments` creates a clean treatment and one treatment per configured attack. Names must
be unique. Each non-clean treatment references one configured attack. `decision_steps` limits
execution length without changing the selected metadata. `strict=False` retains selection and
execution failures and continues, while `strict=True` propagates them.
An offline preparation error propagates in either mode, because that attack has no usable artifacts.
Preparation requires at least one attack to select segments. Use `ClosedLoopRunner` for a clean
run with an explicitly supplied start.

The results contain no active services or simulator runtime:

| Class or property | Fields or meaning |
|---|---|
| `PreparedExperiment` | `configuration`, `metadata: Mapping[str, tuple[AttackSegment, ...]]`, `artifacts: Mapping[str, Any]`, `selections: tuple[SelectionDecision, ...]` |
| `SelectionDecision` | `attack_name`, `scenario_id`, `status: Literal["selected", "not_selected", "error"]`, `reasons: tuple[str, ...]`, `error: str \| None` |
| `TreatmentResult` | `attack_name`, `segment_index`, `segment`, `treatment`, `seed`, `log: RunLog`, `error: str \| None` |
| `TreatmentResult.group` | `(attack_name, segment_index, seed)` identifying the comparison group |
| `ExperimentResult` | `runs: tuple[TreatmentResult, ...]`, `selections: tuple[SelectionDecision, ...]` |
| `ExperimentResult.groups` | Maps each comparison group to its named treatment logs |
| `ExperimentResult.failures` | Treatment results whose logs did not complete |
| `ExperimentResult.selection_errors` | Selection decisions with status `error` |

Preparation uses the first configured seed for each attack's selection and runs its offline
phase once. The metabuilder does not automatically inherit the experiment's base environment;
it can request an environment explicitly through closed-loop services. Prepared metadata and
artifacts are reused for the other seeds. At each segment's start, execution checks its role IDs
against that seed's actors and records a failure if any are absent; it does not reselect.

Each attack selects independently. A clean treatment runs on that attack's selected segments.
Different attacks' selections are not silently combined or treated as the same comparison group.
Each run uses its experiment seed for hooks and sensor-environment transforms. Sensor randomness
also uses the absolute dataset frame position (`segment.start.position + decision_index`),
so offline and online requests at the same position share the same realization. Directly supplied
`HookSet` objects are copied without copying their handlers, whose state must reset for each run.
`ExperimentConfiguration.configuration_summary` computes an independent summary on demand;
configuration construction does not inspect attack parameters or enumerate dataset scenarios.
Saving and loading request it for the dataset type and scenario list, simulator identity, and
each attack's class, configured parameters and environment. Saving records it in `prepared.json`
(schema version 2). Loading rejects mismatches before reading artifacts. Output location and
recording policy can change without invalidating preparation. Models, factories and live resources
are not serialized.

If a Python attack cannot describe its parameters, saving emits a `RuntimeWarning`, records its
type and environment, and sets `parameters: null` and `parameters_verified: false`. Its metadata
and artifacts are still saved. Loading rejects an unverified saved or supplied description rather
than treating matching class names as compatible parameters. Implement
`Attack.configuration_parameters()` and save again to enable validated loading.

The existing CLI still calls `avsectester.legacy_experiment`, documented under
[Run functions](#run-functions).

## Data plane

`avsectester.closed_loop.datatypes` holds the messages between the simulator and the driving system
and the run records. It imports no simulator or model package. Sensor and state payloads keep their
adapter-specific types.

| Class | Constructor | Meaning |
|---|---|---|
| `Control` | `Control(throttle: float = 0.0, steer: float = 0.0, brake: float = 0.0, trajectory: Any = None)` | The command. Actuator commands use `throttle`, `steer` and `brake`. Trajectory commands use `trajectory` in the selected adapter's coordinates and time convention |
| `Observation` | `Observation(t: float, frame: int, sensor_data: dict[str, Any] = {}, calibration: dict[str, Any] = {}, vehicle_state: Any = None, ego_speed: float = 0.0)` | The model-visible input. `sensor_data` and `calibration` are keyed by sensor ID. `vehicle_state` and `ego_speed` are the estimated state and the reported speed in m/s |
| `ActorState` | `ActorState(pose: Any, velocity: tuple[float, float, float] \| None = None, category: str \| None = None, extent: tuple[float, float, float] \| None = None)` | Dataset record in `avsectester.datasets`. `pose` is a world-from-actor transform; velocity is in world axes. `extent` gives full `(length, width, height)` in metres, with finite nonnegative dimensions; `None` means unknown |
| `WorldSnapshot` | `WorldSnapshot(t: float, frame: int, vehicle_state: Any = None, ego_speed: float = 0.0, actors: Mapping[str, ActorState] = {})` | Frozen. An independent snapshot of the physical ego state and, in `actors`, the other actors' states by stable ID. Editing it never moves a vehicle |
| `StateEstimate` | `StateEstimate(vehicle_state: Any = None, ego_speed: float = 0.0)` | Localization output. Its fields replace the observation's state before the `observation` stage |
| `FrameRecord` | `FrameRecord(frame: int = 0, t: float = 0.0, n_detections: int = 0, speed: float = 0.0, throttle: float = 0.0, brake: float = 0.0, steer: float = 0.0)` | One driving step. `frame` is the zero-based driving-step index. `t` and `speed` are the physical time and speed after the step. The actuator fields are the executed command. `n_detections` is optional stack telemetry |
| `RecordedOutput` | `RecordedOutput(name: str, value: Any, step: int, stage: str, frame: int, t: float, sensor: str \| None = None)` | Frozen. A value recorded by `context.record_output`. `step` is the driving-step index. `frame` and `t` describe the physical world at the callback |
| `Trace` | `Trace(records: list[FrameRecord] = [], replay_scenario: dict \| None = None, outputs: list[RecordedOutput] = [], provenance: dict[str, Any] = {}, run_log: Any = None)` | Compatibility view of one run; `run_log` references the captured RunLog on shared-runner results |

| Member | Returns or does |
|---|---|
| `Observation.set_calibration(sensor_id: str, calibration: Any) -> None` | Replaces `calibration[sensor_id]` and, when the payload has a `calibration` attribute, the payload's calibration. Raises `KeyError` for a sensor absent from `sensor_data` |
| `Trace.final_speed` | Speed of the last record, or `0.0` without records |
| `Trace.peak_speed` | Highest recorded speed, or `0.0` |
| `Trace.braking_frames` | Number of records with `throttle == 0.0` and `brake > 0.0` |
| `Trace.mean_detections` | Mean `n_detections` over the records |
| `Trace.run_log` | Source `RunLog` when returned by the shared runner; `None` by default for manually constructed traces |

`record_run(..., collect=True)` also sets a `frames` attribute on the returned `Trace`. Reading
records, outputs and provenance is described in
[Driving traces](../guides/evaluation.md#driving-traces). Copy semantics of observations are in
[Native resources and copying](runtime.md#native-resources-and-copying).

## Driving loop and runners

### Adapter interfaces

`avsectester.closed_loop.simulator` defines the simulator interface, `Simulator` and
`SimulatorRuntime` (see [Simulator](#simulator)), and the legacy single-class `WorldBackend`, which
the runner still accepts. `avsectester.closed_loop.avstack` defines `AVStack`.
[Write a custom simulator adapter](../guides/simulators.md#write-a-custom-simulator-adapter) and
[Use a custom driving system](../guides/driving-systems.md#use-a-custom-driving-system) describe
how to implement them.

Legacy `WorldBackend` members:

| Member | Returns or does |
|---|---|
| `reset() -> Observation` | Abstract. Builds the world, spawns the ego and traffic and returns the first observation |
| `step(control: Control) -> Observation` | Abstract. Executes the command, advances the world and returns the next observation |
| `close() -> None` | Releases simulator resources. The default does nothing |
| `ground_truth() -> WorldSnapshot` | An independent physical-state snapshot. The default copies the state of the last raw `reset` or `step` observation and raises `RuntimeError` before the first one |
| `copy_observation(observation: Observation) -> Observation` | Detaches model-visible data from simulator state. The default is `deepcopy` |
| `command_modes` | Property, `frozenset[str] \| None`. Accepted command modes (`actuator`, `trajectory`). `None`, the default, means undeclared |
| `validate_command(command: Control) -> None` | Calls `validate_command_modes(command, self.command_modes)` |
| `configure_conditions(world: Mapping[str, Any]) -> None` | Applies world conditions to later resets. The default raises `ValueError` for a nonempty mapping |
| `prepare_clean_attack_pair() -> None` | Prepares consecutive resets to replay one initial scene. The default does nothing |
| `setup_ticks: int = 0` | Class attribute. Simulator-managed world-setup ticks. See [World setup ticks](runtime.md#world-setup-ticks) |
| `supported_stages` | Property, `frozenset[str]`. Stages the adapter emits beyond the core stages. The default is empty |
| `native` | Property, `Mapping[str, Any]`. Native resources available to handlers. The default is empty |
| `bind_hooks(dispatcher: HookDispatcher)` | Context manager that binds the adapter to the hook dispatcher of one run. Raises `RuntimeError` if it is already bound |
| `_emit(stage: str, value: Any, sensor: str \| None = None) -> Any` | Passes `value` through the handlers at `stage` of the bound run and returns the result. Returns `value` unchanged when no run is bound |

`AVStack` members:

| Member | Returns or does |
|---|---|
| `__call__(observation: Observation) -> Control` | Abstract. Decides the command |
| `reset(observation: Observation) -> None` | Optional warmup with the first processed observation. The default does nothing |
| `command_modes` | Property, `frozenset[str] \| None`. Produced command modes, or `None` when undeclared |
| `models` | Property, `Mapping[str, Any]`. Loaded native models available to handlers. The default is empty |
| `supported_stages`, `native`, `bind_hooks`, `_emit` | As for `WorldBackend` |

```python
# avsectester.closed_loop.datatypes
def validate_command_modes(command: Control, modes: frozenset[str] | None) -> None: ...

# avsectester.closed_loop.avstack
def validate_command_compatibility(
    simulator_runtime: SimulatorRuntime | WorldBackend, avstack: AVStack, hooks: HookSet | None = None,
) -> None: ...
class CommandCompatibilityError(ValueError): ...
```

`validate_command_modes` raises `TypeError` when `command` is not a `Control`. It raises
`ValueError` when the command sets a trajectory or a nonzero actuator field that `modes` excludes.
`validate_command_compatibility` raises `CommandCompatibilityError` when the AVStack declares a
mode the simulator does not accept. It skips the check when either side is undeclared or `hooks`
has a `command` hook. The rules are in [Command compatibility](runtime.md#command-compatibility).

### Simulator

A simulator has two parts in `avsectester.closed_loop.simulator`:

- `Simulator` is the configured simulator: settings and declarations, without world state. It can
  be reused. The supplied ones are `CarlaSimulator`, `NuRecSimulator` and `RecordedReplaySimulator`.
- `SimulatorRuntime` is one live world of a simulator: world state, clock, current capture and
  native connections. `Simulator.create_runtime(seed=None)` creates fresh worlds for runs, renders
  and rollouts; sequential effect-free reads within a service session may reuse a scan runtime. Its creator
  closes it, also with `with simulator.create_runtime() as runtime:`.
  `reset` starts a run and may be called again for a further run on the same world. The supplied
  runtimes are `CarlaSimulatorRuntime`, `NuRecSimulatorRuntime` and `RecordedReplaySimulatorRuntime`.

`Simulator` members:

| Member | Returns or does |
|---|---|
| `create_runtime(seed: int \| None = None) -> SimulatorRuntime` | Abstract. A new live world with this configuration. CARLA uses `seed` as the client seed when the scenario sets none; NuRec and recorded replay are deterministic and ignore it |
| `validate_start(start) -> None` | Rejects a `ScenarioStart` of another dataset kind or time step before any world exists |
| `validate_world_settings(world: Mapping[str, Any]) -> None` | Checks supported native world keys and values without opening a runtime. Custom simulators override it for their own value rules; world-dependent checks remain in runtime initialization |
| `dataset_kinds`, `decision_time_step`, `capture_ticks`, `can_render`, `command_modes`, `supported_stages`, `supported_world_settings`, `identity` | Accepted `ScenarioStart` kinds (dataset class names such as `NuRecDataset`), decision step, ticks between `simulator.setup` and the first capture, sensor-view support, accepted command modes, emitted stages, supported native world-environment keys, and the identity used in provenance and cache keys |

`SimulatorRuntime` members (declarations are read from its `simulator`):

| Member | Returns or does |
|---|---|
| `reset(start=None, initialization=None, *, deployment=None, world=None) -> None` | Starts a run from a `ScenarioStart` or the configured origin: native initialization, pre-roll, the deployment window (`simulator.setup` and its ticks), the capture ticks and the first capture. `world` contains native world settings only, such as `{"weather": "ClearNoon"}`; the enclosing `{world, sensors}` format and `sensors` are rejected. See [selected-origin initialization](../guides/scenarios.md#selected-origin-deployment-and-reset-ownership) |
| `step(command: Control) -> None` | Emits `simulator.step.pre`, advances one decision, emits `simulator.step.post` and captures |
| `observe() -> Observation` | Independent copy of the current capture, without a tick or repeated render hooks |
| `state() -> WorldSnapshot` | Independent physical state of the ego and, in `actors`, the other actors, without advancing time. Available after a reset and during a reset or step, so stage handlers can read it |
| `clock` | Frozen `ClockSnapshot(t: float, frame: int)` of physical simulator time and frame |
| `exclusive()` | Context manager reserving the world in this process; nested operations raise `RuntimeError`. `resource_owner`, when set, owns the reservation |
| `sensor_view(sensor: str) -> SensorView` | The view of the current capture of `sensor`. The base raises `NotImplementedError`; CARLA and NuRec implement it |
| `control_actor(actor_id: str, control: Control) -> None` | Drives another actor with an actuator command from the next world step; it stops following its prescribed or traffic-manager motion. CARLA only; the ego is rejected |
| `set_actor_state(actor_id: str, state: ActorState) -> None` | Places another actor at a canonical world state, visible from the next capture; it stops following its prescribed motion. CARLA only; the ego is rejected |
| `setup_tick() -> None` | Consumes one deployment tick; only during `simulator.setup` |
| `generate_frame_data(start, frames, *, world=None, deployment=None) -> FrameDataSequence` | Records actual frame data by replaying the start's prescribed motion over consecutive `frames` beginning at `start.frame`, without hooks, attacks or a driving system. `world` contains native world settings only. Shared by all runtimes; a runtime supports it by implementing `_replay_tick` and `_capture_frame_data` (CARLA and recorded replay). Used for datasets without frame data, such as CARLA recipes |
| `close() -> None` | Releases the world and native connections |

Custom `Simulator` classes declare `supported_world_settings` (empty when no native overrides
are supported) and implement value checks in `validate_world_settings`. Concrete runtimes implement
`_initialize_world`, `_advance_native`, `_observe` and `_state`, and cannot override `reset` or
`step`. Their reset receives only world settings; sensor-environment transforms belong to the
runner and services.

```python
# avsectester.closed_loop.simulator
@dataclass(frozen=True)
class SensorView:
    sensor: str                         # key of this capture in Observation.sensor_data
    data: Any                           # captured RGB uint8 (H, W, 3)
    camera: Camera                      # project/unproject; pinhole or f-theta
    cam_from_world: np.ndarray          # world to optical camera axes
    actors: Mapping[str, ActorPose]     # world poses by stable actor ID
    ego: ActorPose                      # the ego cuboid
    time: float
    frame: int
    depth: np.ndarray | None = None     # optical-axis depth in metres, when measured
    native: Mapping[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class DeploymentPlan:
    ticks: int = 0
```

A `SensorView` contains no insertion rendering. Attack effects render into it; see
[Insertion rendering](#insertion-rendering). `native` gives simulator-specific access and gives up
portability. Legacy `WorldBackend` implementations run in `ClosedLoopRunner` through an internal
adapter with the same surface, which supports plain resets only and provides no sensor views.
NuRec checkpoint restoration captures immediately; subsequent observations only copy that capture.

### Closed-loop requests

`avsectester.closed_loop.runner` provides the shared scheduler and direct-run configuration:

```python
@dataclass(frozen=True)
class RecordingPolicy:
    sensor_ids: tuple[str, ...] = ()
    representations: tuple[str, ...] = ("model_input",)
    every_n_steps: int = 1

@dataclass(frozen=True)
class RunRequest:
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

class ClosedLoopRunner:
    def run(
        self,
        simulator_runtime: SimulatorRuntime | WorldBackend,
        avstack: AVStack,
        request: RunRequest,
        *,
        recorder: Recorder | None = None,
        perturb: Callable[[Observation], Observation] | None = None,
        on_step: Callable[[int, Observation, Control], None] | None = None,
        record_views: Mapping[str, Callable[[Observation], Any]] | None = None,
        view_sink: Callable[[int, str, Any], Any] | None = None,
        release_owned_resources: Callable[[], None] | None = None,
    ) -> RunLog: ...

def trace_from_log(log: RunLog) -> Trace: ...
```

`decision_steps` must be a nonnegative integer. `hooks` accepts `HookSet` (and its `Runtime`
compatibility name). `attacks` supplies already-created online `AttackRuntime` instances; the runner
applies their attack effects at their points of action and records each request and receipt in the
step's `effects`. `start`, `initialization` and `deployment` are passed to
`SimulatorRuntime.reset`. A request's `environment` uses `{world, sensors}`: the runner passes
only `environment["world"]` as `reset(world=...)` and applies `environment["sensors"]` after sensor attack
effects and before Vehicle I/O hooks. See [Experiment environment](../guides/conditions.md#experiment-environment). The runner owns the per-run lifecycle, while the simulator runtime and the
AVStack remain caller-owned: create the runtime with `Simulator.create_runtime()` and close it. It checks stages/command compatibility, transforms the first input
once, resets driving history with that same input and records each completed decision.

`RecordingPolicy` selects named sensors and `"raw"`/`"model_input"` representations. It records
those payloads every `every_n_steps`, starting at decision 0; commands, state and available
component outputs are recorded every decision. No sensor payload is selected by default. The
interval must be a positive integer; unknown requested sensors fail at capture.

`output_path` enables Parquet persistence. `retain_steps=False` requires an output path and leaves
returned `RunLog.steps` and `events` empty; use the path for analysis. The default `True` keeps
in-memory records, including when a path is supplied. Passing an explicit `recorder` makes that
recorder's path, retention, codecs and metadata authoritative. The default recorder uses
`avsectester.runlog.codecs.default_codecs()`.

`record_views` supplies named functions of the processed input. A `view_sink`, when supplied,
receives `(step, name, captured_view)` and returns the reference stored in the log. `on_step`
receives independent input/command copies before actuation. Failures preserve the original
exception and attach the partial `run_log` when possible; failure/cancellation is recorded.

`trace_from_log` derives `FrameRecord` values from physical post-state and executed commands,
and gathers recorded intervention outputs. It uses retained records; for a streamed run, first
call `RunLog.read(path, load_payloads=False)` or analyze selected columns directly. See
[Run logs and saved results](../guides/evaluation.md#run-logs-and-saved-results).

### Environment API

`avsectester.closed_loop.environment` owns the shared environment structure and its application.
Built-in image operators live separately in `avsectester.closed_loop.sensor_transforms`.

| Member | Returns or does |
|---|---|
| `normalize_environment(value: Mapping[str, Any] \| None = None) -> dict[str, Any]` | Validates and copies `{world, sensors}`. Sensor entries use `type`, optional `options` and optional `camera` |
| `merge_environments(*values: Mapping[str, Any]) -> dict[str, Any]` | Merges world settings recursively. Each explicit later sensor list replaces the previous list, including an empty list |
| `validate_environment(value: Mapping[str, Any], simulator: Simulator) -> dict[str, Any]` | Normalizes settings and calls the simulator's world-setting validation before creating a runtime |
| `SensorEnvironment(environment: Mapping[str, Any], seed: int = 0)` | Constructs one set of RGB transforms for a run or service query |
| `SensorEnvironment.apply(observation: Observation, position: int) -> Observation` | Transforms the selected cameras. `position` is the absolute dataset frame position, not the relative decision index. Without a selected start, positions begin at zero |

Sensor transforms run after sensor attack effects and before Vehicle I/O hooks. Their random
realization is determined by the seed, absolute dataset position and transform index, matching
service queries and formal runs at the same frame. If `camera` is omitted, the first sensor in
`Observation.sensor_data` is used; it must have a readable RGB payload or the request fails.

### Closed-loop services

`avsectester.closed_loop.services` provides isolated operations to the metabuilder and offline
phase. The [services guide](../guides/scenarios.md#shared-closed-loop-services) describes simulator
support, gradients and selected-origin replay.

| Class or member | Signature or fields |
|---|---|
| `ClosedLoopServices` | `(dataset: Dataset, simulator: Simulator, driving_adapter: DrivingAdapter \| None = None, frame_data_cache: FrameDataCache \| None = None, *, deployment: DeploymentPlan = DeploymentPlan())` |
| `services.capabilities` | `ClosedLoopCapabilities(dataset_kinds, sensor_data, render, rollout, components, scorers)` |
| `services.scenario` | `(scenario_id: str, *, seed: int) -> ScenarioView` |
| `services.session` | `(seed: int, history: str = "fresh") -> ClosedLoopSession`. Only fresh history is supported |
| `services.close` | `() -> None`. Closes every owned session |
| `ensure_fresh` | `(resource: Any) -> Any`. Marks a newly constructed mutable resource and rejects reuse of an instance already used by a service or run |
| `ScenarioView` | `scenario_id`, `services`, `seed`, `frames`, `reasons` |
| `view.frame_data` | `(frame: FrameReference) -> FrameData`. Resolves recorded data or preparation-owned generated data |
| `view.record_reason` | `(text: str) -> None`. Records a nonempty selection explanation |
| `session.initialization` | `(start: ScenarioStart, environment: Mapping[str, Any]) -> Initialization \| None`. Supplies cached selected-origin data for generated CARLA scenarios |
| `session.frame_data` | `(scenario: str, frame: FrameReference, environment: Mapping[str, Any] \| None = None) -> FrameData` |
| `session.sensor_data` | `(request: SensorRequest) -> SensorSample` |
| `session.render` | `(request: RenderRequest) -> RenderResult` |
| `session.components` | `() -> Sequence[ComponentDescriptor]` |
| `session.scorers` | `() -> Sequence[ScorerDescriptor]` |
| `session.evaluate` | `(component_id: str, inputs: Any) -> ComponentResult` |
| `session.score` | `(scorer_id: str, inputs: Any, target: Any, require_grad: bool = False) -> ScoreResult` |
| `session.rollout` | `(request: CandidateRequest) -> RunLog` |
| `session.close` | `() -> None`. Discards an active scan and releases resources |

Services and sessions are context managers. Sequential effect-free sensor reads may retain one
scan runtime. Rendering, candidate rollout and component/scorer evaluation create fresh resources.
The factory must not return an already used instance. An active CARLA scan reserves its endpoint
until discarded or closed. Sensor results and candidate logs carry source, simulator, environment
and seed provenance. Component/scorer results record the query seed.

Request and result dataclasses:

| Record | Fields, in constructor order |
|---|---|
| `SensorRequest` | `scenario_id: str`, `frame: FrameReference`, `sensor_ids: tuple[str, ...]`, `environment: Mapping[str, Any] = {}` |
| `RenderRequest` | `scenario_id: str`, `frame: FrameReference`, `sensor_id: str`, `effects: Sequence[AttackEffect] = ()`, `environment: Mapping[str, Any] = {}` |
| `CandidateRequest` | `segment: AttackSegment`, `seed: int`, `decision_steps: int`, `environment: Mapping[str, Any]`, `recording: RecordingPolicy`, `attack: Any = None`, `artifacts: Any = None`, `output_path: str \| Path \| None = None` |
| `SensorSample` | `frame`, `observation`, `provenance` |
| `RenderResult` | `frame`, `observation`, `view: SensorView`, `provenance` |
| `ComponentResult` | `value`, `descriptor: ComponentDescriptor`, `provenance` |
| `ScoreResult` | `value`, `descriptor: ScorerDescriptor`, `provenance` |

`CandidateRequest.decision_steps` must equal its segment length. Artifacts require an attack,
which must provide `create_runtime(segment, artifacts)`.
Candidate rollout calls that attack's `create_runtime` and uses the same effects as formal runs.
Rendering accepts world and sensor effects, with world effects inside the declared deployment
window. It returns the changed observation and a detached view of the capture before sensor
effects or environment transforms. `RenderRequest` here is distinct from the hook-stage
`CameraRenderRequest` used by `render.pre`.

`DrivingAdapter(factory, *, components=(), scorers=())` stores a factory `(seed: int) -> AVStack`.
In YAML, `avstack.type` names an AVStack class or AVStack-returning function, constructed with
`options` only; the configuration wrapper does not pass the seed to that constructor. Optional
`avstack.bindings.type` is called as `function(avstack_factory, **options) -> DrivingAdapter`.
See [Driving specification](configuration.md#driving-specification).
Each binding pairs a descriptor with a callback: `(avstack, inputs)` for components, or
`(avstack, inputs, target)` for scorers. `ComponentDescriptor` fields are `id`, `inputs`, `outputs`,
`codecs` and optional `differentiable_inputs=()`. `ScorerDescriptor` adds `output_shape=()` and
`direction="minimize"` (`"maximize"` is also accepted). `require_grad=True` rejects an undeclared
gradient path before constructing a model, and rejects detached scorer output. Returned tensors
retain their graph.

### Attack

```python
# avsectester.attack
class Attack(ABC):
    assets: Any = None
    environment: Mapping[str, Any] = {}
    @abstractmethod
    def build_metadata(self, scenario: ScenarioView) -> Sequence[AttackSegment]: ...
    def run_offline(self, context: OfflineContext) -> Any: ...       # default: self.assets
    def create_runtime(self, segment, artifacts) -> AttackRuntime: ...  # default: no effects
    def save_artifacts(self, artifacts: Any, directory: str | Path) -> None: ...
    def load_artifacts(self, directory: str | Path) -> Any: ...
    def configuration_parameters(self) -> Mapping[str, Any]: ...

@dataclass(frozen=True)
class OfflineContext:
    services: ClosedLoopServices
    metadata: tuple[AttackSegment, ...]

@dataclass(frozen=True)
class AttackSegment:
    scenario_id: str
    start: FrameReference
    stop: FrameReference
    roles: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    placement: Any = None
    selection_score: float | None = None
    selection_rank: int | None = None
    user_data: Mapping[str, Any] = field(default_factory=dict)
    @property
    def decision_steps(self) -> int: ...
```

Every attack has the metabuilder (`build_metadata`, once per scenario), the offline phase
(`run_offline`, once) and the online phase (`create_runtime`, once per run). The offline and
online defaults make those phases trivial. An `AttackSegment` is one selected attack instance:
a scenario, its decision interval `[start, stop)` and the actors bound to roles. All selected
segments together are the attack metadata, which `run_offline` receives; candidate rollouts use
the same record. `AttackRuntime`, its `setup`/`step`/`stop` lifecycle
and the `WorldAttackEffect`, `SensorAttackEffect` and `AVStackAttackEffect` classes in
`avsectester.attack.runtime` are described in [Online phase](runtime.md#online-phase).

`selection_score` is an optional finite number, with larger values better. `selection_rank` is an
optional zero-based rank. The framework validates their formats but does not score, sort or
rerank segments. `user_data` is an independently copied mapping for attack-specific information.
Preparation validates the scenario, actual frame boundaries and role IDs. Offline changes to
artifacts do not trigger selection again. See
[Experiment preparation and execution](#experiment-preparation-and-execution).

`save_artifacts(artifacts, directory)` saves the explicitly supplied offline result. The default
format is pickle with a content hash. `load_artifacts(directory)` returns that result without
changing the Attack object. Override both methods to use a custom format. Prepared artifacts live
in `PreparedExperiment.artifacts` and are passed explicitly to each runtime; Attack holds no
second copy. Only load pickle artifacts from trusted attack implementations.

`configuration_parameters()` supplies the stable parameter mapping for a directly supplied Python
attack. It is requested only when describing saved or loaded preparations, never by ordinary
configuration construction or an in-memory experiment. The default reads named `__init__`
parameters from attributes of the same name and includes non-`None` assets; it does not walk
`__dict__` or include counters and caches. Override it when constructor parameters are renamed,
hidden, or collected in `*args`/`**kwargs`, returning configured inputs rather than execution state.
An undescribable parameter mapping warns and produces an unverified description during saving;
loading requires a complete description. YAML-created attacks retain their resolved `type` and
`options` declaration, independent of changes to the constructed attack's attributes.

#### Segment helper

```python
# avsectester.attack.segments, also exported by avsectester.attack
def split_segments(
    frames: Sequence[FrameReference],
    eligible: Callable[[FrameReference], bool],
    *,
    decision_steps: int,
    keep_tail: bool = False,
) -> tuple[AttackSegment, ...]: ...
```

An optional helper for `build_metadata`. It only cuts segments; the attack decides which frames
qualify. `frames` is an ordered, consecutive timeline of one scenario, normally
`ScenarioView.frames`, and includes the terminal frame. `eligible(frame)` is called once for every
frame except the last, in order, and returns whether an attack decision may be made at that frame.
It can use anything available to the metabuilder, for example `ScenarioView.frame_data(frame)` or
closed-loop sensor data. Precomputed values can be passed as `lambda frame: flags[frame.position]`.

Each run of consecutive eligible frames is cut into segments of `decision_steps` decisions. A
segment stops at the frame after its last decision frame, so it needs `decision_steps + 1` actual
frames; the last frame of the timeline is never a decision frame. An ineligible frame ends a run;
no segment crosses it. `keep_tail=True` also returns the shorter remainder of each run. No frame
is padded or extrapolated.

For 80 frames at positions 0–79, with frames eligible from position 5, `decision_steps=20` and
`keep_tail=True` return `[5,25)`, `[25,45)`, `[45,65)` and `[65,79)`. The last segment has 14
decisions. Role bindings, placements, scores and ranks remain decisions of the attack.

```python
from avsectester.attack import split_segments


def build_metadata(self, scenario):
    def eligible(frame):
        actors = scenario.frame_data(frame).actors
        return self.host in actors and actors[self.host].category == "vehicle"

    return split_segments(scenario.frames, eligible, decision_steps=self.decision_steps)
```

### Run functions

```python
# avsectester.backend (compatibility entry point)
def run(
    backend: WorldBackend,
    stack: AVStack,
    frames: int,
    perturb: Callable[[Observation], Observation] | None = None,
    on_step: Callable[[int, Observation, Control], None] | None = None,
    *,
    runtime: Runtime | None = None,
) -> Trace: ...

# avsectester.runner
RuntimeFactory = Runtime | Callable[[WorldBackend], Runtime] | None

def run_pair(
    backend: WorldBackend,
    stack_factory: Callable[[], AVStack],
    frames: int,
    *,
    clean_runtime: RuntimeFactory = None,
    attacked_runtime: RuntimeFactory = None,
    run_fn: Callable[..., Any] | None = None,
    close_backend: bool = True,
) -> tuple[Any, Any]: ...

def run_prepared(
    scenario: ScenarioInstance,
    stack_factory: Callable[[], AVStack],
    frames: int,
    *,
    runtimes: Mapping[str, RuntimeFactory] | None = None,
    run_fn: Callable[..., Any] | None = None,
) -> dict[str, Any]: ...

# avsectester.legacy_experiment
def run_configured(
    config: Mapping[str, Any],
    frames: int,
    *,
    mode: ExecutionMode | str | None = None,
    treatment: TreatmentMode | str | None = None,
    backend_factory: Callable[[dict[str, Any]], WorldBackend] | None = None,
    stack_factory: Callable[[], AVStack] | None = None,
    clean_runtime: Runtime | None = None,
    treated_runtime: Runtime | None = None,
    run_fn: Callable[..., Trace] | None = None,
    provenance: Mapping[str, Any] | None = None,
    scenario: ScenarioInstance | None = None,
) -> dict[str, Trace]: ...

# avsectester.carla_runner
def run_scenario(
    scenario: dict,
    attacks: list[dict] | None = None,
    frames: int = 40,
    settle_iters: int = 100,
    patches: list[dict] | None = None,
    *,
    runtime: Runtime | None = None,
) -> Trace: ...

def run_scenario_pair(
    scenario: dict,
    frames: int = 40,
    *,
    clean_runtime: Runtime | None = None,
    attacked_runtime: Runtime | None = None,
    settle_iters: int = 100,
    stack_factory: Callable[[], AVStack] | None = None,
) -> tuple[Trace, Trace]: ...
```

| Function | Returns or does | Backend | Raises |
|---|---|---|---|
| `run` | Runs one treatment for `frames` driving steps and returns its `Trace`. `perturb` becomes an `observation` hook placed before the runtime's hooks. `on_step(i, seen, control)` receives copies after command handlers and before execution | The caller owns and closes it | `ValueError` if `frames` is not a nonnegative integer. `UnsupportedStageError` or `CommandCompatibilityError` before reset |
| `run_pair` | Calls `backend.prepare_clean_attack_pair()`, then runs the clean and the attacked runtime, each with a fresh stack from `stack_factory`. Returns both results. A runtime factory receives the backend after pair preparation and before the first reset. `run_fn` replaces `run` and must accept `runtime=` | Closed, unless `close_backend=False` | As `run`. `TypeError` if a runtime factory does not return a `Runtime` |
| `run_prepared` | Runs named treatments of a prepared scenario and returns results by name. Wraps every runtime with `scenario.conditions.apply_runtime` and supplies `scenario.match` as its case when the runtime has none. Prepares a shared origin only for two or more treatments | Created with `scenario.make_backend()` and closed | As `run_pair`. `ValueError` for an empty mapping or an empty or non-string name |
| `run_configured` | Prepares the scenario, unless `scenario` is supplied, and the enabled interventions. Then runs the selected treatments through `run_prepared` | As `run_prepared` | `ConfigurationError` for an invalid configuration, including a non-clean treatment without an enabled intervention of the required kind |
| `run_scenario` | One CARLA run of the modular stack, recorded with `run_logged`. Only the `attacks`, `patches` and `runtime` arguments enable interventions. YAML `interventions`, `runtime` and `clean_runtime` are ignored. Without them the run is clean | As `run_prepared` | As `run_configured` |
| `run_scenario_pair` | A configured CARLA pair: mode `pair`, treatment `attack`. Returns `(clean, attack)`. The Python runtimes replace the configured treatments | As `run_prepared` | As `run_configured` |

`run_configured` options:

- `mode` and `treatment` override `config["mode"]` and `config["treatment"]`, which default to
  `pair` and `attack`. The treatments are described in
  [Treatments and modes](../guides/experiments.md#treatments-and-modes).
- `scenario` supplies a prepared scenario. Its factory, match and conditions are kept. The
  configuration's scene and selection settings are not reapplied, and `backend_factory` and
  `provenance` are ignored.
- `backend_factory(scene_config)` replaces the built-in backend construction.
  `stack_factory` defaults to `configured_stack(config)`.
- `clean_runtime` replaces the baseline runtime. `treated_runtime` replaces all configured
  intervention hooks, and no intervention is prepared.
- A pair with treatment `attack` and no attack hooks runs as a baseline comparison.

The shared loop fills `FrameRecord.n_detections` when a perception component output is available.
`run_scenario` and `run_scenario_pair` also set `Trace.replay_scenario`. See
[Python entry points](../guides/simulators.md#python-entry-points).

```python
# avsectester.legacy_experiment
class ExecutionMode(str, Enum):
    SINGLE = "single"
    PAIR = "pair"

class TreatmentMode(str, Enum):
    CLEAN = "clean"
    ATTACK = "attack"
    DEFENSE = "defense"
    COMBINED = "attack-defense"

    @property
    def kinds(self) -> tuple[str, ...]: ...  # (), ("attack",), ("defense",), ("attack", "defense")

class ConfigurationError(ValueError): ...
```

### Result keys

| Call | Result keys |
|---|---|
| `run_prepared(..., runtimes=None)` | `"clean"` |
| `run_prepared(..., runtimes={...})` | The keys of `runtimes`, in the same order |
| `run_configured`, mode `single` | The treatment value: `"clean"`, `"attack"`, `"defense"` or `"attack-defense"` |
| `run_configured`, mode `pair`, treatment `attack`, `defense` or `attack-defense` | `"clean"` and the treatment value |
| `run_configured`, mode `pair`, treatment `clean` | `"clean"` and `"baseline"` |

Each value is the result of `run_fn`, a `Trace` by default. No runner computes a metric, a
verdict or an exit code.

### Provenance written by runners

When a result is a `Trace`, or a tuple whose first element is a `Trace`, the runners update it:

| Writer | `Trace` changes |
|---|---|
| `run_pair`, `run_prepared` | `replay_scenario` from `backend.replay_scenario` when unset. `provenance["runtime_seed"]` and `provenance["runtime_stages"]`, the stages of the effective runtime's hooks in order |
| `run_prepared` | The scenario's `provenance` entries, then `provenance["treatment"]` (the result key) and `provenance["conditions"]` |
| `run_configured` | `provenance["interventions"]`: a list of `{"name", "kind", "provenance"}` for the prepared interventions. Empty for `clean` and when `treated_runtime` is supplied |

How to read these records is described in
[Intervention outputs and provenance](../guides/evaluation.md#intervention-outputs-and-provenance).

### Supplied adapters

| Class | Constructor | Guide |
|---|---|---|
| `CarlaSimulator` (`avsectester.closed_loop.simulators.carla`) | `CarlaSimulator(scenario: dict, settle_iters: int = 100, *, simulator_version="0.9.16", depth_camera: str \| None = None)`. `depth_camera` adds an aligned depth camera to that ego RGB camera and replaces stock RGB cameras with undistorted `PinholeRgbCamera` sensors | [CARLA](../guides/simulators.md#carla) |
| `NuRecSimulator` (`avsectester.closed_loop.simulators.nurec`) | `NuRecSimulator(config: dict, renderer: Renderer \| None = None, dynamics: EgoDynamics \| None = None)`. `renderer` is a configuration template: each runtime renders through its own `renderer.copy()` | [NuRec](../guides/simulators.md#nurec) |
| `StubRenderer` | `StubRenderer(cameras: list[str] \| None = None, height: int = 480, width: int = 640)` | [Renderers](../guides/simulators.md#renderers) |
| `NuRecRenderer` | `NuRecRenderer(endpoint: str = "127.0.0.1:50051", scene_id: str \| None = None, cameras: list[str] = ["camera_front_wide_120fov"], calibration: dict = {}, image_quality: float = 95.0, start_timestamp_us: int \| None = None, start_transform: Any = None)` | [Renderers](../guides/simulators.md#renderers) |
| `Renderer` | Abstract base: `load_scene(scene)`, `render(pose, camera)` (abstract), `close()` | [Renderers](../guides/simulators.md#renderers) |
| `KinematicBicycle` | `KinematicBicycle(wheelbase: float = 2.8, max_accel: float = 3.0, max_brake: float = 8.0, max_steer: float = 0.6)`. Command mode `actuator` | [Dynamics and time step](../guides/simulators.md#dynamics-and-time-step) |
| `TrajectoryFollower` | `TrajectoryFollower(decel: float = 4.0)`. Command mode `trajectory` | [Dynamics and time step](../guides/simulators.md#dynamics-and-time-step) |
| `EgoDynamics` | Protocol: `step(pose: EgoPose, control: Control, dt: float) -> EgoPose`, with an optional `command_modes` | [Dynamics and time step](../guides/simulators.md#dynamics-and-time-step) |
| `EgoPose` | `EgoPose(x: float = 0.0, y: float = 0.0, yaw: float = 0.0, speed: float = 0.0, t: float = 0.0)` | [NuRec](../guides/simulators.md#nurec) |
| `ModularAVStack` (`avsectester.closed_loop.avstacks.modular`) | `ModularAVStack(pipeline_cfg: dict)` | [Modular driving system](../guides/driving-systems.md#modular-driving-system) |
| `AlpamayoAVStack` (`avsectester.closed_loop.avstacks.alpamayo`) | `AlpamayoAVStack(checkpoint_path: str \| None = None, camera_ids: list[str] \| None = None, device: str = "cuda:0", context_length: int = 4, output_frequency_hz: int = 10, model: Any \| None = None)` | [Alpamayo](../guides/driving-systems.md#alpamayo) |

`avsectester.closed_loop.simulators` re-exports `CarlaSimulator`, `EgoPose`, `KinematicBicycle`,
`NuRecSimulator`, `NuRecRenderer`, `Renderer`, `StubRenderer` and `TrajectoryFollower`.
`avsectester.closed_loop.avstacks` re-exports `AlpamayoAVStack` and `ModularAVStack`. Both stacks
provide `component_log()`, and `ModularAVStack.instrument(stages=("perception", "tracking",
"planning", "control"))` selects the captured stages. See [Component logging
interface](../guides/driving-systems.md#component-logging-interface).

## RunLog and recording

`avsectester.runlog` captures independent snapshots and optionally writes incremental Parquet
records. Sensor arrays and byte payloads are stored in separate bounded shards.

| Record | Fields |
|---|---|
| `StepRecord` | Required `step`; optional `raw_observation`, `model_observation`, `physical_pre`, `physical_post`, `produced_command`, `executed_command`; dictionaries `components`, `telemetry`; lists `effects` (attack effect requests and receipts), `outputs` |
| `EventRecord` | Required `kind`; optional `step=None`; `data={}` |
| `RunLog` | `metadata={}`, `steps=[]`, `events=[]`, `status="incomplete"`, `path=None` |

`StepRecord.step` is a decision index. Each observation/state carries its own frame and time, so
the physical post-state is not confused with the preceding model input. Missing component data
remain missing; the recorder does not trigger inference to produce them.

```python
class Recorder:
    def __init__(
        self,
        path: str | Path | None = None,
        *,
        metadata: dict[str, Any] | None = None,
        codecs: CodecRegistry | None = None,
        chunk_steps: int = 128,
        chunk_bytes: int = 8 * 1024 * 1024,
        payload_shard_bytes: int = 64 * 1024 * 1024,
        retain_steps: bool = True,
    ) -> None: ...

class RunLog:
    @classmethod
    def read(
        cls,
        path: str | Path,
        *,
        codecs: CodecRegistry | None = None,
        load_payloads: bool = True,
    ) -> RunLog: ...

def read_table(
    path: str | Path,
    columns: Sequence[str] | None = None,
    *,
    stream: str = "steps",
) -> Any: ...  # pyarrow.Table
```

| Recorder member | Behavior |
|---|---|
| `log` | Current `RunLog` with path, status and retained records |
| `capture(value: Any) -> Any` | Static method. Snapshots mutable data immediately; tensor values are detached |
| `record_step(record: StepRecord) -> StepRecord` | Snapshots and writes one record; decision indices must be nonnegative and strictly increasing |
| `record_event(kind: str, *, step: int \| None = None, **data) -> EventRecord` | Snapshots an initialization, termination or other event |
| `flush() -> None` | Publishes current chunks without marking the run complete |
| `finish(status="completed", **details) -> RunLog` | Records termination, publishes pending chunks and writes final metadata; accepts `completed`, `failed` or `cancelled` |
| `abort() -> None` | Discards unpublished chunks; completed parts remain readable with incomplete status |
| Context manager | Finishes on normal exit; records failure/cancellation on exceptional exit |

The output directory must be empty or absent. Set `retain_steps=False` with a path to avoid
retaining the full run: both `log.steps` and `log.events` remain empty. Retention defaults to
`True` for compatibility. Chunk limits bound buffered disk records and shard sizes, not the size
of a single input value supplied to `capture`. Neither recorder cleanup nor runner cleanup closes
caller-owned simulator/model resources.

`read_table` returns selected top-level columns with typed struct/list values and disk payload
references; it does not decode native objects or load payload bytes. `stream` is `"steps"` or
`"events"`. Unknown columns raise `ValueError`. `RunLog.read` reconstructs core plane records and,
by default, loads arrays. With `load_payloads=False`, arrays/bytes remain reference dictionaries
with shard chunks and array dtype/shape where applicable. It still loads all published records
into memory. Both readers ignore unpublished `.open` files and reject unknown schema versions,
including version 1 logs, whose step field `actions` was renamed to `effects` in version 2.

This executable example records and reads data without a simulator:

<!-- runnable -->
```python
from pathlib import Path
from tempfile import TemporaryDirectory

from avsectester.closed_loop.runner import trace_from_log
from avsectester.runlog.codecs import default_codecs
from avsectester.closed_loop.datatypes import Control, WorldSnapshot
from avsectester.runlog import Recorder, RunLog, StepRecord, read_table

with TemporaryDirectory() as directory:
    path = Path(directory) / "run"
    with Recorder(path, codecs=default_codecs(), retain_steps=False, chunk_steps=1) as recorder:
        recorder.record_step(StepRecord(
            step=0,
            physical_post=WorldSnapshot(t=0.1, frame=1, ego_speed=2.0),
            executed_command=Control(throttle=0.5),
        ))
    assert recorder.log.steps == [] and recorder.log.status == "completed"
    assert read_table(path, columns=["step"]).num_rows == 1
    restored = RunLog.read(path, load_payloads=False)
    assert trace_from_log(restored).final_speed == 2.0
```

### Native adapter codecs

`avsectester.runlog.codecs.default_codecs() -> CodecRegistry` creates lazy, data-only encoders for
supported Modular and Alpamayo outputs. Registry creation does not import AVStack, AlpaSim,
Torch or simulators; types are registered from adapter modules when those modules become loaded.
The default runner installs this registry. A directly constructed `Recorder` uses an empty
registry unless `codecs` is supplied.

Register additional native values with
`CodecRegistry.register(native_type, encode, *, name=None, decode=None) -> CodecRegistry`.
Encoders return plain data, core records or arrays. Unknown native objects raise `TypeError`
during persistence instead of being pickled or stringified. Stable codec names identify their
representation. A decoder is optional; default native codecs deliberately decode to ordinary
data, allowing offline analysis without importing native model classes. In-memory snapshots
retain their native types. Custom native-object metrics must account for this distinction.

## Runtime

`avsectester.closed_loop.hooks` holds the handler registration and lifecycle classes. The
[Runtime reference](runtime.md) owns stage values, ordering, the handler contract, lifecycle,
contexts and randomness.

```python
# avsectester.closed_loop.hooks
@dataclass(frozen=True)
class Hook:
    stage: str
    handler: Any

class Runtime:
    def __init__(
        self,
        hooks: Iterable[Hook] = (),
        *,
        localizer: Callable[[Observation, StageContext], StateEstimate] | Plugin | None = None,
        case: Any = None,
        seed: int | None = 0,
    ) -> None: ...
    @classmethod
    def from_config(cls, config: Mapping[str, Any] | None = None, *, case: Any = None) -> Runtime: ...
    def with_hooks(self, hooks: Iterable[Hook], *, prepend: bool = False) -> Runtime: ...
    def validate_stages(self, simulator: WorldBackend, avstack: AVStack) -> frozenset[str]: ...
    @property
    def setup_ticks(self) -> int: ...
    def session(
        self, simulator: WorldBackend, avstack: AVStack,
        perturb: Callable[[Observation], Observation] | None = None,
    ) -> HookDispatcher: ...

class Plugin(ABC):
    def reset(self, context: RunContext) -> None: ...
    @abstractmethod
    def transform(self, value: Any, context: StageContext) -> Any: ...
    def after_step(self, context: StageContext) -> None: ...
    def close(self) -> None: ...
```

| Name | Returns or does | Raises |
|---|---|---|
| `Hook(stage, handler)` | Registers `handler` at `stage`. A handler is a callable `handler(value, context)` or an object with `transform(value, context)` | `ValueError` for an empty stage. `TypeError` for any other handler |
| `Runtime(...)` | Stores `hooks` as a tuple in registration order, an optional localizer, the default `context.case` and the handler random seed, as the attributes `hooks`, `localizer`, `case` and `seed` | `TypeError` for an entry that is not a `Hook`, or a localizer that is neither callable nor has `transform` |
| `Runtime.from_config(config, *, case)` | Builds a runtime from a `runtime` configuration mapping, not the whole document. Same as `runtime_from_config` | `ValueError`. See [runtime keys](configuration.md#runtime) |
| `with_hooks(hooks, *, prepend)` | A new `Runtime` with `hooks` appended, or prepended, keeping localizer, case and seed | — |
| `validate_stages(simulator_runtime, avstack)` | Checks command compatibility and that every hook stage is available. Returns the available stages | `CommandCompatibilityError`, `UnsupportedStageError` |
| `setup_ticks` | Sum of `setup_ticks` declared by the distinct `simulator.setup` handlers | `ValueError` for a negative or non-integer count |
| `session(simulator_runtime, avstack, perturb)` | A `HookDispatcher` for one run. A `perturb` function is prepended as an `ObservationTransform` at `observation` | As `validate_stages` |
| `Plugin` | Base class for stateful handlers. Only `transform` is abstract. See [Plugin lifecycle](runtime.md#plugin-lifecycle) | — |

| Class | Signature | Notes |
|---|---|---|
| `RunContext` | `RunContext(simulator_runtime: SimulatorRuntime \| WorldBackend, avstack: AVStack, case: Any, rng: np.random.Generator)` | Frozen. Property `native`. See [Contexts](runtime.md#contexts) |
| `StageContext` | `StageContext(run: RunContext, step: int, stage: str, physical_state: WorldSnapshot, sensor: str \| None = None, observation: Observation \| None = None, control: Control \| None = None)` | Frozen. Method `record_output(name: str, value: Any) -> None`. Properties `simulator_runtime`, `avstack`, `case`, `rng`, `native` and `sim_time`. See [Contexts](runtime.md#contexts) and [Recording outputs](runtime.md#recording-outputs) |
| `CallInputs` | `CallInputs(args: tuple = (), kwargs: dict[str, Any] = {})` | The value at component `.pre` stages. `TypeError` unless `args` is a tuple and `kwargs` a dict |
| `CameraRenderRequest` | `CameraRenderRequest(pose: Any, camera: str)` | The value at `render.pre`. `ValueError` for an empty camera name |
| `HookDispatcher` | Created by `HookSet.session` | `CORE_STAGES = frozenset({"sensors.post", "localization.post", "observation", "command"})`. `emit(stage, value, sensor=None)` runs the handlers at `stage`. Adapters reach it through `_emit` |
| `UnsupportedStageError` | Subclass of `ValueError` | A hook uses a stage the selected adapters do not provide |

`StageContext.record_output` raises `ValueError` for an empty name and `RuntimeError` outside an
active callback.

### Runtime adapters

```python
# avsectester.interventions
class ObservationTransform:
    def __init__(self, transform: Callable[[Observation], Observation]) -> None: ...

# avsectester.closed_loop.avstacks.native_hook
class NativeHook(Plugin):
    def __init__(self, config: Mapping[str, Any], *, phase: str = "post") -> None: ...

# avsectester.interventions
class LegacyNativeHook(NativeHook):
    def __init__(self, config: Mapping[str, Any], *, phase: str = "post",
                 imports: Sequence[str] = ()) -> None: ...

# avsectester.attacks.patch.physical_patch
class PhysicalPatchPlugin(Plugin):
    def __init__(self, patches: Sequence[dict]) -> None: ...
```

| Adapter | Behavior | Raises |
|---|---|---|
| `ObservationTransform(transform)` | Wraps a one-argument `Observation -> Observation` function as a handler and calls it without the context. Its `lifecycle_owner` is the wrapped function, so the runtime calls that object's `reset`, `after_step` and `close` when they exist | — |
| `NativeHook(config, *, phase="post")` | Builds an avstack `HOOKS` hook from a copy of `config` at every `reset`. With `phase="post"` it calls the hook on the stage value and unwraps its one-element tuple. With `phase="pre"` it calls the hook with the `CallInputs` arguments and returns `CallInputs(args, kwargs)`. `close` calls the hook's `close` when it exists. Requires avstack | `ValueError` for another phase. `TypeError` if a pre-hook value is not `CallInputs` or a post-hook does not return a one-element tuple |
| `PhysicalPatchPlugin(patches)` | A `simulator.setup` plugin. Spawns and textures one CARLA panel per patch mapping on its target actor, with one `backend.setup_tick()` per panel. `setup_ticks` is `len(patches)`. `close` destroys the panels | `ValueError` for a patch that is not a dict, a backend without `setup_tick`, a missing CARLA world or an unavailable target |

`NativeHook` requires the implementation's registration module to be imported explicitly.
`avsectester.interventions.LegacyNativeHook` adapts legacy intervention configurations and
accepts an explicit `imports` list. Neither class registers `PhantomInjection` automatically:
import `avsectester.attacks.pipeline.phantom`, or list it in a configured hook's `imports`.

Patch mapping keys are in [patches](configuration.md#patches). Target resolution and panel
behavior are in [Physical patch](../examples/physical-patch.md).

## Preparation

`avsectester.preparation` separates reusable artifacts from per-run execution. Writing an
intervention is described in
[Package the attack as an intervention](../guides/attacks.md#package-the-attack-as-an-intervention).

```python
# avsectester.preparation
class Intervention(ABC):
    def prepare(self, scenario: ScenarioInstance) -> Any: ...
    @abstractmethod
    def hooks(self, artifacts: Any) -> Sequence[Hook]: ...
    def export_artifacts(self, artifacts: Any) -> Any: ...
    def import_artifacts(self, data: Any) -> Any: ...

@dataclass(frozen=True)
class PreparedIntervention:
    method: Intervention
    artifacts: Any
    kind: str = "attack"
    name: str = ""
    provenance: dict[str, Any] = field(default_factory=dict)
    position: str = "after"

    def hooks(self) -> tuple[Hook, ...]: ...
    def save(self, path: str | Path) -> None: ...
    @classmethod
    def load(cls, method: Intervention, path: str | Path) -> PreparedIntervention: ...

def prepare_intervention(
    method: Intervention,
    scenario: ScenarioInstance,
    *,
    kind: str = "attack",
    name: str | None = None,
    artifacts: Any = _UNSET,
    position: str = "after",
) -> PreparedIntervention: ...

@dataclass(frozen=True)
class Treatment:
    interventions: tuple[PreparedIntervention, ...] = ()

    def subset(self, kinds: Iterable[str]) -> Treatment: ...
    def runtime(
        self, base: Runtime | None = None, case: Any = None, *,
        kinds: Iterable[str] | None = None,
    ) -> Runtime: ...
```

| Name | Returns or does | Raises |
|---|---|---|
| `Intervention.prepare(scenario)` | Creates reusable artifacts from the complete prepared scenario, whose `match` may be `None`. The default returns `None` | — |
| `Intervention.hooks(artifacts)` | Abstract. The ordered hooks for one run. Handlers must be fresh or reset their own state | — |
| `Intervention.export_artifacts(artifacts)`, `import_artifacts(data)` | The artifact codec. The defaults return the value unchanged | — |
| `PreparedIntervention(...)` | Artifacts with the method that executes them and their role. An empty `name` becomes the method's class name | `ValueError` for `kind` other than `attack` or `defense`, or `position` other than `before` or `after`. `TypeError` if `method` has no `hooks` or `name` is not a string |
| `PreparedIntervention.hooks()` | Calls `method.hooks(artifacts)` without preparing again | `TypeError` for an entry that is not a `Hook` |
| `prepare_intervention(...)` | Calls `method.prepare(scenario)` once, unless `artifacts` is supplied, including `None`. Sets `provenance` to `{"scenario": <copy of scenario.provenance>}`. `name` defaults to the method's class name | `ValueError` for an invalid `kind`, `position` or an empty `name`. `TypeError` if `method` lacks `hooks`, or lacks `prepare` when preparation is needed |
| `Treatment(interventions)` | Ordered prepared interventions | `TypeError` for other entries |
| `Treatment.subset(kinds)` | The entries whose `kind` is in `kinds`, without preparing them again | `ValueError` for kinds other than `attack` and `defense` |
| `Treatment.runtime(base, case, *, kinds)` | Builds fresh hooks in this order: `position="before"` entries, `base.hooks`, `position="after"` entries. Keeps the localizer and seed of `base`. Uses `case` when it is not `None`, otherwise `base.case`. `kinds` selects a subset first | As `subset` |

Ordering across configured sources is described in
[Ordering at a shared stage](runtime.md#ordering-at-a-shared-stage).

### Artifact codec

`PreparedIntervention.save(path)` writes one JSON record and creates missing parent directories.
`PreparedIntervention.load(method, path)` restores the record with the supplied method. No code is
imported from the file.

| Record field | Content |
|---|---|
| `name` | `PreparedIntervention.name` |
| `kind` | `"attack"` or `"defense"` |
| `provenance` | The preparation provenance, a mapping. `prepare_intervention` stores the scenario's provenance under `scenario` |
| `position` | `"before"` or `"after"` |
| `artifacts` | `method.export_artifacts(artifacts)`, or the artifacts when the method has no `export_artifacts` |

```json
{
  "name": "speed-feedback",
  "kind": "attack",
  "provenance": {"scenario": {"scene": "example", "conditions": {"world": {}, "seed": 0, "sensors": []}}},
  "position": "after",
  "artifacts": {"target_speed": 5.0, "gain": 0.5, "max_bias": 5.0, "initial_bias": 0.0}
}
```

- The default codec stores JSON values: mappings, lists, strings, finite numbers, booleans and
  `None`. Tuples load as lists, and non-string mapping keys load as strings.
- `save` serializes with `allow_nan=False` before it creates or replaces the file. A
  non-serializable value raises `TypeError`, and NaN or infinity raises `ValueError`. The file is
  then left unchanged.
- Override `export_artifacts` and `import_artifacts` for arrays, tensors or custom objects. They
  may write and read external files. A stored file path is not copied with the record.
- `load` calls `method.import_artifacts`. It raises `ValueError` unless the record is a mapping with
  exactly the five fields above and a mapping `provenance`. The restored `PreparedIntervention`
  validates `kind` and `position` again.
- A configured `interventions` entry with `artifacts` passes them through `import_artifacts` and
  skips `prepare`. See [interventions](configuration.md#interventions).

Usage is described in [Save and reuse artifacts](../guides/attacks.md#save-and-reuse-artifacts).

### Supplied interventions

| Class | Constructor | `prepare` returns | `hooks(artifacts)` returns |
|---|---|---|---|
| `SpeedFeedbackAttack` (`avsectester.attacks.adaptive`) | `SpeedFeedbackAttack(target_speed: float, gain: float = 0.1, max_bias: float = 5.0, initial_bias: float = 0.0)` | The four options as a dict | `(Hook("observation", SpeedFeedbackPlugin(**artifacts)),)` |
| `PhysicalPatchIntervention` (`avsectester.attacks.patch.physical_patch`) | `PhysicalPatchIntervention(patches: Sequence[dict])` | A copy of `patches`, after validation | `(Hook("simulator.setup", PhysicalPatchPlugin(artifacts)),)` |
| `NativeHookIntervention` (`avsectester.interventions`) | `NativeHookIntervention(stage: str, config: Mapping[str, Any], *, imports: Sequence[str] = ())` | A copy of `config` | `(Hook(stage, LegacyNativeHook(artifacts)),)` |

`SpeedFeedbackPlugin(target_speed: float, gain: float = 0.1, max_bias: float = 5.0,
initial_bias: float = 0.0)` adds `bias` to the observed `ego_speed` and updates `bias` from the
physical speed in `after_step`. It raises `ValueError` for non-finite or negative values or
`initial_bias` outside `[0, max_bias]`. `SpeedFeedbackAttack` validates its options the same way.
See [Speed-feedback adaptive attack](../examples/speed-feedback-attack.md).

`PhantomInjection` (`avsectester.attacks.pipeline.phantom`, also importable from
`avsectester.attacks`) is an avstack `HOOKS` hook, not an `Intervention`. Use it through
`NativeHook` or a convenience `attacks` entry. Importing it requires avstack.

```text
PhantomInjection(
    target_xyz: tuple[float, float, float] = (6.0, 0.0, -1.5),
    obj_type: str = "Car",
    score: float = 0.9,
    extent: tuple[float, float, float] = (1.6, 1.8, 4.0),  # h, w, l
    oid: int = 90002,
)
PhantomInjection.__call__(detections: Any) -> tuple[Any]
```

The call appends one phantom detection and returns `(detections,)`. It raises `ValueError` when
`detections.source_reference` is missing. See [Phantom injection](../examples/phantom-injection.md).

## Configuration helpers

These functions turn configuration mappings into the objects above. Keys and validation of the
mappings are in the [Configuration reference](configuration.md).

```python
# avsectester.interventions
def runtime_from_config(config: Mapping[str, Any] | None, *, case: Any = None) -> Runtime: ...

def prepare_from_config(
    config: Mapping[str, Any], scenario: ScenarioInstance, *,
    kinds: Iterable[str] | None = None,
) -> Treatment: ...

def configured_runtime(
    scenario: Mapping[str, Any], *, case: Any = None, instance: ScenarioInstance | None = None,
    prepared: Treatment | None = None, kinds: Iterable[str] | None = None,
) -> Runtime: ...

def configured_clean_runtime(
    config: Mapping[str, Any], *, case: Any = None, attacked_runtime: Runtime | None = None,
) -> Runtime: ...

def configured_runtime_pair(
    scenario: Mapping[str, Any],
    *,
    clean_runtime: Runtime | None = None,
    attacked_runtime: Runtime | None = None,
    instance: ScenarioInstance | None = None,
    prepared: Treatment | None = None,
    kinds: Iterable[str] | None = None,
) -> tuple[Runtime, Runtime]: ...

# avsectester.legacy_experiment
def configured_stack(config: Mapping[str, Any]) -> AVStack: ...

# avsectester.scenarios.setup
def scene_config(config: Mapping[str, Any]) -> dict[str, Any]: ...

def source_from_config(
    config: Mapping[str, Any],
    *,
    backend_factory: Callable[[dict[str, Any]], WorldBackend] | None = None,
    provenance: Mapping[str, Any] | None = None,
    candidate_provider: Callable | None = None,
    dataset: Dataset | None = None,
) -> ScenarioSource: ...

def scenario_from_config(
    config: Mapping[str, Any],
    *,
    backend_factory: Callable[[dict[str, Any]], WorldBackend] | None = None,
    provenance: Mapping[str, Any] | None = None,
    candidate_provider: Callable | None = None,
    dataset: Dataset | None = None,
) -> ScenarioInstance: ...
```

In `configured_runtime` and `configured_runtime_pair`, the parameter `scenario` is the
configuration mapping, and `instance` is the prepared scenario. `source_from_config` and
`scenario_from_config` are also exported by `avsectester.scenarios`.

| Function | Returns or does | Raises |
|---|---|---|
| `runtime_from_config(config, *, case)` | A `Runtime` from a `runtime` mapping. `None` gives `Runtime(case=case)`. `seed` defaults to `0`. A `factory` handler is imported and called with `options` as keyword arguments. An `avstack` handler becomes `LegacyNativeHook`, after importing its explicit `imports` modules, with `phase="pre"` at stages ending in `.pre` | `ValueError` for unknown keys, a malformed hook entry or a factory that cannot be imported |
| `prepare_from_config(config, scenario, *, kinds)` | Prepares the configured interventions once and returns a `Treatment` in this order: each convenience `attacks` entry, one entry for all `patches`, then the `interventions` entries. Convenience entries use `position="before"`. `kinds` limits preparation to `attack`, `defense` or both. An entry with `artifacts` skips `prepare` | `ValueError` for a top-level `defenses` key, invalid entries or invalid kinds |
| `configured_runtime(scenario, *, case, instance, prepared, kinds)` | The treated runtime: the `runtime` settings with the enabled interventions assembled by `Treatment.runtime`. Prepares again unless `prepared` is given. When `kinds` excludes `attack`, raw `runtime.hooks` are dropped. `case` defaults to `instance.match`. Without `instance`, preparation receives a placeholder scenario whose factory raises `RuntimeError` | As `prepare_from_config` |
| `configured_clean_runtime(config, *, case, attacked_runtime)` | The baseline runtime. Uses `clean_runtime` when configured. Otherwise uses `attacked_runtime`'s localizer, seed and case without hooks. Otherwise uses the `runtime` localizer and seed without hooks | `ValueError` for an invalid `runtime` mapping |
| `configured_runtime_pair(scenario, *, ...)` | `(clean, treated)`, building each runtime that is not supplied | As above |
| `configured_stack(config)` | The configured stack. Without `stack`: `ModularAVStack(config["ego"]["pipeline"])` with perception instrumented. With `type: AlpamayoAVStack`: `AlpamayoAVStack(**options)`. With `factory`: the user's stack. See [Driving system keys](configuration.md#driving-system-keys) | `ConfigurationError` |
| `scene_config(config)` | A detached copy of the configuration without experiment keys. See [File layout](configuration.md#file-layout) | `ValueError` for a non-mapping |
| `source_from_config(config, ...)` | Without `selection`: a `ConfiguredScenarioSource`. With `selection`: a source that applies the configured requirement through `CarlaScenarioBuilder`, or `DatasetFilter` when `dataset` is given. Its `scenarios()` takes no requirement or the same one | `ValueError` for invalid selection settings, selection for a non-CARLA backend without `dataset`, or `dataset` combined with source options or `backend_factory` |
| `scenario_from_config(config, ...)` | The first scenario from `source_from_config`. Releases selection resources before it returns. Without selection, no backend is constructed until `make_backend()` | As `source_from_config`. `ValueError` if no scenario satisfies the requirement |

| Exception | Module | Raised when |
|---|---|---|
| `ConfigurationError(ValueError)` | `avsectester.legacy_experiment` | `run_configured` or `configured_stack` cannot prepare the configuration |
| `ScenarioConfigurationError(ValueError)` | `avsectester.scenarios.setup` | The built-in backend factory cannot construct a backend from the scene settings. Raised when the backend is built, for example by `make_backend()` |
| `ConditionConfigurationError(ValueError)` | `avsectester.conditions` | A backend rejects a configured world condition |

## Scenarios

The names in `avsectester.scenarios.__all__` can be imported from `avsectester.scenarios`. Each
block below gives the defining module. [Prepare scenarios](../guides/scenarios.md) explains
selection workflows.

### Prepared scenario

```python
# avsectester.scenarios.source
@dataclass(frozen=True)
class ScenarioInstance:
    factory: Callable[[], WorldBackend]
    match: ScenarioMatch | None = None
    provenance: dict[str, Any] = field(default_factory=dict)
    selection: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    conditions: ScenarioConditions = field(default_factory=ScenarioConditions)

    def make_backend(self) -> WorldBackend: ...
```

| Field | Meaning |
|---|---|
| `factory` | Callable that constructs a fresh backend. Stored unchanged. Calling it directly does not apply conditions |
| `match` | The selected `ScenarioMatch`, or `None` without selection. Runtimes receive it as `context.case` |
| `provenance` | A deep copy of the supplied metadata, with `conditions` set to `conditions.provenance()` |
| `selection` | Selection evidence from `SelectionResult.summary()`. Empty without selection |
| `config` | A deep copy of the scene and backend settings, when the source has them. Never contains intervention settings |
| `conditions` | The `ScenarioConditions` shared by every treatment |

Construction raises `TypeError` if `factory` is not callable or `conditions` is not a
`ScenarioConditions`.

`make_backend()` calls `factory()`, then `conditions.configure_backend(backend)`, and returns the
backend. If configuration fails, it closes the backend and re-raises the error, a
`ConditionConfigurationError` when the backend rejects a world condition. `run_prepared` and
`run_configured` build backends this way.

The fields are frozen. Assigning one raises `dataclasses.FrozenInstanceError`. Use
`dataclasses.replace(scenario, factory=...)` or `replace(scenario, conditions=...)` instead.
Replacement runs the same initialization, so `provenance["conditions"]` follows the new
conditions. The `config`, `provenance` and `selection` dictionaries stay mutable. Changing
`config` does not rebuild a factory that already captured its settings.

| Source | `provenance` | `selection` | `config` |
|---|---|---|---|
| `ConfiguredScenarioSource` | The `provenance` argument | `{}` | The scene view of the configuration |
| `DatasetFilter` | The selected scene's `source` | The selection summary | `{}` |
| `CarlaScenarioBuilder` | The scene's `source` with `backend: "carla"` and `validation: "prepared"` or `"analytic_preview"` | The selection summary | The replayable CARLA scene settings |

A source created by `source_from_config` with `selection` adds its `provenance` argument to each
scenario. How to use the prepared scenario is described in
[The prepared scenario](../guides/scenarios.md#the-prepared-scenario).

### Sources

```python
# avsectester.scenarios.source
class ScenarioSource(ABC):
    selection_log: list[dict]
    @abstractmethod
    def scenarios(
        self, req: ScenarioRequirement, limit: int | None = None
    ) -> Iterator[ScenarioInstance]: ...
    def save_selection(self, path) -> None: ...

class DatasetFilter(ScenarioSource):
    def __init__(
        self,
        dataset: Dataset,
        prepare_scene: Callable | None = None,
        prepare_context: Callable[[FilterContext], FilterContext] | None = None,
        *,
        conditions: ScenarioConditions | None = None,
    ): ...

class CarlaScenarioBuilder(ScenarioSource):
    def __init__(
        self,
        base_scenario: dict | None = None,
        samples: int = 1,
        lateral_range: tuple[float, float] = (0.0, 3.0),
        ego_speed: float = 0.0,
        min_gap: float = 6.0,
        seed: int = 0,
        prepare_scene: Callable | None = None,
        *,
        candidate_provider: Callable | None = None,
        sample_scene: bool = False,
        analytic_preview: bool = False,
        prepare_context: Callable[[FilterContext], FilterContext] | None = None,
        conditions: ScenarioConditions | None = None,
    ): ...

class Dataset(ABC):
    @abstractmethod
    def scenes(self) -> Iterator[SceneGT]: ...
    @abstractmethod
    def make_backend(self, scene: SceneGT) -> WorldBackend: ...
    def context(self, scene: SceneGT) -> FilterContext: ...
    def initial_sequence(self, scene: SceneGT, frames: int) -> tuple[SceneGT, ...]: ...

# avsectester.scenarios.setup
class ConfiguredScenarioSource(ScenarioSource):
    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        backend_factory: Callable[[dict[str, Any]], WorldBackend] | None = None,
        provenance: Mapping[str, Any] | None = None,
    ) -> None: ...
```

| Name | Returns or does | Raises |
|---|---|---|
| `scenarios(req, limit=None)` | Yields qualifying prepared scenarios, at most `limit`. Clears and refills `selection_log`, one entry per evaluated candidate | `ValueError` for a negative or non-integer `limit` |
| `save_selection(path)` | Writes `selection_log` as JSON, without native clients or images | — |
| `ConfiguredScenarioSource` | Yields one scenario without preview or filters. `req` must be `None`. `backend_factory(scene_config)` defaults to the built-in CARLA or NuRec factory | `ValueError` when `req` is given. `TypeError` for a non-callable factory |
| `DatasetFilter` | Evaluates `req` on recorded scenes of `dataset`. `prepare_scene` adapts each scene. `prepare_context` augments each filter context | `ValueError` if `conditions` has world settings. `TypeError` if `prepare_context` returns another type |
| `CarlaScenarioBuilder` | Instantiates CARLA candidates and evaluates `req` on them. Without a `candidate_provider` it uses `CarlaCandidateProvider`. `sample_scene=True` samples lead distance and lateral offset. `analytic_preview=True` evaluates an analytic scene without CARLA | `ValueError` for `samples < 1`, a nonzero `ego_speed`, an unordered `lateral_range`, `analytic_preview` with a `candidate_provider`, or world conditions other than `weather` |
| `Dataset` | Base class of a recorded-data provider. `context` returns `FilterContext(scene=scene, dataset=self)`. `initial_sequence` returns `(scene,)` | — |

Supplied providers:

```python
# avsectester.scenarios.datasets.nurec
class NuRecDataset(Dataset):
    def __init__(
        self,
        usdz_paths: list[str],
        sensor: str = "camera_front_wide_120fov",
        camera: str = "front",
        keyframe: float = 0.5,
        vehicle_labels: tuple[str, ...] = _VEHICLE_LABELS,
        max_scenes: int | None = None,
        endpoint: str = "127.0.0.1:50051",
        time_step: float = 0.1,
    ) -> None: ...
    @classmethod
    def from_glob(cls, pattern: str, **kwargs) -> NuRecDataset: ...
    def insertion_renderer(self, scene, backend, insertions, bindings=None, compositor=None): ...

# avsectester.scenarios.datasets.nuscenes
class NuScenesDataset(Dataset):
    def __init__(self, dataroot: str, version: str = "v1.0-trainval", camera: str = "CAM_FRONT",
                 max_samples: int | None = None) -> None: ...
# avsectester.closed_loop.simulators.recorded
class RecordedReplaySimulator(Simulator):  # open-loop replay; creates RecordedReplaySimulatorRuntime
    def __init__(self, start: ScenarioStart | None = None) -> None: ...

# avsectester.scenarios.carla_provider
class CarlaCandidateProvider:
    def __init__(self, *, initial_frames=1, advance_initial=None, backend_factory=None, conditions=None): ...
```

`CarlaCandidateProvider` creates runtimes of `CarlaSimulator(scenario, depth_camera="front")` unless
`backend_factory` is given.

`_VEHICLE_LABELS` is `("automobile", "car", "truck", "bus", "trailer", "van", "vehicle")`.
`NuRecDataset` raises `ValueError` for `keyframe` outside `[0, 1]` or a negative `max_scenes`.
`CarlaCandidateProvider` raises `ValueError` for `initial_frames` below 1, or above 1 without
`advance_initial`. See [CARLA](../guides/scenarios.md#carla),
[NuRec datasets](../guides/scenarios.md#nurec-datasets) and
[Initial windows and providers](../guides/scenarios.md#initial-windows-and-providers).

### Requirements, roles and matches

```python
# avsectester.scenarios.requirement
@dataclass
class ScenarioRequirement:
    name: str
    target: TargetSpec | None = None
    constraints: list[Constraint] = field(default_factory=list)
    description: str = ""
    roles: dict[str, RoleSpec] = field(default_factory=dict)
    insertions: tuple[Insertion, ...] = ()
    window: InitialWindow = field(default_factory=InitialWindow)
    max_bindings: int = 1000
    camera: str | None = None

    def evaluate(self, context: FilterContext | SceneGT) -> SelectionResult: ...
    def match(self, context: FilterContext | SceneGT) -> ScenarioMatch | None: ...

@dataclass
class RoleSpec:
    ids: tuple[str, ...] | None = None
    count: int = 1
    category: str = "vehicle"
    select: str = "nearest_ahead"
    camera: str | None = "front"

@dataclass
class TargetSpec:
    category: str = "vehicle"
    camera: str | None = "front"
    select: str = "nearest_ahead"
    kind: str = "object"

@dataclass(frozen=True)
class InitialWindow:
    frames: int = 1

@dataclass
class ScenarioMatch:
    scene: SceneGT
    target: TargetGeometry | None
    camera: str | None
    bindings: dict[str, tuple[TargetGeometry, ...]] = field(default_factory=dict)
    insertions: tuple[Insertion, ...] = ()
    window: InitialWindow = field(default_factory=InitialWindow)

    @property
    def binding_ids(self): ...

@dataclass
class SelectionResult:
    status: str
    match: ScenarioMatch | None
    candidates: list[CandidateResult]
    reason: str = ""

    def summary(self) -> dict: ...

@dataclass
class CandidateResult:
    target_id: str | None
    checks: list[tuple[str, FilterResult]]
    bindings: dict[str, tuple[str, ...]] = field(default_factory=dict)
```

| Name | Returns or does | Raises |
|---|---|---|
| `ScenarioRequirement.evaluate(context)` | Evaluates every role assignment on every frame of the initial window and returns a `SelectionResult`. A `SceneGT` is wrapped in `FilterContext(scene)` | `ValueError` for duplicate insertion IDs or an inconsistent initial sequence. `TypeError` if a filter returns another type |
| `ScenarioRequirement.match(context)` | `evaluate(context).match` | As `evaluate` |
| `ScenarioRequirement(...)` | — | `ValueError` for both `target` and `roles`, a role named `victim`, `max_bindings < 1` or duplicate insertion IDs |
| `RoleSpec(...)` | Binds explicit `ids`, or selects `count` actors of `category` ranked by `select` | `ValueError` for `count < 1`, an unknown `select`, `largest` without a camera, or `ids` that are not distinct or do not match `count` |
| `TargetSpec(...)` | Single-target shorthand. `select` is `nearest_ahead`, `nearest` or `largest`. `kind` is `object` or `placement` | `ValueError` for other values, or `largest` without a camera |
| `InitialWindow(frames)` | Requires filters to pass on the first `frames` prepared frames | `ValueError` unless `frames` is a positive integer |
| `ScenarioMatch.binding_ids` | `{role: (track_id, ...)}` for the bound actors | — |
| `SelectionResult.status` | `"pass"`, `"fail"` or `"unknown"` | — |
| `SelectionResult.summary()` | JSON-compatible evidence with the keys `status`, `reason`, `selected_target`, `bindings`, `insertions`, `initial_frames` and `candidates` | — |
| `CandidateResult.status` | The combined status of `checks`, or `"pass"` without checks | — |

See [Compose a requirement](../guides/scenarios.md#compose-a-requirement) and
[Roles and bindings](../guides/scenarios.md#roles-and-bindings).

### Filters

```python
# avsectester.scenarios.filters
@dataclass(frozen=True)
class FilterResult:
    status: Literal["pass", "fail", "unknown"]
    reason: str = ""
    @classmethod
    def from_bool(cls, passed: bool, reason: str) -> FilterResult: ...

class Constraint(ABC):
    @abstractmethod
    def evaluate(self, context: FilterContext) -> FilterResult: ...
    def holds(self, context: FilterContext | SceneGT, target=None) -> bool: ...

@dataclass
class All(Constraint):
    filters: list[Constraint] = field(default_factory=list)

@dataclass
class Any(Constraint):
    filters: list[Constraint] = field(default_factory=list)

@dataclass
class Not(Constraint):
    filter: Constraint
```

`FilterResult` raises `ValueError` for another status. `Constraint.holds` returns `True` when
`evaluate` passes. Given a `SceneGT`, it builds a context with `target` bound as `attacker`. The
combinator `Any` shadows `typing.Any` when imported by name.

The built-in filters are `InView`, `DistanceRange`, `ImageAreaFrac`, `ViewpointRear`,
`MinVisibility`, `EgoMoving` and `ClearLaneAhead`. Their fields, semantics and the combinator
results are in [Filters](../guides/scenarios.md#filters).

```python
# avsectester.scenarios.serialize
def register_constraint(cls: type[Constraint]) -> type[Constraint]: ...
def requirement_to_dict(req: ScenarioRequirement) -> dict[str, Any]: ...
def requirement_from_dict(value: dict[str, Any]) -> ScenarioRequirement: ...
def constraint_vocabulary() -> list[dict[str, Any]]: ...

# avsectester.scenarios.nl
LLM = Callable[[str], str]
def interpret(description: str, llm: LLM, name: str = "custom") -> ScenarioRequirement: ...
```

| Function | Returns or does | Raises |
|---|---|---|
| `register_constraint(cls)` | Registers a filter dataclass for dictionary serialization and returns it, so it works as a decorator | `TypeError` unless `cls` is a `Constraint` dataclass. `ValueError` if another class has the same name |
| `requirement_to_dict(req)` | A dictionary of the requirement. Insertions serialize only with a `PlaneAsset` | — |
| `requirement_from_dict(value)` | The inverse of `requirement_to_dict` | Errors of the constructors |
| `constraint_vocabulary()` | Registered filter names, fields, defaults and first docstring lines | — |
| `interpret(description, llm, name)` | Prompts `llm`, parses the JSON in its answer and returns the requirement via `requirement_from_dict` | `ValueError` when the answer has no JSON object. Constructor errors for invalid fields |

See [Selection in YAML](../guides/scenarios.md#selection-in-yaml) and
[Natural-language requirements](../guides/scenarios.md#natural-language-requirements).

### Scene state and filter context

Scene coordinates use the ego frame: x forward, y left, z up, in metres.

```text
# avsectester.scenarios.scene
@dataclass
class SceneGT:
    frame: int
    t: float
    ego: EgoState
    cameras: dict[str, CameraCalib]
    objects: list[ObjectGT]
    source: dict[str, Any] = field(default_factory=dict)
    placements: list[PlacementCandidate] = field(default_factory=list)
    def objects_of(self, category: str) -> list[ObjectGT]

@dataclass
class EgoState:
    speed: float = 0.0
    pose: Any = None
    extent: tuple[float, float, float] = (0.0, 0.0, 0.0)
    center: tuple[float, float, float] = (0.0, 0.0, 0.0)

@dataclass
class CameraCalib:
    name: str
    width: int
    height: int
    model: Any = None
    cam_to_ego: Any = None

@dataclass
class TargetGeometry:  # base class of ObjectGT and PlacementCandidate
    track_id: str
    category: str
    center: tuple[float, float, float]
    extent: tuple[float, float, float] = (0.0, 0.0, 0.0)
    yaw: float = 0.0
    box2d: dict[str, tuple[float, float, float, float]] = field(default_factory=dict)
    visibility: Visibility | None = None
    pose: Any = None
    @property
    def distance(self) -> float
    @property
    def ahead(self) -> bool
    def image_area_frac(self, camera: str, calib: CameraCalib) -> float | None
```

| Name | Meaning |
|---|---|
| `SceneGT.source` | Provenance of the scene. It becomes the scenario's `provenance` in recorded selection |
| `EgoState.pose` | Optional 4×4 world-from-ego transform. Inserted geometry requires it. `center` locates the bounding box in ego axes |
| `CameraCalib` | Calibration in optical axes (x right, y down, z forward). `model` is a projection model, a pinhole K or a provider lens. `cam_to_ego` maps optical coordinates into the ego frame |
| `TargetGeometry` | An object in the ego frame. `extent` is (length, width, height). `yaw` is the heading relative to the ego. `box2d` maps camera names to `(x1, y1, x2, y2)`. `pose` is an optional 4×4 ego-from-object transform |
| `ObjectGT`, `PlacementCandidate` | An existing object identified by `track_id`, and a proposed insertion with a stable placement ID |
| `distance`, `ahead` | Ground-plane distance from the ego origin, and whether `center` has positive x |
| `image_area_frac(camera, calib)` | Image-clipped `box2d` area as a fraction of the image, or `None` without a box |

```python
# avsectester.scenarios.context
@dataclass
class FilterContext:
    scene: SceneGT
    bindings: dict[str, tuple[TargetGeometry, ...]] = field(default_factory=dict)
    insertions: tuple[Insertion, ...] = ()
    sequence: tuple[SceneGT, ...] = ()
    metadata: Any = None
    dataset: Any = None
    backend: Any = None
    renderer: Any = None
    native: Mapping[str, Any] = field(default_factory=dict)
    visibility_estimator: GeometryVisibilityEstimator | None = None
    visibility_provider: (
        Callable[[FilterContext, str, str], Visibility | VisibilityEvidence | None] | None
    ) = None
    frame_context: Callable[[SceneGT], FilterContext] | None = None
    initial_contexts: Callable[[], Iterator[FilterContext]] | None = None
    target: TargetGeometry | None = None
    frame_index: int = 0
```

| Member | Returns or does |
|---|---|
| `own(resource)` | Registers `resource.close()` for this scope and returns `resource` |
| `close()`, `with context:` | Releases owned resources once, in reverse order |
| `derive(**changes) -> FilterContext` | A copy with a fresh geometry cache and shared native resources |
| `at(scene: SceneGT, frame_index: int = 0) -> FilterContext` | Rebinds the same identities in another prepared frame |
| `actor_aliases -> dict[str, str]` | Single-member roles and indexed aliases such as `hosts[0]`, mapped to track IDs |
| `actors -> dict` | World `ActorPose` objects keyed by track ID and role alias |
| `world_from_ego` | `scene.ego.pose` as an array. `ValueError` without a pose |
| `victim` | The ego `ActorPose` in the world frame |
| `resolved_insertions -> dict` | `resolve_insertion` results keyed by insertion ID |
| `subjects(ids: tuple[str, ...] = ()) -> tuple[TargetGeometry, ...]` | Resolved insertions when there are insertions, otherwise the target or scene placements |
| `in_view(subject: TargetGeometry, camera: str) -> bool` | Whether opaque insertion pixels fall inside the image, ignoring occlusion |
| `visibility(subject: TargetGeometry, camera: str) -> Visibility \| None` | Visibility from `visibility_provider`, `visibility_estimator` or the subject's own field, cached per subject and camera |

See [Custom filters and native access](../guides/scenarios.md#custom-filters-and-native-access).

## Conditions

The new Python experiment API uses [Environment API](#environment-api). `ScenarioConditions`
below belongs to the legacy configured-runner path; its seed and ordering options are separate.

```python
# avsectester.conditions (both classes are also exported by avsectester.scenarios)
@dataclass(frozen=True)
class ScenarioConditions:
    world: Mapping[str, Any] = field(default_factory=dict)
    sensors: tuple[SensorCondition, ...] = ()
    seed: int = 0

    @classmethod
    def from_config(cls, config: Mapping[str, Any] | None = None) -> ScenarioConditions: ...
    def configure_backend(self, backend: WorldBackend) -> None: ...
    def apply_runtime(self, runtime: Runtime) -> Runtime: ...
    def provenance(self) -> dict[str, Any]: ...
    def selection_scope(self): ...
    def selection_context(self, context: FilterContext) -> FilterContext: ...
    @staticmethod
    def selection_observation(context: FilterContext) -> Observation: ...

@dataclass(frozen=True)
class SensorCondition:
    factory: Callable[[], Any]
    stage: str = "observation"
    order: str = "after"
    seed: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def image(
        cls,
        pipeline: Any,
        *,
        camera: str | None = None,
        stage: str = "observation",
        order: str = "after",
    ) -> SensorCondition: ...

class ConditionConfigurationError(ValueError): ...
```

| Member | Returns or does | Raises |
|---|---|---|
| `ScenarioConditions(...)` | Copies `world` into a dict and `sensors` into a tuple | `ValueError` for a non-mapping `world` or a seed that is not a nonnegative integer. `TypeError` for a sensor entry that is not a `SensorCondition` |
| `from_config(config)` | Parses a `conditions` mapping. `None` gives empty conditions. See [Condition keys](configuration.md#condition-keys) | `ValueError` for unknown keys or invalid entries |
| `configure_backend(backend)` | Calls `backend.configure_conditions(world)` when `world` is nonempty | `ConditionConfigurationError` wrapping the backend's `ValueError` |
| `apply_runtime(runtime)` | A new runtime with a fresh handler from every sensor condition's factory. `order="before"` handlers are prepended and `order="after"` handlers appended. Each handler gets its own random generator from its seed and the driving step | — |
| `provenance()` | `{"world": ..., "seed": ..., "sensors": [...]}`. Each sensor entry is its metadata with `stage`, `order` and the effective `seed` | — |
| `selection_scope()` | A context manager, callable as `(context, frame_index)`, that applies the sensor conditions to each selection frame's `native["observation"]`. The result is stored in `native["conditioned_observation"]` | `TypeError` or `ValueError` for a mismatched selection observation |
| `selection_context(context)` | Applies the conditions to one selection frame | As `selection_scope` |
| `selection_observation(context)` | `context.native["conditioned_observation"]` | `ValueError` when the provider supplies no observation |
| `SensorCondition(...)` | A factory of sensor handlers at `sensors.post` or `observation`. `seed=None` uses the scenario's `seed` | `TypeError` for a non-callable factory. `ValueError` for another stage or order, or an invalid seed |
| `SensorCondition.image(pipeline, *, camera, stage, order)` | Wraps a copy of an `AugmentationPipeline`. The pipeline's seed becomes the condition seed, and the metadata lists its corruptions. With `camera=None` the first sensor is selected and must be readable RGB | `TypeError` for another pipeline type. At run time, `ValueError` when the camera is missing or unreadable |

How conditions are configured, ordered and seeded is described in
[Apply operating conditions](../guides/conditions.md).

### Image corruption API

The corruption operators are in `avsectester.closed_loop.sensor_transforms`. The legacy image
augmentation pipelines and robustness suites remain in `avsectester.perturbations.image`.

```python
# avsectester.closed_loop.sensor_transforms
class Corruption(ABC):
    category: str = "sensor"
    def __init__(self, severity: float = 0.5) -> None: ...
    @property
    def name(self) -> str: ...
    @abstractmethod
    def apply(self, rgb: np.ndarray, rng: np.random.Generator) -> np.ndarray: ...

CORRUPTIONS: dict[str, type[Corruption]]

# avsectester.perturbations.image
class AugmentationPipeline:
    def __init__(self, corruptions: Iterable[Corruption], seed: int = 0) -> None: ...
    @property
    def name(self) -> str: ...
    def apply(self, rgb: np.ndarray, frame: int = 0) -> np.ndarray: ...

def common_corruptions(severity: float = 0.5) -> list[AugmentationPipeline]: ...
def albumentations_available() -> bool: ...

class AlbumentationsCorruption(Corruption):
    def __init__(self, build: Callable[[float], Any], severity: float = 0.5,
                 category: str = "sensor", label: str = "albumentations") -> None: ...

def albumentations_corruptions(severity: float = 0.5) -> list[AugmentationPipeline]: ...

# avsectester.closed_loop.datatypes
def read_camera_rgb(observation: Observation, camera: str | None = None) -> tuple[str, np.ndarray] | None: ...
def write_camera_rgb(observation: Observation, key: str, rgb: np.ndarray) -> Observation: ...

# avsectester.perturbations.image
def sensor_augmentation(pipeline: AugmentationPipeline | Iterable[Corruption],
                        camera: str | None = None) -> Callable[[Observation], Observation]: ...
def compose(*perturbs: Callable[[Observation], Observation] | None) -> Callable[[Observation], Observation]: ...
```

| Name | Returns or does |
|---|---|
| `Corruption` | An image operator on `(H, W, 3)` uint8 RGB. `severity` is clipped to `[0, 1]`. `name` is `"<Class>@<severity>"` |
| `CORRUPTIONS` | Operator classes by name. The catalog is in [Built-in image corruptions](../guides/conditions.md#built-in-image-corruptions) |
| `AugmentationPipeline` | Applies `corruptions` in order with a generator seeded from `(seed, frame)`. `name` joins the operator names with `+`, or is `"identity"` |
| `common_corruptions(severity)` | One single-operator pipeline per `CORRUPTIONS` entry |
| `albumentations_available()` | Whether `albumentations` can be imported |
| `AlbumentationsCorruption` | Wraps `build(severity)`, an Albumentations transform, as a `Corruption` seeded from the pipeline's generator |
| `albumentations_corruptions(severity)` | One pipeline per Albumentations-backed operator. Raises `RuntimeError` when Albumentations is not installed |
| `read_camera_rgb(observation, camera)` | `(sensor_id, rgb)` for `camera`, or for the first sensor when `camera` is absent. Reads an ndarray payload or a payload's `rgb_image`. `None` when unreadable |
| `write_camera_rgb(observation, key, rgb)` | A copy of `observation` with that sensor's image replaced. A non-ndarray payload is shallow-copied with its `data` replaced |
| `sensor_augmentation(pipeline, camera)` | An `Observation -> Observation` function that corrupts one camera, using `observation.frame` as the frame index. Leaves the observation unchanged without a readable camera |
| `compose(*perturbs)` | Applies the functions left to right and skips `None` |

`frame_perturbation`, also in `avsectester.perturbations.image`, is listed under
[Insertion rendering](#insertion-rendering).

## Insertion and rendering

World and actor frames use metres and right-handed axes: x forward, y left, z up. An actor pose
is centred on its bounding box. Assets use the same axes, with their front along +x and their top
along +z. Camera geometry uses optical axes: x right, y down, z forward. Pixel `(column, row)` has
its centre at `(column + 0.5, row + 0.5)`.

How to specify insertions is described in
[Specify inserted objects](../guides/scenarios.md#specify-inserted-objects). How to feed them to
the driving system is described in [Image insertion](../examples/image-insertion.md).

### Insertion specification

```python
# avsectester.insertion
@dataclass(frozen=True)
class Insertion:
    id: str
    asset: PlanarAsset
    placement: AttachedPlacement | WorldPlacement
    orientation: Orientation = field(default_factory=Orientation)

@dataclass(frozen=True)
class PlaneAsset:
    texture: np.ndarray | str | Path
    width_m: float
    height_m: float
    texture_path: str | None = field(default=None, init=False)
    def planes(self) -> tuple[PlaneSurface, ...]: ...

@dataclass(frozen=True)
class PlaneSurface:
    corners: np.ndarray
    texture: np.ndarray

class PlanarAsset(Protocol):
    def planes(self) -> Sequence[PlaneSurface]: ...

@dataclass(frozen=True)
class AttachedPlacement:
    host: str
    anchor: str = "center"
    offset_m: tuple[float, float, float] = (0.0, 0.0, 0.0)

@dataclass(frozen=True)
class WorldPlacement:
    position_m: tuple[float, float, float]

@dataclass(frozen=True)
class Orientation:
    mode: str = "fixed_world"
    rotation_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)
    target_offset_m: tuple[float, float, float] = (0.0, 0.0, 0.0)

@dataclass(frozen=True)
class ResolvedInsertion:
    id: str
    pose: np.ndarray
    asset: PlanarAsset
    host_id: str | None = None
    def planes(self) -> tuple[PlaneSurface, ...]: ...

def resolve_insertion(
    insertion: Insertion,
    actors: Mapping[str, ActorPose],
    victim: ActorPose,
) -> ResolvedInsertion: ...

# avsectester.geometry.poses
@dataclass(frozen=True)
class ActorPose:
    transform: np.ndarray
    extent: tuple[float, float, float] = (0.0, 0.0, 0.0)

def rotation_matrix(rotation_deg) -> np.ndarray: ...
def validated_vector(value, name: str) -> np.ndarray: ...
def validated_transform(value) -> np.ndarray: ...
```

| Type | Meaning | Validation |
|---|---|---|
| `Insertion` | One inserted object with its asset, placement and orientation. Placement is never searched or adjusted | `ValueError` for an empty `id` or `follow_host` without an `AttachedPlacement`. `TypeError` for an asset without `planes()` or another placement type |
| `PlaneAsset` | An RGBA rectangle centred at the origin in the local yz plane, front +x, top +z. `width_m` runs along y and `height_m` along z. A path is loaded once with Pillow and kept in `texture_path`. Transparent pixels are outside the silhouette | `ValueError` for a non-finite or nonpositive size, and for the texture checks of `PlaneSurface` |
| `PlaneSurface` | Four corners, shape `(4, 3)`, ordered top-left, top-right, bottom-right, bottom-left as seen from the front, and a uint8 RGBA texture. Both arrays are copied and read-only. Unpacks as `(corners, texture)` | `ValueError` for a wrong shape, non-finite corners, an empty, non-RGBA or non-uint8 texture, or corners that do not form a rectangle |
| `PlanarAsset` | Protocol for custom assets. `planes()` returns surfaces in one asset-local frame | — |
| `AttachedPlacement` | A point in the host's local frame: the anchor scaled by the host extent, plus `offset_m` in host axes. `host` is a track ID or a bound role alias | `ValueError` for an empty host, an unknown anchor or a non-finite offset |
| `WorldPlacement` | A fixed point in the provider's world frame | `ValueError` for a non-finite position |
| `Orientation` | The rotation mode and its parameters | `ValueError` for an unknown mode, a nonzero `rotation_deg` with `face_victim` or a nonzero `target_offset_m` with another mode |
| `ActorPose` | A world pose of a bounding-box centre, a right-handed rigid 4×4 transform, and its full extent `(length, width, height)` in metres; defined in `avsectester.geometry.poses` | `ValueError` for an invalid transform or a negative extent |
| `validated_vector(value, name)` | A finite NumPy three-vector, for canonical positions, extents or angles | `ValueError` for another shape or non-finite values |
| `validated_transform(value)` | An independent, read-only 4×4 right-handed rigid transform | `ValueError` for an invalid shape, rotation or homogeneous last row |
| `ResolvedInsertion` | An insertion's world pose at one instant. `planes()` returns world-frame surfaces | `ValueError` for an invalid pose |

| Anchor | Local point, as fractions of (length, width, height) |
|---|---|
| `center` | (0, 0, 0) |
| `front_center` | (0.5, 0, 0) |
| `rear_center` | (-0.5, 0, 0) |
| `left_center` | (0, 0.5, 0) |
| `right_center` | (0, -0.5, 0) |
| `top_center` | (0, 0, 0.5) |
| `bottom_center` | (0, 0, -0.5) |

| Orientation mode | Rotation | Constraint |
|---|---|---|
| `fixed_world` (default) | `rotation_matrix(rotation_deg)` in the world frame | — |
| `follow_host` | The host rotation multiplied by `rotation_matrix(rotation_deg)` | Requires `AttachedPlacement` |
| `face_victim` | +x points at the victim-local point `target_offset_m`, with +z as close to world up as possible. A vertical direction uses world +y as the secondary reference | `rotation_deg` must be zero |

`rotation_matrix(rotation_deg)`, in `avsectester.geometry.poses`, takes `(roll, pitch, yaw)` in
degrees and returns
`Rz(yaw) @ Ry(pitch) @ Rx(roll)`. `resolve_insertion` computes the pose without changing its
inputs. It raises `KeyError` when the host is missing from `actors`, and never substitutes another
actor. It raises `ValueError` when a `face_victim` target coincides with the insertion position.

#### Payload builders

```python
# avsectester.attacks.object_insertion.sign_spoof
STOP_SIGN: Path  # avsectester/assets/signs/mutcd_r1-1_stop.png
def sign_rgba(path: str | Path = STOP_SIGN, size: int | None = None) -> np.ndarray: ...
def post_rgba(height: int = 256, width: int = 16, color: tuple = (150, 152, 155)) -> np.ndarray: ...

@dataclass
class SignAsset:
    face: np.ndarray = field(default_factory=sign_rgba, repr=False)
    width: float = 0.9
    mount_height: float = 1.5
    post: bool = True
    post_width: float = 0.07
    n_posts: int = 1
    def planes(self) -> Sequence[PlaneSurface]: ...

# avsectester.attacks.object_insertion.person_poster
def load_cutout(path: str | Path) -> np.ndarray: ...
def poster_rgba(person: np.ndarray, aspect: float = 1.5, fill: float = 0.86,
                paper: tuple = (228, 226, 218), border: tuple = (40, 40, 42),
                border_frac: float = 0.025) -> np.ndarray: ...
def standee(person: np.ndarray, height: float = 1.75) -> SignAsset: ...
def billboard(person: np.ndarray, width: float = 1.4, mount_height: float = 0.6) -> SignAsset: ...

# avsectester.attacks.object_insertion.traffic_light
def signal_head_rgba(lit: str = "red", height: int = 300) -> np.ndarray: ...
def traffic_lights_rgba(n: int = 3, lit: str = "red", height: int = 300, gap_frac: float = 0.25,
                        board: tuple | None = (35, 37, 40)) -> np.ndarray: ...
def aspect_of(face: np.ndarray) -> float: ...
def roadside_rig(face: np.ndarray, width: float = 2.4, mount_height: float = 2.2) -> SignAsset: ...

# avsectester.attacks.patch.physical_patch
def checkerboard_rgba(size: int = 256, squares: int = 8,
                      color_a: tuple = (255, 0, 0), color_b: tuple = (255, 255, 255)) -> np.ndarray: ...
def image_rgba(path: str, size: int = 256) -> np.ndarray: ...
```

| Builder | Returns |
|---|---|
| `SignAsset` | A local planar sign with optional posts, usable as an `Insertion` asset. The origin is on the ground below the face. +x is the face normal, +y the viewer's right and +z up. `width` is the face width, and the face height follows the texture aspect. `mount_height` is the face's bottom edge. `planes()` raises `ValueError` for non-finite or invalid sizes or `n_posts < 1` |
| `sign_rgba(path, size)` | A sign face as `(H, W, 4)` uint8 RGBA. The default is the STOP face. `size` resizes it to a square |
| `post_rgba(height, width, color)` | A shaded grey post texture |
| `load_cutout(path)` | A person cut-out as RGBA |
| `poster_rgba(person, ...)` | An opaque framed poster, 900 px tall with height/width `aspect`, showing the person at `fill` of its height |
| `standee(person, height)` | `SignAsset(face=person, ..., mount_height=0.0, post=False)`, with the width from the cut-out's aspect |
| `billboard(person, width, mount_height)` | `SignAsset(face=poster_rgba(person), ..., post=True, post_width=0.09, n_posts=2)` |
| `signal_head_rgba(lit, height)` | One vertical three-lens signal head with the `lit` lens (`red`, `yellow` or `green`) on |
| `traffic_lights_rgba(n, lit, ...)` | `n` signal heads in a row on a board. `board=None` gives a transparent background |
| `aspect_of(face)` | Height divided by width |
| `roadside_rig(face, width, mount_height)` | `SignAsset(face=face, ..., post_width=0.12)` with one post |
| `checkerboard_rgba(...)`, `image_rgba(path, size)` | Square RGBA patch textures: a checkerboard, or an image file resized to `size` |

`PhysicalPatch` is the CARLA panel payload deployed by `PhysicalPatchPlugin`. It is not an
`Insertion` asset.

```python
# avsectester.attacks.patch.physical_patch
@dataclass
class PhysicalPatch:
    prop: str = DEFAULT_PROP            # "static.prop.ironplank"
    location: tuple = DEFAULT_LOCATION  # (-2.3, 0.0, 1.0), relative to the target actor
    rotation: tuple = DEFAULT_ROTATION  # (90.0, 0.0, 0.0), pitch, yaw, roll in degrees
    texture: dict = field(default_factory=lambda: {"pattern": "checkerboard", "size": 256})
    emissive: bool = False

    def apply(
        self, world: Any, target_actor: Any, *,
        tick: Callable[[], Any] | None = None,
        on_spawn: Callable[[Any], None] | None = None,
    ) -> list: ...

def build_patch(cfg: dict | PhysicalPatch) -> PhysicalPatch: ...
def to_carla_texture(rgba: np.ndarray) -> Any: ...
```

`apply` spawns the panel, paints it and returns the spawned actors. It raises `ValueError` for a
prop missing from the blueprint library and `RuntimeError` when the panel cannot be painted.
`build_patch` drops the configuration keys `type`, `target`, `gap` and `lead_vehicle` and
constructs a `PhysicalPatch`. See [Payloads](../examples/image-insertion.md#payloads) and
[Physical patch](../examples/physical-patch.md).

### Insertion rendering

```python
# avsectester.geometry.cameras (Camera is also exported by avsectester.geometry)
class Camera(Protocol):
    width: int
    height: int
    def project(self, pts_cam: np.ndarray) -> np.ndarray: ...
    def unproject(self, uv: np.ndarray) -> np.ndarray: ...
```

| Member | Contract |
|---|---|
| `width`, `height` | Image size in pixels |
| `project(pts_cam)` | `(N, 3)` optical-frame points to `(N, 2)` pixel coordinates |
| `unproject(uv)` | `(N, 2)` pixel coordinates to `(N, 3)` unit rays |
| `max_angle` | Optional. The field-of-view limit in radians from the optical axis |

A custom camera needs only these members. It does not have to inherit from a supplied class.

```python
# avsectester.geometry.cameras
@dataclass
class PinholeCamera:
    K: np.ndarray
    width: int
    height: int

@dataclass
class FThetaCamera:
    cx: float
    cy: float
    angle_to_pixeldist: tuple
    width: int
    height: int
    pixeldist_to_angle: tuple = ()
    max_angle: float = math.pi / 2
    linear_cde: tuple = (1.0, 0.0, 0.0)
    @classmethod
    def from_nurec(cls, spec: Any) -> FThetaCamera: ...
    def in_view(self, pts_cam: np.ndarray) -> np.ndarray: ...

def camera_from_calibration(calib) -> Camera: ...
def quat_to_matrix(w: float, x: float, y: float, z: float) -> np.ndarray: ...
def make_pose(rotation: np.ndarray, translation) -> np.ndarray: ...
def pose_from_proto(pose: Any) -> np.ndarray: ...
def planar_rig_pose(x: float, y: float, yaw: float, z: float = 0.0) -> np.ndarray: ...
def transform(T: np.ndarray, pts: np.ndarray) -> np.ndarray: ...
def project_to_pixels(pts_cam: np.ndarray, K: np.ndarray) -> np.ndarray: ...
def cam_coords(pts_world: np.ndarray, world_to_cam: np.ndarray) -> np.ndarray: ...
def carla_cam_coords(pts_world: np.ndarray, cam_inverse_matrix: np.ndarray) -> np.ndarray: ...
```

| Name | Returns or does |
|---|---|
| `PinholeCamera` | An ideal pinhole lens with 3×3 intrinsics `K` |
| `FThetaCamera` | The NuRec f-theta lens. Polynomials list coefficients in increasing order. `from_nurec(spec)` builds it from a NuRec camera specification. `in_view(pts_cam)` tests z > 0 and `max_angle`. `unproject` raises `NotImplementedError` without `pixeldist_to_angle` |
| `camera_from_calibration(calib)` | A `Camera` from a `CameraCalib`. Returns `calib.model` when it already projects and unprojects, converts a NuRec dataset lens, or builds a `PinholeCamera` from a 3×3 K or `(f, cx, cy)`. Raises `ValueError` for a missing or malformed model |
| `quat_to_matrix`, `make_pose`, `pose_from_proto` | A 3×3 rotation from a quaternion, a 4×4 pose from rotation and translation, and a 4×4 pose from a NuRec `Pose` message |
| `planar_rig_pose(x, y, yaw, z)` | The world-from-rig 4×4 pose of a planar ego pose |
| `transform(T, pts)` | Applies a 4×4 transform to `(N, 3)` points |
| `project_to_pixels`, `cam_coords`, `carla_cam_coords` | Pinhole projection, world-to-optical coordinates, and the same for a CARLA camera inverse matrix |

```python
# avsectester.rendering.types (all are also exported by avsectester.rendering)
class InsertionGeometry(NamedTuple):
    actors: Mapping[str, ActorPose]
    victim: ActorPose
    cam_from_world: np.ndarray

@dataclass(frozen=True)
class VisibilityEvidence:
    visibility: Visibility | None
    reference_mask: np.ndarray
    visible_mask: np.ndarray
    target_depth: np.ndarray
    reference_pixels: int
    reason: str = ""
    sampled_rgba: np.ndarray | None = None

class GeometryProvider(Protocol):
    def __call__(self, observation: Observation) -> InsertionGeometry: ...

class EvidenceProvider(Protocol):
    def __call__(
        self,
        observation: Observation,
        resolved: tuple[ResolvedInsertion, ...],
        geometry: InsertionGeometry,
    ) -> Mapping[str, VisibilityEvidence]: ...
```

| `InsertionGeometry` field | Contract |
|---|---|
| `actors` | World-frame `ActorPose` objects keyed by stable actor IDs or bound role aliases |
| `victim` | The victim's `ActorPose` in the same world frame |
| `cam_from_world` | 4×4 transform from world coordinates to optical camera coordinates |

All values describe the current observation, in metres. Tuple unpacking is supported.

| `VisibilityEvidence` field | Contract |
|---|---|
| `visibility` | A `Visibility`, or `None` when visibility is unknown |
| `reference_mask` | Image-sized boolean mask of the unoccluded silhouette inside the image |
| `visible_mask` | Image-sized boolean mask of the visible target pixels |
| `target_depth` | Image-sized optical-axis z of the target surface in metres, infinite outside the target |
| `reference_pixels` | Pixel count of the full silhouette, including its part outside the image |
| `reason` | Why visibility is unknown, or an empty string |
| `sampled_rgba` | Optional image-sized RGBA surface samples. The compositor reuses them. Custom providers may omit them |

An `EvidenceProvider` returns one `VisibilityEvidence` per insertion ID, for the current camera,
pose and frame.

```python
# avsectester.rendering.compositor
class InsertionRenderer:
    def __init__(
        self,
        insertions: Sequence[Insertion],
        camera: Camera,
        geometry: GeometryProvider,
        compositor: PatchCompositor | None = None,
        evidence_provider: EvidenceProvider | None = None,
    ): ...
    def __call__(self, observation: Observation, rgb: np.ndarray) -> np.ndarray: ...

def render_resolved(
    frame: np.ndarray,
    camera: Camera,
    cam_from_world: np.ndarray,
    resolved: ResolvedInsertion,
    evidence: VisibilityEvidence | None = None,
    *,
    compositor: PatchCompositor | None = None,
) -> tuple[np.ndarray, np.ndarray]: ...

def render_plane(
    frame_rgb: np.ndarray,
    camera: Any,
    cam_from_world: np.ndarray,
    corners_world: np.ndarray,
    texture_rgba: np.ndarray,
    soften: float = 0.0,
    margin: int = 2,
): ...

def warp_patch(frame_rgb: np.ndarray, dst_quad: np.ndarray, patch_rgba: np.ndarray): ...

# avsectester.rendering.geometry
def insertion_image(subject, camera: Camera, cam_from_world) -> tuple[np.ndarray, np.ndarray]: ...
def projected_silhouette(subject, camera: Camera, cam_from_world) -> np.ndarray: ...
```

| Name | Returns or does | Raises |
|---|---|---|
| `InsertionRenderer(...)` | Holds the insertions, camera, geometry provider, optional compositor and optional evidence provider | `ValueError` for duplicate insertion IDs |
| `InsertionRenderer.__call__(observation, rgb)` | Calls `geometry(observation)`, resolves every insertion with `resolve_insertion`, gets evidence from `evidence_provider`, and composites each insertion with `render_resolved`. Returns the image. The attributes `resolved` and `evidence` keep the last frame's values, and `camera` may be replaced between frames. Runs no filters and does not change the observation | `TypeError` if `geometry` returns another type. `KeyError` for a missing host |
| `render_resolved(...)` | Composites one resolved insertion using per-pixel surface depth. With `evidence`, paints only `visible_mask` pixels and reuses `sampled_rgba`. Without it, only the insertion's own surfaces occlude each other. The compositor's softening and harmonization act on the painted pixels only. Returns `(rgb uint8, mask uint8)`, with 255 where painted | `ValueError` when the frame size differs from the camera, `evidence.visibility` is `None`, or the mask size differs |
| `render_plane(...)` | Ray-casts a world-space textured rectangle, corners in top-left, top-right, bottom-right, bottom-left order, through a pinhole or f-theta camera, and alpha-composites it. `soften` is a Gaussian sigma in pixels. Returns `(rgb, mask)`, with an empty mask when the plane is out of view. No scene occlusion | `ValueError` when the frame size differs from the camera |
| `warp_patch(frame_rgb, dst_quad, patch_rgba)` | Perspective-warps an RGBA patch onto an image quad, shape `(4, 2)`, top-left, top-right, bottom-right, bottom-left, and alpha-composites it. Returns `(rgb, mask)` | — |
| `insertion_image(subject, camera, cam_from_world)` | Image-sized RGBA and optical depth of the nearest opaque surface per pixel | — |
| `projected_silhouette(subject, camera, cam_from_world)` | Boolean mask of opaque insertion pixels inside the image, before occlusion | — |

The rendering functions need the `rendering` extra (OpenCV and Pillow).

Model-input adapters return functions that change the observation passed to the driving system:

```python
# avsectester.perturbations.image
def frame_perturbation(
    insert: Callable[[Observation, Any], Any], camera: str | None = None,
    base: Callable[[Observation], Any] | None = None,
) -> Callable[[Observation], Observation]: ...

# avsectester.rendering.legacy_carla
def insertion_perturbation(simulator_runtime, insertions, *, bindings=None, camera="front",
                           compositor=None): ...

# avsectester.scenarios.datasets.nurec
NuRecDataset.insertion_renderer(self, scene, backend, insertions, bindings=None, compositor=None)

# avsectester.rendering.legacy_nurec
class NuRecInsertions(InsertionRenderer):
    def __init__(self, insertions, camera, geometry, compositor=None,
                 visibility_estimator=None, camera_name="front"): ...
```

| Name | Returns or does | Raises |
|---|---|---|
| `frame_perturbation(insert, camera, base)` | An `Observation -> Observation` function. It reads one camera frame (`camera`, or the first sensor) with `base`, by default `simulators.viz.camera_view`, and replaces the payload with `insert(observation, rgb)`. Unchanged when no frame is readable | — |
| `insertion_perturbation(simulator_runtime, insertions, *, ...)` | An `Observation -> Observation` function that renders `insertions` into a CARLA camera image at every driving step, with depth-based visibility. The runtime must provide `selection_context()`, as a runtime of `CarlaSimulator(depth_camera=...)` does. `bindings` maps roles to actor-ID tuples, normally `match.binding_ids` | `TypeError` for a string binding. `KeyError` for unavailable bound actors. `ValueError` for a missing camera or depth, or mismatched frames |
| `NuRecDataset.insertion_renderer(...)` | A `NuRecInsertions` renderer with cuboid visibility, callable as `(observation, rgb) -> rgb`. It can be created before the backend is reset. Geometry follows the bound actor IDs at the effective render time and camera pose | `ValueError` when a role alias conflicts with a track ID |
| `NuRecInsertions(...)` | An `InsertionRenderer` whose evidence provider calls `visibility_estimator.estimate` for each insertion. Without an estimator there is no evidence provider | — |

Wrap these functions in `ObservationTransform` at the `observation` stage, or wrap a renderer with
`frame_perturbation`. See [Insert into the model input](../examples/image-insertion.md#insert-into-the-model-input).

### Compositing and harmonization

```python
# avsectester.rendering.compositor
class PatchCompositor:
    def __init__(
        self, harmonizer: Harmonizer | None = None, *, soften: float = 0.0,
        harmonize: bool = True,
    ) -> None: ...

# avsectester.rendering.harmonizers
class Harmonizer(ABC):
    @abstractmethod
    def __call__(
        self, composite_rgb: np.ndarray, mask: np.ndarray, background_rgb: np.ndarray
    ) -> np.ndarray: ...

class ClassicHarmonizer(Harmonizer):
    def __init__(
        self,
        color_transfer: bool = True,
        poisson: bool = True,
        preserve_chroma: bool = False,
        blend: str = "poisson",
        feather: float = 1.2,
        gain_range: tuple = (0.35, 1.2),
    ) -> None: ...

class PCTNetHarmonizer(Harmonizer):
    def __init__(self, device: int = 0, weights: str | None = None, strict: bool = False) -> None: ...
```

| `PatchCompositor` attribute | Meaning |
|---|---|
| `harmonizer` | The harmonizer called after alpha compositing. `None` at construction selects `ClassicHarmonizer()` |
| `harmonize` | Boolean switch. When `False`, the harmonizer is neither called nor loaded. It can change between frames |
| `soften` | Gaussian blur sigma in pixels, applied to the sampled insertion before compositing. It never expands the visibility mask |

`PatchCompositor` raises `TypeError` when `harmonize` is not a bool and `ValueError` for a
negative or non-finite `soften`. Without a compositor, `render_resolved` and `InsertionRenderer`
neither soften nor harmonize, so texture colors are preserved.

A `Harmonizer` receives the alpha-composited image, the 0/255 mask of painted pixels and the
original frame, and returns an RGB image. `render_resolved` copies only the painted pixels from it.

| `ClassicHarmonizer` option | Default | Effect |
|---|---|---|
| `color_transfer` | `True` | Matches the Lab mean and standard deviation of the painted pixels to a surrounding ring (the mask dilated by a 25×25 kernel, minus the mask) |
| `poisson` | `True` | Poisson seamless cloning with OpenCV. Active only with `blend="poisson"` |
| `preserve_chroma` | `False` | With `color_transfer`, scales only Lab lightness by the ratio of ring mean to object mean, clipped to `gain_range`. Hue and internal contrast are kept |
| `blend` | `"poisson"` | `"poisson"`, or `"feather"` for an edge blended with a Gaussian-blurred mask instead of Poisson cloning |
| `feather` | `1.2` | Gaussian sigma in pixels for `blend="feather"` |
| `gain_range` | `(0.35, 1.2)` | Lower and upper bounds of the lightness gain |

| `PCTNetHarmonizer` option | Default | Effect |
|---|---|---|
| `device` | `0` | CUDA device index. Runs on `cuda:<device>` when CUDA is available, otherwise on the CPU |
| `weights` | `None` | Path to `PCTNet.pth`. `None` uses `third_party/libcom/libcom/image_harmonization/pretrained_models/PCTNet.pth` and downloads it from the Hugging Face repository `BCMIZB/Libcom_pretrained_models` when missing |
| `strict` | `False` | `False` logs a warning and falls back to `ClassicHarmonizer()` on any loading or inference error. `True` raises `RuntimeError` |

`PCTNetHarmonizer` loads PCTNet in-process from the `third_party/libcom` submodule at its first
call. It requires PyTorch. Weight setup is described in
[PCTNet harmonizer](../getting-started/installation.md#pctnet-harmonizer), and usage in
[Harmonization](../examples/image-insertion.md#harmonization).

### Visibility

```python
# avsectester.rendering.types (Visibility is also exported by avsectester.rendering
# and avsectester.scenarios)
@dataclass(frozen=True)
class Visibility:
    fraction: float
    source: str
    camera: str | None = None
    label: str | None = None

class GeometryVisibilityEstimator(Protocol):
    def estimate(
        self,
        subject: ResolvedInsertion,
        camera: Camera,
        cam_from_world: np.ndarray,
        *,
        occluders: Mapping[str, ActorPose],
        camera_name: str,
        other_insertions: Sequence[ResolvedInsertion] = (),
    ) -> VisibilityEvidence: ...

# avsectester.rendering.visibility
# Both estimators take the constructor
#     (self, *, tolerance_m: float = 0.002, max_reference_pixels: int = 4_000_000)
class CuboidVisibilityEstimator:
    def estimate(
        self,
        subject: ResolvedInsertion,
        camera: Camera,
        cam_from_world: np.ndarray,
        *,
        occluders: Mapping[str, ActorPose],
        camera_name: str,
        other_insertions: Sequence[ResolvedInsertion] = (),
    ) -> VisibilityEvidence: ...

class DepthVisibilityEstimator:
    def estimate(
        self,
        subject: ResolvedInsertion,
        camera: Camera,
        cam_from_world: np.ndarray,
        *,
        scene_depth: np.ndarray,
        depth_convention: str,
        camera_name: str,
        other_insertions: Sequence[ResolvedInsertion] = (),
    ) -> VisibilityEvidence: ...

def visibility_from_masks(
    reference, visible, *, camera: str, image_bounds=None
) -> Visibility | None: ...

def visibility_from_depth(
    footprint,
    target_depth,
    scene_depth,
    *,
    camera: str,
    tolerance_m: float = 0.0002,
    image_bounds=None,
) -> Visibility | None: ...
```

| Name | `source` | Returns or does | Raises |
|---|---|---|---|
| `Visibility` | — | A visible fraction with its source, its camera (`None` for source-wide) and an optional label | `ValueError` unless `fraction` is finite and in `[0, 1]` |
| `CuboidVisibilityEstimator.estimate` | `cuboid_estimate` | Tests target rays against the boxes in `occluders`, which should include the host, and against the surfaces of `other_insertions` | — |
| `DepthVisibilityEstimator.estimate` | `depth_comparison` | Compares the target distance with `scene_depth` for the same camera pose, lens and frame. `depth_convention` is `"z"` for optical-axis depth or `"range"` for ray distance. Other insertions also occlude. Invalid depth at a target pixel makes visibility unknown | `ValueError` for another convention or a `scene_depth` shape that differs from the image |
| `visibility_from_masks` | `rendered_masks` | Visible in-image reference pixels divided by all reference pixels. `image_bounds=(left, top, right, bottom)` limits the numerator to the real image. `None` for an empty reference | `ValueError` for misaligned masks or invalid bounds |
| `visibility_from_depth` | `depth_comparison` | Builds the visible mask from `target_depth <= scene_depth + tolerance_m` and applies `visibility_from_masks` | `ValueError` for misaligned arrays, a negative tolerance, or invalid depth inside the in-image footprint |

The estimator constructors raise `ValueError` for a negative or non-finite `tolerance_m` or a
`max_reference_pixels` that is not a positive integer. Both estimators return
`VisibilityEvidence`. A target fully behind the camera has visibility `0.0`. A target crossing the
camera plane, a footprint larger than `max_reference_pixels` or an undefined projection gives
`visibility=None` with a `reason`. The measurement definition and provider table are in
[Visibility](../guides/scenarios.md#visibility).

## Metrics and evaluation

### Driving-impact metric

```python
# avsectester.evaluation.metric
@dataclass
class Impact:
    clean_peak_speed: float
    clean_final_speed: float
    attacked_final_speed: float
    clean_brake_frames: int
    attacked_brake_frames: int
    clean_drove: bool
    induced_braking: bool
    induced_stop: bool
    suppressed_stop: bool = False
    overdrive_frames: int = 0

    @property
    def attack_succeeded(self) -> bool: ...
    @property
    def verdict(self) -> str: ...
    def comparison(self, treated_label: str) -> str: ...

def impact(
    clean: Trace,
    attacked: Trace,
    stop_speed: float = 0.5,
    baseline_speed: float = 1.0,
    min_overdrive_frames: int = 3,
) -> Impact: ...

def plot_impact(
    clean: Trace,
    attacked: Trace,
    path: str,
    title: str | None = None,
    result: Impact | None = None,
    *,
    treated_label: str = "attacked",
    show_verdict: bool = True,
) -> str: ...
```

| `impact` parameter | Default | Meaning |
|---|---|---|
| `stop_speed` | `0.5` | Speed in m/s at or below which the ego counts as stopped |
| `baseline_speed` | `1.0` | Peak speed in m/s the clean run must reach for `clean_drove`. Also the speed above which the treated ego counts as moving for `overdrive_frames` |
| `min_overdrive_frames` | `3` | Minimum `overdrive_frames` for `suppressed_stop` |

| `Impact` field | Value |
|---|---|
| `clean_peak_speed`, `clean_final_speed` | `clean.peak_speed`, `clean.final_speed` |
| `attacked_final_speed` | `attacked.final_speed` |
| `clean_brake_frames`, `attacked_brake_frames` | `braking_frames` of each trace |
| `clean_drove` | `clean.peak_speed >= baseline_speed` |
| `induced_braking` | `attacked.braking_frames > clean.braking_frames` |
| `induced_stop` | The treated final speed is at most `stop_speed` and the clean final speed is above it |
| `suppressed_stop` | The clean final speed is at most `stop_speed` and `overdrive_frames >= min_overdrive_frames` |
| `overdrive_frames` | Records, paired by index, where the clean speed is at most `stop_speed` and the treated speed exceeds `baseline_speed` |

| `Impact` member | Returns |
|---|---|
| `attack_succeeded` | `clean_drove and (induced_stop or suppressed_stop)` |
| `verdict` | A one-line verdict. It is inconclusive when `clean_drove` is `False` |
| `comparison(treated_label)` | Two lines of measurements, labelled `clean:` and `treated_label`, without a verdict |
| `str(impact)` | `comparison("attacked")` followed by `=> <verdict>` |

`plot_impact` draws ego speed and brake command over driving steps for both traces and saves the
figure to `path`, creating directories. It returns the absolute path. `result` defaults to
`impact(clean, attacked)`. `treated_label` names the second trace. `show_verdict=False` omits the
verdict caption. The default title is `Adversarial attack — driving impact (clean vs
<treated_label>)`, or `Driving comparison (clean vs <treated_label>)` without the verdict. An
explicit `title` replaces it. These options do not change the scores. It requires matplotlib, the
`viz` extra. The verdict semantics are in
[Driving-impact metric](../guides/evaluation.md#driving-impact-metric) and usage in
[Plot a comparison](../guides/evaluation.md#plot-a-comparison).

### Robustness evaluation

```python
# avsectester.evaluation.robustness (evaluate_robustness and RobustnessReport are also
# exported by avsectester.evaluation)
Perturb = Callable[["Observation"], "Observation"]
AttackFor = Callable[["ScenarioMatch | None", "WorldBackend"], Runtime | Perturb | None]
OutcomeStatus = Literal["success", "failure", "skipped", "inconclusive"]

def evaluate_robustness(
    source: ScenarioSource,
    req: ScenarioRequirement,
    attack_for: AttackFor | Intervention | PreparedIntervention,
    stack: Callable[[], AVStack],
    frames: int,
    *,
    conditions: Iterable[AugmentationPipeline] | None = None,
    severity: float = 0.5,
    limit: int | None = None,
    run_fn: Callable[..., Trace] = _run_loop,  # avsectester.backend.run
) -> RobustnessReport: ...

@dataclass
class RobustnessReport:
    per_condition: dict[str, ConditionResult] = field(default_factory=dict)
    rows: list[ScenarioResult] = field(default_factory=list)
    baseline: str = "clean"

    def record(self, scenario_id: str, condition: str, status: OutcomeStatus,
               reason: str = "") -> None: ...
    @property
    def baseline_asr(self) -> float: ...
    def resilience(self, condition: str) -> float: ...
    @property
    def mean_resilience(self) -> float: ...
    def summary(self) -> str: ...

@dataclass
class ConditionResult:
    condition: str
    successes: int = 0
    failures: int = 0
    skipped: int = 0
    inconclusive: int = 0
    @property
    def n(self) -> int: ...
    @property
    def asr(self) -> float: ...

@dataclass(frozen=True)
class ScenarioResult:
    scenario_id: str
    condition: str
    status: OutcomeStatus
    reason: str = ""
```

`evaluate_robustness` runs a clean and an attacked treatment for every scenario from
`source.scenarios(req, limit=limit)` and every condition: first the scenario's own conditions,
labelled `clean`, then each pipeline in `conditions` added as an image condition after the other
handlers. `conditions=None` uses `common_corruptions(severity)`. Each pair is scored with `impact`
and recorded in the returned `RobustnessReport`. It raises `ValueError` for an invalid `frames` or
a `PreparedIntervention` whose kind is not `attack`, and `TypeError` for another `attack_for` type
or a condition that is not an `AugmentationPipeline`.

| `attack_for` | Behavior |
|---|---|
| `Intervention` | Prepared once per scenario. Fresh hooks for each attacked run |
| `PreparedIntervention` | Its artifacts are reused for every scenario |
| `AttackFor` callback | Called once per pair after world conditions are configured. Returns a `Runtime`, a one-argument observation function, or `None` to skip the pair |

| Member | Returns |
|---|---|
| `RobustnessReport.record(...)` | Appends a `ScenarioResult` and updates the condition's counters |
| `baseline_asr` | `per_condition[baseline].asr`, or NaN without that condition |
| `resilience(condition)` | `asr(condition) / baseline_asr`, at most 1. NaN when either is NaN. `0.0` when the baseline ASR is zero |
| `mean_resilience` | Mean resilience over the non-baseline conditions, or `0.0` without them |
| `summary()` | A text table of valid, skipped and inconclusive counts, ASR and resilience per condition |
| `ConditionResult.n`, `asr` | `successes + failures`, and `successes / n`, NaN when `n` is zero |

The scenario ID in each row is the scenario provenance's `scene_id`, `sample_token` or
`image_path`, or the scenario's index. Statuses, ASR and resilience are explained in
[Statuses, ASR and resilience](../guides/evaluation.md#statuses-asr-and-resilience), and the
parameters in [Parameters](../guides/evaluation.md#parameters).

### Component logging

```python
# avsectester.evaluation.component_log (all names are also exported by avsectester.evaluation)
def run_logged(
    simulator: WorldBackend,
    avstack: AVStack,
    frames: int,
    perturb: Callable[[Observation], Observation] | None = None,
    *,
    hooks: HookSet | None = None,
) -> tuple[Trace, ComponentTrace]: ...

@runtime_checkable
class InstrumentedAVStack(Protocol):
    def component_log(self) -> dict[str, Any] | None: ...

@dataclass
class StepLog:
    frame: int
    stages: dict[str, Any] = field(default_factory=dict)
    def count(self, stage: str) -> int: ...

@dataclass
class ComponentTrace:
    steps: list[StepLog] = field(default_factory=list)
    @property
    def stage_names(self) -> list[str]: ...
    def counts(self, stage: str) -> list[int]: ...
    def degradation(self, other: ComponentTrace, stage: str) -> list[int]: ...
    def performance(self, stage: str, truths: list[Any], assign_radius: float = 4.0) -> list[Any]: ...
```

`run_logged` uses `ClosedLoopRunner` and returns compatibility views of the captured RunLog.
For an AVStack implementing `InstrumentedAVStack`, each `StepLog` reads the corresponding record's
snapshotted `components`; no second collector or inference runs. Otherwise the `ComponentTrace`
is empty. The caller owns the simulator. `StepLog.frame` is the driving-step index.
The `ComponentTrace` methods and timing rules are described in
[Component outputs](../guides/evaluation.md#component-outputs).

## Visualization

```python
# avsectester.closed_loop.simulators.viz
View = Callable[[Observation], Any]
REPO_TMP: Path  # <repo>/tmp

def record_run(
    simulator: WorldBackend,
    avstack: AVStack,
    frames: int,
    *,
    out_dir: str | Path | None = None,
    visualize: View = camera_view,
    perturb: Callable[[Observation], Observation] | None = None,
    prefix: str = "frame",
    collect: bool = False,
    hooks: HookSet | None = None,
) -> Trace: ...

def camera_view(observation: Observation, camera: str | None = None) -> Any: ...
def as_rgb(frame: Any) -> Any: ...
def annotate(image: Any, detections: Any, threshold: float = 0.3) -> Any: ...
def scene_labels(scene: Any, camera: str | None = None, target: Any = None,
                 color=(40, 200, 40), target_color=(235, 64, 52)) -> list: ...
def box3d_corners(center: Any, extent: Any, yaw: float) -> Any: ...
def ego_projector(model: Any) -> Any: ...
def draw_boxes3d(image: Any, scene: Any, camera: str | None = None, target: Any = None,
                 subdiv: int = 8, max_distance: float | None = None,
                 color=(40, 200, 40), target_color=(235, 64, 52)) -> Any: ...
def labels_view(get_scene: Callable[[Observation], Any], base: View = camera_view,
                camera: str | None = None, target: Any = None, boxes3d: bool = True,
                max_distance: float | None = None) -> View: ...
def detections_view(
    detect: Callable[[Any], Any], base: View = camera_view, threshold: float = 0.3
) -> View: ...
def save_image(image: Any, path: Path) -> None: ...
def filmstrip(images: list, cols: int = 4, pad: int = 6, bg=(20, 20, 20)) -> Any: ...
def save_gif(images: list, path: str | Path, fps: int = 5) -> None: ...
def save_sequence(frames: list, out_dir: str | Path, name: str = "sequence",
                  cols: int = 4, fps: int = 4) -> tuple[str, str] | None: ...

# avsectester.closed_loop.simulators.carla
def camera_view(observation: Observation, camera: str | None = None) -> Any: ...
def lidar_bev(observation: Observation, size: int = 800, meters: float = 60.0) -> Any: ...
```

| Function | Returns or does |
|---|---|
| `record_run(...)` | Runs the same loop as `run` and saves `visualize(seen)` of each driving step's model input as `<out_dir>/<prefix>_<step:04d>.png`. `out_dir` defaults to `<repo>/tmp/frames`. `collect=True` also stores the images in `trace.frames`. Returns the `Trace`. The caller owns the simulator |
| `camera_view(observation, camera)` | The RGB frame of `camera`, or of the first sensor, when the payload is an ndarray. Otherwise `None` |
| `as_rgb(frame)` | An `(H, W, 3)` uint8 array from an ndarray payload, or `None` |
| `annotate(image, detections, threshold)` | Draws `(xyxy, score, label)` detections, colored by `score >= threshold`, or `(xyxy, score, label, color)` labels |
| `scene_labels(scene, camera, target, ...)` | `annotate` tuples for the 2D boxes of a `SceneGT`. `target` is drawn last in `target_color` |
| `box3d_corners(center, extent, yaw)` | The eight corners `(8, 3)` of a box in the ego frame |
| `ego_projector(model)` | A function mapping ego points to `(pixels, valid)` for a camera model or a 3×3 K, or `None` |
| `draw_boxes3d(image, scene, ...)` | Draws projected 3D boxes of a `SceneGT`, skipping boxes that do not project fully. Falls back to 2D boxes when the model cannot project |
| `labels_view(get_scene, base, ...)` | A view that overlays ground-truth boxes from `get_scene(observation)` on `base` |
| `detections_view(detect, base, threshold)` | A view that overlays `detect(rgb)` detections on `base` |
| `save_image(image, path)` | Writes an image with matplotlib, creating directories |
| `filmstrip(images, cols, pad, bg)` | A grid image of the frames |
| `save_gif(images, path, fps)` | Writes an animated GIF |
| `save_sequence(frames, out_dir, name, cols, fps)` | Writes `<name>_filmstrip.png` and `<name>.gif` in `out_dir` and returns both paths, or `None` for no frames |
| `simulators.carla.camera_view` | The RGB image of a CARLA `ImageData` payload |
| `simulators.carla.lidar_bev(observation, size, meters)` | A top-down image of a CARLA lidar point cloud, or `None` |

The clean-versus-treated figure is `metric.plot_impact`, listed under
[Driving-impact metric](#driving-impact-metric). Scene recording is described in
[Scene recordings](../guides/evaluation.md#scene-recordings).

## Module map

| Module | Contents | Guide |
|---|---|---|
| `avsectester.experiment` | `Experiment`, `ExperimentConfiguration`, `TreatmentConfiguration`, `load_configuration`, `PreparedExperiment`, `ExperimentResult`, `SelectionDecision`, `TreatmentResult` | [Experiment preparation and execution](#experiment-preparation-and-execution) |
| `avsectester.closed_loop.datatypes` | `Observation`, `Control`, `WorldSnapshot`, `StateEstimate`, `FrameRecord`, `RecordedOutput`, `Trace`, `read_camera_rgb`, `write_camera_rgb` | [Driving traces](../guides/evaluation.md#driving-traces) |
| `avsectester.closed_loop.avstack` | `AVStack`, `CommandCompatibilityError`, `validate_command_compatibility` | [Use a custom driving system](../guides/driving-systems.md#use-a-custom-driving-system) |
| `avsectester.backend` | Compatibility entry point `run`, returning a `Trace` from the shared loop | [Run functions](#run-functions) |
| `avsectester.datasets` | `Dataset`, `FrameReference`, `FrameData`, `ActorState`, `SensorDataReference`, `ScenarioStart`, `ScenarioDescriptor`, `Unavailable`; `nurec`, `nuscenes` and `carla` datasets | [Stateless datasets](../guides/scenarios.md#stateless-datasets) |
| `avsectester.closed_loop.frame_data` | `FrameDataCache`, `FrameDataSequence`, `Initialization` | [Stateless datasets](../guides/scenarios.md#stateless-datasets) |
| `avsectester.closed_loop.simulator` | `Simulator`, `SimulatorRuntime`, `SensorView`, `DeploymentPlan`, `ClockSnapshot` | [Simulator](#simulator) |
| `avsectester.attack` | Re-exports the user-facing attack API: `Attack`, `AttackSegment`, `OfflineContext` (module `attack.attack`), `split_segments` (module `attack.segments`) and the names of `attack.runtime` | [Attack](#attack) |
| `avsectester.attack.runtime` | `AttackRuntime`, `OnlineContext`, `ResultSubscription`, `AttackEffect`, `WorldAttackEffect`, `SensorAttackEffect`, `AVStackAttackEffect` | [Online phase](runtime.md#online-phase) |
| `avsectester.closed_loop.environment` | `normalize_environment`, `merge_environments`, `validate_environment`, `SensorEnvironment` | [Environment API](#environment-api) |
| `avsectester.closed_loop.sensor_transforms` | `Corruption`, `CORRUPTIONS` and the built-in RGB corruption classes | [Image corruption API](#image-corruption-api) |
| `avsectester.closed_loop.services` | `ClosedLoopServices`, `ClosedLoopSession`, `ScenarioView`, `CandidateRequest`, `SensorRequest`, `RenderRequest`, `RenderResult`, `DrivingAdapter`, `ensure_fresh` | [Shared closed-loop services](../guides/scenarios.md#shared-closed-loop-services) |
| `avsectester.closed_loop.runner` | `ClosedLoopRunner`, `RunRequest`, `RecordingPolicy`, `trace_from_log` | [Run logs and saved results](../guides/evaluation.md#run-logs-and-saved-results) |
| `avsectester.runlog` | `RunLog`, `StepRecord`, `EventRecord`, `Recorder`, `CodecRegistry`, `read_table` | [RunLog and recording](#runlog-and-recording) |
| `avsectester.runlog.codecs` | `default_codecs` | [Native adapter codecs](#native-adapter-codecs) |
| `avsectester.closed_loop.hooks` | `HookSet`, `Runtime`, `Hook`, `Plugin`, `RunContext`, `StageContext`, `CallInputs`, `CameraRenderRequest`, `HookDispatcher` | [Runtime reference](runtime.md) |
| `avsectester.interventions` | `ObservationTransform`, `LegacyNativeHook`, `NativeHookIntervention`, configuration helpers | [Package the attack as an intervention](../guides/attacks.md#package-the-attack-as-an-intervention) |
| `avsectester.preparation` | `Intervention`, `PreparedIntervention`, `prepare_intervention`, `Treatment` | [Package the attack as an intervention](../guides/attacks.md#package-the-attack-as-an-intervention) |
| `avsectester.runner` | `run_pair`, `run_prepared` | [Custom treatments](../guides/experiments.md#custom-treatments), [Direct runs](../guides/experiments.md#direct-runs-and-resource-ownership) |
| `avsectester.legacy_experiment` | `run_configured`, `configured_stack`, `ExecutionMode`, `TreatmentMode`, `ConfigurationError` | [Configured runs](../guides/experiments.md#configured-runs) |
| `avsectester.carla_runner` | `run_scenario`, `run_scenario_pair` | [Python entry points](../guides/simulators.md#python-entry-points) |
| `avsectester.conditions` | `ScenarioConditions`, `SensorCondition`, `ConditionConfigurationError` | [Apply operating conditions](../guides/conditions.md) |
| `avsectester.insertion` | `Insertion`, assets, placements, `Orientation`, `resolve_insertion` | [Specify inserted objects](../guides/scenarios.md#specify-inserted-objects) |
| `avsectester.evaluation.metric` | `impact`, `Impact`, `plot_impact` | [Driving-impact metric](../guides/evaluation.md#driving-impact-metric) |
| `avsectester.cli` | The `avsectester run` and `avsectester version` commands | [CLI reference](cli.md) |
| `avsectester.scenarios` | Re-exports of the scenario, filter, condition and serialization API | [Prepare scenarios](../guides/scenarios.md) |
| `avsectester.scenarios.source` | `ScenarioInstance`, `ScenarioSource`, `Dataset`, `DatasetFilter`, `CarlaScenarioBuilder` | [Select a scenario](../guides/scenarios.md#select-a-scenario) |
| `avsectester.scenarios.setup` | `scenario_from_config`, `source_from_config`, `scene_config`, `ConfiguredScenarioSource`, `ScenarioConfigurationError` | [Configured scenarios](../guides/scenarios.md#configured-scenarios) |
| `avsectester.scenarios.requirement` | `ScenarioRequirement`, `RoleSpec`, `TargetSpec`, `InitialWindow`, `ScenarioMatch`, `SelectionResult`, `CandidateResult` | [Compose a requirement](../guides/scenarios.md#compose-a-requirement) |
| `avsectester.scenarios.filters` | `FilterResult`, `Constraint`, `All`, `Any`, `Not` and the built-in filters | [Filters](../guides/scenarios.md#filters) |
| `avsectester.scenarios.context` | `FilterContext` | [Custom filters and native access](../guides/scenarios.md#custom-filters-and-native-access) |
| `avsectester.scenarios.scene` | `SceneGT`, `EgoState`, `CameraCalib`, `TargetGeometry`, `ObjectGT`, `PlacementCandidate` | [Support another dataset or provider](../guides/scenarios.md#support-another-dataset-or-provider) |
| `avsectester.scenarios.serialize` | `requirement_to_dict`, `requirement_from_dict`, `register_constraint`, `constraint_vocabulary` | [Selection in YAML](../guides/scenarios.md#selection-in-yaml) |
| `avsectester.scenarios.nl` | `interpret`, `build_prompt`, `parse_response` | [Natural-language requirements](../guides/scenarios.md#natural-language-requirements) |
| `avsectester.scenarios.presets` | `PHYSICAL_PATCH_HIDE_VEHICLE`, `REQUIREMENTS` | [Compose a requirement](../guides/scenarios.md#compose-a-requirement) |
| `avsectester.scenarios.carla_provider` | `CarlaCandidateProvider`, `carla_filter_context`, `depth_visibility` | [CARLA](../guides/scenarios.md#carla) |
| `avsectester.scenarios.carla_gt` | `carla_scene_gt`, `predict_scene_gt`, `carla_actor_pose`, `carla_rgb_sensor` | [CARLA](../guides/scenarios.md#carla) |
| `avsectester.scenarios.datasets.nurec` | `NuRecDataset` and its rig-frame `FThetaCamera` | [NuRec datasets](../guides/scenarios.md#nurec-datasets) |
| `avsectester.scenarios.datasets.nuscenes` | `NuScenesDataset`, `scene_from_cam_boxes` | [Support another dataset or provider](../guides/scenarios.md#support-another-dataset-or-provider) |
| `avsectester.scenarios.estimators` | `resolved_to_target` | [Visibility](../guides/scenarios.md#visibility) |
| `avsectester.rendering` | Re-exports `Camera`, `InsertionGeometry`, `Visibility`, `VisibilityEvidence` and the provider protocols | [Insert into the model input](../examples/image-insertion.md#insert-into-the-model-input) |
| `avsectester.geometry` | `ActorPose`, `Camera`, `FThetaCamera`, `PinholeCamera`, `rotation_matrix` | [Insertion and rendering](#insertion-and-rendering) |
| `avsectester.geometry.cameras` | `Camera`, `PinholeCamera`, `FThetaCamera`, `camera_from_calibration`, pose and projection helpers | [Insertion rendering](#insertion-rendering) |
| `avsectester.geometry.poses` | `ActorPose`, `rotation_matrix`, `validated_vector`, `validated_transform` | [Insertion specification](#insertion-specification) |
| `avsectester.rendering.types` | `Visibility`, `InsertionGeometry`, `VisibilityEvidence`, `GeometryProvider`, `EvidenceProvider`, `GeometryVisibilityEstimator` | [Visibility](../guides/scenarios.md#visibility) |
| `avsectester.rendering.geometry` | Shared surface sampling: `insertion_image`, `projected_silhouette` and internal helpers | [Rendering scope](../examples/image-insertion.md#rendering-scope) |
| `avsectester.rendering.legacy_carla` | `insertion_perturbation`, retained for existing concrete insertion attacks | [Insertion rendering](#insertion-rendering) |
| `avsectester.rendering.legacy_nurec` | `NuRecInsertions`, `insertion_renderer`, retained for existing concrete insertion attacks | [Insertion rendering](#insertion-rendering) |
| `avsectester.rendering.compositor` | `InsertionRenderer`, `PatchCompositor`, `render_resolved`, `render_plane`, `warp_patch` | [Insert into the model input](../examples/image-insertion.md#insert-into-the-model-input) |
| `avsectester.rendering.harmonizers` | `Harmonizer`, `ClassicHarmonizer`, `PCTNetHarmonizer` | [Harmonization](../examples/image-insertion.md#harmonization) |
| `avsectester.rendering.visibility` | `CuboidVisibilityEstimator`, `DepthVisibilityEstimator`, `visibility_from_masks`, `visibility_from_depth` | [Visibility](../guides/scenarios.md#visibility) |
| `avsectester.perturbations.image` | `AugmentationPipeline`, corruption suites, Albumentations wrappers, `sensor_augmentation`, `compose`, `frame_perturbation`; compatibility exports of sensor corruption operators | [Built-in image corruptions](../guides/conditions.md#built-in-image-corruptions) |
| `avsectester.closed_loop.simulators.carla` | `CarlaSimulator`, `CarlaSimulatorRuntime`, `carla_sensor_view`, `register_pinhole_camera`, `decode_carla_depth`, `camera_view`, `lidar_bev`, `prepare_scenario`, `resolve_scenario_seed`, `set_perception_gpu` | [CARLA](../guides/simulators.md#carla) |
| `avsectester.closed_loop.simulators.nurec` | `NuRecSimulator`, `NuRecSimulatorRuntime`, `Renderer`, `StubRenderer`, `NuRecRenderer`, `EgoPose`, `EgoDynamics`, `KinematicBicycle`, `TrajectoryFollower`, `nurec_sensor_view` | [NuRec](../guides/simulators.md#nurec) |
| `avsectester.closed_loop.simulators.viz` | `record_run`, views, `filmstrip`, `save_gif`, `save_sequence` | [Scene recordings](../guides/evaluation.md#scene-recordings) |
| `avsectester.closed_loop.avstacks.modular` | `ModularAVStack` | [Modular driving system](../guides/driving-systems.md#modular-driving-system) |
| `avsectester.closed_loop.avstacks.alpamayo` | `AlpamayoAVStack` | [Alpamayo](../guides/driving-systems.md#alpamayo) |
| `avsectester.closed_loop.avstacks.native_hook` | `NativeHook`, adapter for explicitly registered avstack hooks | [Runtime adapters](#runtime-adapters) |
| `avsectester.closed_loop.avstacks.scorers` | `MMDetInferenceScorer`, `MMDetRPNObjectnessScorer`, region `TargetSpec` protocol | [Current optimization interfaces](../guides/optimization.md#current-optimization-interfaces) |
| `avsectester.closed_loop.avstacks.mmdet3d` | `register_carla_dataset` | [Pipeline configuration](../guides/driving-systems.md#pipeline-configuration) |
| `avsectester.attacks` | `PhantomInjection`, imported on first access | [Phantom injection](../examples/phantom-injection.md) |
| `avsectester.attacks.pipeline.phantom` | `PhantomInjection` | [Phantom injection](../examples/phantom-injection.md) |
| `avsectester.attacks.patch.physical_patch` | `PhysicalPatch`, `PhysicalPatchPlugin`, `PhysicalPatchIntervention`, `build_patch`, patch textures | [Physical patch](../examples/physical-patch.md) |
| `avsectester.attacks.object_insertion` | `sign_spoof`, `person_poster` and `traffic_light` payload builders | [Payloads](../examples/image-insertion.md#payloads) |
| `avsectester.attacks.adaptive` | `SpeedFeedbackAttack`, `SpeedFeedbackPlugin` | [Speed-feedback adaptive attack](../examples/speed-feedback-attack.md) |
| `avsectester.attacks.optim` | `AdvSample`, `AttackObjective`, `AttackResult`, `DataSource`, `GradientAttack`, `HideObject`, `Perturbation`, `Scorer`, `SpoofObject`, `TargetSpec`. Submodules add `PGD`, `NES`, `PatchPerturbation`, `LinfImage`, `CarlaRenderPerturbation` and `FrameSource`. MMDetection scorer implementations are in `closed_loop.avstacks.scorers`. This `TargetSpec` differs from the scenario `TargetSpec` | [Current optimization interfaces](../guides/optimization.md#current-optimization-interfaces) |
| `avsectester.evaluation` | `evaluate_robustness`, `RobustnessReport`, `run_logged`, `ComponentTrace`, `StepLog`, `InstrumentedAVStack` | [Evaluate results](../guides/evaluation.md) |

The repository layout beyond the package is described in
[Repository layout](../development/contributing.md#repository-layout).
