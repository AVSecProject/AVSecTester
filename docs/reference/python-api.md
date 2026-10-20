# Python API reference

> **Audience:** Python experiment, attack and adapter authors.
>
> **Outcome:** Locate public contracts and their current signatures.
>
> **Prerequisites:** [Architecture](../concepts/architecture.md).

## Module map

Imports are from the owning modules below. Heavy SDK/model imports belong to implementations; plain
data and boundary modules can be imported without simulator servers or weights.

| Module | Public purpose |
|---|---|
| avsectester.experiment | Experiment, ExperimentConfiguration, PreparedExperiment and result types |
| avsectester.experiment.config | load_configuration and treatment/configuration validation |
| avsectester.attack | Attack, AttackSegment, OfflineContext, AttackRuntime and effects |
| avsectester.attack.segments | split_segments |
| avsectester.attack.placement | AttachedPlacement, WorldPlacement and Orientation |
| avsectester.attack.implemented_attacks.<attack_folder> | Concrete Attack class exported by its folder |
| avsectester.attack.insertion | Assets, Insertion, compositing, visibility and harmonization |
| avsectester.attack.optimization | Generic optimization contracts, PGD, NES, perturbations and SessionScorer |
| avsectester.datasets | Dataset and shared frame/actor/start records |
| avsectester.datasets.carla | CarlaDataset, CarlaRecipe and PrescribedMotion |
| avsectester.datasets.nurec | NuRecDataset |
| avsectester.datasets.nuscenes | NuScenesDataset |
| avsectester.geometry.cameras | Camera, PinholeCamera, FThetaCamera and pose/projection helpers |
| avsectester.geometry.poses | ActorPose and canonical rigid-pose validation |
| avsectester.closed_loop.datatypes | Observation, Control, WorldSnapshot, StateEstimate, Detection, RGB transport |
| avsectester.closed_loop.simulator | Simulator, SimulatorRuntime, SensorView, SensorGeometry and DeploymentPlan |
| avsectester.closed_loop.simulators.carla | CarlaSimulator and CarlaSimulatorRuntime |
| avsectester.closed_loop.simulators.nurec | NuRecSimulator, NuRecSimulatorRuntime, Renderer, StubRenderer, NuRecRenderer and ego dynamics |
| avsectester.closed_loop.simulators.recorded | RecordedReplaySimulator and RecordedReplaySimulatorRuntime |
| avsectester.closed_loop.avstack | AVStack and command compatibility checks |
| avsectester.closed_loop.avstacks | ModularAVStack, AlpamayoAVStack and CameraForwardCollisionAVStack |
| avsectester.closed_loop.avstacks.native_hook | NativeHook for explicitly registered native hooks |
| avsectester.closed_loop.avstacks.scorers | Driving-model scorer implementations |
| avsectester.closed_loop.avstacks.mmdet3d | Native MMDetection3D model and dataset registration |
| avsectester.closed_loop.vehicle_io | VehicleIO and localizer lifecycle |
| avsectester.closed_loop.hooks | HookSet, Hook, Plugin, contexts and stage values |
| avsectester.closed_loop.runner | ClosedLoopRunner, RunRequest and RecordingPolicy |
| avsectester.closed_loop.services | ClosedLoopServices/session, ScenarioView, requests and component/scorer bindings |
| avsectester.closed_loop.driving_adapters | Modular/Alpamayo component-binding constructors |
| avsectester.closed_loop.environment | Environment validation/merge and SensorEnvironment |
| avsectester.closed_loop.sensor_transforms | Built-in image corruptions and AlbumentationsCorruption |
| avsectester.closed_loop.frame_data | Preparation-owned frame cache, sequence and initialization records |
| avsectester.runlog | RunLog, Recorder, StepRecord, EventRecord, CodecRegistry and read_table |
| avsectester.runlog.codecs | default_codecs for native adapter values |
| avsectester.evaluation.metric | ComparisonGroup, Metric and built-in metrics |
| avsectester.evaluation.report | EvaluationConfiguration, EvaluationReport and evaluate |
| avsectester.evaluation.visualization | Log-only frames, overlays, filmstrips and animation exports |

## Experiment API

Use Experiment.prepare(config) and execute(prepared). Configuration can be an existing typed object,
mapping or YAML path. PreparedExperiment owns explicit metadata and artifacts and no live resources.
Save/load are methods on PreparedExperiment. ExperimentResult.load(output, config) loads saved runs
for evaluation without starting a simulator or loading artifacts for deployment.

Configuration summaries are lazy. Direct Python attacks override configuration_parameters when
constructor inputs cannot be read from same-named attributes. Unverifiable parameters warn on save,
are explicitly marked unverified, and prevent validated preparation loading. They do not prevent
configuration construction or in-memory execution.

Configurations contain exactly one attack. Treatments enable its attack_name and optional
environment overrides on shared metadata. Every treatment pair is evaluated; no baseline is
configured. The same hooks apply to every treatment. Result failures and selection errors remain
available independently.

ConfigurationMismatchError identifies an incompatible saved attack/configuration summary. Other
result-reading exceptions remain read failures rather than configuration errors.
ExperimentResult.notices contains ordinary messages for skipped saved runs that the configuration or
saved preparation does not expect, such as removed treatments. run_directory defines the only run
layout; saved_run_directories lists saved runs by that layout.

## Attack API

Attack parameters and assets remain reusable configuration. build_metadata(scenario) is the
metabuilder; run_offline(context) defaults to assets; create_runtime(segment, artifacts) defaults to
an empty runtime. The explicit artifact argument is the sole saved/executed value.

AttackSegment start/stop belong to the same scenario, stop > start and decision_steps is their
position difference. roles maps names to stable actor-ID tuples. placement, selection_score,
selection_rank and user_data are optional and interpreted only by the metabuilder. The framework
validates their structure without ranking or changing their meaning.

AttackRuntime setup returns only world effects; step returns typed effects once per decision; stop
closes the online lifecycle. All history and bound AVStack callbacks share one runtime instance.
subscriptions supports observation and physical_state snapshots. OnlineContext gives physical
frame/time and record_output but no live time-advancing resources.

Attack.deployment_ticks(segment) declares physical setup needs. OfflineContext includes metadata,
selection seed, resolved experiment/attack environment and the common DeploymentPlan.

World effects receive SimulatorRuntime and an ExitStack for cleanup. Sensor effects receive the
current sensor value and a SensorView containing the original capture. AVStack effects receive the
current value at their declared stage and OnlineContext. See [Runtime](runtime.md).

## Dataset API

Dataset methods expose recorded records or recipes and never own simulation. N decisions require N+1
supported FrameReference records. Frames use canonical timeline position separately from native
frame_id. ActorState.pose is world-from-actor; box_center is its local bounding-box center, extent
is full length/width/height in metres and velocity is world-frame m/s. box_rotation describes local
box orientation. box_transform() composes the world-from-box pose; box_pose() preserves the optional
extent in ActorPose. Unknown extent is None, distinct from explicit zero dimensions. Pose-only
operations accept unknown extent; attached placement and cuboid occlusion require known dimensions.
FrameData.actors excludes ego, which is only in .ego.

NuRecDataset uses a fixed time_step timeline, interpolates only within recording support, and uses
camera_id/camera_alias for recorded id/model-visible name. CARLA generated data lives in
FrameDataCache, while CarlaDataset itself remains recipe-only.

## Simulator API

Simulator.create_runtime(seed) creates a new owned world. SimulatorRuntime.reset receives
ScenarioStart, optional initialization/deployment and only world settings. SimulatorRuntime.step
advances and captures; observe returns an independent current capture and never renders or advances.
state returns physical WorldSnapshot, including ego and other actors. The creator closes the
simulator runtime.

Custom simulator runtime classes implement native operations, not reset/step. Simulator declarations
include dataset_kinds, decision_time_step, command_modes, supported_stages,
supported_world_settings, can_render, capture_ticks and identity. validate_world_settings checks
supported names and values without native startup; validate_start checks dataset/time-step
compatibility.

SensorGeometry contains sensor, typed camera and cam_from_world. camera_sensors declares eligible
cameras; sensor_geometry(sensor) reads geometry without copying image payloads.

SensorView contains sensor/data/camera/cam_from_world/actors/ego/time/frame, optional depth and
native. Optical camera axes are x right, y down, z forward. Canonical world/actor axes are x
forward, y left, z up. Camera project/unproject use Nx3 points and Nx2 pixel coordinates.

## Driving-system API

AVStack.__call__(observation) returns Control. reset receives the first already processed input.
VehicleIO owns sensor/localization/observation processing and localizer lifecycle. AVStack also
declares command_modes, supported_stages, models, component_log and detections. None detections
means unsupported; an empty sequence is valid zero detections. Base close() is a no-op.
AVStack.vehicle_io is an assignable VehicleIO property. Detection is a true capture-time ego-frame
box center, full extent and yaw in radians, plus optional category/score. Conversion uses sensor
mounting extrinsics, independently of model-visible localization.

The configured AVStack constructor receives only options and creates a fresh instance per run.
DrivingAdapter separately binds component/scorer callables. Its Python factory accepts a seed; YAML
avstack.bindings receives the loader's configured factory and returns DrivingAdapter. Scorers
declare differentiable inputs and direction; services reject unavailable gradients.

## Closed-loop service API

ClosedLoopServices(dataset, simulator, driving_adapter=None, ...) exposes capabilities and
session(seed, history="fresh"). Each session owns cleanup. Forward sensor-only scans can reuse one
simulator runtime; effectful rendering and rollouts create independent worlds. Reservations span
operation cleanup. Native handles do not escape closed operations in RenderResult views.

SensorRequest selects scenario/frame/sensor_ids/environment. RenderRequest additionally selects
sensor_id and world/sensor effects. AVStack effects require a rollout. CandidateRequest declares
segment, seed, decision_steps, environment, recording and optional Attack/artifacts/output_path. Its
decision count must equal the segment interval. The simulator receives ScenarioStart, never the
segment metadata.

ScenarioView supplies scenario_id, seed, frames, frame_data and selection reasons. ComponentResult
and ScoreResult return value, descriptor and provenance. evaluate uses fresh driving state, while
score reuses one isolated scoring AVStack within its session and releases it at session close.
Rollouts remain fresh; unknown bindings or gradient mismatch fail explicitly.

## Hook and environment API

Hook binds stage and callable/plugin. HookSet preserves registration order, validates adapter stages
and owns one lifecycle per shared handler owner. StageContext supplies simulator_runtime, avstack,
physical_state, model-visible input/control, RNG and copied outputs. NativeHook is specific to
avstack and requires explicit registration imports.

normalize_environment, merge_environments and validate_environment manage the `{world, sensors}`
shape. SensorEnvironment.apply(observation, position) receives absolute timeline position. Custom
transforms implement apply(rgb, rng). read_camera_rgb returns None for an unknown explicit camera;
it never substitutes another camera. read_camera_rgb/write_camera_rgb live in
closed_loop/datatypes.py and preserve sensor payload transport conventions.

## RunLog API

RunLog holds metadata, steps, events, status and optional path. read loads published Parquet parts;
schema version 5 is checked explicitly. read_table selects Parquet columns without loading all
payloads. Recorder captures independent in-memory snapshots, records steps/events, flushes closed
parts, finishes or aborts. It does not close caller-owned simulators/models.

StepRecord fields include raw/model observations, pre/post physical state, produced/executed
commands, components, canonical detections, effects/receipts, outputs, typed sensor_geometry and
telemetry. sensor_geometry maps declared camera names to lightweight typed SensorGeometry records,
not SensorView images. physical_pre and physical_post are StepRecord fields, not fields on RunLog.
Payload retention is controlled by RecordingPolicy. Nested native values require codecs; persisted
values can remain plain data for analysis without Python SDK classes.

## Evaluation API

Metric.evaluate receives ComparisonGroup and returns per-treatment scalar dictionaries.
UnavailableMetricValue(reason, value_type="number" or "boolean") carries missing evidence without
mutable metric state. MetricScalar is one scalar or unavailable value; FieldMeasurements maps field
names to these values, and TreatmentMeasurements maps treatment names to FieldMeasurements.
PairMetric evaluates TreatmentPair; TrajectoryDeviation is symmetric, while StopImpact returns both
directional outcomes. Pair differences for single-run values are B minus A.

report.evaluate(result, configuration) writes report.json, runs.csv and plots. Metric
specifications, notes and errors remain in the report. EvaluationReport.format_summary() prints
treatment tables, pair difference matrices and notes; treatment_summaries and pair_summaries use
population standard deviation (ddof=0). as_dict returns data, and save writes it. No inference or
dataset read occurs.

Visualization functions consume StepRecord/RunLog: camera_view, lidar_bev, labels_view,
detections_view, filmstrip, record_run, save_gif and save_sequence. sensor_view reconstructs only
logged geometric evidence. record_run exports recorded frames; it does not execute an experiment.

## Public signatures

Properties appear as name: Type. Method signatures omit self/cls. Dataclass constructor fields are
listed in declaration order; optional inherited behavior is described in the owning guide. Concrete
Attack classes are imported from their folder, for example `from
avsectester.attack.implemented_attacks.patch_attack import PatchAttack`.

### avsectester.experiment.config

| Public name | Signature or fields |
|---|---|
| `TreatmentConfiguration` | `name: str`<br>`attack_name: str \| None = None`<br>`environment: Mapping[str, Any] = field(default_factory=dict)` |
| `ExperimentConfiguration` | `dataset: Dataset`<br>`simulator: Simulator`<br>`avstack: DrivingAdapter`<br>`attacks: Mapping[str, Attack] = field(default_factory=dict)`<br>`treatments: tuple[TreatmentConfiguration, ...] = ()`<br>`seeds: tuple[int, ...] = (0,)`<br>`decision_steps: int \| None = None`<br>`environment: Mapping[str, Any] = field(default_factory=dict)`<br>`recording: RecordingPolicy = field(default_factory=RecordingPolicy)`<br>`evaluation: EvaluationConfiguration = field(default_factory=EvaluationConfiguration)`<br>`output: Path \| None = None`<br>`strict: bool = False`<br>`hooks: HookConfiguration = None` |
| `ExperimentConfiguration.configuration_summary` | `configuration_summary: dict[str, Any]` |
| `ExperimentConfiguration.describe_configuration` | `describe_configuration(*, scenarios: Sequence[str] \| None = None) -> dict[str, Any]` |
| `load_configuration` | `load_configuration(config: ExperimentConfiguration \| Mapping[str, Any] \| str \| Path) -> ExperimentConfiguration` |
| `resolve_type` | `resolve_type(value: Any) -> Callable[..., Any]` |
| `attack_directory_name` | `attack_directory_name(class_name: str) -> str` |
| `available_attacks` | `available_attacks() -> tuple[str, ...]` |

### avsectester.experiment.engine

| Public name | Signature or fields |
|---|---|
| `validate_configuration_environment` | `validate_configuration_environment(config: ExperimentConfiguration) -> None` |
| `Experiment` | Class; see methods below |
| `Experiment.prepare` | `prepare(config: ExperimentConfiguration \| Mapping[str, Any] \| str \| Path) -> PreparedExperiment` |
| `Experiment.execute` | `execute(prepared: PreparedExperiment) -> ExperimentResult` |

### avsectester.experiment.models

| Public name | Signature or fields |
|---|---|
| `SelectionDecision` | `attack_name: str`<br>`scenario_id: str`<br>`status: Literal['selected', 'not_selected', 'error']`<br>`reasons: tuple[str, ...] = ()`<br>`error: str \| None = None` |
| `SelectionDecision.as_dict` | `as_dict() -> dict[str, Any]` |
| `SelectionDecision.from_dict` | `from_dict(value: Mapping[str, Any]) -> SelectionDecision` |
| `runs_directory` | `runs_directory(root: Path) -> Path` |
| `run_directory` | `run_directory(root: Path, attack_name: str, segment_index: int, seed: int, treatment: str) -> Path` |
| `saved_run_directories` | `saved_run_directories(root: Path) -> tuple[tuple[tuple[str, int, int, str], Path], ...]` |
| `treatment_applies` | `treatment_applies(treatment: Any, attack_name: str) -> bool` |
| `PreparedExperiment` | `configuration: ExperimentConfiguration`<br>`metadata: Mapping[str, tuple[AttackSegment, ...]]`<br>`artifacts: Mapping[str, Any]`<br>`selections: tuple[SelectionDecision, ...] = ()` |
| `PreparedExperiment.save` | `save(directory: str \| Path) -> Path` |
| `PreparedExperiment.load` | `load(config: ExperimentConfiguration \| Mapping[str, Any] \| str \| Path, directory: str \| Path) -> PreparedExperiment` |
| `TreatmentResult` | `attack_name: str`<br>`segment_index: int`<br>`segment: AttackSegment`<br>`treatment: str`<br>`seed: int`<br>`log: RunLog`<br>`error: str \| None = None` |
| `TreatmentResult.group` | `group: tuple[str, int, int]` |
| `ExperimentResult` | `runs: tuple[TreatmentResult, ...] = ()`<br>`selections: tuple[SelectionDecision, ...] = ()`<br>`notices: tuple[str, ...] = ()` |
| `ExperimentResult.groups` | `groups: dict[tuple[str, int, int], dict[str, RunLog]]` |
| `ExperimentResult.failures` | `failures: tuple[TreatmentResult, ...]` |
| `ExperimentResult.selection_errors` | `selection_errors: tuple[SelectionDecision, ...]` |
| `ExperimentResult.load` | `load(output: str \| Path, configuration: ExperimentConfiguration) -> ExperimentResult` |

### avsectester.experiment.persistence

| Public name | Signature or fields |
|---|---|
| `ConfigurationMismatchError` | `ConfigurationMismatchError(ValueError)`; raised when a saved preparation and the supplied configuration describe different attacks or inputs |

### avsectester.attack.attack

| Public name | Signature or fields |
|---|---|
| `AttackSegment` | `scenario_id: str`<br>`start: FrameReference`<br>`stop: FrameReference`<br>`roles: Mapping[str, tuple[str, ...]] = field(default_factory=dict)`<br>`placement: Any = None`<br>`selection_score: float \| None = None`<br>`selection_rank: int \| None = None`<br>`user_data: Mapping[str, Any] = field(default_factory=dict)` |
| `AttackSegment.decision_steps` | `decision_steps: int` |
| `OfflineContext` | `services: ClosedLoopServices`<br>`metadata: tuple[AttackSegment, ...]`<br>`seed: int = 0`<br>`environment: Mapping[str, Any] = field(default_factory=dict)`<br>`deployment: DeploymentPlan \| None = None` |
| `Attack` | `assets: Any = None`<br>`environment: Mapping[str, Any] = MappingProxyType({})` |
| `Attack.build_metadata` | `build_metadata(scenario: ScenarioView) -> Sequence[AttackSegment]` |
| `Attack.run_offline` | `run_offline(context: OfflineContext) -> Any` |
| `Attack.deployment_ticks` | `deployment_ticks(segment: AttackSegment) -> int` |
| `Attack.create_runtime` | `create_runtime(segment: AttackSegment, artifacts: Any) -> AttackRuntime` |
| `Attack.configuration_parameters` | `configuration_parameters() -> Mapping[str, Any]` |
| `Attack.save_artifacts` | `save_artifacts(artifacts: Any, directory: str \| Path) -> None` |
| `Attack.load_artifacts` | `load_artifacts(directory: str \| Path) -> Any` |

### avsectester.attack.runtime

| Public name | Signature or fields |
|---|---|
| `ResultSubscription` | `kind: Literal['observation', 'physical_state']` |
| `OnlineContext` | `step: int`<br>`stage: str`<br>`sim_time: float`<br>`frame: int`<br>`observation: Observation \| None = None`<br>`physical_state: WorldSnapshot \| None = None` |
| `OnlineContext.record_output` | `record_output(name: str, value: Any) -> None` |
| `AttackEffect` | Class; see methods below |
| `WorldAttackEffect` | `setup_ticks: int = 0` |
| `WorldAttackEffect.apply` | `apply(simulator_runtime: SimulatorRuntime, resources: ExitStack) -> None` |
| `SensorAttackEffect` | `sensor: str` |
| `SensorAttackEffect.apply` | `apply(data: Any, view: SensorView) -> Any` |
| `SensorAttackEffect.render_metadata` | `render_metadata() -> Mapping[str, Any]` |
| `AVStackAttackEffect` | `stage: str`<br>`apply: Callable[[Any, OnlineContext], Any]` |
| `AttackRuntime` | `subscriptions: Sequence[ResultSubscription] = ()`<br>`setup_ticks: int = 0` |
| `AttackRuntime.setup` | `setup(context: OnlineContext) -> Sequence[WorldAttackEffect]` |
| `AttackRuntime.step` | `step(context: OnlineContext) -> Sequence[AttackEffect]` |
| `AttackRuntime.stop` | `stop(context: OnlineContext) -> None` |
| `AttackDispatcher` | Class; see methods below |
| `AttackDispatcher.__init__` | `__init__(attacks: Iterable[AttackRuntime] = (), *, observation_copy: Callable[[Observation], Observation] = deepcopy, output_recorder: Callable[[OnlineContext, str, Any], None] \| None = None) -> None` |
| `AttackDispatcher.setup_ticks` | `setup_ticks: int` |
| `AttackDispatcher.started` | `started: bool` |
| `AttackDispatcher.validate` | `validate(available_stages: Iterable[str]) -> None` |
| `AttackDispatcher.hooks` | `hooks: tuple[Hook, ...]` |
| `AttackDispatcher.drain_records` | `drain_records() -> dict[str, list[dict[str, Any]]]` |
| `AttackDispatcher.setup` | `setup(context: OnlineContext, simulator_runtime: SimulatorRuntime) -> None` |
| `AttackDispatcher.step` | `step(context: OnlineContext) -> None` |
| `AttackDispatcher.record_end` | `record_end(context: OnlineContext) -> None` |
| `AttackDispatcher.apply_sensor_effects` | `apply_sensor_effects(observation: Observation, simulator_runtime: Any) -> Observation` |
| `AttackDispatcher.finish_decision` | `finish_decision() -> None` |
| `AttackDispatcher.stop` | `stop() -> None` |

### avsectester.attack.segments

| Public name | Signature or fields |
|---|---|
| `split_segments` | `split_segments(frames: Sequence[FrameReference], eligible: Callable[[FrameReference], bool], *, decision_steps: int, keep_tail: bool = False) -> tuple[AttackSegment, ...]` |
| `validate_segment_options` | `validate_segment_options(decision_steps: int, max_segments: int \| None, keep_tail: bool) -> None` |

### avsectester.attack.placement

| Public name | Signature or fields |
|---|---|
| `validate_attached_mount` | `validate_attached_mount(anchor: str, offset_m: tuple[float, float, float]) -> tuple[float, float, float]` |
| `AttachedPlacement` | `host: str`<br>`anchor: str = 'center'`<br>`offset_m: tuple[float, float, float] = (0.0, 0.0, 0.0)` |
| `AttachedPlacement.local_position` | `local_position(extent: tuple[float, float, float]) -> np.ndarray` |
| `WorldPlacement` | `position_m: tuple[float, float, float]` |
| `Orientation` | `mode: str = 'fixed_world'`<br>`rotation_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)`<br>`target_offset_m: tuple[float, float, float] = (0.0, 0.0, 0.0)` |
| `placement_from_spec` | `placement_from_spec(value: Any) -> AttachedPlacement \| WorldPlacement` |
| `orientation_from_spec` | `orientation_from_spec(value: Any) -> Orientation` |

### avsectester.datasets.base

| Public name | Signature or fields |
|---|---|
| `nonnegative` | `nonnegative(value: int, name: str) -> None` |
| `FrameReference` | `scenario_id: str`<br>`position: int`<br>`frame_id: str \| int`<br>`timestamp: float` |
| `ScenarioDescriptor` | `id: str`<br>`kind: str`<br>`payload: Mapping[str, Any] = field(default_factory=dict)` |
| `Unavailable` | `reason: str` |
| `SensorDataReference` | `uri: str`<br>`sensor: str`<br>`timestamp: float`<br>`metadata: Mapping[str, Any] = field(default_factory=dict)` |
| `ActorState` | `pose: Any`<br>`velocity: tuple[float, float, float] \| None = None`<br>`category: str \| None = None`<br>`extent: tuple[float, float, float] \| None = None`<br>`box_center: tuple[float, float, float] = (0.0, 0.0, 0.0)`<br>`box_rotation: tuple[float, float, float] = (0.0, 0.0, 0.0)` |
| `ActorState.box_transform` | `box_transform() -> np.ndarray` |
| `ActorState.box_pose` | `box_pose() -> ActorPose` |
| `FrameData` | `frame: FrameReference`<br>`actors: Mapping[str, ActorState] = field(default_factory=dict)`<br>`sensor_data: Mapping[str, SensorDataReference] = field(default_factory=dict)`<br>`provenance: Mapping[str, Any] = field(default_factory=dict)`<br>`ego: ActorState \| None = None` |
| `ScenarioStart` | `kind: str`<br>`scenario_id: str`<br>`frame: FrameReference`<br>`payload: Mapping[str, Any]`<br>`schema_version: int = 1` |
| `Dataset` | Class; see methods below |
| `Dataset.scenarios` | `scenarios() -> Iterator[ScenarioDescriptor]` |
| `Dataset.frames` | `frames(scenario_id: str) -> Sequence[FrameReference]` |
| `Dataset.frame_data` | `frame_data(scenario_id: str, frame: FrameReference) -> FrameData \| Unavailable` |
| `Dataset.native` | `native(scenario_id: str) -> Any` |
| `Dataset.scenario_start` | `scenario_start(scenario_id: str, frame: FrameReference) -> ScenarioStart` |
| `Dataset.checked_frame` | `checked_frame(scenario_id: str, frame: FrameReference) -> FrameReference` |
| `Dataset.decision_window` | `decision_window(scenario_id: str, start: FrameReference, decision_steps: int) -> tuple[FrameReference, ...]` |

### avsectester.datasets.carla

| Public name | Signature or fields |
|---|---|
| `PrescribedMotion` | `controls: tuple[Control, ...] = ()`<br>`trajectory: tuple[ActorState, ...] = ()`<br>`constant_control: Control \| None = None` |
| `PrescribedMotion.expanded` | `expanded(decision_steps: int) -> PrescribedMotion` |
| `PrescribedMotion.validate` | `validate(decision_steps: int) -> None` |
| `CarlaRecipe` | `scenario: dict`<br>`decision_steps: int`<br>`time_step: float`<br>`motion: Mapping[str, PrescribedMotion]`<br>`metadata: Mapping = field(default_factory=dict)` |
| `CarlaDataset` | Class; see methods below |
| `CarlaDataset.__init__` | `__init__(recipes: Mapping[str, CarlaRecipe]) -> None` |
| `CarlaDataset.scenarios` | `scenarios()` |
| `CarlaDataset.frames` | `frames(scenario_id)` |
| `CarlaDataset.frame_data` | `frame_data(scenario_id, frame)` |
| `CarlaDataset.native` | `native(scenario_id)` |
| `CarlaDataset.scenario_start` | `scenario_start(scenario_id, frame)` |

### avsectester.datasets.nurec

| Public name | Signature or fields |
|---|---|
| `NuRecDataset` | Class; see methods below |
| `NuRecDataset.__init__` | `__init__(usdz_paths: list[str], camera_id: str = 'camera_front_wide_120fov', camera_alias: str = 'front', vehicle_labels: tuple[str, ...] = _VEHICLE_LABELS, time_step: float = 0.1) -> None` |
| `NuRecDataset.from_glob` | `from_glob(pattern: str, **kwargs) -> NuRecDataset` |
| `NuRecDataset.open_archive` | `open_archive(path: str) -> zipfile.ZipFile` |
| `NuRecDataset.raw_metadata` | `raw_metadata(path: str) -> dict[str, Any]` |
| `NuRecDataset.camera_geometry` | `camera_geometry(path: str)` |
| `NuRecDataset.camera_timestamps` | `camera_timestamps(path: str) -> np.ndarray` |
| `NuRecDataset.scene_id` | `scene_id(path: str) -> str` |
| `NuRecDataset.scenarios` | `scenarios()` |
| `NuRecDataset.frames` | `frames(scenario_id)` |
| `NuRecDataset.frame_data` | `frame_data(scenario_id, frame)` |
| `NuRecDataset.native` | `native(scenario_id)` |
| `NuRecDataset.scenario_start` | `scenario_start(scenario_id, frame)` |
| `NuRecDataset.ego_state` | `ego_state(path: str, timestamp_us: int) -> ActorState` |
| `NuRecDataset.actor_poses` | `actor_poses(path: str, timestamp_us: int)` |
| `NuRecDataset.actor_states` | `actor_states(path: str, timestamp_us: int) -> dict[str, ActorState]` |

### avsectester.datasets.nuscenes

| Public name | Signature or fields |
|---|---|
| `NuScenesDataset` | Class; see methods below |
| `NuScenesDataset.__init__` | `__init__(dataroot: str, version: str = 'v1.0-trainval', camera: str = 'CAM_FRONT') -> None` |
| `NuScenesDataset.scenarios` | `scenarios()` |
| `NuScenesDataset.frames` | `frames(scenario_id)` |
| `NuScenesDataset.frame_data` | `frame_data(scenario_id, frame)` |
| `NuScenesDataset.native` | `native(scenario_id)` |
| `NuScenesDataset.scenario_start` | `scenario_start(scenario_id, frame)` |

### avsectester.closed_loop.datatypes

| Public name | Signature or fields |
|---|---|
| `Control` | `throttle: float = 0.0`<br>`steer: float = 0.0`<br>`brake: float = 0.0`<br>`trajectory: Any = None` |
| `Observation` | `t: float`<br>`frame: int`<br>`sensor_data: dict[str, Any] = field(default_factory=dict)`<br>`calibration: dict[str, Any] = field(default_factory=dict)`<br>`vehicle_state: Any = None`<br>`ego_speed: float = 0.0` |
| `Observation.set_calibration` | `set_calibration(sensor_id: str, calibration: Any) -> None` |
| `WorldSnapshot` | `t: float`<br>`frame: int`<br>`vehicle_state: Any = None`<br>`ego_speed: float = 0.0`<br>`actors: Mapping[str, Any] = field(default_factory=dict)`<br>`ego: ActorState \| None = None` |
| `StateEstimate` | `vehicle_state: Any = None`<br>`ego_speed: float = 0.0` |
| `RecordedOutput` | `name: str`<br>`value: Any`<br>`step: int`<br>`stage: str`<br>`frame: int`<br>`t: float`<br>`sensor: str \| None = None` |
| `Detection` | `position: tuple[float, float, float]`<br>`extent: tuple[float, float, float]`<br>`yaw: float`<br>`category: str \| None = None`<br>`score: float \| None = None` |
| `validate_command_modes` | `validate_command_modes(command: Control, modes: frozenset[str] \| None) -> None` |
| `read_camera_rgb` | `read_camera_rgb(observation: Observation, camera: str \| None = None) -> tuple[str, np.ndarray] \| None` |
| `write_camera_rgb` | `write_camera_rgb(observation: Observation, key: str, rgb: np.ndarray) -> Observation` |

### avsectester.closed_loop.simulator

| Public name | Signature or fields |
|---|---|
| `DeploymentPlan` | `ticks: int = 0` |
| `can_deploy_at` | `can_deploy_at(position: int, deployment: DeploymentPlan, capture_ticks: int) -> bool` |
| `ClockSnapshot` | `t: float`<br>`frame: int` |
| `SensorGeometry` | `sensor: str`<br>`camera: Camera`<br>`cam_from_world: np.ndarray` |
| `SensorView` | `sensor: str`<br>`data: Any`<br>`camera: Camera`<br>`cam_from_world: np.ndarray`<br>`actors: Mapping[str, ActorPose]`<br>`ego: ActorPose`<br>`time: float`<br>`frame: int`<br>`depth: np.ndarray \| None = None`<br>`native: Mapping[str, Any] = field(default_factory=dict)` |
| `Simulator` | `dataset_kinds: tuple[str, ...] = ()`<br>`can_render: bool = False`<br>`supported_world_settings: frozenset[str] = frozenset()` |
| `Simulator.create_runtime` | `create_runtime(seed: int \| None = None) -> SimulatorRuntime` |
| `Simulator.decision_time_step` | `decision_time_step: float \| None` |
| `Simulator.command_modes` | `command_modes: frozenset[str] \| None` |
| `Simulator.supported_stages` | `supported_stages: frozenset[str]` |
| `Simulator.identity` | `identity: Mapping[str, Any]` |
| `Simulator.validate_world_settings` | `validate_world_settings(settings: Mapping[str, Any]) -> None` |
| `Simulator.validate_start` | `validate_start(start: ScenarioStart \| None) -> None` |
| `SimulatorRuntime` | `resource_owner: object \| None = None` |
| `SimulatorRuntime.__init__` | `__init__(simulator: Simulator) -> None` |
| `SimulatorRuntime.dataset_kinds` | `dataset_kinds: tuple[str, ...]` |
| `SimulatorRuntime.can_render` | `can_render: bool` |
| `SimulatorRuntime.supported_world_settings` | `supported_world_settings: frozenset[str]` |
| `SimulatorRuntime.capture_ticks` | `capture_ticks: int` |
| `SimulatorRuntime.decision_time_step` | `decision_time_step: float \| None` |
| `SimulatorRuntime.command_modes` | `command_modes: frozenset[str] \| None` |
| `SimulatorRuntime.supported_stages` | `supported_stages: frozenset[str]` |
| `SimulatorRuntime.identity` | `identity: Mapping[str, Any]` |
| `SimulatorRuntime.validate_start` | `validate_start(start: ScenarioStart \| None) -> None` |
| `SimulatorRuntime.validate_command` | `validate_command(command: Control) -> None` |
| `SimulatorRuntime.copy_observation` | `copy_observation(observation: Observation) -> Observation` |
| `SimulatorRuntime.close` | `close() -> None` |
| `SimulatorRuntime.exclusive` | `exclusive() -> Iterator[None]` |
| `SimulatorRuntime.clock` | `clock: ClockSnapshot` |
| `SimulatorRuntime.state` | `state() -> WorldSnapshot` |
| `SimulatorRuntime.camera_sensors` | `camera_sensors: frozenset[str]` |
| `SimulatorRuntime.sensor_geometry` | `sensor_geometry(sensor: str) -> SensorGeometry` |
| `SimulatorRuntime.sensor_view` | `sensor_view(sensor: str) -> SensorView` |
| `SimulatorRuntime.control_actor` | `control_actor(actor_id: str, control: Control) -> None` |
| `SimulatorRuntime.set_actor_state` | `set_actor_state(actor_id: str, state: ActorState) -> None` |
| `SimulatorRuntime.reset` | `reset(start: ScenarioStart \| None = None, initialization: Any = None, *, deployment: DeploymentPlan \| None = None, world: Mapping[str, Any] \| None = None) -> None` |
| `SimulatorRuntime.setup_tick` | `setup_tick() -> None` |
| `SimulatorRuntime.step` | `step(control: Control) -> None` |
| `SimulatorRuntime.observe` | `observe() -> Observation` |
| `SimulatorRuntime.generate_frame_data` | `generate_frame_data(start: ScenarioStart, frames: Sequence[FrameReference], *, world: Mapping[str, Any] \| None = None, deployment: DeploymentPlan \| None = None) -> FrameDataSequence` |

### avsectester.closed_loop.avstack

| Public name | Signature or fields |
|---|---|
| `AVStack` | `provides_detections: bool \| None = None` |
| `AVStack.vehicle_io` | `vehicle_io: VehicleIO` |
| `AVStack.reset` | `reset(observation: Observation) -> None` |
| `AVStack.command_modes` | `command_modes: frozenset[str] \| None` |
| `AVStack.models` | `models: Mapping[str, Any]` |
| `AVStack.component_log` | `component_log() -> Mapping[str, Any] \| None` |
| `AVStack.detections` | `detections() -> Sequence[Detection] \| None` |
| `AVStack.close` | `close() -> None` |
| `AVStack.__call__` | `__call__(observation: Observation) -> Control` |
| `CommandCompatibilityError` | Class; see methods below |
| `validate_command_compatibility` | `validate_command_compatibility(simulator_runtime: SimulatorRuntime, avstack: AVStack, hooks: HookSet \| None = None) -> None` |

### avsectester.closed_loop.hooks

| Public name | Signature or fields |
|---|---|
| `StageEmitter` | Class; see methods below |
| `StageEmitter.emit` | `emit(stage: str, value: Any, sensor: str \| None = None) -> Any` |
| `UnsupportedStageError` | Class; see methods below |
| `CallInputs` | `args: tuple = ()`<br>`kwargs: dict[str, Any] = field(default_factory=dict)` |
| `CameraRenderRequest` | `pose: Any`<br>`camera: str` |
| `RunContext` | `simulator_runtime: SimulatorRuntime`<br>`avstack: AVStack`<br>`rng: np.random.Generator` |
| `RunContext.native` | `native: Mapping[str, Mapping[str, Any]]` |
| `StageContext` | `run: RunContext`<br>`step: int`<br>`stage: str`<br>`physical_state: WorldSnapshot`<br>`sensor: str \| None = None`<br>`observation: Observation \| None = None`<br>`control: Control \| None = None` |
| `StageContext.record_output` | `record_output(name: str, value: Any) -> None` |
| `StageContext.simulator_runtime` | `simulator_runtime: SimulatorRuntime` |
| `StageContext.avstack` | `avstack: AVStack` |
| `StageContext.rng` | `rng: np.random.Generator` |
| `StageContext.native` | `native: Mapping[str, Mapping[str, Any]]` |
| `StageContext.sim_time` | `sim_time: float` |
| `Plugin` | Class; see methods below |
| `Plugin.reset` | `reset(context: RunContext) -> None` |
| `Plugin.transform` | `transform(value: Any, context: StageContext) -> Any` |
| `Plugin.after_step` | `after_step(context: StageContext) -> None` |
| `Plugin.close` | `close() -> None` |
| `Hook` | `stage: str`<br>`handler: Any` |
| `HookSet` | Class; see methods below |
| `HookSet.__init__` | `__init__(hooks: Iterable[Hook] = (), *, seed: int \| None = 0) -> None` |
| `HookSet.with_hooks` | `with_hooks(hooks: Iterable[Hook], *, prepend: bool = False) -> Self` |
| `HookSet.validate_stages` | `validate_stages(simulator_runtime: SimulatorRuntime, avstack: AVStack) -> frozenset[str]` |
| `HookSet.setup_ticks` | `setup_ticks: int` |
| `HookSet.session` | `session(simulator_runtime: SimulatorRuntime, avstack: AVStack) -> HookDispatcher` |
| `HookDispatcher` | Class; see methods below |
| `HookDispatcher.__init__` | `__init__(hook_set: HookSet, simulator_runtime: SimulatorRuntime, avstack: AVStack) -> None` |
| `HookDispatcher.activate` | `activate() -> Iterator[HookDispatcher]` |
| `HookDispatcher.stage_context` | `stage_context(stage: str, sensor: str \| None = None) -> StageContext` |
| `HookDispatcher.record_output` | `record_output(context: StageContext, name: str, value: Any) -> None` |
| `HookDispatcher.drain_outputs` | `drain_outputs() -> list[RecordedOutput]` |
| `HookDispatcher.emit` | `emit(stage: str, value: Any, sensor: str \| None = None) -> Any` |
| `HookDispatcher.observe` | `observe(observation: Observation) -> Observation` |
| `HookDispatcher.command` | `command(control: Control) -> Control` |
| `HookDispatcher.after_step` | `after_step(observation: Observation) -> None` |

### avsectester.closed_loop.vehicle_io

| Public name | Signature or fields |
|---|---|
| `VehicleIO` | Class; see methods below |
| `VehicleIO.__init__` | `__init__(localizer: Any = None) -> None` |
| `VehicleIO.process` | `process(observation: Observation, dispatcher: HookDispatcher) -> Observation` |

### avsectester.closed_loop.services

| Public name | Signature or fields |
|---|---|
| `CandidateRequest` | `segment: AttackSegment`<br>`seed: int`<br>`decision_steps: int`<br>`environment: Mapping[str, Any]`<br>`recording: RecordingPolicy`<br>`attack: Any = None`<br>`artifacts: Any = None`<br>`output_path: str \| Path \| None = None` |
| `SensorRequest` | `scenario_id: str`<br>`frame: FrameReference`<br>`sensor_ids: tuple[str, ...]`<br>`environment: Mapping[str, Any] = field(default_factory=dict)` |
| `RenderRequest` | `scenario_id: str`<br>`frame: FrameReference`<br>`sensor_id: str`<br>`effects: Sequence[AttackEffect] = ()`<br>`environment: Mapping[str, Any] = field(default_factory=dict)` |
| `SensorSample` | `frame: FrameReference`<br>`observation: Observation`<br>`provenance: Mapping[str, Any]` |
| `RenderResult` | `frame: FrameReference`<br>`observation: Observation`<br>`view: SensorView`<br>`provenance: Mapping[str, Any]` |
| `ComponentDescriptor` | `id: str`<br>`inputs: Mapping[str, str]`<br>`outputs: Mapping[str, str]`<br>`codecs: Mapping[str, str]`<br>`differentiable_inputs: tuple[str, ...] = ()` |
| `ScorerDescriptor` | `output_shape: tuple[int, ...] = ()`<br>`direction: str = 'minimize'` |
| `ComponentResult` | `value: Any`<br>`descriptor: ComponentDescriptor`<br>`provenance: Mapping[str, Any]` |
| `ScoreResult` | `value: Any`<br>`descriptor: ScorerDescriptor`<br>`provenance: Mapping[str, Any]` |
| `ClosedLoopCapabilities` | `dataset_kinds: tuple[str, ...]`<br>`sensor_data: bool`<br>`render: bool`<br>`rollout: bool`<br>`components: tuple[ComponentDescriptor, ...]`<br>`scorers: tuple[ScorerDescriptor, ...]` |
| `DrivingAdapter` | Class; see methods below |
| `DrivingAdapter.__init__` | `__init__(factory: Callable[[int], AVStack], *, components: Sequence[tuple[ComponentDescriptor, Callable[..., Any]]] = (), scorers: Sequence[tuple[ScorerDescriptor, Callable[..., Any]]] = ()) -> None` |
| `ensure_fresh` | `ensure_fresh(resource: Any) -> Any` |
| `ClosedLoopServices` | Class; see methods below |
| `ClosedLoopServices.__init__` | `__init__(dataset: Dataset, simulator: Simulator, driving_adapter: DrivingAdapter \| None = None, frame_data_cache: FrameDataCache \| None = None, *, deployment: DeploymentPlan = DeploymentPlan()) -> None` |
| `ClosedLoopServices.close` | `close() -> None` |
| `ClosedLoopServices.capabilities` | `capabilities: ClosedLoopCapabilities` |
| `ClosedLoopServices.session` | `session(seed: int, history: str = 'fresh') -> ClosedLoopSession` |
| `ClosedLoopServices.scenario` | `scenario(scenario_id: str, *, seed: int) -> ScenarioView` |
| `ClosedLoopSession` | Class; see methods below |
| `ClosedLoopSession.__init__` | `__init__(services: ClosedLoopServices, seed: int) -> None` |
| `ClosedLoopSession.close` | `close() -> None` |
| `ClosedLoopSession.can_deploy_at` | `can_deploy_at(position: int) -> bool` |
| `ClosedLoopSession.frame_data` | `frame_data(scenario: str, frame: FrameReference, environment: Mapping[str, Any] \| None = None) -> FrameData` |
| `ClosedLoopSession.initialization` | `initialization(start: ScenarioStart, environment: Mapping[str, Any], deployment: DeploymentPlan \| None = None) -> Initialization \| None` |
| `ClosedLoopSession.sensor_data` | `sensor_data(request: SensorRequest) -> SensorSample` |
| `ClosedLoopSession.render` | `render(request: RenderRequest) -> RenderResult` |
| `ClosedLoopSession.components` | `components() -> Sequence[ComponentDescriptor]` |
| `ClosedLoopSession.scorers` | `scorers() -> Sequence[ScorerDescriptor]` |
| `ClosedLoopSession.evaluate` | `evaluate(component_id: str, inputs: Any) -> ComponentResult` |
| `ClosedLoopSession.score` | `score(scorer_id: str, inputs: Any, target: Any, require_grad: bool = False) -> ScoreResult` |
| `ClosedLoopSession.rollout` | `rollout(request: CandidateRequest) -> RunLog` |
| `ScenarioView` | `scenario_id: str`<br>`services: ClosedLoopServices`<br>`seed: int` |
| `ScenarioView.record_reason` | `record_reason(text: str) -> None` |
| `ScenarioView.reasons` | `reasons: tuple[str, ...]` |
| `ScenarioView.frames` | `frames: Sequence[FrameReference]` |
| `ScenarioView.frame_data` | `frame_data(frame: FrameReference) -> FrameData` |

### avsectester.closed_loop.runner

| Public name | Signature or fields |
|---|---|
| `RecordingPolicy` | `sensor_ids: tuple[str, ...] = ()`<br>`representations: tuple[str, ...] = ('model_input',)`<br>`every_n_steps: int = 1` |
| `RecordingPolicy.observation` | `observation(value: Observation, step: int, representation: str) -> Observation` |
| `RunRequest` | `decision_steps: int`<br>`hooks: HookSet \| None = None`<br>`attacks: Sequence[Any] = ()`<br>`recording: RecordingPolicy = field(default_factory=RecordingPolicy)`<br>`output_path: str \| Path \| None = None`<br>`retain_steps: bool = True`<br>`metadata: Mapping[str, Any] = field(default_factory=dict)`<br>`start: Any = None`<br>`initialization: Any = None`<br>`deployment: Any = None`<br>`environment: Mapping[str, Any] = field(default_factory=dict)` |
| `ClosedLoopRunner` | Class; see methods below |
| `ClosedLoopRunner.run` | `run(simulator_runtime: SimulatorRuntime, avstack: AVStack, request: RunRequest, *, recorder: Recorder \| None = None, release_owned_resources: Callable[[], None] \| None = None) -> RunLog` |

### avsectester.closed_loop.environment

| Public name | Signature or fields |
|---|---|
| `normalize_environment` | `normalize_environment(value: Mapping[str, Any] \| None = None) -> dict[str, Any]` |
| `merge_environments` | `merge_environments(*values: Mapping[str, Any]) -> dict[str, Any]` |
| `validate_environment` | `validate_environment(value: Mapping[str, Any], simulator: Any) -> dict[str, Any]` |
| `SensorEnvironment` | Class; see methods below |
| `SensorEnvironment.__init__` | `__init__(environment: Mapping[str, Any], seed: int = 0) -> None` |
| `SensorEnvironment.apply` | `apply(observation: Observation, position: int) -> Observation` |

### avsectester.closed_loop.sensor_transforms

| Public name | Signature or fields |
|---|---|
| `Corruption` | `category: str = 'sensor'` |
| `Corruption.__init__` | `__init__(severity: float = 0.5) -> None` |
| `Corruption.name` | `name: str` |
| `Corruption.apply` | `apply(rgb: np.ndarray, rng: np.random.Generator) -> np.ndarray` |
| `Fog` | Class; see methods below |
| `Fog.apply` | `apply(rgb: np.ndarray, rng: np.random.Generator) -> np.ndarray` |
| `Rain` | Class; see methods below |
| `Rain.apply` | `apply(rgb: np.ndarray, rng: np.random.Generator) -> np.ndarray` |
| `Snow` | Class; see methods below |
| `Snow.apply` | `apply(rgb: np.ndarray, rng: np.random.Generator) -> np.ndarray` |
| `Brightness` | Class; see methods below |
| `Brightness.apply` | `apply(rgb: np.ndarray, rng: np.random.Generator) -> np.ndarray` |
| `LowLight` | Class; see methods below |
| `LowLight.apply` | `apply(rgb: np.ndarray, rng: np.random.Generator) -> np.ndarray` |
| `Contrast` | Class; see methods below |
| `Contrast.apply` | `apply(rgb: np.ndarray, rng: np.random.Generator) -> np.ndarray` |
| `Gamma` | Class; see methods below |
| `Gamma.apply` | `apply(rgb: np.ndarray, rng: np.random.Generator) -> np.ndarray` |
| `ColorTemp` | Class; see methods below |
| `ColorTemp.__init__` | `__init__(severity: float = 0.5, warm: bool = True) -> None` |
| `ColorTemp.apply` | `apply(rgb: np.ndarray, rng: np.random.Generator) -> np.ndarray` |
| `GaussianNoise` | Class; see methods below |
| `GaussianNoise.apply` | `apply(rgb: np.ndarray, rng: np.random.Generator) -> np.ndarray` |
| `ShotNoise` | Class; see methods below |
| `ShotNoise.apply` | `apply(rgb: np.ndarray, rng: np.random.Generator) -> np.ndarray` |
| `MotionBlur` | Class; see methods below |
| `MotionBlur.apply` | `apply(rgb: np.ndarray, rng: np.random.Generator) -> np.ndarray` |
| `DefocusBlur` | Class; see methods below |
| `DefocusBlur.apply` | `apply(rgb: np.ndarray, rng: np.random.Generator) -> np.ndarray` |
| `JPEGCompression` | Class; see methods below |
| `JPEGCompression.apply` | `apply(rgb: np.ndarray, rng: np.random.Generator) -> np.ndarray` |
| `ChromaticAberration` | Class; see methods below |
| `ChromaticAberration.apply` | `apply(rgb: np.ndarray, rng: np.random.Generator) -> np.ndarray` |
| `AlbumentationsCorruption` | Class; see methods below |
| `AlbumentationsCorruption.__init__` | `__init__(transform: str, severity: float = 0.5) -> None` |
| `AlbumentationsCorruption.name` | `name: str` |
| `AlbumentationsCorruption.apply` | `apply(rgb: np.ndarray, rng: np.random.Generator) -> np.ndarray` |

### avsectester.closed_loop.simulators.carla

| Public name | Signature or fields |
|---|---|
| `resolve_scenario_seed` | `resolve_scenario_seed(scenario: dict) -> dict` |
| `prepare_scenario` | `prepare_scenario(scenario: dict) -> dict` |
| `SensorAlignmentError` | Class; see methods below |
| `SensorAlignmentError.__init__` | `__init__(report: Mapping[str, Any]) -> None` |
| `register_pinhole_camera` | `register_pinhole_camera()` |
| `decode_carla_depth` | `decode_carla_depth(image)` |
| `CarlaSimulator` | Class; see methods below |
| `CarlaSimulator.__init__` | `__init__(client: dict, settle_iters: int = 100, *, simulator_version = '0.9.16', depth_camera: str \| None = None) -> None` |
| `CarlaSimulator.validate_world_settings` | `validate_world_settings(world: Mapping[str, Any]) -> None` |
| `CarlaSimulator.create_runtime` | `create_runtime(seed: int \| None = None) -> CarlaSimulatorRuntime` |
| `CarlaSimulator.command_modes` | `command_modes: frozenset[str]` |
| `CarlaSimulator.decision_time_step` | `decision_time_step: Any` |
| `CarlaSimulator.identity` | `identity: Any` |
| `CarlaSimulator.validate_start` | `validate_start(start)` |
| `CarlaSimulatorRuntime` | Class; see methods below |
| `CarlaSimulatorRuntime.__init__` | `__init__(simulator: CarlaSimulator, seed: int \| None = None) -> None` |
| `CarlaSimulatorRuntime.resource_key` | `resource_key: Any` |
| `CarlaSimulatorRuntime.native` | `native: dict[str, Any]` |
| `CarlaSimulatorRuntime.control_actor` | `control_actor(actor_id, control)` |
| `CarlaSimulatorRuntime.set_actor_state` | `set_actor_state(actor_id, state)` |
| `CarlaSimulatorRuntime.copy_observation` | `copy_observation(observation: Observation) -> Observation` |
| `CarlaSimulatorRuntime.camera_sensors` | `camera_sensors: frozenset[str]` |
| `CarlaSimulatorRuntime.sensor_geometry` | `sensor_geometry(sensor: str) -> SensorGeometry` |
| `CarlaSimulatorRuntime.sensor_view` | `sensor_view(sensor: str) -> SensorView` |
| `CarlaSimulatorRuntime.close` | `close() -> None` |
| `RecipeReplay` | Class; see methods below |
| `RecipeReplay.__init__` | `__init__(simulator_runtime, recipe)` |
| `RecipeReplay.initialize` | `initialize()` |
| `RecipeReplay.activate` | `activate()` |
| `RecipeReplay.hold_origin` | `hold_origin()` |
| `RecipeReplay.finish_holding` | `finish_holding()` |
| `RecipeReplay.before_step` | `before_step()` |
| `RecipeReplay.after_step` | `after_step()` |
| `RecipeReplay.ego_command` | `ego_command()` |
| `RecipeReplay.release` | `release(identity)` |
| `RecipeReplay.release_ego` | `release_ego()` |

### avsectester.closed_loop.simulators.nurec

| Public name | Signature or fields |
|---|---|
| `EgoPose` | `x: float = 0.0`<br>`y: float = 0.0`<br>`yaw: float = 0.0`<br>`speed: float = 0.0`<br>`t: float = 0.0` |
| `EgoDynamics` | Class; see methods below |
| `EgoDynamics.step` | `step(pose: EgoPose, control: Control, dt: float) -> EgoPose` |
| `KinematicBicycle` | `wheelbase: float = 2.8`<br>`max_accel: float = 3.0`<br>`max_brake: float = 8.0`<br>`max_steer: float = 0.6`<br>`command_modes: ClassVar[frozenset[str]] = frozenset({'actuator'})` |
| `KinematicBicycle.step` | `step(pose: EgoPose, control: Control, dt: float) -> EgoPose` |
| `TrajectoryFollower` | `decel: float = 4.0`<br>`command_modes: ClassVar[frozenset[str]] = frozenset({'trajectory'})` |
| `TrajectoryFollower.step` | `step(pose: EgoPose, control: Control, dt: float) -> EgoPose` |
| `Renderer` | `cameras: ClassVar[list[str]] = ['camera_front']`<br>`calibration: ClassVar[dict] = {}` |
| `Renderer.copy` | `copy() -> Renderer` |
| `Renderer.reset_origin` | `reset_origin(start = None)` |
| `Renderer.origin_transform` | `origin_transform: Any` |
| `Renderer.origin_timestamp_us` | `origin_timestamp_us: Any` |
| `Renderer.origin_bounds` | `origin_bounds: Any` |
| `Renderer.load_scene` | `load_scene(scene: Any) -> None` |
| `Renderer.render` | `render(pose: EgoPose, camera: str) -> Any` |
| `Renderer.camera_model` | `camera_model(camera: str \| None = None) -> Camera \| None` |
| `Renderer.cam_from_world` | `cam_from_world(pose: EgoPose, camera: str \| None = None) -> np.ndarray` |
| `Renderer.close` | `close() -> None` |
| `StubRenderer` | Class; see methods below |
| `StubRenderer.__init__` | `__init__(cameras: list[str] \| None = None, height: int = 480, width: int = 640) -> None` |
| `StubRenderer.render` | `render(pose: EgoPose, camera: str)` |
| `NuRecSimulator` | Class; see methods below |
| `NuRecSimulator.__init__` | `__init__(config: dict, renderer: Renderer \| None = None, dynamics: EgoDynamics \| None = None) -> None` |
| `NuRecSimulator.create_runtime` | `create_runtime(seed: int \| None = None) -> NuRecSimulatorRuntime` |
| `NuRecSimulator.supported_stages` | `supported_stages: frozenset[str]` |
| `NuRecSimulator.command_modes` | `command_modes: frozenset[str] \| None` |
| `NuRecSimulator.decision_time_step` | `decision_time_step: Any` |
| `NuRecSimulator.identity` | `identity: Any` |
| `NuRecSimulatorRuntime` | Class; see methods below |
| `NuRecSimulatorRuntime.__init__` | `__init__(simulator: NuRecSimulator) -> None` |
| `NuRecSimulatorRuntime.native` | `native: dict[str, Any]` |
| `NuRecSimulatorRuntime.render_request` | `render_request(camera: str) -> CameraRenderRequest` |
| `NuRecSimulatorRuntime.camera_sensors` | `camera_sensors: frozenset[str]` |
| `NuRecSimulatorRuntime.sensor_geometry` | `sensor_geometry(sensor: str) -> SensorGeometry` |
| `NuRecSimulatorRuntime.sensor_view` | `sensor_view(sensor: str) -> SensorView` |
| `NuRecSimulatorRuntime.close` | `close() -> None` |
| `NuRecSimulatorRuntime.checkpoint` | `checkpoint() -> dict` |
| `NuRecSimulatorRuntime.restore` | `restore(ckpt: dict) -> None` |
| `NuRecRenderer` | `endpoint: str = '127.0.0.1:50051'`<br>`scene_id: str \| None = None`<br>`cameras: list[str] = field(default_factory=lambda: ['camera_front_wide_120fov'])`<br>`calibration: dict = field(default_factory=dict)`<br>`image_quality: float = 95.0` |
| `NuRecRenderer.copy` | `copy() -> NuRecRenderer` |
| `NuRecRenderer.load_scene` | `load_scene(scene: Any) -> None` |
| `NuRecRenderer.service` | `service: Any` |
| `NuRecRenderer.close` | `close()` |
| `NuRecRenderer.timestamp_us` | `timestamp_us(pose: EgoPose) -> int` |
| `NuRecRenderer.camera_model` | `camera_model(camera: str \| None = None) -> Camera` |
| `NuRecRenderer.rig_transform` | `rig_transform(pose: EgoPose)` |
| `NuRecRenderer.cam_from_world` | `cam_from_world(pose: EgoPose, camera: str \| None = None) -> np.ndarray` |
| `NuRecRenderer.render` | `render(pose: EgoPose, camera: str)` |
| `NuRecRenderer.render_at` | `render_at(world_from_rig, timestamp_us: int, camera: str)` |
| `nurec_sensor_view` | `nurec_sensor_view(simulator_runtime, dataset, sensor = None, data = None)` |

### avsectester.closed_loop.simulators.recorded

| Public name | Signature or fields |
|---|---|
| `RecordedReplaySimulator` | Class; see methods below |
| `RecordedReplaySimulator.__init__` | `__init__(start: ScenarioStart \| None = None) -> None` |
| `RecordedReplaySimulator.create_runtime` | `create_runtime(seed: int \| None = None) -> RecordedReplaySimulatorRuntime` |
| `RecordedReplaySimulatorRuntime` | Class; see methods below |
| `RecordedReplaySimulatorRuntime.__init__` | `__init__(simulator: RecordedReplaySimulator) -> None` |
| `RecordedReplaySimulatorRuntime.camera_sensors` | `camera_sensors: frozenset[str]` |
| `RecordedReplaySimulatorRuntime.sensor_geometry` | `sensor_geometry(sensor: str) -> SensorGeometry` |
| `RecordedReplaySimulatorRuntime.sensor_view` | `sensor_view(sensor: str) -> SensorView` |
| `RecordedReplaySimulatorRuntime.close` | `close()` |

### avsectester.closed_loop.avstacks.modular

| Public name | Signature or fields |
|---|---|
| `ModularAVStack` | Class; see methods below |
| `ModularAVStack.__init__` | `__init__(pipeline_cfg: dict) -> None` |
| `ModularAVStack.reset` | `reset(observation: Observation) -> None` |
| `ModularAVStack.command_modes` | `command_modes: frozenset[str]` |
| `ModularAVStack.supported_stages` | `supported_stages: frozenset[str]` |
| `ModularAVStack.native` | `native: Mapping[str, Any]` |
| `ModularAVStack.models` | `models: Mapping[str, Any]` |
| `ModularAVStack.bind_hooks` | `bind_hooks(dispatcher: StageEmitter) -> Iterator[None]` |
| `ModularAVStack.instrument` | `instrument(stages: tuple[str, ...] = ('perception', 'tracking', 'planning', 'control')) -> None` |
| `ModularAVStack.component_log` | `component_log() -> dict[str, Any] \| None` |
| `ModularAVStack.detections` | `detections() -> Sequence[Detection] \| None` |
| `ModularAVStack.__call__` | `__call__(observation: Observation) -> Control` |

### avsectester.closed_loop.avstacks.alpamayo

| Public name | Signature or fields |
|---|---|
| `AlpamayoAVStack` | Class; see methods below |
| `AlpamayoAVStack.__init__` | `__init__(checkpoint_path: str \| None = None, camera_ids: list[str] \| None = None, device: str = 'cuda:0', context_length: int = 4, output_frequency_hz: int = 10, model: Any \| None = None) -> None` |
| `AlpamayoAVStack.command_modes` | `command_modes: frozenset[str]` |
| `AlpamayoAVStack.supported_stages` | `supported_stages: frozenset[str]` |
| `AlpamayoAVStack.native` | `native: Mapping[str, Any]` |
| `AlpamayoAVStack.models` | `models: Mapping[str, Any]` |
| `AlpamayoAVStack.reset` | `reset(observation: Observation) -> None` |
| `AlpamayoAVStack.close` | `close() -> None` |
| `AlpamayoAVStack.__call__` | `__call__(observation: Observation) -> Control` |
| `AlpamayoAVStack.component_log` | `component_log() -> dict[str, Any] \| None` |

### avsectester.closed_loop.avstacks.camera_forward_collision

| Public name | Signature or fields |
|---|---|
| `CameraForwardCollisionAVStack` | Class; see methods below |
| `CameraForwardCollisionAVStack.__init__` | `__init__(gpu: int = 0, camera: str = 'front', area_brake: float = 0.03, throttle: float = 0.55, detector_dataset: str = 'carla-vehicle', threshold: float = 0.3, area_max: float = 0.3, max_height_frac: float = 0.6) -> None` |
| `CameraForwardCollisionAVStack.models` | `models: dict[str, Any]` |
| `CameraForwardCollisionAVStack.detect` | `detect(rgb: Any) -> list` |
| `CameraForwardCollisionAVStack.reset` | `reset(observation: Observation) -> None` |
| `CameraForwardCollisionAVStack.__call__` | `__call__(observation: Observation) -> Control` |
| `CameraForwardCollisionAVStack.component_log` | `component_log() -> dict[str, Any] \| None` |
| `CameraForwardCollisionAVStack.close` | `close() -> None` |

### avsectester.closed_loop.avstacks.native_hook

| Public name | Signature or fields |
|---|---|
| `NativeHook` | Class; see methods below |
| `NativeHook.__init__` | `__init__(config: Mapping[str, Any], *, phase: str = 'post') -> None` |
| `NativeHook.reset` | `reset(context: RunContext) -> None` |
| `NativeHook.transform` | `transform(value: Any, context: StageContext) -> Any` |
| `NativeHook.close` | `close() -> None` |

### avsectester.closed_loop.avstacks.scorers

| Public name | Signature or fields |
|---|---|
| `TargetSpec` | `box: tuple[float, float, float, float]` |
| `MMDetRPNObjectnessScorer` | Class; see methods below |
| `MMDetRPNObjectnessScorer.__init__` | `__init__(model, strides = FPN_STRIDES)` |
| `MMDetRPNObjectnessScorer.target_score` | `target_score(image: torch.Tensor, target: TargetSpec) -> torch.Tensor` |
| `MMDetInferenceScorer` | Class; see methods below |
| `MMDetInferenceScorer.__init__` | `__init__(model)` |
| `MMDetInferenceScorer.target_score` | `target_score(image, target: TargetSpec) -> float` |

### avsectester.closed_loop.driving_adapters

| Public name | Signature or fields |
|---|---|
| `modular_driving_adapter` | `modular_driving_adapter(factory, *, components = (), scorers = ())` |
| `alpamayo_driving_adapter` | `alpamayo_driving_adapter(factory)` |

### avsectester.closed_loop.frame_data

| Public name | Signature or fields |
|---|---|
| `fingerprint` | `fingerprint(value: Any) -> str` |
| `Initialization` | `expected: FrameData`<br>`resolved_scenario: Mapping[str, Any] = field(default_factory=dict)`<br>`provenance: Mapping[str, Any] = field(default_factory=dict)` |
| `FrameDataSequence` | `frames: tuple[FrameData, ...]`<br>`resolved_scenario: Mapping[str, Any]`<br>`provenance: Mapping[str, Any]` |
| `FrameDataSequence.at` | `at(frame: FrameReference) -> FrameData` |
| `FrameDataCache` | Class; see methods below |
| `FrameDataCache.__init__` | `__init__() -> None` |
| `FrameDataCache.key` | `key(start: ScenarioStart, simulator_identity: Mapping[str, Any], seed: int) -> str` |
| `FrameDataCache.get_or_generate` | `get_or_generate(key: str, generate: Callable[[], FrameDataSequence]) -> FrameDataSequence` |
| `FrameDataCache.get` | `get(key: str) -> FrameDataSequence` |

### avsectester.attack.implemented_attacks.phantom_injection_attack.attack

| Public name | Signature or fields |
|---|---|
| `PhantomInjectionAttack` | Class; see methods below |
| `PhantomInjectionAttack.__init__` | `__init__(*, decision_steps: int, max_segments: int \| None = None, target_xyz = (6.0, 0.0, -0.7), object_type = 'Car', score = 0.9, extent = (4.0, 1.8, 1.6), object_id = 90002, keep_tail: bool = False)` |
| `PhantomInjectionAttack.build_metadata` | `build_metadata(scenario)` |
| `PhantomInjectionAttack.create_runtime` | `create_runtime(segment, artifacts)` |

### avsectester.attack.implemented_attacks.speed_feedback_attack.attack

| Public name | Signature or fields |
|---|---|
| `SpeedFeedbackAttack` | Class; see methods below |
| `SpeedFeedbackAttack.__init__` | `__init__(*, decision_steps: int, target_speed: float, gain: float = 0.1, max_bias: float = 5.0, initial_bias: float = 0.0, max_segments: int \| None = None, keep_tail: bool = False)` |
| `SpeedFeedbackAttack.build_metadata` | `build_metadata(scenario)` |
| `SpeedFeedbackAttack.create_runtime` | `create_runtime(segment, artifacts)` |

### avsectester.attack.implemented_attacks.patch_attack.attack

| Public name | Signature or fields |
|---|---|
| `PatchAttack` | Class; see methods below |
| `PatchAttack.__init__` | `__init__(*, decision_steps, texture = None, candidate_hosts = None, hosts_per_window = 1, sensor = 'front', width_m = 1.0, height_m = 0.5, anchor = 'rear_center', offset_m = (-0.01, 0.0, 0.0), orientation = None, max_distance = None, min_visible_fraction = 0.0, max_windows = None, keep_tail = False, optimization = None, prop = DEFAULT_PROP, emissive = True, compositor = None)` |
| `PatchAttack.placement_for` | `placement_for(host)` |
| `PatchAttack.insertion` | `insertion(artifacts, *, host = None, placement = None)` |
| `PatchAttack.build_metadata` | `build_metadata(scenario)` |
| `PatchAttack.run_offline` | `run_offline(context)` |
| `PatchAttack.deployment_ticks` | `deployment_ticks(segment)` |
| `PatchAttack.create_runtime` | `create_runtime(segment, artifacts)` |

### avsectester.attack.implemented_attacks.object_insertion_attack.attack

| Public name | Signature or fields |
|---|---|
| `ObjectInsertionAttack` | Class; see methods below |
| `ObjectInsertionAttack.__init__` | `__init__(*, decision_steps, placement, assets = None, orientation = None, sensor = 'front', min_visible_fraction = 0.0, max_segments = None, keep_tail = False, placement_relative_to_ego_origin = False, compositor = None)` |
| `ObjectInsertionAttack.insertion` | `insertion(artifacts, placement = None)` |
| `ObjectInsertionAttack.build_metadata` | `build_metadata(scenario)` |
| `ObjectInsertionAttack.create_runtime` | `create_runtime(segment, artifacts)` |

### avsectester.attack.implemented_attacks.object_insertion_attack.assets

| Public name | Signature or fields |
|---|---|
| `post_rgba` | `post_rgba(height_pixels: int = 256, width_pixels: int = 16, color: tuple = (150, 152, 155)) -> np.ndarray` |
| `SignAsset` | `face: np.ndarray \| str \| Path = field(default=STOP_SIGN, repr=False)`<br>`width_m: float = 0.9`<br>`mount_height_m: float = 1.5`<br>`post: bool = True`<br>`post_width_m: float = 0.07`<br>`post_count: int = 1` |
| `SignAsset.planes` | `planes() -> Sequence[PlaneSurface]` |
| `poster_rgba` | `poster_rgba(person: np.ndarray, aspect: float = 1.5, fill: float = 0.86, paper: tuple = (228, 226, 218), border: tuple = (40, 40, 42), border_frac: float = 0.025) -> np.ndarray` |
| `standee` | `standee(person: np.ndarray, height_m: float = 1.75) -> SignAsset` |
| `billboard` | `billboard(person: np.ndarray, width_m: float = 1.4, mount_height_m: float = 0.6) -> SignAsset` |
| `signal_head_rgba` | `signal_head_rgba(lit: str = 'red', texture_height_pixels: int = 300) -> np.ndarray` |
| `traffic_lights_rgba` | `traffic_lights_rgba(head_count: int = 3, lit: str = 'red', texture_height_pixels: int = 300, gap_fraction: float = 0.25, board: tuple \| None = (35, 37, 40)) -> np.ndarray` |
| `roadside_rig` | `roadside_rig(face: np.ndarray, width_m: float = 2.4, mount_height_m: float = 2.2) -> SignAsset` |
| `PersonAsset` | `image: np.ndarray \| str \| Path`<br>`mode: str = 'billboard'`<br>`height_m: float = 1.75`<br>`width_m: float = 1.4`<br>`mount_height_m: float = 0.6` |
| `PersonAsset.planes` | `planes()` |
| `TrafficLightAsset` | `head_count: int = 3`<br>`lit: str = 'red'`<br>`texture_height_pixels: int = 300`<br>`gap_fraction: float = 0.25`<br>`board: tuple \| None = (35, 37, 40)`<br>`width_m: float = 2.4`<br>`mount_height_m: float = 2.2` |
| `TrafficLightAsset.planes` | `planes()` |

### avsectester.attack.insertion.assets

| Public name | Signature or fields |
|---|---|
| `load_rgba_image` | `load_rgba_image(value: np.ndarray \| str \| Path \| object) -> np.ndarray` |
| `ImageTexture` | `image: str \| Path` |
| `ImageTexture.rgba` | `rgba() -> np.ndarray` |
| `CheckerboardTexture` | `size: int = 256`<br>`cells: int = 8`<br>`colors: tuple = ((255, 0, 0), (255, 255, 255))` |
| `CheckerboardTexture.rgba` | `rgba() -> np.ndarray` |
| `PlaneSurface` | `corners: np.ndarray`<br>`texture: np.ndarray` |
| `PlanarAsset` | Class; see methods below |
| `PlanarAsset.planes` | `planes() -> Sequence[PlaneSurface]` |
| `PlaneAsset` | `texture: np.ndarray \| str \| Path`<br>`width_m: float`<br>`height_m: float` |
| `PlaneAsset.planes` | `planes() -> tuple[PlaneSurface, ...]` |
| `Insertion` | `id: str`<br>`asset: PlanarAsset`<br>`placement: AttachedPlacement \| WorldPlacement`<br>`orientation: Orientation = field(default_factory=Orientation)` |
| `ResolvedInsertion` | `id: str`<br>`pose: np.ndarray`<br>`asset: PlanarAsset`<br>`host_id: str \| None = None` |
| `ResolvedInsertion.planes` | `planes() -> tuple[PlaneSurface, ...]` |
| `resolve_insertion` | `resolve_insertion(insertion: Insertion, actors: Mapping[str, poses.ActorPose], ego: poses.ActorPose) -> ResolvedInsertion` |

### avsectester.attack.insertion.types

| Public name | Signature or fields |
|---|---|
| `Visibility` | `fraction: float`<br>`source: str`<br>`camera: str \| None = None`<br>`label: str \| None = None` |
| `VisibilityEvidence` | `visibility: Visibility \| None`<br>`reference_mask: np.ndarray`<br>`visible_mask: np.ndarray`<br>`target_depth: np.ndarray`<br>`reference_pixels: int`<br>`reason: str = ''`<br>`sampled_rgba: np.ndarray \| None = None` |

### avsectester.attack.insertion.geometry

| Public name | Signature or fields |
|---|---|
| `edge_points` | `edge_points(corners: np.ndarray) -> np.ndarray` |
| `camera_ray_mask` | `camera_ray_mask(camera: Camera, rays: np.ndarray) -> np.ndarray` |
| `SurfaceSampler` | `corners: np.ndarray`<br>`texture: np.ndarray`<br>`bounds: tuple[int, int, int, int] \| None`<br>`reason: str = ''` |
| `SurfaceSampler.sample` | `sample(rays: np.ndarray) -> tuple[np.ndarray, np.ndarray]` |
| `prepare_surface` | `prepare_surface(surface: PlaneSurface, camera: Camera, cam_from_world) -> SurfaceSampler` |
| `Projection` | `reference: np.ndarray`<br>`distance: np.ndarray`<br>`rays: np.ndarray`<br>`rgba: np.ndarray`<br>`x0: int`<br>`y0: int` |
| `rasterize` | `rasterize(subject, camera: Camera, cam_from_world, max_reference_pixels)` |
| `sample_image` | `sample_image(sampler: SurfaceSampler, camera: Camera, *, margin: int = 2) -> tuple[np.ndarray, np.ndarray]` |
| `insertion_image` | `insertion_image(subject, camera: Camera, cam_from_world) -> tuple[np.ndarray, np.ndarray]` |
| `box_intersections` | `box_intersections(actor: ActorPose, rays: np.ndarray, world_from_cam) -> np.ndarray` |

### avsectester.attack.insertion.compositor

| Public name | Signature or fields |
|---|---|
| `warp_patch` | `warp_patch(frame_rgb: np.ndarray, dst_quad: np.ndarray, patch_rgba: np.ndarray)` |
| `PatchCompositor` | Class; see methods below |
| `PatchCompositor.__init__` | `__init__(harmonizer: Harmonizer \| None = None, *, soften: float = 0.0, harmonize: bool = True) -> None` |
| `render_resolved` | `render_resolved(frame: np.ndarray, camera: Camera, cam_from_world: np.ndarray, resolved: ResolvedInsertion, evidence: VisibilityEvidence \| None = None, *, compositor: PatchCompositor \| None = None) -> tuple[np.ndarray, np.ndarray]` |

### avsectester.attack.insertion.harmonizers

| Public name | Signature or fields |
|---|---|
| `Harmonizer` | Class; see methods below |
| `Harmonizer.__call__` | `__call__(composite_rgb: np.ndarray, mask: np.ndarray, background_rgb: np.ndarray) -> np.ndarray` |
| `ClassicHarmonizer` | Class; see methods below |
| `ClassicHarmonizer.__init__` | `__init__(color_transfer: bool = True, poisson: bool = True, preserve_chroma: bool = False, blend: str = 'poisson', feather: float = 1.2, gain_range: tuple = (0.35, 1.2)) -> None` |
| `ClassicHarmonizer.__call__` | `__call__(composite_rgb, mask, background_rgb)` |
| `PCTNetHarmonizer` | Class; see methods below |
| `PCTNetHarmonizer.__init__` | `__init__(device: int = 0, weights: str \| None = None, strict: bool = False) -> None` |
| `PCTNetHarmonizer.__call__` | `__call__(composite_rgb, mask, background_rgb)` |

### avsectester.attack.insertion.visibility

| Public name | Signature or fields |
|---|---|
| `CuboidVisibilityEstimator` | Class; see methods below |
| `CuboidVisibilityEstimator.estimate` | `estimate(subject: ResolvedInsertion, camera: Camera, cam_from_world: np.ndarray, *, occluders: Mapping[str, ActorPose], camera_name: str, other_insertions: Sequence[ResolvedInsertion] = ()) -> VisibilityEvidence` |
| `DepthVisibilityEstimator` | Class; see methods below |
| `DepthVisibilityEstimator.__init__` | `__init__(*, tolerance_m: float = 0.0002, max_reference_pixels: int = 4000000)` |
| `DepthVisibilityEstimator.estimate` | `estimate(subject: ResolvedInsertion, camera: Camera, cam_from_world: np.ndarray, *, scene_depth: np.ndarray, depth_convention: str, camera_name: str, other_insertions: Sequence[ResolvedInsertion] = ()) -> VisibilityEvidence` |
| `visibility_from_masks` | `visibility_from_masks(reference, visible, *, camera: str, image_bounds = None) -> Visibility \| None` |
| `visibility_from_depth` | `visibility_from_depth(footprint, target_depth, scene_depth, *, camera: str, tolerance_m: float = 0.0002, image_bounds = None) -> Visibility \| None` |

### avsectester.attack.insertion.effect

| Public name | Signature or fields |
|---|---|
| `InsertionAttackEffect` | `sensor: str`<br>`insertion: Insertion`<br>`compositor: PatchCompositor \| None = None`<br>`rendered: bool = field(default=False, init=False)`<br>`reason: str = field(default='', init=False)` |
| `InsertionAttackEffect.render_metadata` | `render_metadata()` |
| `InsertionAttackEffect.resolve` | `resolve(view)` |
| `InsertionAttackEffect.evidence` | `evidence(view, resolved = None)` |
| `InsertionAttackEffect.apply` | `apply(data: Any, view)` |

### avsectester.attack.optimization.interface

| Public name | Signature or fields |
|---|---|
| `TargetSpec` | `box: tuple[float, float, float, float]`<br>`label: str \| None = None` |
| `AdversarialSample` | `x: Tensor`<br>`target: TargetSpec`<br>`context: dict = field(default_factory=dict)` |
| `DataSource` | Class; see methods below |
| `DataSource.sample` | `sample(batch: int) -> list[AdversarialSample]` |
| `Perturbation` | Class; see methods below |
| `Perturbation.init` | `init() -> Tensor` |
| `Perturbation.apply` | `apply(sample: AdversarialSample, delta: Tensor) -> Tensor` |
| `Perturbation.project` | `project(delta: Tensor) -> Tensor` |
| `Perturbation.export` | `export(delta: Tensor) -> np.ndarray \| None` |
| `Scorer` | `differentiable: bool = True` |
| `Scorer.target_score` | `target_score(image: Tensor, target: TargetSpec) -> Tensor` |
| `OptimizationObjective` | Class; see methods below |
| `OptimizationObjective.loss` | `loss(score: Tensor) -> Tensor` |
| `HideObject` | Class; see methods below |
| `HideObject.loss` | `loss(score: Tensor) -> Tensor` |
| `SpoofObject` | Class; see methods below |
| `SpoofObject.loss` | `loss(score: Tensor) -> Tensor` |
| `OptimizationResult` | `delta: Tensor`<br>`export: np.ndarray \| None`<br>`history: list[float]`<br>`final_score: float` |
| `GradientOptimizer` | Class; see methods below |
| `GradientOptimizer.run` | `run(data: DataSource, perturbation: Perturbation, scorer: Scorer, objective: OptimizationObjective) -> OptimizationResult` |

### avsectester.attack.optimization.scorers

| Public name | Signature or fields |
|---|---|
| `SessionScorer` | Class; see methods below |
| `SessionScorer.__init__` | `__init__(session, scorer_id, *, require_grad, input_format = None, device = 'cpu')` |
| `SessionScorer.target_score` | `target_score(image, target)` |

### avsectester.attack.optimization.perturbations

| Public name | Signature or fields |
|---|---|
| `PatchPerturbation` | Class; see methods below |
| `PatchPerturbation.__init__` | `__init__(size = (128, 128), device = 'cuda:0', init = 'random', seed = 0, alpha = None)` |
| `PatchPerturbation.init` | `init() -> torch.Tensor` |
| `PatchPerturbation.apply` | `apply(sample: AdversarialSample, delta: torch.Tensor) -> torch.Tensor` |
| `PatchPerturbation.project` | `project(delta: torch.Tensor) -> torch.Tensor` |
| `PatchPerturbation.export` | `export(delta: torch.Tensor) -> np.ndarray` |
| `LinfImage` | Class; see methods below |
| `LinfImage.__init__` | `__init__(epsilon = 8 / 255, image_hw = (600, 800), device = 'cuda:0')` |
| `LinfImage.init` | `init() -> torch.Tensor` |
| `LinfImage.apply` | `apply(sample: AdversarialSample, delta: torch.Tensor) -> torch.Tensor` |
| `LinfImage.project` | `project(delta: torch.Tensor) -> torch.Tensor` |

### avsectester.attack.implemented_attacks.patch_attack.offline

| Public name | Signature or fields |
|---|---|
| `PatchOptimization` | `algorithm: GradientOptimizer`<br>`scorer_id: str \| None = None`<br>`size: tuple[int, int] = (64, 64)`<br>`device: str \| None = None`<br>`target_box: tuple[float, float, float, float] \| None = None`<br>`brightness: tuple[float, float] = (0.8, 1.2)`<br>`contrast: tuple[float, float] = (0.8, 1.2)`<br>`validate_rollout: bool = False`<br>`init_texture: Any = None` |
| `render_candidate` | `render_candidate(session, attack, segment, frame, texture, environment, observation = None)` |
| `optimize_patch` | `optimize_patch(attack, context, options: PatchOptimization)` |
| `to_rgba_uint8` | `to_rgba_uint8(delta: np.ndarray) -> np.ndarray` |
| `run_offline` | `run_offline(attack, context)` |

### avsectester.attack.implemented_attacks.patch_attack.optimization_data

| Public name | Signature or fields |
|---|---|
| `plane_sampling_grid` | `plane_sampling_grid(view, corners)` |
| `projected_host_box` | `projected_host_box(view, host_id)` |
| `native_sampling_grid` | `native_sampling_grid(black, white)` |
| `SessionDataSource` | Class; see methods below |
| `SessionDataSource.__init__` | `__init__(session, attack, metadata, *, size, device, target_box = None, brightness = (0.8, 1.2), contrast = (0.8, 1.2), seed = 0, at_segment_start = False, environment = None, require_patch_grid = True)` |
| `SessionDataSource.sample` | `sample(batch)` |

### avsectester.attack.optimization.whitebox

| Public name | Signature or fields |
|---|---|
| `PGD` | `steps: int = 200`<br>`step_size: float = 4.0 / 255.0`<br>`batch: int = 4`<br>`tv_weight: float = 0.0`<br>`seed: int = 0`<br>`verbose: bool = False`<br>`log_every: int = 25` |
| `PGD.run` | `run(data: DataSource, perturbation: Perturbation, scorer: Scorer, objective: OptimizationObjective) -> OptimizationResult` |

### avsectester.attack.optimization.blackbox

| Public name | Signature or fields |
|---|---|
| `NES` | `steps: int = 40`<br>`population_size: int = 20`<br>`sigma: float = 0.08`<br>`learning_rate: float = 0.03`<br>`seed: int = 0`<br>`verbose: bool = False`<br>`log_every: int = 5` |
| `NES.run` | `run(data: DataSource, perturbation: Perturbation, scorer: Scorer, objective: OptimizationObjective) -> OptimizationResult` |

### avsectester.runlog.log

| Public name | Signature or fields |
|---|---|
| `StepRecord` | `step: int`<br>`raw_observation: Any = None`<br>`model_observation: Any = None`<br>`physical_pre: Any = None`<br>`physical_post: Any = None`<br>`produced_command: Any = None`<br>`executed_command: Any = None`<br>`components: dict[str, Any] = field(default_factory=dict)`<br>`detections: Sequence[Detection] \| None = None`<br>`sensor_geometry: Mapping[str, SensorGeometry] = field(default_factory=dict)`<br>`effects: list[Any] = field(default_factory=list)`<br>`outputs: list[Any] = field(default_factory=list)`<br>`telemetry: dict[str, Any] = field(default_factory=dict)` |
| `EventRecord` | `kind: str`<br>`step: int \| None = None`<br>`data: dict[str, Any] = field(default_factory=dict)` |
| `RunLog` | `metadata: dict[str, Any] = field(default_factory=dict)`<br>`steps: list[StepRecord] = field(default_factory=list)`<br>`events: list[EventRecord] = field(default_factory=list)`<br>`status: str = 'incomplete'`<br>`path: Path \| None = None` |
| `RunLog.read` | `read(path: str \| Path, *, codecs: CodecRegistry \| None = None, load_payloads: bool = True) -> RunLog` |
| `CodecRegistry` | Class; see methods below |
| `CodecRegistry.__init__` | `__init__() -> None` |
| `CodecRegistry.register` | `register(native_type: type, encode: Callable[[Any], Any], *, name: str \| None = None, decode: Callable[[Any], Any] \| None = None, snapshot: Callable[[Any, Callable[[Any], Any]], Any] \| None = None) -> CodecRegistry` |
| `CodecRegistry.encoder` | `encoder(value: Any) -> tuple[str, Callable[[Any], Any]] \| None` |
| `CodecRegistry.snapshotter` | `snapshotter(value: Any) -> Callable[[Any, Callable[[Any], Any]], Any] \| None` |
| `Recorder` | Class; see methods below |
| `Recorder.__init__` | `__init__(path: str \| Path \| None = None, *, metadata: dict[str, Any] \| None = None, codecs: CodecRegistry \| None = None, chunk_steps: int = 128, chunk_bytes: int = 8 * 1024 * 1024, payload_shard_bytes: int = 64 * 1024 * 1024, retain_steps: bool = True) -> None` |
| `Recorder.capture` | `capture(value: Any) -> Any` |
| `Recorder.record_step` | `record_step(record: StepRecord) -> StepRecord` |
| `Recorder.record_event` | `record_event(kind: str, *, step: int \| None = None, **data) -> EventRecord` |
| `Recorder.flush` | `flush() -> None` |
| `Recorder.finish` | `finish(status: str = 'completed', **details) -> RunLog` |
| `Recorder.abort` | `abort() -> None` |
| `read_table` | `read_table(path: str \| Path, columns: Sequence[str] \| None = None, *, stream: str = 'steps') -> Any` |

### avsectester.evaluation.metric

| Public name | Signature or fields |
|---|---|
| `UnavailableMetricValue` | `reason: str`<br>`value_type: Literal['number', 'boolean'] = 'number'` |
| `ComparisonGroup` | `attack_name: str`<br>`segment: AttackSegment`<br>`seed: int`<br>`runs: Mapping[str, RunLog]` |
| `TreatmentPair` | `treatment_a: str`<br>`treatment_b: str`<br>`log_a: RunLog`<br>`log_b: RunLog` |
| `Metric` | Class; see methods below |
| `Metric.evaluate` | `evaluate(data: Any) -> Mapping[str, Any]` |
| `PairMetric` | Class; see methods below |
| `PairMetric.evaluate` | `evaluate(pair: TreatmentPair) -> FieldMeasurements` |
| `speeds` | `speeds(log: RunLog) -> list[float] \| None` |
| `executed_brakes` | `executed_brakes(log: RunLog) -> list[float] \| None` |
| `brake_steps` | `brake_steps(log: RunLog) -> int \| None` |
| `DrivingMetrics` | Class; see methods below |
| `DrivingMetrics.__init__` | `__init__(stop_speed: float = 0.5) -> None` |
| `DrivingMetrics.evaluate` | `evaluate(group: ComparisonGroup) -> TreatmentMeasurements` |
| `TrajectoryDeviation` | Class; see methods below |
| `TrajectoryDeviation.evaluate` | `evaluate(pair: TreatmentPair) -> FieldMeasurements` |
| `DetectionMetrics` | Class; see methods below |
| `DetectionMetrics.__init__` | `__init__(match_distance: float = 2.0, max_range: float \| None = 50.0, category_map: Mapping[str, str] \| None = None) -> None` |
| `DetectionMetrics.evaluate` | `evaluate(group: ComparisonGroup) -> TreatmentMeasurements` |
| `StopImpact` | Class; see methods below |
| `StopImpact.__init__` | `__init__(stop_speed: float = 0.5, reference_speed: float = 1.0, min_overdrive_steps: int = 3) -> None` |
| `StopImpact.evaluate` | `evaluate(pair: TreatmentPair) -> FieldMeasurements` |

### avsectester.evaluation.report

| Public name | Signature or fields |
|---|---|
| `EvaluationConfiguration` | `metrics: tuple[Metric, ...] \| None = None` |
| `EvaluationReport` | `groups: list[dict[str, Any]] = field(default_factory=list)`<br>`treatment_summaries: list[dict[str, Any]] = field(default_factory=list)`<br>`pair_summaries: list[dict[str, Any]] = field(default_factory=list)`<br>`selections: list[dict[str, Any]] = field(default_factory=list)`<br>`metrics: list[dict[str, Any]] = field(default_factory=list)`<br>`notes: list[str] = field(default_factory=list)`<br>`metric_errors: list[dict[str, Any]] = field(default_factory=list)`<br>`path: Path \| None = None` |
| `EvaluationReport.as_dict` | `as_dict() -> dict[str, Any]` |
| `EvaluationReport.format_summary` | `format_summary() -> str` |
| `EvaluationReport.save` | `save(directory: str \| Path) -> Path` |
| `evaluate` | `evaluate(result: ExperimentResult, configuration: ExperimentConfiguration) -> EvaluationReport` |
| `plot_group` | `plot_group(runs: Mapping[str, Any], path: str \| Path, *, title: str \| None = None) -> Path` |

### avsectester.evaluation.visualization

| Public name | Signature or fields |
|---|---|
| `as_rgb` | `as_rgb(frame: Any) -> np.ndarray \| None` |
| `camera_view` | `camera_view(record: StepRecord \| Observation, camera: str \| None = None) -> np.ndarray \| None` |
| `lidar_bev` | `lidar_bev(record: StepRecord \| Observation, size: int = 800, meters: float = 60.0) -> np.ndarray \| None` |
| `annotate` | `annotate(image: Any, detections: Sequence[Any], threshold: float = 0.3) -> np.ndarray` |
| `box3d_corners` | `box3d_corners(center: Any, extent: Any, yaw: float) -> np.ndarray` |
| `draw_boxes3d` | `draw_boxes3d(image: Any, view: SensorView, *, actors: Mapping[str, Any] \| None = None, target: str \| None = None, subdiv: int = 8, max_distance: float \| None = None, color: tuple[int, int, int] = (40, 200, 40), target_color: tuple[int, int, int] = (235, 64, 52)) -> np.ndarray` |
| `sensor_view` | `sensor_view(record: StepRecord, camera: str \| None = None) -> SensorView \| None` |
| `labels_view` | `labels_view(record: StepRecord, camera: str \| None = None, *, target: str \| None = None, max_distance: float \| None = None) -> np.ndarray \| None` |
| `detections_view` | `detections_view(record: StepRecord, camera: str \| None = None, *, threshold: float = 0.3) -> np.ndarray \| None` |
| `save_image` | `save_image(image: Any, path: str \| Path) -> None` |
| `record_run` | `record_run(log: RunLog, out_dir: str \| Path, *, visualize: View = camera_view, prefix: str = 'frame') -> tuple[Path, ...]` |
| `filmstrip` | `filmstrip(source: RunLog \| Sequence[Any], cols: int = 4, pad: int = 6, bg: tuple[int, int, int] = (20, 20, 20), *, visualize: View = camera_view) -> np.ndarray` |
| `save_gif` | `save_gif(source: RunLog \| Sequence[Any], path: str \| Path, fps: int = 5, *, visualize: View = camera_view) -> None` |
| `save_sequence` | `save_sequence(log: RunLog, out_dir: str \| Path, name: str = 'sequence', cols: int = 4, fps: int = 4, *, visualize: View = camera_view) -> tuple[str, str] \| None` |

### avsectester.geometry.poses

| Public name | Signature or fields |
|---|---|
| `validated_vector` | `validated_vector(value: ArrayLike, name: str) -> np.ndarray` |
| `validated_transform` | `validated_transform(value: ArrayLike) -> np.ndarray` |
| `rotation_matrix` | `rotation_matrix(rotation_deg: ArrayLike) -> np.ndarray` |
| `ActorPose` | `transform: np.ndarray`<br>`extent: tuple[float, float, float] \| None = None` |
| `rotation_to_carla` | `rotation_to_carla(rotation: ArrayLike) -> tuple[float, float, float]` |

### avsectester.geometry.cameras

| Public name | Signature or fields |
|---|---|
| `Camera` | `width: int`<br>`height: int` |
| `Camera.project` | `project(pts_cam: np.ndarray) -> np.ndarray` |
| `Camera.unproject` | `unproject(uv: np.ndarray) -> np.ndarray` |
| `PinholeCamera` | `K: np.ndarray`<br>`width: int`<br>`height: int` |
| `PinholeCamera.project` | `project(pts_cam: np.ndarray) -> np.ndarray` |
| `PinholeCamera.unproject` | `unproject(uv: np.ndarray) -> np.ndarray` |
| `FThetaCamera` | `cx: float`<br>`cy: float`<br>`angle_to_pixeldist: tuple`<br>`width: int`<br>`height: int`<br>`pixeldist_to_angle: tuple = ()`<br>`max_angle: float = math.pi / 2`<br>`linear_cde: tuple = (1.0, 0.0, 0.0)` |
| `FThetaCamera.from_nurec` | `from_nurec(spec: Any) -> FThetaCamera` |
| `FThetaCamera.project` | `project(pts_cam: np.ndarray) -> np.ndarray` |
| `FThetaCamera.in_view` | `in_view(pts_cam: np.ndarray) -> np.ndarray` |
| `FThetaCamera.unproject` | `unproject(uv: np.ndarray) -> np.ndarray` |
| `quat_to_matrix` | `quat_to_matrix(w: float, x: float, y: float, z: float) -> np.ndarray` |
| `make_pose` | `make_pose(rotation: np.ndarray, translation: ArrayLike) -> np.ndarray` |
| `pose_from_proto` | `pose_from_proto(pose: Any) -> np.ndarray` |
| `planar_rig_pose` | `planar_rig_pose(x: float, y: float, yaw: float, z: float = 0.0) -> np.ndarray` |
| `transform` | `transform(T: np.ndarray, pts: np.ndarray) -> np.ndarray` |
