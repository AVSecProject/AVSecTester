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
| avsectester.geometry.boxes | Oriented-box ray distances and overlap |
| avsectester.geometry.cameras | Camera, PinholeCamera, FThetaCamera and lens/ray conventions |
| avsectester.geometry.mesh | MeshRaycaster and calibrated camera_depth |
| avsectester.geometry.poses | Canonical transforms, quaternions, interpolation and Y-axis mirroring |
| avsectester.geometry.projection | Plane grids, box projection and plane visibility |
| avsectester.serialization | SerializationRegistry and DEFAULT_REGISTRY for JSON encoding/decoding and independent snapshots |
| avsectester.type_paths | Callable imports using explicit module and attribute paths |
| avsectester.validation | Reusable integer and finite real-number parameter validation |
| avsectester.data_types | Shared observations, commands, world/actor state, detections, availability and image targets |
| avsectester.datasets | Dataset and frame/start records |
| avsectester.datasets.base | Dataset interface, timeline records and ScenarioStart |
| avsectester.datasets.carla | CarlaDataset, CarlaRecipe and PrescribedMotion |
| avsectester.datasets.nurec | NuRecDataset and camera-independent read_scene_mesh |
| avsectester.datasets.nuscenes | NuScenesDataset |
| avsectester.maps | ScenarioMap, Lane, TrafficControl, StopLine, STOP_LINE_KIND_BY_CONTROL_KIND and read_opendrive |
| avsectester.maps.clipgt | NuRec clipgt semantic-map archive reader |
| avsectester.maps.opendrive | OpenDRIVE semantic-map reader |
| avsectester.maps.scenario_map | Semantic-map records and stopping-control classification |
| avsectester.rendering | Object assets, surface rendering, composition and harmonizers |
| avsectester.rendering.compositing | ObjectVisibility, CompositePlan, CompositeResult, plan_composite, composite_objects and blend_rgba |
| avsectester.rendering.harmonizers | Harmonizer, ClassicHarmonizer and PCTNetHarmonizer |
| avsectester.rendering.objects | PlaneSurface, ObjectAsset and PlacedObject |
| avsectester.rendering.surfaces | Calibrated planar sampling and render_surfaces |
| avsectester.rendering.textures | Texture, CheckerboardTexture and load_rgba_image |
| avsectester.closed_loop.avstacks.base | AVStack, ObservationPipeline and command compatibility checks |
| avsectester.closed_loop.avstacks | AVStack, ObservationPipeline and driving-adapter interfaces |
| avsectester.closed_loop.avstacks.alpamayo | AlpamayoAVStack and its component bindings |
| avsectester.closed_loop.avstacks.camera_forward_collision | CameraForwardCollisionAVStack |
| avsectester.closed_loop.avstacks.modular | ModularAVStack, native hooks, dataset registration and component bindings |
| avsectester.closed_loop.avstacks.scorers | Driving-model scorer implementations |
| avsectester.closed_loop.avstacks.driving_adapter | DrivingAdapter, ComponentDescriptor and ScorerDescriptor |
| avsectester.closed_loop.simulators.ground_contact | Mesh-based ego ground constraint |
| avsectester.closed_loop.simulators.navigation | RouteGenerator: map or recorded navigation intent |
| avsectester.closed_loop.simulators.ego_dynamics | PlanarEgoState, EgoDynamics and vehicle-motion implementations |
| avsectester.closed_loop.environment | Environment validation/merge and SensorEnvironment |
| avsectester.closed_loop.hooks | HookSet, Hook, Plugin, contexts and stage values |
| avsectester.closed_loop.sensor_corruptions | Built-in image corruptions and AlbumentationsCorruption |
| avsectester.closed_loop.simulators | Simulator/runtime, sensor-view, initialization and deployment interfaces |
| avsectester.closed_loop.simulators.base | Simulator/runtime, sensor views, initialization, frame sequences and deployment plans |
| avsectester.closed_loop.simulators.carla | CarlaSimulator and CarlaSimulatorRuntime exports |
| avsectester.closed_loop.simulators.carla.conversions | CARLA rotation and canonical-state conversion |
| avsectester.closed_loop.simulators.carla.recipe_replay | RecipeReplay |
| avsectester.closed_loop.simulators.carla.scenario | CARLA scenario preparation and seed resolution |
| avsectester.closed_loop.simulators.carla.sensors | SensorCapture, sensor alignment, depth decoding and pinhole registration |
| avsectester.closed_loop.simulators.carla.simulator | CARLA simulator configuration and runtime |
| avsectester.closed_loop.simulators.nurec | NuRecSimulator, NuRecSimulatorRuntime and NuRecRenderer exports |
| avsectester.closed_loop.simulators.nurec.renderer | Renderer, NuRecRenderer and RPC pose/camera conversion |
| avsectester.closed_loop.simulators.nurec.simulator | NuRec simulator configuration and runtime |
| avsectester.closed_loop.simulators.recorded | RecordedReplaySimulator and RecordedReplaySimulatorRuntime |
| avsectester.attack | Attack, AttackSegment, OfflineContext, AttackRuntime and actions |
| avsectester.attack.attack | Attack, AttackSegment and OfflineContext |
| avsectester.attack.insertion | InsertObject, sensor insertion, assets, placement, compositing and visibility |
| avsectester.attack.insertion.action | InsertObject and InsertionAttackAction |
| avsectester.attack.insertion.assets | Planar assets, insertion descriptors and placement resolution |
| avsectester.attack.insertion.compositor | PatchCompositor and sensor insertion rendering |
| avsectester.attack.insertion.placement | AttachedPlacement, WorldPlacement and Orientation |
| avsectester.attack.insertion.types | Visibility and visibility evidence records |
| avsectester.attack.insertion.visibility | Depth/cuboid visibility estimators and mask evidence |
| avsectester.attack.optimization | Generic optimization contracts, PGD, NES, perturbations and SessionScorer |
| avsectester.attack.optimization.blackbox | NES optimizer |
| avsectester.attack.optimization.interface | Generic data, perturbation, scoring and objective contracts |
| avsectester.attack.optimization.perturbations | PatchPerturbation and LinfImage |
| avsectester.attack.optimization.scorers | SessionScorer |
| avsectester.attack.optimization.whitebox | PGD optimizer |
| avsectester.attack.runtime | AttackRuntime, actions, subscriptions and OnlineContext |
| avsectester.attack.segments | split_segments |
| avsectester.runlog | RunLog, Recorder, StepRecord, EventRecord and read_table |
| avsectester.runlog.log | RunLog, Recorder, step/event records and selected-column reads |
| avsectester.execution.frame_data_cache | Preparation-owned frame data cache and fingerprints |
| avsectester.execution.runner | ClosedLoopRunner, RunRequest and RecordingPolicy |
| avsectester.execution.services | ClosedLoopServices/session, ScenarioView and request/result records |
| avsectester.implemented_attacks.<attack_folder> | Concrete Attack class exported by its folder |
| avsectester.evaluation | RunGroup, evaluate_groups, metrics and report records |
| avsectester.evaluation.evaluator | Standalone ordered log groups and metric evaluation |
| avsectester.evaluation.metric | ComparisonGroup, Metric and built-in metrics |
| avsectester.evaluation.report | EvaluationConfiguration, EvaluationReport, statistics and formatting |
| avsectester.evaluation.visualization | Group plots, log-only frames, overlays, filmstrips and animation exports |
| avsectester.experiment | Experiment, configuration/prepared/result records and evaluate |
| avsectester.experiment.config | Configuration records, YAML structure, validation and loading |
| avsectester.experiment.type_registry | Built-in type resolution and concrete attack discovery |
| avsectester.experiment.configuration_summary | Configuration value and type descriptions |
| avsectester.experiment.result_evaluation | Order experiment logs, select metrics and save reports/plots |
| avsectester.experiment.engine | Experiment preparation and execution |
| avsectester.experiment.models | Prepared/result records and preparation provenance |
| avsectester.experiment.output_directory | Experiment output layout, prepared metadata, artifacts, maps and result loading |

## Experiment API

Use Experiment.prepare(config) and execute(prepared). Configuration can be an existing typed object,
mapping or YAML path. PreparedExperiment owns explicit metadata and artifacts and no live resources.
Save/load are methods on PreparedExperiment. ExperimentResult.load(output, config) loads saved runs
for evaluation without starting a simulator or loading artifacts for deployment.

ExperimentConfiguration contains the dataset, simulator, avstack, attacks, hooks, evaluation and a
frozen RunsConfiguration under runs. Read or change treatments, seeds, decision_steps, environment,
recording, output and strict through configuration.runs. YAML uses the same runs nesting; old
top-level run settings are rejected rather than converted. Relative runs.output paths resolve
against the containing YAML file.

Constructor path resolution follows Path/os.PathLike annotations, including unions and sequence
item types. Strings and Path values in these parameters become absolute path strings relative to
the containing YAML, with ~ expanded; URI strings containing :// and non-path objects are unchanged.
Plain str, Any, unannotated parameters, **kwargs and ordinary nested dictionaries are not path
signals. Unresolved annotations mentioning Path are configuration errors; their names must be
available at runtime. Other unresolved annotations are ignored. This applies to every framework
type/options constructor. See
[filesystem parameters](configuration.md#constructor-filesystem-parameters).

Configuration summaries are lazy. Direct Python attacks override configuration_parameters when
constructor inputs cannot be read from same-named attributes. Unverifiable parameters warn on save,
are explicitly marked unverified, and prevent validated preparation loading. They do not prevent
configuration construction or in-memory execution.

PreparedExperiment.preparation records the preparation inputs (seed and offline_environments) as
canonical data; [Saved preparation compatibility](configuration.md#saved-preparation-compatibility)
defines what reuse checks.

OfflineContext.environment merges runs.environment with the attack environment. Metabuilder
sensor/render requests use the original world unless they specify an environment explicitly.
ExperimentResult.load and CLI evaluate read historical logs through load_prepared_metadata without
constraining runs.

Configurations contain exactly one attack. Treatments enable its attack_name and optional
environment overrides on shared metadata. Every treatment pair is evaluated; no baseline is
configured. The same hooks apply to every treatment. Result failures and selection errors remain
available independently.

ConfigurationMismatchError identifies an incompatible saved summary or preparation input. Other
result-reading exceptions remain read failures rather than configuration errors.
ExperimentResult.notices contains ordinary messages for skipped saved runs that the configuration or
saved preparation does not expect, such as removed treatments. The output_directory module defines
the run layout through run_directory; saved_run_directories lists saved runs by that layout.

## Attack API

Attack parameters remain reusable configuration. build_metadata(scenario) is the metabuilder;
run_offline(context) defaults to None; create_runtime(segment, artifacts) defaults to
an empty runtime. The explicit artifact argument is the sole saved/executed value.

AttackSegment start/stop belong to the same scenario, stop > start and decision_steps is their
position difference. Its only fields are scenario_id, start, stop and user_data. Actor IDs,
placement parameters, scores and other attack-specific data belong to user_data. Persisted custom
data must contain only JSON-like values or numpy arrays. The framework checks boundaries without
interpreting custom data, checking selected actors or ranking candidates.

AttackRuntime setup returns only world actions; step returns typed actions once per decision; stop
closes the online lifecycle. All history and bound AVStack callbacks share one runtime instance.
subscriptions supports observation and physical_state snapshots. OnlineContext gives physical
frame/time and record_output but no live time-advancing resources.

Attack.deployment_ticks(segment) declares physical setup needs. OfflineContext includes metadata,
selection seed, resolved experiment/attack environment and the common DeploymentPlan.

World actions receive SimulatorRuntime and an ExitStack for cleanup. Sensor actions receive the
current sensor value and a SensorView containing the original capture, for cameras, LiDAR, GNSS
and IMU alike. AVStack actions can target localization.pre through command, excluding sensors.post;
they receive the
current value at their declared stage and OnlineContext. See [Runtime](runtime.md).

## Dataset API

Dataset methods expose recorded records or recipes and never own simulation. Shared ActorState
and Unavailable records are imported from avsectester.data_types. N decisions require N+1
supported FrameReference records. Frames use canonical timeline position separately from native
frame_id. ActorState.pose is world-from-actor; box_center is its local bounding-box center, extent
is full length/width/height in metres and velocity is world-frame m/s. box_rotation describes local
box orientation. box_transform() composes the world-from-box-center pose; pose remains the actor
origin. Unknown extent is None, distinct from explicit zero dimensions. Pose-only
operations accept unknown extent; attached placement and cuboid occlusion require known dimensions.
FrameData.actors excludes ego, which is only in .ego.

Dataset.scenario_map returns ScenarioMap or Unavailable(reason); its default provides no map.
ScenarioView.scenario_map forwards that method for selection. Map records use canonical world
metres, frozen dataclasses and copied read-only geometry arrays. NuRec reads clipgt where present;
CARLA reads exported OpenDRIVE through its optional client package. See [Semantic
maps](../guides/scenarios.md#semantic-maps) for source coverage and coordinate conventions.

ExperimentResult.scenario_maps retains one map or unavailable reason per selected scene.
Persistent results store it under maps/<scenario_id>.json. Evaluation uses these saved values through
ComparisonGroup.scenario_map without dataset access. Maps are not online attack or AVStack inputs.
WorldSnapshot.traffic_lights stores dynamic signal states keyed by TrafficControl.id.

read_scene_mesh(usdz_path) in datasets.nurec merges mesh.ply and mesh_ground.ply into world-coordinate
vertices (N×3) and indexed triangle faces (M×3), without a camera or calibration lookup.
NuRecDataset.scene_mesh and the simulator share this reader. NuRecSimulator depth="mesh" is the
default and requires the optional nurec dependencies at construction; depth="none" disables it.
sensor_view computes optical-axis mesh/actor-box depth on demand and caches it per capture and camera. A
two-scene process cache holds static mesh BVHs. See [Mesh depth](../guides/simulators.md#mesh-depth).
In mesh mode, NuRecSimulatorRuntime.reset with a recorded Dataset ScenarioStart verifies that
mesh.ply and mesh_ground.ply exist in its USDZ before the first NRE request. Missing entries raise
ValueError naming the files. depth="none" disables this depth preflight, while ego ground contact
still requires the selected ground_mesh file unless physics_update_mode="none".

NuRecDataset uses a fixed time_step timeline, interpolates only within recording support, and uses
the native camera_id in FrameData.sensor_references. Dataset public camera and trajectory methods
take scenario_id. Sensor aliases belong only to simulators: NuRecSimulator.camera_aliases maps
recorded camera IDs, and RecordedReplaySimulator.camera_aliases maps names such as CAM_FRONT to
model-visible names such as front. CARLA generated data lives in FrameDataCache, while CarlaDataset
itself remains recipe-only.

## Simulator API

closed_loop.simulators exports interfaces from simulators.base. Import concrete simulators from
their implementation modules, such as simulators.carla, simulators.nurec or simulators.recorded.

SimulatorRuntime.world_key identifies the underlying world. Its default is ("simulator", id(self));
runtimes attached to the same external CARLA world share a key. claim_world(owner) assigns that
world to an owner within the process. The same owner object may nest claims; a different owner is
rejected immediately. The world becomes unclaimed after the outermost context exits. exclusive()
allows one operation
at a time on that simulator runtime and can be used inside a world claim.

Simulator.create_runtime(seed) creates a new owned world. SimulatorRuntime.reset receives
ScenarioStart, optional initialization/deployment and only world settings. SimulatorRuntime.step
advances and captures; observe returns an independent current capture and never renders or advances.
state returns physical WorldSnapshot, including ego, other actors and separate inserted objects.
The creator closes the simulator runtime.

Simulator.can_insert_objects enables shared camera composition. insert_object registers an ObjectAsset
under a unique ID and rigid world_from_object pose; remove_object removes it. state().objects reflects
the registration immediately, while images change at the next capture. Each inserted ActorState has
zero velocity and box_center=(0, 0, height/2). sensor_view returns composed camera data. Existing
native depth is merged with object depth by pixelwise minimum; missing native depth stays None.
object_visibility(object_id) returns the latest capture's camera-to-ObjectVisibility mapping; see
[Inserted world objects](../guides/simulators.md#inserted-world-objects). InsertObject registers
cleanup. See [Insert a world object](../guides/attacks.md#insert-a-world-object) for a runnable
setup action.

Custom simulator runtime classes implement native operations, not reset/step. Simulator declarations
include dataset_kinds, decision_time_step, command_modes, supported_stages,
supported_world_settings, can_render, can_insert_objects, can_replay, can_observe_past,
capture_ticks and identity.
validate_world_settings checks supported names and values without native startup; validate_start checks
dataset/time-step compatibility.

SensorGeometry contains sensor, typed camera and cam_from_world. camera_sensors declares eligible
cameras. The base sensor_geometry derives it from _sensor_view; an optional lightweight override
can avoid image or depth work. Implementations supply native _sensor_view and leave public
sensor_view to the shared simulator-runtime composition path.

SimulatorRuntime.sensors declares every sensor supported by sensor_view; camera_sensors is its
camera subset. SensorView contains data, actors, ego, time, frame, native metadata and the
objects composited into this camera (see
[Inserted world objects](../guides/simulators.md#inserted-world-objects)). Camera views provide
camera and cam_from_world (optical
x right, y down, z forward). sensor_from_world maps world
points to the sensor body (x forward, y left, z up), including for cameras, and is None if its pose
is unavailable. Optional depth belongs to camera views. Camera project/unproject use Nx3 points
and Nx2 pixel coordinates. BODY_FROM_OPTICAL in geometry/cameras.py converts optical points to
body axes. CARLA LiDAR views snapshot raw SDK bytes without changing their axes; the native wrapper
retains its flipy option. See [CARLA LiDAR data](runtime.md#carla-lidar-data) for the byte layout,
coordinate convention and a replacement action.

start_replay initializes and captures a recorded or recipe boundary while retaining prescribed ego
motion. Both replay methods reject bound hooks. advance_replay(count) advances that motion and
captures without stage hooks; it requires an initialized runtime and can_replay=True. CARLA and
recorded playback declare True; NuRec declares False and supports raw
frame reads through repeated start_replay calls. SimulatorRuntime owns both public methods;
implementations supply _replay_tick rather than overriding them.

Simulator.can_observe_past declares access to recorded observations before the selected start.
After reset, observe_past(count) returns that many observations from oldest to newest, with negative
run-relative t and frame values. It requires start.frame.position >= count and nonempty
start.recorded_ego. Reading the past neither changes the current capture/world nor includes inserted
objects or attack actions. NuRec supports this capability; CARLA and recorded replay do not.
Implementations supply _observe_recorded(position), leaving observe_past to the base lifecycle.

## Driving-system API

closed_loop.avstacks exports AVStack, ObservationPipeline, command compatibility and driving-adapter
interfaces. Import concrete driving systems from avstacks.modular, avstacks.alpamayo or
avstacks.camera_forward_collision.

AVStack.__call__(observation) returns Control. reset(observation, past=()) receives the first already
processed input and preceding recorded observations ordered oldest to newest. AVStack.past_steps is a
class-level declaration, defaulting to 0. Each past input receives the treatment's sensor corruptions
at its own absolute frame position, without stage hooks, ObservationPipeline or attacks. The
initialization event records past_steps and past_observations containing t and vehicle_state; past
images are not recorded there.
ObservationPipeline applies sensors.post hooks, localization, localization hooks and observation
hooks, and owns the localizer lifecycle. It does not process output commands. AVStack also
declares command_modes, supported_stages, models, component_log and detections. None detections
means unsupported; an empty sequence is valid zero detections. Base close() is a no-op.
AVStack.observation_pipeline is an assignable ObservationPipeline property. Detection uses the
true capture-time ego frame
box center, full extent and yaw in radians, plus optional category/score. Conversion uses sensor
mounting extrinsics, independently of model-visible localization.

The configured AVStack constructor receives only options and creates a fresh instance per run.
DrivingAdapter separately binds component/scorer callables. Its Python factory accepts a seed; YAML
avstack.bindings receives the loader's configured factory and returns DrivingAdapter. Scorers
declare differentiable inputs and direction; services reject unavailable gradients.
DrivingAdapter.past_steps reads factory.avstack_type.past_steps when avstack_type is an AVStack
subclass, otherwise its explicit past_steps constructor option. This inspection never creates an
AVStack or loads model weights. ClosedLoopSession.can_start_at(position) combines this history
requirement with the shared deployment-window check. The Alpamayo binding explicitly supplies a
15-step fallback for factories without AVStack-type metadata.

## Closed-loop service API

The execution package owns runner, services, candidate dispatch and preparation caches. Its package
initializer stays lightweight; import public contracts from execution.runner or execution.services.

ClosedLoopServices from execution.services exposes capabilities and
session(seed, history="fresh"). Each session owns cleanup. Sensor-only scans reuse a simulator
runtime when its simulator declares can_replay=True; NuRec resets to each requested frame with
start_replay. Rendering with attack actions and rollouts create independent worlds.
World claims span operation cleanup. Native handles do not escape closed operations in returned views.
capabilities.insert_objects reflects Simulator.can_insert_objects and can be checked during
selection. An unsupported InsertObject returned by setup raises NotImplementedError during reset
and fails the run; dynamically created setup actions cannot be inspected in preflight.

SensorRequest selects scenario/frame/sensor_ids/environment and optional views=False. With
views=True, SensorSample.views maps each requested sensor to a detached SensorView with native={};
its data precedes sensor corruptions, while observation contains the transformed input.
The default returns an empty views mapping to avoid depth computation and copying. A simulator
without can_render rejects views=True with the same NotImplementedError as render.
See [Use the selected scenario](../guides/attacks.md#use-the-selected-scenario) for a runnable
multi-frame optimization-sampling example.

read_sensors reads the action-free world and can reuse supported forward scans; render applies
attack actions in
a newly created world for each request. RenderRequest additionally selects
sensor_id and world/sensor actions. AVStack actions require a rollout. CandidateRequest declares
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
corruptions implement apply(rgb, rng). read_camera_rgb returns None for an unknown explicit camera;
it never substitutes another camera. write_sensor_data replaces arrays directly or shallow-copies
native wrappers and replaces their data, without inspecting array shapes or converting dtypes.
The runner chooses write_camera_rgb only when SensorView.camera is present, keeping
camera writes RGB uint8; other sensor writes use write_sensor_data.
read_camera_rgb, write_camera_rgb and write_sensor_data live in
data_types.py and preserve sensor payload transport conventions.

## RunLog API

RunLog holds metadata, steps, events, status and optional path. read loads published Parquet parts.
read_table selects Parquet columns without loading all
payloads. Recorder captures independent in-memory snapshots, records steps/events, flushes closed
parts, finishes or aborts. It does not close caller-owned simulators/models.

StepRecord fields include raw/model observations, pre/post physical state, produced/executed
commands, components, canonical detections, actions/receipts, outputs, typed sensor_geometry,
object_visibility and telemetry. sensor_geometry maps declared camera names to lightweight typed
SensorGeometry records, not SensorView images. physical_pre and physical_post are StepRecord
fields, not fields on RunLog. Receipt details and object_visibility are described in
[RunLog](runtime.md#runlog); RunLog.read restores those typed records using DEFAULT_REGISTRY.
Recorder and RunLog.read accept an explicit registry when isolation is needed. The same registry
encodes prepared metadata and saved maps. Register user dataclasses before writing, and import
their defining and registration modules before reading. Native encoders are registered by their
adapters and persist ordinary data for analysis without Python SDK classes.
Payload retention is controlled by RecordingPolicy. Older RunLog files and experiment output
directories may no longer be readable after the record layout change; there is no format conversion.

## Evaluation API

RunGroup contains all treatment logs in their supplied order, including unfinished or failed runs,
plus optional errors and a scenario map. evaluate_groups(groups, metrics, notes=()) evaluates
already available logs without an Experiment or dataset. It passes only completed logs to the
ComparisonGroup used by metrics and returns an EvaluationReport with statuses and summaries.
The evaluation package exports RunGroup and evaluate_groups.

Metric.evaluate receives ComparisonGroup and returns per-treatment scalar dictionaries.
UnavailableMetricValue(reason, value_type="number" or "boolean") carries missing evidence without
mutable metric state. MetricScalar is one scalar or unavailable value; FieldMeasurements maps field
names to these values, and TreatmentMeasurements maps treatment names to FieldMeasurements.
PairMetric evaluates TreatmentPair; TrajectoryDeviation is symmetric. StopImpact reports stopping
comparisons in both directions. Pair differences for single-run values are B minus A.

evaluate(result, configuration), exported by avsectester.experiment and implemented in
experiment.result_evaluation, writes report.json, runs.csv and plots. The evaluation package owns
RunGroup evaluation, metrics, report records, statistics and visualization without depending on
experiment types. Experiment evaluation orders logs, chooses defaults, attaches selections and
saves report/plot files. Metric
specifications, notes and errors remain in the report. EvaluationReport.format_summary() prints
treatment tables, pair difference matrices and notes; treatment_summaries and pair_summaries use
population standard deviation (ddof=0). summarize recalculates summaries; record_error records an
exception with its source and context. as_dict returns data, and save writes it. No inference or
dataset read occurs.

Visualization functions consume StepRecord/RunLog: camera_view, lidar_bev, labels_view,
detections_view, filmstrip, record_run, save_gif and save_sequence. sensor_view reconstructs only
logged geometric evidence. record_run exports recorded frames; it does not execute an experiment.

## Public signatures

Properties appear as name: Type. Method signatures omit self/cls. Dataclass constructor fields are
listed in declaration order; optional inherited behavior is described in the owning guide. Concrete
Attack classes are imported from their folder, for example `from
avsectester.implemented_attacks.patch_attack import PatchAttack`.

### avsectester.experiment.config

| Public name | Signature or fields |
|---|---|
| `TreatmentConfiguration` | `name: str`<br>`attack_name: str \| None = None`<br>`environment: Mapping[str, Any] = field(default_factory=dict)` |
| `RunsConfiguration` | `treatments: tuple[TreatmentConfiguration, ...] = ()`<br>`seeds: tuple[int, ...] = (0,)`<br>`decision_steps: int \| None = None`<br>`environment: Mapping[str, Any] = field(default_factory=dict)`<br>`recording: RecordingPolicy = field(default_factory=RecordingPolicy)`<br>`output: Path \| None = None`<br>`strict: bool = False` |
| `ExperimentConfiguration` | `dataset: Dataset`<br>`simulator: Simulator`<br>`avstack: DrivingAdapter`<br>`attacks: Mapping[str, Attack] = field(default_factory=dict)`<br>`hooks: HookConfiguration = None`<br>`evaluation: EvaluationConfiguration = field(default_factory=EvaluationConfiguration)`<br>`runs: RunsConfiguration = field(default_factory=RunsConfiguration)` |
| `ExperimentConfiguration.configuration_summary` | `configuration_summary: dict[str, Any]` |
| `ExperimentConfiguration.describe_configuration` | `describe_configuration(*, scenarios: Sequence[str] \| None = None) -> dict[str, Any]` |
| `load_configuration` | `load_configuration(config: ExperimentConfiguration \| Mapping[str, Any] \| str \| Path) -> ExperimentConfiguration` |

### avsectester.experiment.type_registry

resolve_type accepts a built-in component name, a discovered concrete Attack name or an explicit
module:attribute path. is_builtin_type_name checks only the built-in component registry; it does
not discover attacks. Third-party registry specifications remain the owning constructor's concern.

| Public name | Signature or fields |
|---|---|
| `resolve_type` | `resolve_type(value: Any) -> Callable[..., Any]` |
| `attack_directory_name` | `attack_directory_name(class_name: str) -> str` |
| `available_attacks` | `available_attacks() -> tuple[str, ...]` |
| `is_builtin_type_name` | `is_builtin_type_name(value: str) -> bool` |

### avsectester.experiment.configuration_summary

| Public name | Signature or fields |
|---|---|
| `summary_value` | `summary_value(value: Any) -> Any` |
| `type_name` | `type_name(value: Any) -> str` |

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
| `preparation_record` | `preparation_record(configuration: ExperimentConfiguration) -> dict[str, Any]` |
| `SelectionDecision` | `attack_name: str`<br>`scenario_id: str`<br>`status: Literal['selected', 'not_selected', 'error']`<br>`reasons: tuple[str, ...] = ()`<br>`error: str \| None = None` |
| `SelectionDecision.as_dict` | `as_dict() -> dict[str, Any]` |
| `SelectionDecision.from_dict` | `from_dict(value: Mapping[str, Any]) -> SelectionDecision` |
| `treatment_applies` | `treatment_applies(treatment: Any, attack_name: str) -> bool` |
| `PreparedExperiment` | `configuration: ExperimentConfiguration`<br>`metadata: Mapping[str, tuple[AttackSegment, ...]]`<br>`artifacts: Mapping[str, Any]`<br>`selections: tuple[SelectionDecision, ...] = ()`<br>`preparation: Mapping[str, Any]` (required, keyword-only) |
| `PreparedExperiment.save` | `save(directory: str \| Path) -> Path` |
| `PreparedExperiment.load` | `load(config: ExperimentConfiguration \| Mapping[str, Any] \| str \| Path, directory: str \| Path) -> PreparedExperiment` |
| `TreatmentResult` | `attack_name: str`<br>`segment_index: int`<br>`segment: AttackSegment`<br>`treatment: str`<br>`seed: int`<br>`log: RunLog`<br>`error: str \| None = None` |
| `TreatmentResult.group` | `group: tuple[str, int, int]` |
| `ExperimentResult` | `runs: tuple[TreatmentResult, ...] = ()`<br>`selections: tuple[SelectionDecision, ...] = ()`<br>`notices: tuple[str, ...] = ()`<br>`scenario_maps: Mapping[str, ScenarioMap \| Unavailable] = field(default_factory=dict)` |
| `ExperimentResult.groups` | `groups: dict[tuple[str, int, int], dict[str, RunLog]]` |
| `ExperimentResult.failures` | `failures: tuple[TreatmentResult, ...]` |
| `ExperimentResult.selection_errors` | `selection_errors: tuple[SelectionDecision, ...]` |
| `ExperimentResult.load` | `load(output: str \| Path, configuration: ExperimentConfiguration) -> ExperimentResult` |

### avsectester.experiment.output_directory

Defines the experiment output layout and reads/writes its files other than individual RunLog parts.
ExperimentResult.load delegates to load_result here; evaluation restores these saved records
without reading the dataset. PreparedMetadata contains analysis metadata without executable
artifacts; load_prepared always restores the artifacts required by execute. PreparedExperiment
methods provide the usual save/load entry points.

| Public name | Signature or fields |
|---|---|
| `PreparedMetadata` | `metadata: Mapping[str, tuple[AttackSegment, ...]]`<br>`selections: tuple[SelectionDecision, ...]`<br>`preparation: Mapping[str, Any] \| None` |
| `save_prepared` | `save_prepared(prepared: PreparedExperiment, directory: str \| Path) -> Path` |
| `load_prepared` | `load_prepared(configuration: ExperimentConfiguration, directory: str \| Path) -> PreparedExperiment` |
| `load_prepared_metadata` | `load_prepared_metadata(configuration: ExperimentConfiguration, directory: str \| Path) -> PreparedMetadata` |
| `runs_directory` | `runs_directory(root: Path) -> Path` |
| `run_directory` | `run_directory(root: Path, attack_name: str, segment_index: int, seed: int, treatment: str) -> Path` |
| `saved_run_directories` | `saved_run_directories(root: Path) -> tuple[tuple[tuple[str, int, int, str], Path], ...]` |
| `load_result` | `load_result(directory: str \| Path, configuration: ExperimentConfiguration) -> ExperimentResult` |
| `ConfigurationMismatchError` | `ConfigurationMismatchError(ValueError)`; raised when a saved preparation and the supplied configuration describe different attacks or inputs |
| `save_scenario_maps` | `save_scenario_maps(maps: Mapping[str, ScenarioMap \| Unavailable], directory: str \| Path) -> None` |
| `load_scenario_maps` | `load_scenario_maps(directory: str \| Path, scenario_ids: tuple[str, ...]) -> dict[str, ScenarioMap \| Unavailable]` |

### avsectester.experiment.result_evaluation

| Public name | Signature or fields |
|---|---|
| `evaluate` | `evaluate(result: ExperimentResult, configuration: ExperimentConfiguration) -> EvaluationReport` |

### avsectester.attack.attack

| Public name | Signature or fields |
|---|---|
| `AttackSegment` | `scenario_id: str`<br>`start: FrameReference`<br>`stop: FrameReference`<br>`user_data: Mapping[str, Any] = field(default_factory=dict)` |
| `AttackSegment.decision_steps` | `decision_steps: int` |
| `OfflineContext` | `services: ClosedLoopServices`<br>`metadata: tuple[AttackSegment, ...]`<br>`seed: int = 0`<br>`environment: Mapping[str, Any] = field(default_factory=dict)`<br>`deployment: DeploymentPlan \| None = None` |
| `Attack` | `environment: Mapping[str, Any] = MappingProxyType({})` |
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
| `AttackAction` | Class; see methods below |
| `AttackAction.receipt_details` | `receipt_details() -> Mapping[str, Any]` |
| `WorldAttackAction` | `setup_ticks: int = 0` |
| `WorldAttackAction.apply` | `apply(simulator_runtime: SimulatorRuntime, resources: ExitStack) -> None` |
| `SensorAttackAction` | `sensor: str` |
| `SensorAttackAction.apply` | `apply(data: Any, view: SensorView) -> Any` |
| `AVStackAttackAction` | `stage: str`<br>`apply: Callable[[Any, OnlineContext], Any]` |
| `AttackRuntime` | `subscriptions: Sequence[ResultSubscription] = ()`<br>`setup_ticks: int = 0` |
| `AttackRuntime.setup` | `setup(context: OnlineContext) -> Sequence[WorldAttackAction]` |
| `AttackRuntime.step` | `step(context: OnlineContext) -> Sequence[AttackAction]` |
| `AttackRuntime.stop` | `stop(context: OnlineContext) -> None` |

### avsectester.attack.segments

| Public name | Signature or fields |
|---|---|
| `split_segments` | `split_segments(frames: Sequence[FrameReference], eligible: Callable[[FrameReference], bool], *, decision_steps: int, keep_tail: bool = False) -> tuple[AttackSegment, ...]` |
| `validate_segment_options` | `validate_segment_options(decision_steps: int, max_segments: int \| None, keep_tail: bool) -> None` |

### avsectester.attack.insertion.placement

| Public name | Signature or fields |
|---|---|
| `validate_attached_mount` | `validate_attached_mount(anchor: str, offset_m: tuple[float, float, float]) -> tuple[float, float, float]` |
| `AttachedPlacement` | `host: str`<br>`anchor: str = 'center'`<br>`offset_m: tuple[float, float, float] = (0.0, 0.0, 0.0)` |
| `AttachedPlacement.local_position` | `local_position(extent: tuple[float, float, float]) -> np.ndarray` |
| `WorldPlacement` | `position_m: tuple[float, float, float]` |
| `Orientation` | `mode: str = 'fixed_world'`<br>`rotation_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)`<br>`target_offset_m: tuple[float, float, float] = (0.0, 0.0, 0.0)` |
| `placement_from_spec` | `placement_from_spec(value: Any) -> AttachedPlacement \| WorldPlacement` |
| `orientation_from_spec` | `orientation_from_spec(value: Any) -> Orientation` |

### avsectester.validation

Framework code and user extensions can share these parameter checks. Integer validators reject
booleans. Number validators accept finite real numbers and reject booleans. Positive values must
exceed zero; nonnegative values may be zero. Invalid inputs raise ValueError naming the parameter
and the required positive/nonnegative integer or number. The functions validate without coercion.

| Public name | Signature or fields |
|---|---|
| `require_nonnegative_integer` | `require_nonnegative_integer(value: int, name: str) -> None` |
| `require_positive_integer` | `require_positive_integer(value: int, name: str) -> None` |
| `require_nonnegative_number` | `require_nonnegative_number(value: float, name: str) -> None` |
| `require_positive_number` | `require_positive_number(value: float, name: str) -> None` |

### avsectester.type_paths

split_type_path validates an explicit package.module:name path and returns the module and
attribute names without importing anything. The attribute name may contain dots, as in
package.module:Container.CustomAttack. import_type_path uses the same validation, imports the
module, resolves nested attributes and requires the value to be callable. Configuration type
resolution and custom sensor environments use this import rule; importing a module executes its
Python code.

| Public name | Signature or fields |
|---|---|
| `split_type_path` | `split_type_path(value: str) -> tuple[str, str]` |
| `import_type_path` | `import_type_path(value: str) -> Any` |

### avsectester.serialization

SerializationRegistry provides JSON-compatible encode/decode and independent snapshot(value).
Dataclasses encode as `{"_avsectester_type": "dataclass", "type": "package.module:Name",
"fields": {...}}`. Dataclass fields, including a field named record, remain inside fields. String
registrations import their types only during decoding. Native encoders produce ordinary data,
which remains ordinary data after decoding. DEFAULT_REGISTRY pre-registers framework records
without importing heavy SDKs or models. See [RunLog](runtime.md#runlog) for user registration.
The optional encode/decode callbacks provide recursive transport for persistence layers.

| Public name | Signature or fields |
|---|---|
| `SerializationRegistry` | Class; see methods below |
| `SerializationRegistry.__init__` | `__init__() -> None` |
| `SerializationRegistry.register_dataclass` | `register_dataclass(cls_or_path: type \| str) -> None` |
| `SerializationRegistry.register_native` | `register_native(module: str, name: str, encode: Callable[[Any], Any], *, snapshot: Callable[[Any, Callable[[Any], Any]], Any] \| None = None) -> None` |
| `SerializationRegistry.snapshot` | `snapshot(value: Any) -> Any` |
| `SerializationRegistry.encode` | `encode(value: Any, *, encode_child: Callable[[Any], Any] \| None = None, encode_field: Callable[[str, Any], Any] \| None = None, encode_leaf: Callable[[Any], Any] \| None = None) -> Any` |
| `SerializationRegistry.decode` | `decode(data: Any, *, decode_child: Callable[[Any], Any] \| None = None) -> Any` |
| `DEFAULT_REGISTRY` | `SerializationRegistry`; shared framework and adapter registrations |

### avsectester.datasets.base

| Public name | Signature or fields |
|---|---|
| `FrameReference` | `scenario_id: str`<br>`position: int`<br>`frame_id: str \| int`<br>`timestamp: float` |
| `ScenarioDescriptor` | `id: str`<br>`kind: str`<br>`payload: Mapping[str, Any] = field(default_factory=dict)` |
| `SensorDataReference` | `uri: str`<br>`sensor: str`<br>`timestamp: float`<br>`metadata: Mapping[str, Any] = field(default_factory=dict)` |
| `FrameData` | `frame: FrameReference`<br>`actors: Mapping[str, ActorState] = field(default_factory=dict)`<br>`sensor_references: Mapping[str, SensorDataReference] = field(default_factory=dict)`<br>`provenance: Mapping[str, Any] = field(default_factory=dict)`<br>`ego: ActorState \| None = None` |
| `ScenarioStart` | `kind: str`<br>`scenario_id: str`<br>`frame: FrameReference`<br>`payload: Mapping[str, Any]`<br>`recorded_ego: tuple[ActorState, ...] = ()`<br>`scenario_map: ScenarioMap \| Unavailable = Unavailable('The dataset provides no scenario map')` |
| `Dataset` | Class; see methods below |
| `Dataset.scenarios` | `scenarios() -> Iterator[ScenarioDescriptor]` |
| `Dataset.frames` | `frames(scenario_id: str) -> Sequence[FrameReference]` |
| `Dataset.frame_data` | `frame_data(scenario_id: str, frame: FrameReference) -> FrameData \| Unavailable` |
| `Dataset.native` | `native(scenario_id: str) -> Any` |
| `Dataset.scenario_start` | `scenario_start(scenario_id: str, frame: FrameReference) -> ScenarioStart` |
| `Dataset.scenario_map` | `scenario_map(scenario_id: str) -> ScenarioMap \| Unavailable` |
| `Dataset.checked_frame` | `checked_frame(scenario_id: str, frame: FrameReference) -> FrameReference` |
| `Dataset.decision_window` | `decision_window(scenario_id: str, start: FrameReference, decision_steps: int) -> tuple[FrameReference, ...]` |

ScenarioStart.recorded_ego is the complete recorded ego trajectory on the decision timeline:
positions and length match Dataset.frames and each state matches FrameData.ego. It is empty when
there is no recording, as for CARLA recipes. scenario_map is read-only and shared without copying;
datasets cache parsed maps per scene. NuRec fills both fields, nuScenes fills the trajectory, and
CARLA fills the map when an exported OpenDRIVE map is available. These are simulator inputs;
online attack context still receives no semantic map.

### avsectester.datasets.carla

| Public name | Signature or fields |
|---|---|
| `CARLA_WEATHER_PRESETS` | `tuple[str, ...]`; curated native weather preset names |
| `CARLA_WORLD_SETTINGS` | `frozenset[str]`; supported native world-setting names |
| `validate_carla_world_settings` | `validate_carla_world_settings(settings: Mapping[str, Any]) -> None` |
| `PrescribedMotion` | `controls: tuple[Control, ...] = ()`<br>`trajectory: tuple[ActorState, ...] = ()`<br>`constant_control: Control \| None = None` |
| `PrescribedMotion.expanded` | `expanded(decision_steps: int) -> PrescribedMotion` |
| `PrescribedMotion.validate` | `validate(decision_steps: int) -> None` |
| `CarlaRecipe` | `scenario: dict`<br>`decision_steps: int`<br>`time_step: float`<br>`motion: Mapping[str, PrescribedMotion]`<br>`metadata: Mapping = field(default_factory=dict)` |
| `CarlaDataset` | Class; see methods below |
| `CarlaDataset.__init__` | `__init__(recipes: Mapping[str, CarlaRecipe], map_directory: str \| Path \| None = None) -> None` |
| `CarlaDataset.scenarios` | `scenarios()` |
| `CarlaDataset.frames` | `frames(scenario_id)` |
| `CarlaDataset.frame_data` | `frame_data(scenario_id, frame)` |
| `CarlaDataset.native` | `native(scenario_id)` |
| `CarlaDataset.scenario_start` | `scenario_start(scenario_id, frame)` |
| `CarlaDataset.scenario_map` | `scenario_map(scenario_id: str) -> ScenarioMap \| Unavailable` |

### avsectester.maps.clipgt

| Public name | Signature or fields |
|---|---|
| `read_clipgt` | `read_clipgt(archive: zipfile.ZipFile) -> ScenarioMap` |

### avsectester.datasets.nurec

| Public name | Signature or fields |
|---|---|
| `read_scene_mesh` | `read_scene_mesh(usdz_path: str \| Path) -> tuple[np.ndarray, np.ndarray]` |
| `NuRecDataset` | Class; see methods below |
| `NuRecDataset.__init__` | `__init__(usdz_paths: list[str \| Path], camera_id: str = 'camera_front_wide_120fov', vehicle_labels: tuple[str, ...] = _VEHICLE_LABELS, time_step: float = 0.1) -> None` |
| `NuRecDataset.from_glob` | `from_glob(pattern: str \| Path, **kwargs) -> NuRecDataset` |
| `NuRecDataset.open_archive` | `open_archive(path: str \| Path) -> zipfile.ZipFile` |
| `NuRecDataset.raw_metadata` | `raw_metadata(scenario_id: str) -> dict[str, Any]` |
| `NuRecDataset.camera_geometry` | `camera_geometry(scenario_id: str)` |
| `NuRecDataset.camera_timestamps` | `camera_timestamps(scenario_id: str) -> np.ndarray` |
| `NuRecDataset.usdz_path` | `usdz_path(scenario_id: str) -> str` |
| `NuRecDataset.scenarios` | `scenarios()` |
| `NuRecDataset.frames` | `frames(scenario_id)` |
| `NuRecDataset.frame_data` | `frame_data(scenario_id, frame)` |
| `NuRecDataset.native` | `native(scenario_id)` |
| `NuRecDataset.scenario_start` | `scenario_start(scenario_id, frame)` |
| `NuRecDataset.scenario_map` | `scenario_map(scenario_id: str) -> ScenarioMap \| Unavailable` |
| `NuRecDataset.scene_mesh` | `scene_mesh(scenario_id: str) -> tuple[np.ndarray, np.ndarray]` |
| `NuRecDataset.ego_state` | `ego_state(scenario_id: str, timestamp_us: int) -> ActorState` |
| `NuRecDataset.actor_states` | `actor_states(scenario_id: str, timestamp_us: int) -> dict[str, ActorState]` |

camera_geometry returns an independent FThetaCamera and rig-from-camera pose, preserving the
archive's native forward and inverse polynomials. Missing native pixeldist_to_angle remains empty;
unproject then raises NotImplementedError instead of using a fitted inverse.

### avsectester.datasets.nuscenes

| Public name | Signature or fields |
|---|---|
| `NuScenesDataset` | Class; see methods below |
| `NuScenesDataset.__init__` | `__init__(dataroot: str \| Path, version: str = 'v1.0-trainval', camera: str = 'CAM_FRONT') -> None` |
| `NuScenesDataset.scenarios` | `scenarios()` |
| `NuScenesDataset.frames` | `frames(scenario_id)` |
| `NuScenesDataset.frame_data` | `frame_data(scenario_id, frame)` |
| `NuScenesDataset.native` | `native(scenario_id)` |
| `NuScenesDataset.scenario_start` | `scenario_start(scenario_id, frame)` |

### avsectester.maps.scenario_map

All records are frozen. Construction copies geometry arrays and makes them read-only. Coordinates
are canonical world metres. Polygon and polyline arrays have shape N×3; poses have shape 4×4.

| Public name | Signature or fields |
|---|---|
| `STOP_LINE_KIND_BY_CONTROL_KIND` | Read-only `Mapping[str, str]`: stop_sign, yield_sign and traffic_light map to themselves |
| `ScenarioMap` | `lanes: Mapping[str, Lane] = field(default_factory=dict)`<br>`drivable_area: tuple[np.ndarray, ...] = ()`<br>`road_boundaries: tuple[np.ndarray, ...] = ()`<br>`crosswalks: tuple[np.ndarray, ...] = ()`<br>`intersections: tuple[np.ndarray, ...] = ()`<br>`traffic_controls: Mapping[str, TrafficControl] = field(default_factory=dict)`<br>`stop_lines: Mapping[str, StopLine] = field(default_factory=dict)`<br>`unavailable_layers: Mapping[str, str] = field(default_factory=dict)` |
| `Lane` | `id: str`<br>`centerline: np.ndarray`<br>`left_boundary: np.ndarray`<br>`right_boundary: np.ndarray`<br>`lane_type: Literal['driving', 'shoulder', 'bike', 'parking', 'other'] = 'driving'`<br>`predecessors: tuple[str, ...] = ()`<br>`successors: tuple[str, ...] = ()`<br>`left_neighbor: str \| None = None`<br>`right_neighbor: str \| None = None`<br>`speed_limit_mps: float \| None = None` |
| `TrafficControl` | `id: str`<br>`kind: Literal['stop_sign', 'yield_sign', 'speed_limit', 'traffic_light', 'other']`<br>`native_category: str`<br>`pose: np.ndarray`<br>`extent: tuple[float, float, float]`<br>`pose_reference: Literal['face_center', 'signal_position']`<br>`lane_ids: tuple[str, ...] = ()`<br>`stop_line_id: str \| None = None` |
| `StopLine` | `id: str`<br>`points: np.ndarray`<br>`native_category: str`<br>`kind: Literal['stop_sign', 'traffic_light', 'yield_sign', 'unknown'] = 'unknown'`<br>`lane_ids: tuple[str, ...] = ()`<br>`control_id: str \| None = None` |

Field values, the pose_reference sources, unavailable_layers keys and per-source mappings are
defined in [Semantic maps](../guides/scenarios.md#semantic-maps) and its
[Data-source coverage](../guides/scenarios.md#data-source-coverage) section.

### avsectester.maps.opendrive

| Public name | Signature or fields |
|---|---|
| `read_opendrive` | `read_opendrive(xodr_text: str, *, name: str) -> ScenarioMap \| Unavailable` |

Uses carla.Map(name, xodr_text) locally, without a server. Missing carla returns Unavailable.
Every lane type is sampled at intervals of at most two metres; lateral neighbors must travel in
the same direction. Each signal reference generates one stop line per governed lane. A control's
lane_ids unions its references; stop_line_id is set only when exactly one line belongs to it.
See [Semantic maps](../guides/scenarios.md#semantic-maps) for lane sampling and stop-line generation.

### avsectester.data_types

Shared data records for every layer: driving-system observations and commands, world and actor
state, detections, unavailable-evidence reasons and image target regions. This module has no
dataset, simulator or model dependencies.

| Public name | Signature or fields |
|---|---|
| `Unavailable` | `reason: str` |
| `ActorState` | `pose: Any`<br>`velocity: tuple[float, float, float] \| None = None`<br>`category: str \| None = None`<br>`extent: tuple[float, float, float] \| None = None`<br>`box_center: tuple[float, float, float] = (0.0, 0.0, 0.0)`<br>`box_rotation: tuple[float, float, float] = (0.0, 0.0, 0.0)` |
| `ActorState.box_transform` | `box_transform() -> np.ndarray` |
| `TargetSpec` | `box: tuple[float, float, float, float]`<br>`label: str \| None = None` |
| `Control` | `throttle: float = 0.0`<br>`steer: float = 0.0`<br>`brake: float = 0.0`<br>`trajectory: Any = None` |
| `Observation` | `t: float`<br>`frame: int`<br>`sensor_data: dict[str, Any] = field(default_factory=dict)`<br>`calibration: dict[str, Any] = field(default_factory=dict)`<br>`vehicle_state: Any = None`<br>`ego_speed: float = 0.0`<br>`route: np.ndarray \| None = None` |
| `Observation.set_calibration` | `set_calibration(sensor_id: str, calibration: Any) -> None` |
| `WorldSnapshot` | `t: float`<br>`frame: int`<br>`vehicle_state: Any = None`<br>`ego_speed: float = 0.0`<br>`actors: Mapping[str, Any] = field(default_factory=dict)`<br>`ego: ActorState \| None = None`<br>`traffic_lights: Mapping[str, str] = field(default_factory=dict)`<br>`objects: Mapping[str, ActorState] = field(default_factory=dict)` |
| `StateEstimate` | `vehicle_state: Any = None`<br>`ego_speed: float = 0.0` |
| `RecordedOutput` | `name: str`<br>`value: Any`<br>`step: int`<br>`stage: str`<br>`frame: int`<br>`t: float`<br>`sensor: str \| None = None` |
| `Detection` | `position: tuple[float, float, float]`<br>`extent: tuple[float, float, float]`<br>`yaw: float`<br>`category: str \| None = None`<br>`score: float \| None = None` |
| `validate_command_modes` | `validate_command_modes(command: Control, modes: frozenset[str] \| None) -> None` |
| `read_camera_rgb` | `read_camera_rgb(observation: Observation, camera: str \| None = None) -> tuple[str, np.ndarray] \| None` |
| `write_sensor_data` | `write_sensor_data(observation: Observation, sensor: str, data: Any) -> Observation` |
| `write_camera_rgb` | `write_camera_rgb(observation: Observation, key: str, rgb: np.ndarray) -> Observation` |

### avsectester.closed_loop.simulators.base

| Public name | Signature or fields |
|---|---|
| `Initialization` | `expected: FrameData`<br>`resolved_scenario: Mapping[str, Any] = field(default_factory=dict)`<br>`provenance: Mapping[str, Any] = field(default_factory=dict)` |
| `FrameDataSequence` | `frames: tuple[FrameData, ...]`<br>`resolved_scenario: Mapping[str, Any]`<br>`provenance: Mapping[str, Any]` |
| `FrameDataSequence.at` | `at(frame: FrameReference) -> FrameData` |
| `DeploymentPlan` | `ticks: int = 0` |
| `can_deploy_at` | `can_deploy_at(position: int, deployment: DeploymentPlan, capture_ticks: int) -> bool` |
| `ClockSnapshot` | `t: float`<br>`frame: int` |
| `SensorGeometry` | `sensor: str`<br>`camera: Camera`<br>`cam_from_world: np.ndarray` |
| `SensorView` | `sensor: str`<br>`data: Any`<br>`camera: Camera \| None`<br>`cam_from_world: np.ndarray \| None`<br>`actors: Mapping[str, ActorState]`<br>`ego: ActorState`<br>`time: float`<br>`frame: int`<br>`sensor_from_world: np.ndarray \| None = None`<br>`depth: np.ndarray \| None = None`<br>`native: Mapping[str, Any] = field(default_factory=dict)`<br>`objects: Mapping[str, PlacedObject] = field(default_factory=dict)` |
| `Simulator` | `dataset_kinds: tuple[str, ...] = ()`<br>`can_render: bool = False`<br>`can_insert_objects: bool = False`<br>`can_replay: bool = False`<br>`can_observe_past: bool = False`<br>`supported_world_settings: frozenset[str] = frozenset()` |
| `Simulator.create_runtime` | `create_runtime(seed: int \| None = None) -> SimulatorRuntime` |
| `Simulator.decision_time_step` | `decision_time_step: float \| None` |
| `Simulator.command_modes` | `command_modes: frozenset[str] \| None` |
| `Simulator.supported_stages` | `supported_stages: frozenset[str]` |
| `Simulator.identity` | `identity: Mapping[str, Any]` |
| `Simulator.validate_world_settings` | `validate_world_settings(settings: Mapping[str, Any]) -> None` |
| `Simulator.validate_start` | `validate_start(start: ScenarioStart \| None) -> None` |
| `SimulatorRuntime` | Class; see methods below |
| `SimulatorRuntime.__init__` | `__init__(simulator: Simulator) -> None` |
| `SimulatorRuntime.dataset_kinds` | `dataset_kinds: tuple[str, ...]` |
| `SimulatorRuntime.can_render` | `can_render: bool` |
| `SimulatorRuntime.can_insert_objects` | `can_insert_objects: bool` |
| `SimulatorRuntime.can_replay` | `can_replay: bool` |
| `SimulatorRuntime.can_observe_past` | `can_observe_past: bool` |
| `SimulatorRuntime.supported_world_settings` | `supported_world_settings: frozenset[str]` |
| `SimulatorRuntime.capture_ticks` | `capture_ticks: int` |
| `SimulatorRuntime.decision_time_step` | `decision_time_step: float \| None` |
| `SimulatorRuntime.command_modes` | `command_modes: frozenset[str] \| None` |
| `SimulatorRuntime.supported_stages` | `supported_stages: frozenset[str]` |
| `SimulatorRuntime.termination_details` | `termination_details: Mapping[str, Any]` |
| `SimulatorRuntime.identity` | `identity: Mapping[str, Any]` |
| `SimulatorRuntime.validate_start` | `validate_start(start: ScenarioStart \| None) -> None` |
| `SimulatorRuntime.validate_command` | `validate_command(command: Control) -> None` |
| `SimulatorRuntime.copy_observation` | `copy_observation(observation: Observation) -> Observation` |
| `SimulatorRuntime.close` | `close() -> None` |
| `SimulatorRuntime.world_key` | `world_key: Hashable` |
| `SimulatorRuntime.claim_world` | `claim_world(owner: object) -> Iterator[None]` |
| `SimulatorRuntime.exclusive` | `exclusive() -> Iterator[None]` |
| `SimulatorRuntime.clock` | `clock: ClockSnapshot` |
| `SimulatorRuntime.state` | `state() -> WorldSnapshot` |
| `SimulatorRuntime.sensors` | `sensors: frozenset[str]` |
| `SimulatorRuntime.camera_sensors` | `camera_sensors: frozenset[str]` |
| `SimulatorRuntime.sensor_geometry` | `sensor_geometry(sensor: str) -> SensorGeometry` |
| `SimulatorRuntime.sensor_view` | `sensor_view(sensor: str) -> SensorView` |
| `SensorView.composite_plan` | `composite_plan(objects: Sequence[PlacedObject]) -> CompositePlan` |
| `SimulatorRuntime.control_actor` | `control_actor(actor_id: str, control: Control) -> None` |
| `SimulatorRuntime.set_actor_state` | `set_actor_state(actor_id: str, state: ActorState) -> None` |
| `SimulatorRuntime.insert_object` | `insert_object(object_id: str, asset: ObjectAsset, world_from_object: np.ndarray, *, harmonizer: Harmonizer \| None = None) -> None` |
| `SimulatorRuntime.remove_object` | `remove_object(object_id: str) -> None` |
| `SimulatorRuntime.object_visibility` | `object_visibility(object_id: str) -> Mapping[str, ObjectVisibility]` |
| `SimulatorRuntime.reset` | `reset(start: ScenarioStart \| None = None, initialization: Any = None, *, deployment: DeploymentPlan \| None = None, world: Mapping[str, Any] \| None = None) -> None` |
| `SimulatorRuntime.start_replay` | `start_replay(start: ScenarioStart, initialization: Initialization \| None = None, *, world: Mapping[str, Any] \| None = None, deployment: DeploymentPlan \| None = None) -> None` |
| `SimulatorRuntime.advance_replay` | `advance_replay(count: int) -> None` |
| `SimulatorRuntime.setup_tick` | `setup_tick() -> None` |
| `SimulatorRuntime.step` | `step(control: Control) -> None` |
| `SimulatorRuntime.observe` | `observe() -> Observation` |
| `SimulatorRuntime.observe_past` | `observe_past(count: int) -> tuple[Observation, ...]` |
| `SimulatorRuntime.generate_frame_data` | `generate_frame_data(start: ScenarioStart, frames: Sequence[FrameReference], *, world: Mapping[str, Any] \| None = None, deployment: DeploymentPlan \| None = None) -> FrameDataSequence` |

### avsectester.closed_loop.avstacks.base

| Public name | Signature or fields |
|---|---|
| `ObservationPipeline` | Class; see methods below |
| `ObservationPipeline.__init__` | `__init__(localizer: Any = None) -> None` |
| `ObservationPipeline.process` | `process(observation: Observation, dispatcher: HookDispatcher) -> Observation` |
| `AVStack` | `provides_detections: bool \| None = None`<br>`past_steps: ClassVar[int] = 0` |
| `AVStack.observation_pipeline` | `observation_pipeline: ObservationPipeline` |
| `AVStack.reset` | `reset(observation: Observation, past: Sequence[Observation] = ()) -> None` |
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
| `CameraPoseRequest` | `pose: Any`<br>`camera: str`; camera and render pose editable at render.pre |
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

### avsectester.closed_loop.avstacks.driving_adapter

| Public name | Signature or fields |
|---|---|
| `ComponentDescriptor` | `id: str`<br>`inputs: Mapping[str, str]`<br>`outputs: Mapping[str, str]`<br>`codecs: Mapping[str, str]`<br>`differentiable_inputs: tuple[str, ...] = ()` |
| `ScorerDescriptor` | `output_shape: tuple[int, ...] = ()`<br>`direction: str = 'minimize'` |
| `DrivingAdapter` | Class; see methods below |
| `DrivingAdapter.__init__` | `__init__(factory: Callable[[int], AVStack], *, components: Sequence[tuple[ComponentDescriptor, Callable[..., Any]]] = (), scorers: Sequence[tuple[ScorerDescriptor, Callable[..., Any]]] = (), past_steps: int = 0) -> None` |
| `DrivingAdapter.past_steps` | `past_steps: int` |

### avsectester.execution.services

| Public name | Signature or fields |
|---|---|
| `CandidateRequest` | `segment: AttackSegment`<br>`seed: int`<br>`decision_steps: int`<br>`environment: Mapping[str, Any]`<br>`recording: RecordingPolicy`<br>`attack: Any = None`<br>`artifacts: Any = None`<br>`output_path: str \| Path \| None = None` |
| `SensorRequest` | `scenario_id: str`<br>`frame: FrameReference`<br>`sensor_ids: tuple[str, ...]`<br>`environment: Mapping[str, Any] = field(default_factory=dict)`<br>`views: bool = False` |
| `RenderRequest` | `scenario_id: str`<br>`frame: FrameReference`<br>`sensor_id: str`<br>`actions: Sequence[AttackAction] = ()`<br>`environment: Mapping[str, Any] = field(default_factory=dict)` |
| `SensorSample` | `frame: FrameReference`<br>`observation: Observation`<br>`provenance: Mapping[str, Any]`<br>`views: Mapping[str, SensorView] = field(default_factory=dict)` |
| `RenderResult` | `frame: FrameReference`<br>`observation: Observation`<br>`view: SensorView`<br>`provenance: Mapping[str, Any]`<br>`object_visibility: Mapping[str, Mapping[str, ObjectVisibility]] = field(default_factory=dict)` |
| `ComponentResult` | `value: Any`<br>`descriptor: ComponentDescriptor`<br>`provenance: Mapping[str, Any]` |
| `ScoreResult` | `value: Any`<br>`descriptor: ScorerDescriptor`<br>`provenance: Mapping[str, Any]` |
| `ClosedLoopCapabilities` | `dataset_kinds: tuple[str, ...]`<br>`sensor_data: bool`<br>`render: bool`<br>`insert_objects: bool`<br>`rollout: bool`<br>`components: tuple[ComponentDescriptor, ...]`<br>`scorers: tuple[ScorerDescriptor, ...]` |
| `ensure_fresh` | `ensure_fresh(resource: Any) -> Any` |
| `create_simulator_runtime` | `create_simulator_runtime(simulator: Simulator, seed: int \| None, resources: ExitStack) -> SimulatorRuntime` |
| `create_avstack` | `create_avstack(adapter: DrivingAdapter, seed: int, resources: ExitStack) -> AVStack` |
| `stop_unstarted_attacks` | `stop_unstarted_attacks(runtimes: Sequence[Any], frame: FrameReference, error: BaseException) -> None` |
| `ClosedLoopServices` | Class; see methods below |
| `ClosedLoopServices.__init__` | `__init__(dataset: Dataset, simulator: Simulator, driving_adapter: DrivingAdapter \| None = None, frame_data_cache: FrameDataCache \| None = None, *, deployment: DeploymentPlan = DeploymentPlan()) -> None` |
| `ClosedLoopServices.close` | `close() -> None` |
| `ClosedLoopServices.capabilities` | `capabilities: ClosedLoopCapabilities` |
| `ClosedLoopServices.session` | `session(seed: int, history: str = 'fresh') -> ClosedLoopSession` |
| `ClosedLoopServices.scenario` | `scenario(scenario_id: str, *, seed: int) -> ScenarioView` |
| `ClosedLoopSession` | Class; see methods below |
| `ClosedLoopSession.__init__` | `__init__(services: ClosedLoopServices, seed: int) -> None` |
| `ClosedLoopSession.close` | `close() -> None` |
| `ClosedLoopSession.can_start_at` | `can_start_at(position: int) -> bool` |
| `ClosedLoopSession.frame_data` | `frame_data(scenario: str, frame: FrameReference, environment: Mapping[str, Any] \| None = None) -> FrameData` |
| `ClosedLoopSession.initialization` | `initialization(start: ScenarioStart, environment: Mapping[str, Any], deployment: DeploymentPlan \| None = None) -> Initialization \| None` |
| `ClosedLoopSession.read_sensors` | `read_sensors(request: SensorRequest) -> SensorSample` |
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
| `ScenarioView.scenario_map` | `scenario_map() -> ScenarioMap \| Unavailable` |

### avsectester.execution.runner

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

### avsectester.closed_loop.sensor_corruptions

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

### avsectester.closed_loop.simulators.carla.simulator

| Public name | Signature or fields |
|---|---|
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
| `CarlaSimulatorRuntime.world_key` | `world_key: tuple[str, str, int]` |
| `CarlaSimulatorRuntime.native` | `native: dict[str, Any]` |
| `CarlaSimulatorRuntime.control_actor` | `control_actor(actor_id, control)` |
| `CarlaSimulatorRuntime.set_actor_state` | `set_actor_state(actor_id, state)` |
| `CarlaSimulatorRuntime.copy_observation` | `copy_observation(observation: Observation) -> Observation` |
| `CarlaSimulatorRuntime.sensors` | `sensors: frozenset[str]` |
| `CarlaSimulatorRuntime.camera_sensors` | `camera_sensors: frozenset[str]` |
| `CarlaSimulatorRuntime.sensor_geometry` | `sensor_geometry(sensor: str) -> SensorGeometry` |
| `CarlaSimulatorRuntime.sensor_view` | `sensor_view(sensor: str) -> SensorView` |
| `CarlaSimulatorRuntime.close` | `close() -> None` |

### avsectester.closed_loop.simulators.carla.scenario

| Public name | Signature or fields |
|---|---|
| `resolve_scenario_seed` | `resolve_scenario_seed(scenario: dict) -> dict` |
| `prepare_scenario` | `prepare_scenario(scenario: dict) -> dict` |

### avsectester.closed_loop.simulators.carla.sensors

| Public name | Signature or fields |
|---|---|
| `SensorCapture` | `native_type: str`<br>`raw_bytes: bytes`<br>`frame: int`<br>`timestamp: float`<br>`metadata: dict[str, Any] = field(default_factory=dict)` |
| `SensorCapture.raw_data` | `raw_data: memoryview` |
| `SensorAlignmentError` | Class; see methods below |
| `SensorAlignmentError.__init__` | `__init__(report: Mapping[str, Any]) -> None` |
| `register_pinhole_camera` | `register_pinhole_camera()` |
| `decode_carla_depth` | `decode_carla_depth(image)` |

### avsectester.closed_loop.simulators.carla.conversions

| Public name | Signature or fields |
|---|---|
| `rotation_to_carla` | `rotation_to_carla(rotation: ArrayLike) -> tuple[float, float, float]` |

### avsectester.closed_loop.simulators.carla.recipe_replay

| Public name | Signature or fields |
|---|---|
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

### avsectester.closed_loop.simulators.nurec.simulator

| Public name | Signature or fields |
|---|---|
| `NuRecSimulator` | Class; see methods below |
| `NuRecSimulator.__init__` | `__init__(config: dict, renderer: Renderer, dynamics: EgoDynamics \| None = None, *, depth: str = 'mesh', route_generator: str = 'map', physics_update_mode: str = 'ego_only', ground_mesh: str = 'mesh') -> None` |
| `NuRecSimulator.create_runtime` | `create_runtime(seed: int \| None = None) -> NuRecSimulatorRuntime` |
| `NuRecSimulator.supported_stages` | `supported_stages: frozenset[str]` |
| `NuRecSimulator.command_modes` | `command_modes: frozenset[str] \| None` |
| `NuRecSimulator.decision_time_step` | `decision_time_step: Any` |
| `NuRecSimulator.identity` | `identity: Any` |
| `NuRecSimulatorRuntime` | Class; see methods below |
| `NuRecSimulatorRuntime.__init__` | `__init__(simulator: NuRecSimulator) -> None` |
| `NuRecSimulatorRuntime.native` | `native: dict[str, Any]` |
| `NuRecSimulatorRuntime.render_request` | `render_request(camera: str) -> CameraPoseRequest` |
| `NuRecSimulatorRuntime.sensors` | `sensors: frozenset[str]` |
| `NuRecSimulatorRuntime.camera_sensors` | `camera_sensors: frozenset[str]` |
| `NuRecSimulatorRuntime.sensor_geometry` | `sensor_geometry(sensor: str) -> SensorGeometry` |
| `NuRecSimulatorRuntime.sensor_view` | `sensor_view(sensor: str) -> SensorView` |
| `NuRecSimulatorRuntime.set_actor_state` | `set_actor_state(actor_id: str, state: ActorState) -> None` |
| `NuRecSimulatorRuntime.close` | `close() -> None` |
| `NuRecSimulatorRuntime.checkpoint` | `checkpoint() -> dict` |
| `NuRecSimulatorRuntime.restore` | `restore(ckpt: dict) -> None` |

NuRecRenderer is explicitly configured on NuRecSimulator; it is not constructed automatically.
The package root exports only NuRecSimulator, NuRecSimulatorRuntime and NuRecRenderer.
Renderer and the RPC conversion functions are imported from nurec.renderer.

dynamic_objects is a sequence of (track_id, world_from_actor) pairs. NuRecRenderer serializes each
pose into both endpoints of DynamicObject.pose_pair, retaining the dataset's world pose and
quaternion convention without a box-height shift. Runtime rendering supplies every currently
present tracked actor, including unchanged tracks. NRE must enable actor editing; see
[NuRec actor movement](../guides/simulators.md#move-recorded-actors).

NuRecSimulatorRuntime.set_actor_state accepts an existing non-ego track and an ActorState with
explicit world velocity. Its pose is anchored at call time and advances linearly from the next
capture; repeated calls replace the override. Unknown IDs raise KeyError, ego and missing velocity
raise ValueError. control_actor remains unsupported. state() reports current actors, while sensor
views retain the actor poses used for their capture. checkpoint/restore includes these overrides.

### avsectester.closed_loop.simulators.ego_dynamics

| Public name | Signature or fields |
|---|---|
| `PlanarEgoState` | `x: float = 0.0`<br>`y: float = 0.0`<br>`yaw: float = 0.0`<br>`speed: float = 0.0`<br>`t: float = 0.0`<br>`lateral_speed: float = 0.0`<br>`yaw_rate: float = 0.0`<br>`acceleration: float = 0.0` |
| `EgoDynamics` | Class; see methods below |
| `EgoDynamics.reset` | `reset(state: PlanarEgoState) -> None` |
| `EgoDynamics.step` | `step(pose: PlanarEgoState, control: Control, dt: float) -> PlanarEgoState` |
| `KinematicBicycle` | `wheelbase: float = 2.8`<br>`max_accel: float = 3.0`<br>`max_brake: float = 8.0`<br>`max_steer: float = 0.6`<br>`command_modes: ClassVar[frozenset[str]] = frozenset({'actuator'})` |
| `KinematicBicycle.reset` | `reset(state: PlanarEgoState) -> None` |
| `KinematicBicycle.step` | `step(pose: PlanarEgoState, control: Control, dt: float) -> PlanarEgoState` |
| `TrajectoryFollower` | `decel: float = 4.0`<br>`command_modes: ClassVar[frozenset[str]] = frozenset({'trajectory'})` |
| `TrajectoryFollower.reset` | `reset(state: PlanarEgoState) -> None` |
| `TrajectoryFollower.step` | `step(pose: PlanarEgoState, control: Control, dt: float) -> PlanarEgoState` |
| `AlpaSimController` | `mpc_implementation: str = 'linear'`<br>`n_horizon: int = 20`<br>`dt_mpc: float = 0.1`<br>`gains: Mapping[str, float \| int] \| None = None`<br>`command_modes: ClassVar[frozenset[str]] = frozenset({'trajectory'})`<br>`EPOCH_US: ClassVar[int] = 10000000` |
| `AlpaSimController.reset` | `reset(state: PlanarEgoState) -> None` |
| `AlpaSimController.step` | `step(pose: PlanarEgoState, control: Control, dt: float) -> PlanarEgoState` |

PlanarEgoState uses world X/Y and yaw for the rig origin, ego-axis longitudinal/lateral speeds,
seconds for t, radians per second for yaw_rate and m/s² for longitudinal acceleration. NuRec
initializes these from the recording and calls the dynamics reset method when present.
KinematicBicycle and TrajectoryFollower need no warmup; their step results include difference-based
yaw_rate and acceleration. TrajectoryFollower is an explicitly selected ideal follower.

AlpaSimController tracks t0-led trajectory commands through AlpaSim's in-process MPC and dynamic
bicycle model. Its defaults match AlpaSim affc2ea ControllerConfig. gains overrides named MPCGains
fields, retaining unspecified defaults. VehicleModel.Parameters uses the official Ford Fusion
values rather than USDZ measurements. The first reset lazily imports AlpaSim and creates a system;
subsequent resets create fresh systems. An internal 10-second offset keeps protobuf timestamps
nonnegative while returned state times remain on the simulator clock. State time must match the
controller's last propagated time, and plans must start at t0 with a zero position and unit rotation.
See [controller installation](../getting-started/installation.md#driver-environment) for the editable
workspace package and required linear entry point.

### avsectester.closed_loop.simulators.nurec.renderer

| Public name | Signature or fields |
|---|---|
| `ftheta_camera_from_spec` | `ftheta_camera_from_spec(spec: Any) -> FThetaCamera` |
| `pose_from_proto` | `pose_from_proto(pose: Any) -> np.ndarray` |
| `planar_rig_pose` | `planar_rig_pose(x: float, y: float, yaw: float, z: float = 0.0) -> np.ndarray` |
| `Renderer` | `cameras: ClassVar[list[str]] = ['camera_front']`<br>`calibration: ClassVar[dict] = {}` |
| `Renderer.copy` | `copy() -> Renderer` |
| `Renderer.reset_origin` | `reset_origin(start = None)` |
| `Renderer.origin_transform` | `origin_transform: Any` |
| `Renderer.origin_timestamp_us` | `origin_timestamp_us: Any` |
| `Renderer.origin_bounds` | `origin_bounds: Any` |
| `Renderer.load_scene` | `load_scene(scene: Any) -> None` |
| `Renderer.render` | `render(world_from_rig: np.ndarray, timestamp_us: int, camera: str, dynamic_objects=()) -> Any` |
| `Renderer.camera_model` | `camera_model(camera: str \| None = None) -> Camera \| None` |
| `Renderer.cam_from_world` | `cam_from_world(world_from_rig: np.ndarray, camera: str \| None = None) -> np.ndarray` |
| `Renderer.close` | `close() -> None` |
| `NuRecRenderer` | `endpoint: str = '127.0.0.1:50051'`<br>`scene_id: str \| None = None`<br>`cameras: list[str] = field(default_factory=lambda: ['camera_front_wide_120fov'])`<br>`calibration: dict = field(default_factory=dict)`<br>`image_quality: float = 95.0`<br>`resolution_hw: Mapping[str, tuple[int, int]] = field(default_factory=dict)` |
| `NuRecRenderer.copy` | `copy() -> NuRecRenderer` |
| `NuRecRenderer.load_scene` | `load_scene(scene: Any) -> None` |
| `NuRecRenderer.service` | `service: Any` |
| `NuRecRenderer.close` | `close()` |
| `NuRecRenderer.camera_model` | `camera_model(camera: str \| None = None) -> Camera` |
| `NuRecRenderer.cam_from_world` | `cam_from_world(world_from_rig: np.ndarray, camera: str \| None = None) -> np.ndarray` |
| `NuRecRenderer.render` | `render(world_from_rig: np.ndarray, timestamp_us: int, camera: str, dynamic_objects=())` |
| `NuRecRenderer.render_at` | `render_at(world_from_rig, timestamp_us: int, camera: str, dynamic_objects=())` |

### avsectester.closed_loop.simulators.recorded

| Public name | Signature or fields |
|---|---|
| `RecordedReplaySimulator` | Class; see methods below |
| `RecordedReplaySimulator.__init__` | `__init__(start: ScenarioStart \| None = None, *, camera_aliases: Mapping[str, str] = {}) -> None` |
| `RecordedReplaySimulator.create_runtime` | `create_runtime(seed: int \| None = None) -> RecordedReplaySimulatorRuntime` |
| `RecordedReplaySimulatorRuntime` | Class; see methods below |
| `RecordedReplaySimulatorRuntime.__init__` | `__init__(simulator: RecordedReplaySimulator) -> None` |
| `RecordedReplaySimulatorRuntime.sensors` | `sensors: frozenset[str]` |
| `RecordedReplaySimulatorRuntime.camera_sensors` | `camera_sensors: frozenset[str]` |
| `RecordedReplaySimulatorRuntime.sensor_geometry` | `sensor_geometry(sensor: str) -> SensorGeometry` |
| `RecordedReplaySimulatorRuntime.sensor_view` | `sensor_view(sensor: str) -> SensorView` |
| `RecordedReplaySimulatorRuntime.close` | `close()` |

### avsectester.closed_loop.avstacks.modular

| Public name | Signature or fields |
|---|---|
| `register_carla_dataset` | `register_carla_dataset() -> type` |
| `NativeHook` | Class; see methods below |
| `NativeHook.__init__` | `__init__(config: Mapping[str, Any], *, phase: str = 'post') -> None` |
| `NativeHook.reset` | `reset(context: RunContext) -> None` |
| `NativeHook.transform` | `transform(value: Any, context: StageContext) -> Any` |
| `NativeHook.close` | `close() -> None` |
| `modular_driving_adapter` | `modular_driving_adapter(factory, *, components = (), scorers = ())` |
| `ModularAVStack` | Class; see methods below |
| `ModularAVStack.__init__` | `__init__(pipeline_cfg: dict) -> None` |
| `ModularAVStack.reset` | `reset(observation: Observation, past: Sequence[Observation] = ()) -> None` |
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
| `alpamayo_driving_adapter` | `alpamayo_driving_adapter(factory)` |
| `AlpamayoAVStack` | Class; see methods below |
| `AlpamayoAVStack.past_steps` | `past_steps: ClassVar[int] = 15` |
| `AlpamayoAVStack.__init__` | `__init__(checkpoint_path: str \| Path \| None = None, camera_ids: list[str] \| None = None, device: str = 'cuda:0', context_length: int = 4, frame_subsample: int = 1, output_frequency_hz: int = 10, model: Any \| None = None) -> None` |
| `AlpamayoAVStack.command_modes` | `command_modes: frozenset[str]` |
| `AlpamayoAVStack.supported_stages` | `supported_stages: frozenset[str]` |
| `AlpamayoAVStack.native` | `native: Mapping[str, Any]` |
| `AlpamayoAVStack.models` | `models: Mapping[str, Any]` |
| `AlpamayoAVStack.reset` | `reset(observation: Observation, past: Sequence[Observation] = ()) -> None` |
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
| `CameraForwardCollisionAVStack.reset` | `reset(observation: Observation, past: Sequence[Observation] = ()) -> None` |
| `CameraForwardCollisionAVStack.__call__` | `__call__(observation: Observation) -> Control` |
| `CameraForwardCollisionAVStack.component_log` | `component_log() -> dict[str, Any] \| None` |
| `CameraForwardCollisionAVStack.close` | `close() -> None` |

### avsectester.closed_loop.avstacks.scorers

| Public name | Signature or fields |
|---|---|
| `MMDetRPNObjectnessScorer` | Class; see methods below |
| `MMDetRPNObjectnessScorer.__init__` | `__init__(model, strides = FPN_STRIDES)` |
| `MMDetRPNObjectnessScorer.target_score` | `target_score(image: torch.Tensor, target: TargetSpec) -> torch.Tensor` |
| `MMDetInferenceScorer` | Class; see methods below |
| `MMDetInferenceScorer.__init__` | `__init__(model)` |
| `MMDetInferenceScorer.target_score` | `target_score(image, target: TargetSpec) -> float` |


### avsectester.execution.frame_data_cache

| Public name | Signature or fields |
|---|---|
| `fingerprint` | `fingerprint(value: Any) -> str` |
| `FrameDataCache` | Class; see methods below |
| `FrameDataCache.__init__` | `__init__() -> None` |
| `FrameDataCache.key` | `key(start: ScenarioStart, simulator_identity: Mapping[str, Any], seed: int) -> str` |
| `FrameDataCache.get_or_generate` | `get_or_generate(key: str, generate: Callable[[], FrameDataSequence]) -> FrameDataSequence` |
| `FrameDataCache.get` | `get(key: str) -> FrameDataSequence` |

### avsectester.implemented_attacks.phantom_injection_attack.attack

| Public name | Signature or fields |
|---|---|
| `PhantomInjectionAttack` | Class; see methods below |
| `PhantomInjectionAttack.__init__` | `__init__(*, decision_steps: int, max_segments: int \| None = None, target_xyz = (6.0, 0.0, -0.7), object_type = 'Car', score = 0.9, extent = (4.0, 1.8, 1.6), object_id = 90002, keep_tail: bool = False)` |
| `PhantomInjectionAttack.build_metadata` | `build_metadata(scenario)` |
| `PhantomInjectionAttack.create_runtime` | `create_runtime(segment, artifacts)` |

### avsectester.implemented_attacks.speed_feedback_attack.attack

| Public name | Signature or fields |
|---|---|
| `SpeedFeedbackAttack` | Class; see methods below |
| `SpeedFeedbackAttack.__init__` | `__init__(*, decision_steps: int, target_speed: float, gain: float = 0.1, max_bias: float = 5.0, initial_bias: float = 0.0, max_segments: int \| None = None, keep_tail: bool = False)` |
| `SpeedFeedbackAttack.build_metadata` | `build_metadata(scenario)` |
| `SpeedFeedbackAttack.create_runtime` | `create_runtime(segment, artifacts)` |

### avsectester.implemented_attacks.patch_attack.attack

| Public name | Signature or fields |
|---|---|
| `PatchAttack` | Class; see methods below |
| `PatchAttack.__init__` | `__init__(*, decision_steps, assets: np.ndarray \| str \| Path \| Texture \| None = None, candidate_hosts = None, hosts_per_window = 1, sensor = 'front', width_m = 1.0, height_m = 0.5, anchor = 'rear_center', offset_m = (-0.01, 0.0, 0.0), orientation = None, max_distance = None, min_visible_fraction = 0.0, max_windows = None, keep_tail = False, optimization = None, prop = DEFAULT_PROP, emissive = True, compositor = None)` |
| `PatchAttack.placement_for` | `placement_for(host)` |
| `PatchAttack.selected_placement` | `selected_placement(segment)` |
| `PatchAttack.insertion` | `insertion(artifacts, *, host = None, placement = None)` |
| `PatchAttack.build_metadata` | `build_metadata(scenario)` |
| `PatchAttack.run_offline` | `run_offline(context)` |
| `PatchAttack.deployment_ticks` | `deployment_ticks(segment)` |
| `PatchAttack.create_runtime` | `create_runtime(segment, artifacts)` |

### avsectester.implemented_attacks.object_insertion_attack.attack

| Public name | Signature or fields |
|---|---|
| `ObjectInsertionAttack` | Class; see methods below |
| `ObjectInsertionAttack.__init__` | `__init__(*, decision_steps, placement, assets = None, orientation = None, sensor = 'front', min_visible_fraction = 0.0, max_segments = None, keep_tail = False, placement_relative_to_ego_origin = False, compositor = None)` |
| `ObjectInsertionAttack.insertion` | `insertion(artifacts, placement = None)` |
| `ObjectInsertionAttack.build_metadata` | `build_metadata(scenario)` |
| `ObjectInsertionAttack.run_offline` | `run_offline(context)` |
| `ObjectInsertionAttack.selected_placement` | `selected_placement(segment)` |
| `ObjectInsertionAttack.create_runtime` | `create_runtime(segment, artifacts)` |

### avsectester.implemented_attacks.object_insertion_attack.assets

SignAsset defaults to the bundled textures/mutcd_r1-1_stop.png in this concrete attack package.
Its bottom-center origin, +X front, +Y left and +Z up match ObjectAsset. object_asset delegates to
ObjectAsset.from_panel; planes returns that object's surfaces, with posts before the face.
post_count is nonnegative; zero omits support posts.

| Public name | Signature or fields |
|---|---|
| `SignAsset` | `face: np.ndarray \| str \| Path = ...`<br>`width_m: float = 0.9`<br>`mount_height_m: float = 1.5`<br>`post_width_m: float = 0.07`<br>`post_count: int = 1` |
| `SignAsset.planes` | `planes() -> Sequence[PlaneSurface]` |
| `SignAsset.object_asset` | `object_asset(category: str) -> ObjectAsset` |
| `poster_rgba` | `poster_rgba(person: np.ndarray, aspect: float = 1.5, fill: float = 0.86, paper: tuple = (228, 226, 218), border: tuple = (40, 40, 42), border_frac: float = 0.025) -> np.ndarray` |
| `standee` | `standee(person: np.ndarray, height_m: float = 1.75) -> SignAsset` |
| `billboard` | `billboard(person: np.ndarray, width_m: float = 1.4, mount_height_m: float = 0.6) -> SignAsset` |
| `signal_head_rgba` | `signal_head_rgba(lit: str = 'red', texture_height_pixels: int = 300) -> np.ndarray` |
| `traffic_lights_rgba` | `traffic_lights_rgba(head_count: int = 3, lit: str = 'red', texture_height_pixels: int = 300, gap_fraction: float = 0.25, board: tuple \| None = (35, 37, 40)) -> np.ndarray` |
| `roadside_rig` | `roadside_rig(face: np.ndarray \| str \| Path, width_m: float = 2.4, mount_height_m: float = 2.2) -> SignAsset` |
| `PersonAsset` | `image: np.ndarray \| str \| Path`<br>`mode: str = 'billboard'`<br>`height_m: float = 1.75`<br>`width_m: float = 1.4`<br>`mount_height_m: float = 0.6` |
| `PersonAsset.planes` | `planes()` |
| `TrafficLightAsset` | `head_count: int = 3`<br>`lit: str = 'red'`<br>`texture_height_pixels: int = 300`<br>`gap_fraction: float = 0.25`<br>`board: tuple \| None = (35, 37, 40)`<br>`width_m: float = 2.4`<br>`mount_height_m: float = 2.2` |
| `TrafficLightAsset.planes` | `planes()` |

### avsectester.attack.insertion.assets

| Public name | Signature or fields |
|---|---|
| `PlanarAsset` | Class; see methods below |
| `PlanarAsset.planes` | `planes() -> Sequence[PlaneSurface]` |
| `PlaneAsset` | `texture: np.ndarray \| str \| Path`<br>`width_m: float`<br>`height_m: float` |
| `PlaneAsset.planes` | `planes() -> tuple[PlaneSurface, ...]` |
| `Insertion` | `id: str`<br>`asset: PlanarAsset`<br>`placement: AttachedPlacement \| WorldPlacement`<br>`orientation: Orientation = field(default_factory=Orientation)` |
| `ResolvedInsertion` | `id: str`<br>`pose: np.ndarray`<br>`asset: PlanarAsset`<br>`host_id: str \| None = None` |
| `ResolvedInsertion.planes` | `planes() -> tuple[PlaneSurface, ...]` |
| `resolve_insertion` | `resolve_insertion(insertion: Insertion, actors: Mapping[str, ActorState], ego: ActorState) -> ResolvedInsertion` |

### avsectester.attack.insertion.types

| Public name | Signature or fields |
|---|---|
| `Visibility` | `fraction: float`<br>`source: str`<br>`camera: str \| None = None`<br>`label: str \| None = None` |
| `VisibilityEvidence` | `visibility: Visibility \| None`<br>`reference_mask: np.ndarray`<br>`visible_mask: np.ndarray`<br>`target_depth: np.ndarray`<br>`reference_pixels: int`<br>`reason: str = ''`<br>`sampled_rgba: np.ndarray \| None = None`<br>`ignored_occluders: tuple[str, ...] = ()` |

Visibility.source uses depth, actor_cuboids or rendered_masks. VisibilityEvidence.ignored_occluders
lists actor IDs whose missing extent prevents cuboid occlusion; they are skipped, not treated as
blocking. CuboidVisibilityEstimator also treats world_objects' alpha-aware surfaces as occluders.

### avsectester.rendering.textures

Texture implements rgba() and can be passed to load_rgba_image alongside an array or image path.
The loader returns an independent uint8 RGBA image.

| Public name | Signature or fields |
|---|---|
| `Texture` | Protocol; see method below |
| `Texture.rgba` | `rgba() -> np.ndarray` |
| `CheckerboardTexture` | `size: int = 256`<br>`cells: int = 8`<br>`colors: tuple = ((255, 0, 0), (255, 255, 255))` |
| `CheckerboardTexture.rgba` | `rgba() -> np.ndarray` |
| `load_rgba_image` | `load_rgba_image(value: np.ndarray \| str \| Path \| Texture) -> np.ndarray` |


### avsectester.rendering.objects

ObjectAsset and PlacedObject use the canonical right-handed world convention. Object-local origin
is the bottom center, with +X front, +Y left and +Z up. Extent is full length, width and height.
PlaneSurface corners and rigid object transforms are copied and read-only. from_panel reads a
texture array, image path or Texture and creates a panel facing +X. Its face height follows the
texture aspect ratio; mount_height_m is the lower face edge height. Optional posts sit 3 cm behind
the face and precede it in surfaces. from_surfaces derives the final extent.

| Public name | Signature or fields |
|---|---|
| `PlaneSurface` | `corners: np.ndarray`<br>`texture: np.ndarray` |
| `ObjectAsset` | `surfaces: tuple[PlaneSurface, ...]`<br>`extent: tuple[float, float, float]`<br>`category: str` |
| `ObjectAsset.from_surfaces` | `from_surfaces(surfaces: Sequence[PlaneSurface], category: str) -> ObjectAsset` |
| `ObjectAsset.from_panel` | `from_panel(texture: np.ndarray \| str \| Path \| Texture, *, category: str, width_m: float, mount_height_m: float = 0.0, post_count: int = 0, post_width_m: float = 0.07) -> ObjectAsset` |
| `PlacedObject` | `object_id: str`<br>`asset: ObjectAsset`<br>`world_from_object: np.ndarray`<br>`harmonizer: Harmonizer \| None = None` |
| `PlacedObject.world_surfaces` | `world_surfaces() -> tuple[PlaneSurface, ...]` |

### avsectester.rendering.surfaces

Every pixel averages SUBSAMPLES x SUBSAMPLES rays; each ray reads a trilinear mipmap of the
premultiplied texture at its own footprint level. Coverage is geometric: rays outside the
rectangle contribute nothing, so filtered edges are antialiased without spreading. Rendering,
visibility and PatchPerturbation call the same sample_texture, so optimization filters the patch
exactly as deployment renders it.

| Public name | Signature or fields |
|---|---|
| `SUBSAMPLES` | `2`; rays per pixel along each image axis |
| `edge_points` | `edge_points(corners: np.ndarray) -> np.ndarray` |
| `SurfaceSampler` | `corners: np.ndarray`<br>`texture: np.ndarray`<br>`bounds: tuple[int, int, int, int] \| None`<br>`reason: str = ''` |
| `SurfaceSampler.intersect` | `intersect(rays: np.ndarray) -> tuple[np.ndarray, np.ndarray]` |
| `SurfaceSampler.sample` | `sample(rays: np.ndarray) -> tuple[np.ndarray, np.ndarray]` |
| `SurfaceSampler.image_bounds` | `image_bounds(camera: Camera) -> tuple[int, int, int, int] \| None` |
| `prepare_surface` | `prepare_surface(surface: PlaneSurface, camera: Camera, cam_from_world) -> SurfaceSampler` |
| `PlaneSampling` | Frozen record: `bounds: tuple[int, int, int, int]`<br>`coordinates: np.ndarray`<br>`derivatives: np.ndarray`<br>`depth: np.ndarray` |
| `plane_sampling` | `plane_sampling(sampler: SurfaceSampler, camera: Camera, bounds: tuple[int, int, int, int]) -> PlaneSampling` |
| `premultiplied_texture` | `premultiplied_texture(rgba: np.ndarray) -> torch.Tensor` |
| `sample_texture` | `sample_texture(texture: torch.Tensor, sampling: PlaneSampling) -> torch.Tensor` |
| `Projection` | `reference: np.ndarray`<br>`depth: np.ndarray`<br>`rays: np.ndarray`<br>`rgba: np.ndarray`<br>`x0: int`<br>`y0: int` |
| `rasterize` | `rasterize(surfaces: Sequence[PlaneSurface], camera: Camera, cam_from_world, max_reference_pixels)` |
| `render_surfaces` | `render_surfaces(surfaces: Sequence[PlaneSurface], camera: Camera, cam_from_world: np.ndarray) -> tuple[np.ndarray, np.ndarray]` |

SurfaceSampler.sample is a point query for occlusion: a ray is blocked where the full-resolution
bilinear alpha exceeds 127. PlaneSampling stores texture-independent footprint geometry, so one
sampling serves any texture size. sample_texture takes and returns premultiplied (4, H, W)
tensors and requires torch. Covered pixels returned by render_surfaces have finite depth; their
alpha is the filtered coverage.

### avsectester.attack.insertion.compositor

| Public name | Signature or fields |
|---|---|
| `PatchCompositor` | Class; see methods below |
| `PatchCompositor.__init__` | `__init__(harmonizer: Harmonizer \| None = None, *, soften: float = 0.0) -> None` |
| `render_resolved` | `render_resolved(frame: np.ndarray, camera: Camera, cam_from_world: np.ndarray, resolved: ResolvedInsertion, evidence: VisibilityEvidence \| None = None, *, compositor: PatchCompositor \| None = None) -> tuple[np.ndarray, np.ndarray]` |

### avsectester.rendering.compositing

The rendering package root exports ObjectAsset, PlacedObject, PlaneSurface, Harmonizer,
ClassicHarmonizer, PCTNetHarmonizer, render_surfaces, ObjectVisibility, CompositePlan, CompositeResult,
blend_rgba, plan_composite, composite_objects, Texture, CheckerboardTexture and load_rgba_image.
Plane surfaces are in world coordinates when passed to render_surfaces. Its RGBA and depth outputs
have calibrated image dimensions; depth is optical-axis z, with inf where no surface is hit.

| Public name | Signature or fields |
|---|---|
| `ObjectVisibility` | Frozen record: `occlusion_source: str`<br>`ignored_occluders: tuple[str, ...]`<br>`visible_pixels: int` |
| `CompositeResult` | `image: np.ndarray`<br>`depth: np.ndarray`<br>`visible_mask: np.ndarray`<br>`occlusion_source: str`<br>`ignored_occluders: tuple[str, ...] = ()`<br>`object_visibility: Mapping[str, ObjectVisibility] = field(default_factory=dict)` |
| `blend_rgba` | `blend_rgba(image: np.ndarray, rgba: np.ndarray, visible_mask: np.ndarray, *, harmonizer: Harmonizer \| None = None, soften: float = 0.0) -> np.ndarray` |
| `composite_objects` | `composite_objects(image: np.ndarray, camera: Camera, cam_from_world: np.ndarray, objects: Sequence[PlacedObject], *, scene_depth: np.ndarray \| None = None, occluders: Mapping[str, ActorState] \| None = None, occlusion_tolerance_m: float = 0.1) -> CompositeResult` |
| `plan_composite` | `plan_composite(camera: Camera, cam_from_world: np.ndarray, objects: Sequence[PlacedObject], *, scene_depth: np.ndarray \| None = None, occluders: Mapping[str, ActorState] \| None = None, occlusion_tolerance_m: float = 0.1) -> CompositePlan` |
| `CompositePlan` | Frozen record: `objects: tuple[PlacedObject, ...]`<br>`depth: np.ndarray`<br>`visible_mask: np.ndarray`<br>`occlusion_source: str`<br>`ignored_occluders: tuple[str, ...]`<br>`object_visibility: Mapping[str, ObjectVisibility]`<br>`footprints: tuple[...]` (internal) |
| `CompositePlan.apply` | `apply(image: np.ndarray) -> CompositeResult` |
| `CompositePlan.composite` | `composite(image: torch.Tensor, textures: Mapping[tuple[str, int], torch.Tensor] \| None = None) -> torch.Tensor` |

composite_objects accepts uint8 RGB, resolves the nearest object surfaces, and blends only visible
pixels. Each PlacedObject carries its own optional harmonizer. Supplied scene_depth is the sole
scene occlusion source (occlusion_source="depth"), and the result depth merges it with object depth.
Without scene_depth, actor cuboids supply estimates only in the candidate projection regions
expanded by two pixels (occlusion_source="actor_cuboids"). The fallback result is inf outside those
regions and is not complete sensor-depth evidence. ignored_occluders lists actors with extent=None
in input mapping order; they cannot supply cuboid occlusion. It is empty in scene-depth mode,
which does not consult the occluders mapping. object_visibility provides one ObjectVisibility per
input object, including its visible-pixel count. PlacedObject.world_surfaces() returns the
asset's surfaces in world coordinates.

composite_objects is plan_composite followed by CompositePlan.apply. The plan fixes everything that
does not depend on surface colours: subpixel sampling, coverage from each asset's alpha,
nearest-surface ownership, occlusion and object visibility. apply blends in floating point, rounds
once to uint8 and then runs each object's harmonizer. composite performs the same blend on a
(3, H, W) tensor in [0, 1] and accepts replacement straight-RGB colours, keyed by (object_id,
surface index in the asset), at the surface texture's size. Rounding its output reproduces apply
without harmonizers, and gradients reach the image and the replacement colours. A plan containing a
harmonized object rejects composite, because harmonizers are not differentiable.
SensorView.composite_plan(objects) plans with the view's camera, depth or actor cuboids and the
capture tolerance, exactly as simulator captures compose world objects.

### avsectester.rendering.harmonizers

| Public name | Signature or fields |
|---|---|
| `Harmonizer` | Class; see methods below |
| `Harmonizer.__call__` | `__call__(composite_rgb: np.ndarray, mask: np.ndarray, background_rgb: np.ndarray) -> np.ndarray` |
| `ClassicHarmonizer` | Class; see methods below |
| `ClassicHarmonizer.__init__` | `__init__(color_transfer: bool = True, poisson: bool = True, preserve_chroma: bool = False, blend: str = 'poisson', feather: float = 1.2, gain_range: tuple = (0.35, 1.2)) -> None` |
| `ClassicHarmonizer.__call__` | `__call__(composite_rgb, mask, background_rgb)` |
| `PCTNetHarmonizer` | Class; see methods below |
| `PCTNetHarmonizer.__init__` | `__init__(libcom_directory: str \| Path, device: int = 0, weights: str \| Path \| None = None, strict: bool = False) -> None` |
| `PCTNetHarmonizer.__call__` | `__call__(composite_rgb, mask, background_rgb)` |

### avsectester.attack.insertion.visibility

| Public name | Signature or fields |
|---|---|
| `CuboidVisibilityEstimator` | Class; see methods below |
| `CuboidVisibilityEstimator.estimate` | `estimate(subject: ResolvedInsertion, camera: Camera, cam_from_world: np.ndarray, *, occluders: Mapping[str, ActorState], camera_name: str, other_insertions: Sequence[ResolvedInsertion] = (), world_objects: Sequence[PlacedObject] = ()) -> VisibilityEvidence` |
| `DepthVisibilityEstimator` | Class; see methods below |
| `DepthVisibilityEstimator.__init__` | `__init__(*, tolerance_m: float = 0.0002, max_reference_pixels: int = 4000000)` |
| `DepthVisibilityEstimator.estimate` | `estimate(subject: ResolvedInsertion, camera: Camera, cam_from_world: np.ndarray, *, scene_depth: np.ndarray, depth_convention: str, camera_name: str, other_insertions: Sequence[ResolvedInsertion] = ()) -> VisibilityEvidence` |
| `visibility_from_masks` | `visibility_from_masks(reference, visible, *, camera: str, image_bounds = None) -> Visibility \| None` |
| `visibility_from_depth` | `visibility_from_depth(footprint, target_depth, scene_depth, *, camera: str, tolerance_m: float = 0.0002, image_bounds = None) -> Visibility \| None` |

### avsectester.attack.insertion.action

| Public name | Signature or fields |
|---|---|
| `InsertObject` | `object_id: str`<br>`asset: ObjectAsset`<br>`world_from_object: np.ndarray`<br>`harmonizer: Harmonizer \| None = None` |
| `InsertObject.apply` | `apply(simulator_runtime: SimulatorRuntime, resources: ExitStack) -> None` |
| `InsertObject.receipt_details` | `receipt_details() -> Mapping[str, Any]` |
| `InsertionAttackAction` | `sensor: str`<br>`insertion: Insertion`<br>`compositor: PatchCompositor \| None = None`<br>`rendered: bool = field(default=False, init=False)`<br>`reason: str = field(default='', init=False)`<br>`occlusion_source: str = field(default='', init=False)`<br>`ignored_occluders: tuple[str, ...] = field(default=(), init=False)` |
| `InsertionAttackAction.receipt_details` | `receipt_details()` |
| `InsertionAttackAction.resolve` | `resolve(view)` |
| `InsertionAttackAction.evidence` | `evidence(view, resolved = None)` |
| `InsertionAttackAction.apply` | `apply(data: Any, view)` |

### avsectester.attack.optimization.interface

| Public name | Signature or fields |
|---|---|
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

### avsectester.implemented_attacks.patch_attack.offline

| Public name | Signature or fields |
|---|---|
| `PatchOptimization` | `algorithm: GradientOptimizer`<br>`scorer_id: str \| None = None`<br>`size: tuple[int, int] = (64, 64)`<br>`device: str \| None = None`<br>`target_box: tuple[float, float, float, float] \| None = None`<br>`brightness: tuple[float, float] = (0.8, 1.2)`<br>`contrast: tuple[float, float] = (0.8, 1.2)`<br>`validate_rollout: bool = False`<br>`init_texture: np.ndarray \| str \| Path \| Texture \| None = None` |
| `render_candidate` | `render_candidate(session, attack, segment, frame, texture, environment, observation = None)` |
| `optimize_patch` | `optimize_patch(attack, context, options: PatchOptimization)` |
| `to_rgba_uint8` | `to_rgba_uint8(delta: np.ndarray) -> np.ndarray` |
| `run_offline` | `run_offline(attack, context)` |

### avsectester.implemented_attacks.patch_attack.optimization_data

| Public name | Signature or fields |
|---|---|
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
| `StepRecord` | `step: int`<br>`raw_observation: Any = None`<br>`model_observation: Any = None`<br>`physical_pre: Any = None`<br>`physical_post: Any = None`<br>`produced_command: Any = None`<br>`executed_command: Any = None`<br>`components: dict[str, Any] = field(default_factory=dict)`<br>`detections: Sequence[Detection] \| None = None`<br>`sensor_geometry: Mapping[str, SensorGeometry] = field(default_factory=dict)`<br>`object_visibility: Mapping[str, Mapping[str, ObjectVisibility]] = field(default_factory=dict)`<br>`actions: list[Any] = field(default_factory=list)`<br>`outputs: list[Any] = field(default_factory=list)`<br>`telemetry: dict[str, Any] = field(default_factory=dict)` |
| `EventRecord` | `kind: str`<br>`step: int \| None = None`<br>`data: dict[str, Any] = field(default_factory=dict)` |
| `RunLog` | `metadata: dict[str, Any] = field(default_factory=dict)`<br>`steps: list[StepRecord] = field(default_factory=list)`<br>`events: list[EventRecord] = field(default_factory=list)`<br>`status: str = 'incomplete'`<br>`path: Path \| None = None` |
| `RunLog.read` | `read(path: str \| Path, *, registry: SerializationRegistry = DEFAULT_REGISTRY, load_payloads: bool = True) -> RunLog` |
| `Recorder` | Class; see methods below |
| `Recorder.__init__` | `__init__(path: str \| Path \| None = None, *, metadata: dict[str, Any] \| None = None, registry: SerializationRegistry = DEFAULT_REGISTRY, chunk_steps: int = 128, chunk_bytes: int = 8 * 1024 * 1024, payload_shard_bytes: int = 64 * 1024 * 1024, retain_steps: bool = True) -> None` |
| `Recorder.snapshot` | `snapshot(value: Any) -> Any` |
| `Recorder.record_step` | `record_step(record: StepRecord) -> StepRecord` |
| `Recorder.record_event` | `record_event(kind: str, *, step: int \| None = None, **data) -> EventRecord` |
| `Recorder.flush` | `flush() -> None` |
| `Recorder.finish` | `finish(status: str = 'completed', **details) -> RunLog` |
| `Recorder.abort` | `abort() -> None` |
| `read_table` | `read_table(path: str \| Path, columns: Sequence[str] \| None = None, *, stream: str = 'steps') -> Any` |

### avsectester.evaluation.evaluator

RunGroup is frozen and retains every log status and its mapping order. ComparisonGroup, passed to
metrics, contains completed logs only.

| Public name | Signature or fields |
|---|---|
| `RunGroup` | `attack_name: str`<br>`segment_index: int`<br>`segment: AttackSegment`<br>`seed: int`<br>`logs: Mapping[str, RunLog]`<br>`errors: Mapping[str, str] = field(default_factory=dict)`<br>`scenario_map: ScenarioMap \| Unavailable = field(default_factory=lambda: Unavailable('No saved scenario map is available'))` |
| `evaluate_groups` | `evaluate_groups(groups: Sequence[RunGroup], metrics: Sequence[Metric], *, notes: Sequence[str] = ()) -> EvaluationReport` |

### avsectester.evaluation.metric

| Public name | Signature or fields |
|---|---|
| `UnavailableMetricValue` | `reason: str`<br>`value_type: Literal['number', 'boolean'] = 'number'` |
| `ComparisonGroup` | `attack_name: str`<br>`segment: AttackSegment`<br>`seed: int`<br>`runs: Mapping[str, RunLog]`<br>`scenario_map: ScenarioMap \| Unavailable = field(default_factory=lambda: Unavailable('No saved scenario map is available'))` |
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
| `EvaluationReport.summarize` | `summarize() -> None` |
| `EvaluationReport.record_error` | `record_error(source: str, error: Exception, **context: Any) -> None` |
| `EvaluationReport.as_dict` | `as_dict() -> dict[str, Any]` |
| `EvaluationReport.format_summary` | `format_summary() -> str` |
| `EvaluationReport.save` | `save(directory: str \| Path) -> Path` |

### avsectester.evaluation.visualization

| Public name | Signature or fields |
|---|---|
| `plot_group` | `plot_group(runs: Mapping[str, Any], path: str \| Path, *, title: str \| None = None) -> Path` |
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

Pose helpers operate on canonical world transforms and accept no simulator SDK objects.
MIRROR_Y reflects the Y axis; mirror_y_transform changes both world and local bases between
CARLA/Unreal and canonical axes. ActorState.pose locates the actor origin. Use box_transform()
when a geometry function needs the box center, including a local center offset and rotation.

| Public name | Signature or fields |
|---|---|
| `validated_vector` | `validated_vector(value: ArrayLike, name: str) -> np.ndarray` |
| `validated_transform` | `validated_transform(value: ArrayLike) -> np.ndarray` |
| `rotation_matrix` | `rotation_matrix(rotation_deg: ArrayLike) -> np.ndarray` |
| `quaternion_to_rotation` | `quaternion_to_rotation(w: float, x: float, y: float, z: float) -> np.ndarray` |
| `rotation_to_quaternion` | `rotation_to_quaternion(rotation: ArrayLike) -> tuple[float, float, float, float]` |
| `interpolate_transform` | `interpolate_transform(first: ArrayLike, second: ArrayLike, fraction: float) -> np.ndarray` |
| `make_pose` | `make_pose(rotation: np.ndarray, translation: ArrayLike) -> np.ndarray` |
| `transform` | `transform(T: np.ndarray, pts: np.ndarray) -> np.ndarray` |
| `MIRROR_Y` | Read-only 4×4 Y-axis reflection matrix |
| `mirror_y_points` | `mirror_y_points(points: ArrayLike) -> np.ndarray` |
| `mirror_y_transform` | `mirror_y_transform(matrix: ArrayLike) -> np.ndarray` |

quaternion_to_rotation takes components in (w, x, y, z) order. rotation_to_quaternion returns that
same order and remains stable near a half-turn. interpolate_transform linearly interpolates the
translation and uses spherical interpolation for rotation.

### avsectester.geometry.cameras

BODY_FROM_OPTICAL is a 4x4 ndarray mapping optical axes (x right, y down, z forward) to sensor-body
axes (x forward, y left, z up). Left-multiply cam_from_world to obtain sensor_from_world.
NEAR_Z is the shared 1e-5 m optical near-depth threshold for geometric projection and surface
sampling. camera_ray_mask accepts finite forward rays and respects an optional lens max_angle.

| Public name | Signature or fields |
|---|---|
| `BODY_FROM_OPTICAL` | `np.ndarray`; 4×4 optical-to-sensor-body transform |
| `NEAR_Z` | `float = 1e-5` |
| `camera_ray_mask` | `camera_ray_mask(camera: Camera, rays: np.ndarray) -> np.ndarray` |
| `Camera` | `width: int`<br>`height: int` |
| `Camera.project` | `project(pts_cam: np.ndarray) -> np.ndarray` |
| `Camera.unproject` | `unproject(uv: np.ndarray) -> np.ndarray` |
| `PinholeCamera` | `K: np.ndarray`<br>`width: int`<br>`height: int` |
| `PinholeCamera.project` | `project(pts_cam: np.ndarray) -> np.ndarray` |
| `PinholeCamera.unproject` | `unproject(uv: np.ndarray) -> np.ndarray` |
| `FThetaCamera` | `cx: float`<br>`cy: float`<br>`angle_to_pixeldist: tuple`<br>`width: int`<br>`height: int`<br>`pixeldist_to_angle: tuple = ()`<br>`max_angle: float = math.pi / 2`<br>`linear_cde: tuple = (1.0, 0.0, 0.0)` |
| `FThetaCamera.project` | `project(pts_cam: np.ndarray) -> np.ndarray` |
| `FThetaCamera.in_view` | `in_view(pts_cam: np.ndarray) -> np.ndarray` |
| `FThetaCamera.unproject` | `unproject(uv: np.ndarray) -> np.ndarray` |

### avsectester.geometry.projection

These helpers accept camera models, world transforms and geometry directly, without SensorView or
attack metadata. Both pinhole and f-theta lenses are supported.

| Public name | Signature or fields |
|---|---|
| `project_box` | `project_box(camera, cam_from_world, world_from_box, extent) -> tuple[float, float, float, float] \| None` |

project_box returns the image rectangle (x0, y0, x1, y1), or None when the box has no visible
projection.

### avsectester.geometry.boxes

These functions receive a world-from-box-center transform and full (length, width, height).
Pass ActorState.box_transform(), rather than its actor-origin pose. Rays are unit vectors in the
camera frame; missed rays and zero-sized boxes return infinity. Overlap tests use both boxes' yaw
and horizontal dimensions, include touching, and ignore height.

| Public name | Signature or fields |
|---|---|
| `box_ray_distances` | `box_ray_distances(world_from_box: ArrayLike, extent: ArrayLike, rays: np.ndarray, world_from_cam: ArrayLike) -> np.ndarray` |
| `oriented_boxes_overlap` | `oriented_boxes_overlap(world_from_box_a: ArrayLike, extent_a: ArrayLike, world_from_box_b: ArrayLike, extent_b: ArrayLike) -> bool` |

### avsectester.geometry.mesh

| Public name | Signature or fields |
|---|---|
| `MeshRaycaster` | Class; see methods below |
| `MeshRaycaster.__init__` | `__init__(vertices: np.ndarray, faces: np.ndarray) -> None` |
| `MeshRaycaster.check_dependencies` | `check_dependencies() -> None` |
| `MeshRaycaster.distances` | `distances(origins: np.ndarray, directions: np.ndarray) -> np.ndarray` |
| `camera_depth` | `camera_depth(camera: Camera, cam_from_world: np.ndarray, raycasters: Sequence[MeshRaycaster]) -> np.ndarray` |

MeshRaycaster copies finite N×3 vertices and integer M×3 triangular faces and builds an Embree
acceleration structure. distances accepts matching N×3 ray origins/directions, normalizes
directions and returns N Euclidean first-hit distances, with inf for a miss. These are ray
distances, not optical-axis z depth. Optional libraries load when the class is used; the explicit
dependency check raises ImportError with the installation extra when they are unavailable.

camera_depth unprojects each pixel center through the supplied lens, casts world-coordinate rays
through every supplied MeshRaycaster, and keeps the nearest hit. Its float64 (H, W) output is
optical-axis z depth, with inf where all raycasters miss. It is shared by the simulator and offline
validation, and needs no native renderer or default-camera configuration.

### avsectester.closed_loop.simulators.navigation

| Name | Signature |
|---|---|
| `RouteGenerator` | `RouteGenerator(kind, recorded_points, scenario_map)` |
| `RouteGenerator.route` | `route(world_from_ego: np.ndarray) -> np.ndarray \| None` |

Routes are model-visible navigation intent in ego coordinates, in metres with Z=0. Map mode
uses driving lanes, resolves lane transitions along the recording, and extends through successors.
Recorded mode uses the recorded path without extension. Each query projects the current physical
ego origin onto the route, samples 20 points across 80 m, and repeats the endpoint when short.
None disables navigation.

### avsectester.closed_loop.simulators.ground_contact

| Name | Signature |
|---|---|
| `constrain_to_ground` | `constrain_to_ground(raycaster: MeshRaycaster, world_from_box: np.ndarray, extent: tuple[float, float, float]) -> tuple[np.ndarray, str]` |

A 4×4 box-bottom grid is raycast vertically in both directions within 100 m. At least six hits
are needed. Plane-fit displacement over 1.5 m or rotation over 10° rejects the update; the
original pose is returned with insufficient_intersections, high_translation or high_rotation.
Successful updates return successful_update and preserve box-centre horizontal position.

SimulatorRuntime.termination_details supplies diagnostics for the termination event; NuRec
records ground_contact_failures grouped by reason.

NuRecRenderer.resolution_hw sets requested height and width per camera; missing entries use
native resolution. Loading probes each camera to obtain the server's actual aspect-preserving
output size. Its optical model uses that size, with lens coefficients and principal point scaled
by returned_height/native_height. Later size changes are rejected. Depth and insertion use this
same scaled model. AlpamayoAVStack.frame_subsample selects image context from a strided recent
window, retaining the current frame. The official single-camera preset uses a factor of 3.

AlpamayoAVStack requires 15 actual past decisions and 0.1 s observation intervals. reset seeds
its image and pose caches; each call appends the current observation. The latest 16 received poses
provide the model history and pose-difference speed and acceleration. It rejects missing images
and a context span `(context_length - 1) * frame_subsample` greater than 15. No temporal padding
is used. Model timestamps use round(t * 1e6) + EPOCH_US.

The policy component bound by alpamayo_driving_adapter takes a mapping with observation and past.
Its DrivingAdapter fallback past_steps is 15 even for an ordinary Python factory without
avstack_type; configured class metadata still takes precedence.
Observation.route supplies rig-relative navigation; the official AlpaSim helper determines the
command. The component log includes route_waypoints, navigation_command, navigation_note,
history_pose_count and selected_image_times (simulation seconds by camera). Control.trajectory
starts at rig-frame origin with identity rotation at t0, followed by full model waypoint poses.
