# Configuration reference

> **Audience:** Experiment authors.
>
> **Outcome:** Construct experiments from YAML or Python.
>
> **Prerequisites:** [Architecture](../concepts/architecture.md).

## Experiment configuration

The same configuration drives CLI and Python. Dataset, simulator, driving system, attacks, hooks
and evaluation are components. The runs section describes what to execute on those components and
how to record it. Attack and AVStack specifications may be inline mappings or YAML file paths.
Component YAML references and runs.output resolve relative to their containing YAML file.
Constructor filesystem options follow the [annotation rules](#constructor-filesystem-parameters).

Type resolution and attack discovery live in avsectester.experiment.type_registry; the
configuration loader owns YAML structure and validation.

Every object constructor uses `type` and `options`. A name without a colon selects a built-in class.
`package.module:name` imports a user class or function. The name may contain dots for nested
attributes, such as `package.module:Container.CustomAttack`; the resolved value must be callable.
Explicit paths use `avsectester.type_paths.import_type_path`. Such imports execute Python and must
be trusted. The same rule applies to nested objects, hooks and sensor corruptions.

```yaml
dataset:
  type: CarlaDataset
  options:
    recipes:
      road:
        type: avsectester.datasets.carla:CarlaRecipe
        options:
          decision_steps: 40
          time_step: 0.05
          scenario:
            client: {map_name: Town03}
            ego:
              type: CarlaMobileActor
              spawn: 0
              vehicle: vehicle.tesla.model3
              sensors:
                - {type: CarlaLidar, name: lidar-0, rotation_frequency: 20}
            lead: {gap: 25.0, vehicle: vehicle.tesla.model3}
          motion:
            ego:
              type: avsectester.datasets.carla:PrescribedMotion
              options:
                constant_control:
                  type: avsectester.data_types:Control
                  options: {throttle: 0.35}
            lead:
              type: avsectester.datasets.carla:PrescribedMotion
              options:
                constant_control:
                  type: avsectester.data_types:Control
                  options: {brake: 1.0}
simulator:
  type: CarlaSimulator
  options:
    client: {connect_ip: 127.0.0.1, connect_port: 2000, rate: 20, synchronous: true}
avstack: driving.yaml
attacks:
  phantom:
    type: PhantomInjectionAttack
    options: {decision_steps: 20, target_xyz: [6, 0, -0.7], max_segments: 1}
evaluation:
  metrics: [{type: DrivingMetrics}, {type: DetectionMetrics}, {type: StopImpact}]
runs:
  treatments:
    - {name: clean}
    - {name: attacked, attack_name: phantom}
  seeds: [0]
  output: ../results/phantom
```

`driving.yaml` is a user-supplied driving specification; the complete supplied version is in
`configs/carla_phantom.yaml` at the repository root.

## Constructor filesystem parameters

For every framework type/options constructor, filesystem resolution follows parameter annotations.
Path and os.PathLike annotations mark paths, including these types inside unions or sequence
elements such as str | Path, list[Path] or tuple[str | Path, ...]. The same rules apply to nested
constructors, hooks, sensor corruptions, assets and other helpers.

For an annotated path, a str or Path value becomes an absolute string relative to the YAML file
containing it. A leading ~ is expanded; an already absolute path keeps its location and is returned
as a string. A string containing :// is retained unchanged. Arrays, Texture objects and other
non-path values stay unchanged.

Unannotated parameters, Any, **kwargs and plain nested dictionaries do not trigger filesystem
resolution. A parameter annotated only as str is ordinary text. Annotation resolution is performed
per parameter: if it fails and the annotation text mentions Path, configuration loading reports an
error. Make the names used by that annotation available at runtime. An unresolved annotation that
does not mention Path is ignored. Component YAML references and runs.output retain their separate
containing-file rules.

## Components

| Key | Meaning |
|---|---|
| `dataset` | Dataset object specification |
| `simulator` | Simulator object specification |
| `avstack` | Driving specification, inline or a file reference |
| `attacks` | Exactly one name mapped to an Attack specification or file reference |
| `hooks` | One stage-hook configuration applied to every treatment |
| `evaluation` | Metrics and all-treatment pair comparisons |

### Dataset and simulator

CARLA scene content is solely in `CarlaRecipe.scenario`. The simulator receives runtime client
settings, optional settling iterations and a depth-camera name. Recipe motion explicitly covers ego
and all other actors through `constant_control`, a control sequence or a state trajectory. A
trajectory contains the terminal state. Traffic Manager is not a prescribed motion source.

NuRec configures its dataset `time_step` and simulator `config.dt` to the same value, typically 0.1
s. Recorded trajectories are interpolated only inside their support. Changing the time step changes
segment positions. NuScenes uses RecordedReplaySimulator for open-loop playback.

See [Scenarios](../guides/scenarios.md) and [Simulators](../guides/simulators.md).

### NuRec closed-loop settings

The supplied NuRec experiments use the AlpaSim linear MPC, map navigation, ego ground contact
against mesh.ply, and the official single-camera Alpamayo preset:

```yaml
simulator:
  type: NuRecSimulator
  options:
    config: {dt: 0.1}
    route_generator: map
    physics_update_mode: ego_only
    ground_mesh: mesh
    renderer:
      type: avsectester.closed_loop.simulators.nurec:NuRecRenderer
      options:
        cameras: [camera_front_wide_120fov]
        resolution_hw: {camera_front_wide_120fov: [1080, 1900]}
    dynamics:
      type: avsectester.closed_loop.simulators.ego_dynamics:AlpaSimController
      options: {mpc_implementation: linear, n_horizon: 20, dt_mpc: 0.1}
avstack:
  type: AlpamayoAVStack
  options:
    camera_ids: [camera_front_wide_120fov]
    frame_subsample: 3
```

route_generator accepts map, recorded or none. Map navigation requires a semantic map with driving
lanes; recorded uses the recorded ego path without extending it. physics_update_mode accepts
ego_only or none; ground_mesh selects mesh or mesh_ground independently of depth. Both ground
contact and mesh depth require the nurec extra. NuRecRenderer.resolution_hw uses native camera IDs
and requested (height, width). NRE may return another width to preserve the camera aspect ratio;
projection, depth and insertion use its actual dimensions.

Alpamayo requires 15 preceding recorded decisions at 0.1 s intervals. Their images and poses are
provided to reset; no startup padding is generated. frame_subsample defaults to 1, and the
configured image span `(context_length - 1) * frame_subsample` must not exceed 15. The supplied
stride of 3 selects four images from the latest ten decisions. Built-in attacks skip starts lacking
the required past; preparation also discards invalid segments. See
[Simulators](../guides/simulators.md#alpasim-defaults) and
[Driving systems](../guides/driving-systems.md#alpamayo).

### Driving specification

`avstack.type` names an AVStack class or a callable returning one. It receives only its `options`,
not a seed. Each run, rollout and component evaluation creates fresh driving state. Scoring calls
within one closed-loop session reuse its isolated scoring AVStack until session close.

```yaml
type: my_project.driving:MyAVStack
options:
  checkpoint_path: ../models/driving
  device: cuda:0
bindings:
  type: my_project.driving:bind_components
  options: {}
```

Optional `bindings.type` references `function(avstack_factory, **options) -> DrivingAdapter`.
Bindings expose components and scorers; they do not replace the AVStack. Optional
avstack.observation_pipeline
constructs ObservationPipeline from avsectester.closed_loop.avstacks.base using a {type, options}
specification; avstack.command_modes asserts the
expected command modes and must match the AVStack declaration. These keys describe the configured
driving boundary, while options are constructor arguments. A driving configuration can be reused
across supported scenarios of one simulator.

### Attack specification

The attacks mapping must contain exactly one entry. Multiple attacks in one experiment are future
work; implement a composite Attack when several actions must be applied together. Empty and
multi-attack configurations fail loading with exit code 2. When treatments are omitted, loading
creates clean and attack treatments under runs. Every treatment shares this attack's metadata,
may enable its attack_name and can change environment settings. A clean treatment has no
attack_name.

An attack specification has `type`, `options` and optional `environment`. Selection parameters,
input assets and optimization belong to the Attack constructor. The framework calls its metabuilder
for each scene, then its offline phase once. Selection, host ranking and score semantics belong
entirely to build_metadata; the framework only validates and saves AttackSegment records. It never
reruns selection after optimization.

```yaml
attacks:
  speed_feedback:
    type: SpeedFeedbackAttack
    options:
      decision_steps: 20
      target_speed: 3.0
      gain: 0.1
      max_bias: 5.0
    environment:
      world: {weather: SoftRainNoon}
```


Unqualified type names first resolve built-in framework types. An unknown name is looked up as an
Attack in implemented_attacks/<snake_case_class_name>/, exported by that folder's
__init__.py. Built-in attacks use the same discovery path as user attacks. MyAttack maps to
my_attack, MyNNAttack to my_nn_attack, XMLAttack to xml_attack and HTTP2Attack to http2_attack and
Attack2D to attack2_d. Continuous capitals form an acronym; the boundary before a following
capital-plus-lowercase word is split. Missing attacks report the available names. User code
elsewhere uses package.module:Class.

Only dataset, simulator, AVStack and metric classes have built-in registry names; attacks use the
folder discovery described above. Other constructors, including assets, textures, placements,
optimizers, compositors and their helpers, use full package.module:Class references with options.
Lowercase shortcuts and inferred placement types are rejected. Image assets can be direct paths;
SignAsset.face, PatchAttack.assets and PatchOptimization.init_texture follow the
[constructor filesystem rules](#constructor-filesystem-parameters).

### Hooks

```yaml
hooks:
  - type: my_project.hooks:Audit
    options: {}
    stage: command
```

A native avstack hook uses `native_hook`, with any required module registration stated explicitly:

```yaml
hooks:
  - stage: perception.post
    imports: [my_project.native_hooks]
    native_hook: {type: MyRegisteredHook}
```

The `native_hook` payload follows the avstack registry format. Prefer an Attack for reproducible
attack preparation and deployment. Hook stage names and contexts are in
[Runtime](runtime.md#hook-stages).

The same hooks apply to every treatment, including treatments with no attack_name. Treatments differ
only in attack enablement and environment settings.

### Evaluation

There is no baseline option. Every ordered-by-configuration treatment pair is compared once, with
single-run metric differences calculated as B minus A. TrajectoryDeviation is symmetric and
StopImpact computes both directions. Unfinished pairs are unavailable with a reason.

Omitting metrics selects one default set for the entire experiment: DrivingMetrics and
TrajectoryDeviation, with DetectionMetrics added when completed logs all provide canonical
detections. StopImpact must be explicitly configured. Explicitly unsupported DetectionMetrics and
duplicate metric types fail configuration loading. Metric implementation errors remain reportable
evaluation failures.

```yaml
evaluation:
  metrics:
    - {type: DrivingMetrics, options: {stop_speed: 0.5}}
    - {type: TrajectoryDeviation}
    - {type: StopImpact}
```

User metrics use type/options, including package.module:Class. See
[Evaluation](../guides/evaluation.md#built-in-metrics) for fields and evidence requirements.

## Run settings

Run settings belong exclusively under runs. Supplying treatments, seeds, decision_steps,
environment, recording, output or strict at the experiment's top level raises "Unknown experiment
keys"; old layouts are not converted. In Python, use the frozen RunsConfiguration as
ExperimentConfiguration.runs.

| Key under runs | Meaning |
|---|---|
| `treatments` | Named clean or single-attack runs with optional environment overrides; defaults to clean and attack |
| `seeds` | Nonempty unique nonnegative integers; default `[0]` |
| `decision_steps` | Optional positive upper bound on each selected segment's decisions; default null |
| `environment` | Shared base `{world, sensors}` settings; default empty |
| `recording` | Sensor retention policy; state and commands remain recorded each decision |
| `output` | Required CLI output directory; optional for in-memory Python execution |
| `strict` | Stop on the first error when true; default false |

runs.output is resolved relative to the YAML file containing it. Preparation uses the first value
of runs.seeds. OfflineContext.environment merges runs.environment with the attack's environment;
metabuilders request sensor/render environments explicitly and otherwise see the original world.

### Environment

The value of runs.environment has shape `{world: {...}, sensors: [...]}`. Resolve experiment base,
enabled Attack environment and treatment overrides in that order. World dictionaries merge
recursively; an explicitly supplied sensor list replaces the preceding list. SimulatorRuntime.reset
receives only the world mapping. See the complete [environment merge
rules](../guides/environments.md#experiment-environment).

For example, the value of runs.environment can be:

```yaml
world: {weather: SoftRainNoon}
sensors:
  - {type: GaussianNoise, options: {severity: 0.2}, camera: front}
```

CARLA supports the native keys declared by `CarlaSimulator.supported_world_settings`, including
weather, lighting, map_layers and road_friction; support is not limited to weather. Unsupported
world settings fail rather than becoming sensor approximations. See [Native CARLA
settings](../guides/environments.md#native-carla-settings).

### Recording

```yaml
runs:
  recording:
    sensor_ids: [front]
    representations: [raw, model_input]
    every_n_steps: 2
```

Sensor payloads are sampled every N decisions. Empty sensor_ids retain no image payloads. Physical
states, commands, actions, outputs, available components and detections are recorded at each
decision. Choose payload retention before a run; reevaluation cannot recover omitted images.

## Saved preparation compatibility

`PreparedExperiment.save(directory)` records dataset type and scenario list, simulator identity,
each attack's type, parameters and environment. prepared.json also stores preparation, captured by
Experiment.prepare from the inputs actually used:

- seed is runs.seeds[0].
- offline_environments maps each attack name to its merged environment,
  merge_environments(runs.environment, attack.environment), recorded as canonical data: world
  settings as written, and each sensor entry as type, options and, when set, camera.

The record is computed before selection starts, so an environment value that is not data (for
example an arbitrary object) fails before any preparation work. Paths, tuples, NumPy values and
dataclasses are recorded as their configuration-summary data.

`PreparedExperiment.load(config, directory)` and CLI --prepared check dataset/scenario, simulator
and attack compatibility, then compare the saved preparation seed and each offline environment
with the supplied configuration. A mismatch raises ConfigurationMismatchError naming the differing
fields and their saved and new values; the CLI maps it to exit code 2. Corrupt or unreadable files
are result-reading failures (exit code 1).

Additional seeds after the first, treatments, recording, output, strict, decision_steps and
evaluation settings may change when reusing preparation. The first seed and normalized per-attack
offline environments must still agree. Saving a loaded preparation preserves its original
preparation snapshot rather than recomputing it from the current run settings. Records are compared
as canonical JSON text, so a list and a tuple with the same values agree while True and 1 differ.
Options are compared as written: writing a transform's default value explicitly differs from
omitting it. prepared.json uses deterministic key order. Values passed to offline preparation and
simulation remain unchanged.

ExperimentResult.load and CLI evaluate read saved logs through load_prepared_metadata without
constraining runs. Historical analysis does not require preparation to be reusable for new runs,
and outputs saved without a preparation record remain readable.

Summaries are calculated only when saving or loading. Constructing or running a configuration does
not enumerate scenarios for this check or require parameter descriptions. YAML constructor options
retain their original description. Direct Python Attack objects use `configuration_parameters()`; if
that cannot describe inputs, saving warns and marks compatibility unverified. Loading such a
preparation fails explicitly. Type equality alone is insufficient.
