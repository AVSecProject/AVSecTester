# Configuration reference

> **Audience:** Experiment authors.
>
> **Outcome:** Construct a complete experiment from YAML or Python.
>
> **Prerequisites:** [Architecture](../concepts/architecture.md).

## Experiment configuration

A configuration contains component specifications and run settings. YAML section order does not
affect loading. `load_configuration` accepts a YAML path, a mapping or an
`ExperimentConfiguration` object.

Every constructor specification uses `type` and optional `options`. A built-in short name
selects a framework component; `package.module:Class` or `package.module:function` selects user
code. `dataset`, `simulator`, `avstack`, `attacks.<name>`, `evaluation.metrics` entries and
`observation_pipeline` can also reference YAML files. Nested constructors use the same
type/options form.
Type paths and hook imports execute Python code; use only trusted sources.

### Component short names

| Category and location | Accepted names |
|---|---|
| `dataset.type` | `CarlaDataset`, `NuRecDataset`, `NuScenesDataset` |
| `simulator.type` | `CarlaSimulator`, `NuRecSimulator`, `RecordedReplaySimulator` |
| `avstack.type` | `ModularAVStack`, `CameraForwardCollisionAVStack`, `AlpamayoAVStack`, `RuleBasedAVStack` |
| `evaluation.metrics[].type` | `DrivingMetrics`, `DetectionMetrics`, `TrajectoryDeviation`, `StopImpact`, `StopLineMetrics`, `ReasoningPhraseRate` |
| `attacks.<name>.type` | Discovered `Attack` subclasses in [class-name-derived directories](../../avsectester/implemented_attacks/README.md) |
| `environment.sensors[].type` | `GaussianNoise`, `ShotNoise`, `Fog`, `Rain`, `Snow`, `Brightness`, `LowLight`, `Contrast`, `Gamma`, `ColorTemp`, `MotionBlur`, `DefocusBlur`, `JPEGCompression`, `ChromaticAberration`, `AlbumentationsCorruption` |

Nested `type` mappings construct only dataset, simulator, driving-system and metric short names
from this table, or `module:name` paths. Attack names, corruption names and other values pass
through as native data.

## Top-level keys

| Key | Default | Meaning |
|---|---|---|
| `device` | omitted | Default for constructors with an explicit device parameter |
| `dataset` | required | Scenario source |
| `metadata` | `save`/`load` both `null` | Export or load one shared segment file |
| `split` | `null` | Training, validation and test segment allocation |
| `simulator` | required | World and sensor provider |
| `avstack` | required | Driving system and optional bindings |
| `attacks` | required, nonempty | Names mapped to `Attack` specifications |
| `hooks` | `null` | Stage handlers shared by all treatments |
| `evaluation` | automatic metric set | `Metric` specifications |
| `runs` | defaults below | Treatments, seeds, environment, recording and output |

This complete example uses the checkout's memory components and needs no simulator service or
model weights. Supplied simulator configurations use the same structure; see the
[configuration list](../examples/README.md).

<!-- runnable -->
```yaml
device: cpu
dataset:
  type: tests.fakes:MemoryDataset
  options: {decision_steps: 6, time_step: 0.1}
simulator:
  type: tests.fakes:MemorySimulator
avstack:
  type: tests.fakes:ConstantAVStack
  options:
    command:
      type: avsectester.data_types:Control
      options: {throttle: 0.2}
attacks:
  probe:
    type: tests.fakes:MinimalAttack
    options: {decision_steps: 2, max_segments: 1}
evaluation:
  metrics:
    - {type: DrivingMetrics}
runs:
  treatments:
    - {name: clean}
    - {name: probe, attack_name: probe}
  seeds: [0]
  recording:
    sensor_ids: [front]
    representations: [model_input]
    every_n_steps: 1
  output: results/memory
```

## Device

The optional top-level `device` is a nonempty string, such as `cuda:0`. It supplies a default to
every type/options constructor whose signature explicitly declares `device`. An explicit option
wins. Constructors without that parameter, including those accepting only `**kwargs`, receive
nothing extra. Omitting the key preserves each constructor's default.

Injection applies to dataset, simulator, `AVStack`, observation pipeline, bindings, attacks,
hooks, metrics and sensor corruptions, including nested constructors and deferred `AVStack`
construction. Directly supplied Python objects are unchanged. The injected options have the same
preparation identity as explicitly written options.

Components with this parameter include `AlpamayoAVStack`, `YOLOXDetector`,
`DetectorPatchOptimization`, `PatchOptimization` and `PCTNetHarmonizer`. Container GPU selection
is separate; see [automatic local services](cli.md#automatic-local-services).

## Dataset

`dataset` selects the scenario source. `NuRecDataset` accepts either `dataroot`, searched
recursively for USDZ archives, or an explicit `usdz_paths` sequence. The two options are
mutually exclusive. Dataset behavior and semantic maps are described
in [Datasets](../guides/datasets.md).

## Metadata

The JSON file contains `format: "avsectester.metadata"`, `version: 1`,
`dataset_type: "module:Class"` and `segments` encoded by `DEFAULT_REGISTRY`. The dataset type
must match. See [Experiments](../guides/experiments.md#save-and-reuse-selected-metadata) for
save, load and validation timing.

`metadata.save` and `metadata.load` both default to `null` and are mutually exclusive. Paths
resolve relative to the containing YAML file. Python configurations use `MetadataConfiguration`.

```yaml
metadata:
  save: ../results/metadata/stop_sign_2601.json
```

```yaml
metadata:
  load: ../results/metadata/stop_sign_2601.json
```

Duplicate complete segments are rejected on save and load. There are no per-attack or
per-treatment overrides.
The preparation records the source path and raw-byte SHA256. Compatibility depends only on that
hash, so moving an unchanged file is allowed. Changing any bytes, including whitespace, changes
its identity. Reusing preparation for execution checks the current file's hash and rechecks
saved segment frames; other checks follow [Input validation](../guides/experiments.md#input-validation).
Reevaluation uses the saved hash and segments without reading this file. Adding or removing
`metadata.load` relative to the saved configuration is incompatible.

The [Python API](python-api.md) provides `save_metadata` and `load_metadata` for user-generated
segments. The [experiment workflow](../guides/experiments.md) describes preparation reuse.

## Data split

Without `split`, each attack receives all its selected segments in its offline phase and all
units are executed. With `split`:

```yaml
split:
  train_segments: 5
  validation_segments: 0
  seed: 0
  execute: [test]
```

| Key | Default | Meaning |
|---|---|---|
| `train_segments` | required | Positive number of training units |
| `validation_segments` | 0 | Nonnegative number of validation units |
| `seed` | 0 | Nonnegative permutation seed |
| `execute` | `[test]` | Unique subset of `train`, `validation` and `test`; an empty list executes no units |

Unknown keys are rejected. Executing validation requires a positive validation count. Training
plus validation must be smaller than the number of units, leaving at least one test unit.

A unit matches on scenario, start, stop and all `user_data`, encoded as canonical JSON. Units
follow dataset scenario order, then start, stop and configured attack order. A seeded NumPy
permutation assigns training units first, validation units next and test units last. Identical
selections share a part. The same window with different `user_data` forms separate units;
training and test may therefore contain the same captured scene.

`OfflineContext.metadata` contains training segments and `validation_metadata` contains
validation segments, each in selection order. `PreparedExperiment.metadata` keeps all selections
and `units` records the part and selecting attacks. `split.execute` affects execution, not
preparation compatibility.

## Simulator

CARLA scene and prescribed motion settings belong to `CarlaRecipe`; `CarlaSimulator` receives
the runtime client settings. For recorded sources, choose a simulator compatible with the
dataset; [Simulators](../guides/simulators.md) describes their behavior.

A simulator's top-level `time_step` parameter inherits `dataset.time_step` when omitted and when
the dataset provides it. This injection does not recurse into nested constructors. Explicit
values are retained, but `ExperimentConfiguration` rejects a dataset/simulator mismatch.

### NuRec closed-loop settings

`NuRecSimulator` requires a renderer and a non-null `time_step`; the loader supplies the latter
from `NuRecDataset`. Reset requires `ScenarioStart`. Constructor defaults are:

| Option | Default | Accepted values or role |
|---|---|---|
| `dynamics` | `KinematicBicycle()` | Ego-dynamics implementation |
| `camera_aliases` | empty | Native camera ID to observation/sensor-view name; output names must be unique |
| `depth` | `mesh` | `mesh`, `none` |
| `route_generator` | `map` | `map`, `recorded`, `none` |
| `physics_update_mode` | `ego_only` | `ego_only`, `none` |
| `ground_mesh` | `mesh` | `mesh`, `mesh_ground` |
| `object_occlusion` | `rendered_probes` | `rendered_probes`, `geometry` |
| `occlusion_tolerance_m` | 0.15 | Nonnegative scene occlusion tolerance in meters |
| `traffic` | `null` | Optional `AlpaSimTraffic` |
| `min_track_duration_s` | 3.0 | Nonnegative moving-track duration threshold; 0 disables filtering |

`nurec_patch.yaml`, `nurec_speed_feedback.yaml` and `nurec_object_insertion.yaml` use
`ground_mesh: mesh` and the default mesh depth. The four stop-sign configurations use
`ground_mesh: mesh_ground` and `depth: none`. All seven explicitly select map navigation and ego
ground contact with `AlpaSimController`.

```yaml
simulator:
  type: NuRecSimulator
  options:
    depth: none
    route_generator: map
    physics_update_mode: ego_only
    ground_mesh: mesh_ground
    renderer:
      type: avsectester.closed_loop.simulators.nurec:NuRecRenderer
      options:
        endpoint: 127.0.0.1:50051
        cameras: [camera_front_wide_120fov]
        probe_directory: ../results/nre-probes
        server_probe_directory: /mnt/nre-probes
    dynamics:
      type: avsectester.closed_loop.simulators.ego_dynamics:AlpaSimController
```

NuRec cameras render at their native resolution.
`probe_directory` is a client-side path resolved relative to the YAML file;
`server_probe_directory` is an unchanged string in the server's filesystem namespace. The server
must be able to read the probe files. See
[automatic local services](cli.md#automatic-local-services) for startup and mounts.

### Reactive traffic

Add the following under `simulator.options`:

```yaml
traffic:
  type: avsectester.closed_loop.simulators.nurec:AlpaSimTraffic
  options:
    endpoint: 127.0.0.1:6200
    history_steps: 16
    rpc_timeout_s: 300.0
```

Traffic defaults to `null`. Enabling it requires 0.1 s decisions and `history_steps - 1` preceding
recorded decisions. Traffic configuration affects preparation identity; `metadata.load` can
share segments between modes. Services are managed by [run](cli.md#automatic-local-services).

## Driving specification

`avstack.type` selects an `AVStack` class or factory. Its options are passed to each fresh
instance. A constructor declaring `seed` receives the run seed; explicitly writing that reserved
option is an error. The top-level device follows the [device rule](#device).

| Key under avstack | Meaning |
|---|---|
| `type` | Required `AVStack` class or callable |
| `options` | Constructor arguments; default empty |
| `observation_pipeline` | Optional `ObservationPipeline` type/options specification |
| `command_modes` | Optional assertion matching the stack's declared modes |
| `bindings` | Optional binding factory, with type/options |

`AVStack.decision_time_step` optionally declares a fixed cadence. `DrivingAdapter` reads it from
`factory.avstack_type`, or uses its explicit fallback without constructing the stack.
`ClosedLoopServices` rejects a driving-system step that differs from the simulator step.
`AlpamayoAVStack` requires 0.1 s decisions. Its supplied configurations use
`frame_subsample: 1` to select camera frames from history without changing decision timing. See
[Past observations](../guides/simulators.md#recorded-past-observations).

A binding has signature `function(avstack_factory, **options) -> DrivingAdapter`. The factory
exposes `avstack_type` and an independent read-only `avstack_options` mapping. Bindings declare
components and scorers without constructing the stack;
[Driving systems](../guides/driving-systems.md) describes the supplied bindings.

## Attack specification

`attacks` is a nonempty mapping of names to `Attack` specifications or YAML paths. Each entry
has `type`, optional `options` and optional `environment`. Each attack selects its own segments
unless `metadata.load` supplies them. See [Writing attacks](../guides/attacks.md) for lifecycle
phases and implementer responsibilities.
Attacks that do not override `build_metadata` require `metadata.load`.

```yaml
attacks:
  speed_feedback:
    type: SpeedFeedbackAttack
    options: {decision_steps: 20, target_speed: 3.0, gain: 0.1, max_bias: 5.0}
    environment:
      world: {weather: SoftRainNoon}
```

See the [configuration list](../examples/README.md) for supplied attacks and
[component construction](#experiment-configuration) for user type paths.

## Hooks

`hooks` is a sequence of stage handlers shared by every treatment. Each mapping declares `stage`
and exactly one of `type` or `native_hook`. A type-based hook accepts `options`. Optional
`imports` lists modules imported before constructing the handler.

```yaml
hooks:
  - stage: perception.post
    imports: [my_project.hooks]
    native_hook:
      type: MyDetectionHook
```

Native hooks require a `.pre` or `.post` stage. Their payload uses avstack's `HOOKS` registry
format and does not accept a separate `options` key. Modules listed in `imports` must register
the hook in `HOOKS`. Stage semantics belong
to [Runtime](runtime.md#hook-stages).

## Evaluation

`evaluation.metrics` accepts `Metric` type/options specifications. Omitting it selects
`DrivingMetrics` and `TrajectoryDeviation`, adding `DetectionMetrics` when every completed log
provides canonical detections. An explicit empty list selects no metrics. Repeated metric types
are rejected. `DetectionMetrics` is rejected during loading when the configured stack declares
no detections.

Custom metrics use `package.module:Class`; see [Experiment configuration](#experiment-configuration).

```yaml
evaluation:
  metrics:
    - {type: StopLineMetrics}
    - {type: ReasoningPhraseRate, options: {phrase: stop sign}}
    - {type: DrivingMetrics}
```

`Metric` fields and comparison semantics are described in [Evaluation](../guides/evaluation.md).

## Run settings

Run controls belong under `runs`, represented in Python by frozen `RunsConfiguration`.

| Key | Default | Meaning |
|---|---|---|
| `treatments` | `clean` plus one treatment per attack | Named runs with optional `attack_name` and environment |
| `seeds` | [0] | Nonempty unique nonnegative integers |
| `decision_steps` | `null` | Optional positive upper bound per selected segment |
| `environment` | empty | Shared world and sensor settings |
| `recording` | defaults below | Sensor payload retention |
| `output` | `null` | Output directory; required by the CLI |
| `strict` | `false` | Stop on the first execution or selection error |

Attack and treatment names must be nonempty, contain no path separators and differ from
`.` and `..`. Treatment names
must be unique. A treatment without `attack_name` has no enabled attack; otherwise that field
must reference a configured attack. Group membership is defined
in [Evaluation](../guides/evaluation.md#comparison-groups).

### Environment

CARLA world settings accept `weather` as a preset name or parameter mapping, `lighting`,
`map_layers` and `road_friction`. NuRec and recorded playback reject every nonempty
world setting.

Environment YAML has `world` and `sensors` keys. The base is `runs.environment`, with optional
attack and treatment overrides; [Environments](../guides/environments.md) describes merge and
validation behavior. Sensor entries use `type`, `options` and an optional `camera`; omitting
camera applies the corruption only to the first `sensor_data` entry, which must be a readable
RGB image.

```yaml
runs:
  environment:
    world: {weather: SoftRainNoon}
    sensors:
      - {type: GaussianNoise, options: {severity: 0.2}, camera: front}
```

Custom corruptions use `package.module:Class`; see [Experiment configuration](#experiment-configuration).
Severity is in [0, 1] and defaults to 0.5. `ColorTemp` also accepts
`warm`, which defaults to `true`. `AlbumentationsCorruption` requires `transform`. The
[Python API](python-api.md) lists the available corruption classes.

### Recording

| Key under recording | Default | Meaning |
|---|---|---|
| `sensor_ids` | [] | Unique sensor names whose payloads are retained |
| `representations` | [`model_input`] | Subset of `raw` and `model_input` |
| `every_n_steps` | 1 | Positive decision interval for retained sensor payloads |

Empty `sensor_ids` retains no image payloads. State, commands, actions, outputs, available
components and detections are recorded every decision. Reevaluation cannot recover omitted
sensor payloads.

```yaml
runs:
  recording:
    sensor_ids: [front]
    representations: [raw, model_input]
    every_n_steps: 2
```

## Constructor filesystem parameters

Filesystem options follow constructor annotations for dataset, simulator, `AVStack`, attack,
hook, metric and nested constructors. `Path` or `os.PathLike` annotations mark paths, including
unions and sequence elements such as `str | Path` or `list[Path]`.

Relative values become absolute strings resolved beside the YAML file that contains them. A
leading `~` is expanded; absolute paths preserve their location; URI strings containing `://`
are unchanged. Plain `str`, unannotated parameters, `Any`, `**kwargs` and untyped mappings do
not trigger path resolution. Unresolvable annotations mentioning Path are errors; make their
names available at runtime. Other unresolved annotations are ignored.

Component YAML references and `runs.output` also resolve relative to their containing YAML file.

## Saved preparation compatibility

The saved identity includes dataset type/scenario IDs, simulator identity and each attack's
type, parameters and environment. Its preparation record includes the first run seed, split
counts/seed and merged offline environments.

Loading a preparation checks these values and recomputes saved units under the configured split.
A current metadata file must have the saved raw-byte hash, but its path may differ. A mismatch
raises `ConfigurationMismatchError`. Saved preparations require an offline report for every
completed attack, including attacks whose offline phase is a no-op.

Additional `seeds`, `treatments`, `recording`, `output`, `strict`, decision limits, `split.execute`
and evaluation settings may change. The first seed, partition and offline environments must match.
`run --prepared` writes the loaded preparation record, units, deployment budget and metadata
source unchanged into the new output. `AVStack` and hook configurations are not part of
saved-preparation compatibility. `Attack` constructor options are compared as written,
including an explicitly written default versus an omitted option. Lists and tuples with
identical contents compare equally; booleans, integers and floating-point numbers remain
distinct, so `true`, `1` and `1.0` are incompatible.
The current hooks' setup ticks plus the attack's deployment requirement must not exceed the
saved deployment tick budget; execution fails otherwise.

YAML declarations retain their constructor options. Direct Python attacks describe their inputs
through `configuration_parameters()`. An undescribable value produces a warning and marks
parameters unverified; loading or evaluating that preparation is then rejected.

Reevaluation checks the saved identity and uses saved metadata and maps without accessing the
external metadata file or the dataset. Workflow details are in
[Experiments](../guides/experiments.md); commands and exit codes are in [CLI](cli.md).
