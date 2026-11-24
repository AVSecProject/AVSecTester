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
selects a framework component. `package.module:Class` or `package.module:function` selects user
code. `dataset`, `simulator`, `avstack`, `attacks.<name>`, `evaluation.metrics` entries and
`observation_pipeline` can also reference YAML files. Nested constructors use the same
type/options form.
Type paths and hook imports execute Python code. Use only trusted sources.

### Component short names

| Category and location | Accepted names |
|---|---|
| `dataset.type` | `CarlaDataset`, `NuRecDataset`, `NuScenesDataset` |
| `simulator.type` | `CarlaSimulator`, `NuRecSimulator`, `RecordedReplaySimulator` |
| `simulator.options.external_driving_system.type` | `AutowareDrivingSystem` |
| `avstack.type` | `ModularAVStack`, `CameraForwardCollisionAVStack`, `AlpamayoAVStack`, `RuleBasedAVStack`, `ExternalAVStack` |
| `evaluation.metrics[].type` | `DrivingMetrics`, `DetectionMetrics`, `TrajectoryDeviation`, `StopImpact`, `StopLineMetrics`, `ReasoningPhraseRate` |
| `attacks.<name>.type` | Discovered `Attack` subclasses in [class-name-derived directories](../../avsectester/implemented_attacks/README.md) |
| `environment.sensors[].type` | `GaussianNoise`, `ShotNoise`, `Fog`, `Rain`, `Snow`, `Brightness`, `LowLight`, `Contrast`, `Gamma`, `ColorTemp`, `MotionBlur`, `DefocusBlur`, `JPEGCompression`, `ChromaticAberration`, `AlbumentationsCorruption` |

Nested `type` mappings construct the registered component short names
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
model weights. Supplied simulator configurations use the same structure. See the
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
    every_n_captures: 1
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
is separate. See [automatic local services](cli.md#automatic-local-services).

## Dataset

`dataset` selects the scenario source. `NuRecDataset` accepts either `dataroot`, searched
recursively for USDZ archives, or an explicit `usdz_paths` sequence. The two options are
mutually exclusive. Dataset behavior and semantic maps are described in
[Datasets](../guides/datasets.md).

| Component | Option | Default or requirement |
|---|---|---|
| `NuRecDataset` | `time_step` | `0.1` seconds |
| `NuRecDataset` | `camera_id` | `camera_front_wide_120fov` |
| `NuRecDataset` | `vehicle_labels` | `[automobile, car, truck, bus, trailer, van, vehicle]` |
| `NuScenesDataset` | `dataroot` | Required dataset root |
| `NuScenesDataset` | `version` | `v1.0-trainval` |
| `NuScenesDataset` | `camera` | `CAM_FRONT` |
| `CarlaDataset` | `recipes` | Required mapping from names to `CarlaRecipe` |
| `CarlaDataset` | `map_directory` | `null` |

Each `CarlaRecipe` requires `scenario`, `decision_steps`, `time_step` and `motion`.
Its `metadata` defaults to an empty mapping.

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
saved segment frames. Other checks follow
[Input validation](../guides/experiments.md#input-validation).
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
| `execute` | `[test]` | Unique subset of `train`, `validation` and `test`. An empty list executes no units |

Unknown keys are rejected. Executing validation requires a positive validation count. Training
plus validation must be smaller than the number of units, leaving at least one test unit.

A unit matches on scenario, start, stop and all `user_data`, encoded as canonical JSON. Units
follow dataset scenario order, then start, stop and configured attack order. A seeded NumPy
permutation assigns training units first, validation units next and test units last. Identical
selections share a part. The same window with different `user_data` forms separate units.
Training and test may therefore contain the same captured scene.

`OfflineContext.metadata` contains training segments and `validation_metadata` contains
validation segments, each in selection order. `PreparedExperiment.metadata` keeps all selections
and `units` records the part and selecting attacks. `split.execute` affects execution, not
preparation compatibility.

## Scheduling

`runs.timing` uses integer microseconds and AlpaSim parameter names and timing semantics.
Defaults use one frame per sensor per decision, instantaneous camera exposure, no intermediate
pose reports, no planning delay and no recorded-motion period.

`TimingConfiguration.sensors` copies the caller's mapping in insertion order and is read-only.
Read values by key, `.get()` or `.items()`. Modification raises `TypeError`. Construct a new
configuration to change timing.

| Key | Default | Meaning |
|---|---|---|
| `decision_period_us` | Simulator period | Decision interval, equal to the dataset grid |
| `sensors.<id>.frame_interval_us` | Decision period | Sensor frame interval |
| `sensors.<id>.shutter_duration_us` | `0` | Camera exposure duration |
| `pose_reporting_interval_us` | `0` | Intermediate controller pose interval |
| `planner_delay_us` | `0` | Command-buffer delay |
| `force_gt_duration_us` | `0` | Initial recorded-motion duration |
| `skip_driver_during_force_gt` | `false` | Skip driving inference during recorded motion |

Frame intervals must be positive. Durations and delays must be nonnegative. The decision
period must equal the dataset grid, so a segment's decision count equals its stop position
minus its start position. See [Runtime](runtime.md#one-decision) for frame, pose and decision
ordering, and [Simulators](../guides/simulators.md) for capture behavior.

| Simulator | Timing capabilities |
|---|---|
| NuRec | Independent frame intervals, camera exposure and pose reports. LiDAR accepts no exposure setting |
| CARLA | Frame/report intervals, planner delay and `force_gt` duration must align with physical ticks. Exposure is zero |
| Recorded replay | Default timing on nominal keyframes only. An explicit decision period must equal the nominal interval |

Unsupported settings fail during configuration. `force_gt_duration_us` must be a whole number
of decision periods and requires a recording covering that interval. When the duration is
positive, simulators without `trajectory` in their declared `command_modes` reject
`skip_driver_during_force_gt: true`.

```yaml
runs:
  timing:
    decision_period_us: 100000
    pose_reporting_interval_us: 10000
    planner_delay_us: 100000
    force_gt_duration_us: 500000
    skip_driver_during_force_gt: false
    sensors:
      camera_front_wide_120fov:
        frame_interval_us: 33000
        shutter_duration_us: 17000
      lidar_gt_top_p128:
        frame_interval_us: 100000
```

This fragment keeps the decision grid at 100 ms and configures independent camera and LiDAR
clocks.

**Not yet implemented.** Camera phase from the recording. See
[Recorded camera phase](../development/TODO.md#recorded-camera-phase).

## Simulator

CARLA scene and prescribed motion settings belong to `CarlaRecipe`. `CarlaSimulator` receives
the runtime client settings. For recorded sources, choose a simulator compatible with the
dataset. [Simulators](../guides/simulators.md) describes their behavior.

A simulator's top-level `time_step` parameter inherits `dataset.time_step` when omitted and when
the dataset provides it. This injection does not recurse into nested constructors. Explicit
values are retained, but `ExperimentConfiguration` rejects a dataset/simulator mismatch.
The loader also injects `decision_period_us` from that grid into the simulator's top-level
constructor when it declares that parameter. Explicit options take precedence.

### CARLA settings

`client.rate` sets the native physics base tick, not the driving decision frequency. It must
produce an integer number of microseconds per tick. The decision period must be an integer
number of base ticks. `runs.timing.sensors` selects acquisition ticks.
See [CARLA capture timing](../guides/simulators.md#carla) for native sensor settings and LiDAR
scans.

The physics substep options belong to `CarlaSimulator.client`. A recipe's `client` may not
set them.

| Client option | Native default | Accepted values |
|---|---|---|
| `connect_ip` | required | Server host, typically `127.0.0.1` |
| `connect_port` | required | RPC port, typically `2000` |
| `traffic_manager_port` | required | Traffic Manager port, typically `8000` |
| `rate` | `20.0` | Positive native ticks per second |
| `synchronous` | `true` | Must be `true` |
| `seed` | `null` | Nonnegative native CARLA seed. Preparation assigns one if omitted |
| `traffic_manager_seed` | `seed` | Optional nonnegative Traffic Manager seed |
| `substepping` | `true` | Boolean |
| `max_substep_delta_time` | `0.01` | Positive finite seconds |
| `max_substeps` | `10` | Positive integer |

CARLA initialization and external-control options are:

| Option | Default | Accepted values or role |
|---|---|---|
| `simulator_version` | `"1.0"` | Participates in saved-preparation compatibility |
| `settle_iters` | `100` | Nonnegative polling iteration budget for native sensor delivery |
| `depth_camera` | `null` | RGB camera selected for aligned depth |
| `occlusion_tolerance_m` | `0.1` | Nonnegative occlusion tolerance in meters |
| `camera_warmup_s` | `4.0` | Nonnegative RGB-camera warmup duration |
| `ground_settle_s` | `2.0` | Nonnegative duration for vehicles to settle before holding |
| `external_startup_s` | `0.0` | Nonnegative fixed hold duration for an external system without a lifecycle plugin |
| `ego_control` | `"framework"` | `"framework"` or `"ros2"` |
| `realtime_factor` | `null` | Positive simulation-time to wall-time ratio on the cumulative pacing schedule, required for `"ros2"` |
| `external_driving_system` | `null` | Lifecycle plugin for `ego_control: ros2` |

The three durations must each correspond to an integer number of ticks at `client.rate`.
See [CARLA tick pacing](runtime.md#carla-tick-pacing) for catch-up and resynchronization.
RGB-camera warmup is omitted when the recipe has no RGB camera. The hold uses the larger of the
applicable camera warmup and `external_startup_s`. An `external_driving_system` then extends it
until its session is ready, so it cannot be combined with `external_startup_s`. See
[CARLA initialization](../guides/simulators.md#carla).

In `"ros2"` mode, use `ExternalAVStack` and set the ego blueprint's `role_name` to
`"hero"` or `"ego"`. `runs.timing.force_gt_duration_us` must be zero. See
[native ROS 2 control](../guides/simulators.md#native-ros-2-control).

Recipe ego and sensor entries accept `blueprint_attributes`, a mapping from native attribute
names to scalar values, converted to native strings. Unknown native attributes raise an error.
The spawned ego's attributes must match the configured values. Sensor blueprint keys that match
constructor options are rejected. Set those values as sensor options so they also set calibration.
Sensor entries also accept `ros: true` to enable their ROS 2 publication after spawning.

```yaml
ego:
  blueprint_attributes:
    role_name: hero
    ros_name: ego
    ros2_ackermann_control: "true"
  sensors:
    - type: CarlaLidar
      name: lidar-0
      blueprint_attributes:
        ros_name: lidar
      ros: true
```

Set a camera's `post_process_profile` through its `blueprint_attributes` mapping. The framework
does not choose a profile when the attribute is omitted.

#### Hosted CARLA driving systems

`scenario.ego_goal` supplies `x`, `y` and `yaw_deg` in the framework world frame. It uses meters
and degrees. These values are already in Autoware map coordinates.

`scenario.ego.ros_rig` lists attached actors separate from framework sensors. Each entry declares
a unique `name`, a `blueprint` and a `parent` of `ego` or an earlier entry. Its `transform`
contains `location: {x, y, z}` and `rotation: {pitch, yaw, roll}` in native CARLA coordinates
relative to the parent. Use `blueprint_attributes` for native attributes, including
`sensor_tick`, and `ros: true` to enable ROS publication.

##### ROS rig sensor noise

CARLA's own noise attributes add noise to `ros_rig` sensors before CARLA publishes their
messages, so no forwarding is involved. Set them in the sensor's `blueprint_attributes`.

| Blueprint | Noise attributes | CARLA default |
|---|---|---|
| `sensor.other.imu` | `noise_accel_stddev_{x,y,z}`, `noise_gyro_stddev_{x,y,z}`, `noise_gyro_bias_{x,y,z}` | No noise |
| `sensor.lidar.ray_cast` | `noise_stddev` and `dropoff_{general_rate,intensity_limit,zero_intensity}` | Dropoff enabled |
| `sensor.other.autoware_gnss` | `noise_{lat,lon,alt}_{bias,stddev}` | No noise |

IMU accelerometer noise uses meters per second squared and gyro noise and bias use radians
per second. LiDAR `noise_stddev` is a range deviation in meters. Biases follow CARLA's sensor
axes, and the published ROS axes can reverse their sign. The supplied configuration sets the
LiDAR dropoff rates to zero. The published IMU and GNSS covariances stay zero with noise
enabled. `sensor.other.autoware_gnss` accepts its noise attributes, but its published pose
does not change. To perturb GNSS, add `/sensing/gnss/pose_with_covariance` to
`modified_topics` and change it with a hook.

When a sensor's blueprint has `noise_seed` and the entry omits it, preparation derives the seed
from the scenario seed, as for framework LiDAR sensors. Treatments and repeats of one run seed
share the derived seed. Each value in `runs.seeds` derives different noise seeds.

Configure `AutowareDrivingSystem` under `external_driving_system`:

| Option | Default | Meaning |
| --- | --- | --- |
| `image` | Pinned Autoware 1.9.0 image | Local Docker image. See [Installation](../getting-started/installation.md#hosted-autoware) |
| `gpu` | required | Host GPU index |
| `ros_domain_id` | required | Domain used by the CARLA server, an integer from 0 through 232 |
| `autoware_data_directory` | required | Writable model directory on the Docker host |
| `assets_directory` | required | Extracted official CARLA Autoware assets |
| `dds_directory` | required | Directory containing `cyclonedds.xml` |
| `work_directory` | required | Per-run logs and traffic signal files |
| `launch_arguments` | `vehicle_model: sample_vehicle`, `sensor_model: awsim_sensor_kit`, `perception_mode: lidar` | Autoware launch arguments to override or extend |
| `modified_topics` | `[]` | Absolute names from the [modifiable topic list](../guides/simulators.md#hosted-ros-topics) |
| `observed_topics` | `[]` | Absolute names of topics to subscribe to for read-only processing |
| `startup_timeout_s` | `300.0` | Positive readiness timeout in wall seconds |
| `engage_timeout_s` | `10.0` | Positive autonomous-mode timeout in simulation seconds |
| `localization_tolerance_m` | `0.5` | Maximum planar localization error in meters |
| `localization_tolerance_deg` | `5.0` | Maximum localization heading error in degrees |
| `distortion_check_s` | `12.0` | Diagnostic check window in wall seconds |

Launch argument names are identifiers and values are scalars. Booleans become lowercase strings.
The plugin fixes `simulator_type`, `launch_simulator_interface`, `rviz` and `map_path`.
You cannot supply an argument used to remap a selected topic.
Preparation compatibility includes the image, launch arguments, configuration override version,
check thresholds and timeouts. When topics are configured, it also includes both topic lists
and the producer rewrite version. The lists must be disjoint and require paced execution.
Each observed topic must resolve to a single message type at startup. See
[hosted ROS topics](../guides/simulators.md#hosted-ros-topics).

`modified_topics`, `observed_topics` and `recording.topics` require absolute ROS topic names.
Each slash-separated component starts with an ASCII letter or underscore and contains only
ASCII letters, digits and underscores.

### NuRec closed-loop settings

`NuRecSimulator` requires a renderer and a non-null `time_step`. The loader supplies the latter
from `NuRecDataset`. Reset requires `ScenarioStart`. Constructor defaults are:

| Option | Default | Accepted values or role |
|---|---|---|
| `dynamics` | `KinematicBicycle()` | Ego-dynamics implementation |
| `camera_aliases` | empty | Native camera ID to observation/sensor-view name. Output names must be unique |
| `depth` | `mesh` | `mesh`, `none` |
| `route_generator` | `map` | `map`, `recorded`, `none` |
| `physics_update_mode` | `ego_only` | `ego_only`, `none` |
| `ground_mesh` | `mesh` | `mesh`, `mesh_ground` |
| `object_insertion` | `composite` | `composite`, `scene`. Scene insertion requires `renderer.asset_directory` |
| `object_occlusion` | `rendered_probes` | `rendered_probes`, `geometry`. Applies only to composite insertion |
| `occlusion_tolerance_m` | 0.15 | Nonnegative scene occlusion tolerance in meters. Applies only to composite insertion |
| `traffic` | `null` | Optional `AlpaSimTraffic` |
| `min_track_duration_s` | 3.0 | Nonnegative moving-track duration threshold. A value of 0 disables filtering |

`nurec_patch.yaml`, `nurec_speed_feedback.yaml` and `nurec_object_insertion.yaml` use
`ground_mesh: mesh` and the default mesh depth. The four stop-sign configurations use
`ground_mesh: mesh_ground` and `depth: none`. All eight supplied NuRec configurations select
`route_generator: map`, `physics_update_mode: ego_only` and `AlpaSimController` explicitly.

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
        lidar: lidar_gt_top_p128
        asset_directory: ../results/nre-assets
    dynamics:
      type: avsectester.closed_loop.simulators.ego_dynamics:AlpaSimController
```

NuRec cameras render at their native resolution.

| `NuRecRenderer` option | Default |
|---|---|
| `endpoint` | `127.0.0.1:50051` |
| `scene_id` | `null` |
| `cameras` | `[camera_front_wide_120fov]` |
| `image_quality` | `95.0` |
| `track_padding_m` | `[1.0, 1.0, 0.25]` |

`NuRecRenderer.lidar` defaults to `null`. Set it to the scene's logical LiDAR sensor name to
collect scans. See [NuRec LiDAR](../guides/simulators.md#lidar) for timing and sensor constraints.

`NuRecRenderer.asset_directory` defaults to `null`. It is the client-side directory for generated
NRE assets, including occlusion probes and scene objects. Its path resolves relative to the YAML
file. `server_asset_directory` also defaults to `null`, using the same absolute path as the
client. Set this unchanged server-side string only when connecting to an externally managed NRE
whose asset mount has a different path. See
[automatic local services](cli.md#automatic-local-services) for startup and mounts.

For insertion behavior, see
[Inserted world objects](../guides/simulators.md#inserted-world-objects).

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

Traffic defaults to `null`. Its fixed traffic step is 0.1 s. Decision periods must be whole
multiples of that step. Selection requires `history_steps - 1` preceding recorded decisions.
Traffic configuration affects preparation identity. `metadata.load` can
share segments between modes. Services are managed by [run](cli.md#automatic-local-services).

## Mesh assets

An attack option can construct a `MeshAsset` with `load_mesh_asset`. This example assumes a
user-provided attack with an `asset` option:

```yaml
attacks:
  vehicle:
    type: my_attacks:MeshInsertionAttack
    options:
      asset:
        type: avsectester.rendering:load_mesh_asset
        options:
          path: ../models/vehicle.glb
          category: vehicle
          scale: 1.0
          forward_axis: "+z"
          up_axis: "+y"
          origin: bottom_center
```

`path` is relative to the YAML file. Supported suffixes are OBJ, glTF/GLB, mesh PLY and STL,
case-insensitively. OBJ materials and textures, and glTF buffers and textures, are resolved
relative to the model file.

| Option | Default | Meaning |
|---|---|---|
| `category` | required | Nonempty category recorded for the object |
| `scale` | 1.0 | Positive finite conversion from file units to meters |
| `forward_axis` | `"+x"` | File axis mapped to forward |
| `up_axis` | `"+z"` | Orthogonal file axis mapped to up |
| `origin` | `bottom_center` | Recenter the bounding box horizontally and place its bottom at zero |
| `color` | `null` | RGB or RGBA fallback required only when the file has no colors |

Axis names are `±x`, `±y` or `±z`, with right-handed file coordinates.
For glTF files with +Y up and +Z forward, use `up_axis: "+y"` and `forward_axis: "+z"`.
The loader does not infer axes. With `origin: file`, the transformed file origin is retained,
including a model's ground clearance, and vertices below z = 0 are rejected.
Supplying `color` for an already colored file is an error.

**Not yet implemented.** FBX and USD loading. See
[TODO](../development/TODO.md#fbx-and-usd-assets).

**Not yet implemented.** 3DGS Gaussian PLY assets. See
[TODO](../development/TODO.md#gaussian-assets).

## Driving specification

`avstack.type` selects an `AVStack` class or factory. Its options are passed to each fresh
instance. A constructor declaring `seed` receives the run seed. Explicitly writing that reserved
option is an error. The top-level device follows the [device rule](#device).

| Key under avstack | Meaning |
|---|---|
| `type` | Required `AVStack` class or callable |
| `options` | Constructor arguments, empty by default |
| `observation_pipeline` | Optional `ObservationPipeline` type/options specification |
| `command_modes` | Optional assertion matching the stack's declared modes |
| `bindings` | Optional binding factory, with type/options |

History requirements, adapter precedence and instance consistency are defined in
[Recorded input history](../guides/driving-systems.md#recorded-input-history).
`ClosedLoopServices` rejects a driving-system period that differs from the simulator period.
Alpamayo context selection is described in [Driving systems](../guides/driving-systems.md#alpamayo).

A binding has signature `function(avstack_factory, **options) -> DrivingAdapter`. The factory
exposes `avstack_type` and an independent read-only `avstack_options` mapping. Bindings declare
components and scorers without constructing the stack.
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
and exactly one of `type` or `native_hook`. For hosted topics, `stage` is the original
absolute ROS topic name. A type-based hook accepts `options`. Optional
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

Custom metrics use `package.module:Class`. See
[Experiment configuration](#experiment-configuration).

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
| `repeats` | `1` | Positive integer. Values above one require simulator support, currently CARLA. See [Repeating treatments](../guides/experiments.md#repeating-treatments) |
| `decision_steps` | `null` | Optional positive upper bound per selected segment |
| `environment` | empty | Shared world and sensor settings |
| `recording` | defaults below | Sensor payload retention |
| `output` | `null` | Output directory, required by the CLI |
| `strict` | `false` | Stop on the first execution or selection error |
| `timing` | default clocks | [Scheduling](#scheduling) rates and delays |

Treatments can be a sequence of entries with `name`, or a mapping keyed by treatment name.
Attack and treatment names must be nonempty, contain no path separators and differ from
`.` and `..`. Treatment names must be unique and must not have the form `repeat-<n>`, which
names [repeat directories](cli.md#output-layout). A treatment without `attack_name` has no
enabled attack. Otherwise that field must reference a configured attack. Group membership is defined
in [Evaluation](../guides/evaluation.md#comparison-groups).

### Environment

`TreatmentConfiguration`, `RunsConfiguration`, `RunRequest`, `CandidateRequest`, `SensorRequest`
and `RenderRequest` copy their environment mappings during construction. Treat the resulting
mappings as read-only and construct a new object to change them. The framework does not enforce
this convention. Direct Python construction of `OfflineContext` retains its supplied mapping,
and assigning `Attack.environment` retains that mapping. Do not modify either after use. YAML
attack environments are copied when loaded.

CARLA world settings accept `weather` as a preset name or parameter mapping, `lighting`,
`map_layers` and `road_friction`. NuRec and recorded playback reject every nonempty
world setting.

`road_friction` requires a positive `friction_force_multiplier` and accepts an optional `actors`
list of stable actor IDs. The multiplier sets each selected vehicle wheel's native friction
force multiplier. Omitting `road_friction` preserves each wheel's vehicle-blueprint value.
Unknown keys are rejected.
A nonempty `map_layers` mapping is rejected on maps without layers, including `Town10HD_Opt`.

Environment YAML has `world` and `sensors` keys. The base is `runs.environment`, with optional
attack and treatment overrides. [Environments](../guides/environments.md) describes merge and
validation behavior. Sensor entries use `type`, `options` and an optional `camera`. Omitting
camera binds to the first initial `sensor_data` entry, which must be a readable RGB camera.

```yaml
runs:
  environment:
    world: {weather: SoftRainNoon}
    sensors:
      - {type: GaussianNoise, options: {severity: 0.2}, camera: front}
```

Custom corruptions use `package.module:Class`. See
[Experiment configuration](#experiment-configuration). Severity defaults to 0.5 and is clipped to
[0, 1]. `ColorTemp` also accepts `warm`, which defaults to `true`. `AlbumentationsCorruption`
requires `transform`. The [Python API](python-api.md) lists the available corruption classes.

### Recording

Python `RecordingPolicy` values are immutable and hashable.

| Key under recording | Default | Meaning |
|---|---|---|
| `sensor_ids` | [] | Unique sensor names whose payloads are retained |
| `representations` | [`model_input`] | Subset of `raw` and `model_input` |
| `every_n_captures` | `1` | Save every Nth acquisition of each selected sensor |
| `topics` | `[]` | Original names of modifiable or observed ROS topics whose payloads are saved |

Empty `sensor_ids` retains no sensor payloads. State, commands, actions, outputs, available
components and detections are recorded every decision. Reevaluation cannot recover omitted
sensor payloads. With an output directory, recorded sensor payloads stay on disk. Logs in
experiment results then have empty observation `sensor_data`, and `RunLog.read(path)` loads it.

`topics` accepts names from `modified_topics` and `observed_topics` on the hosted driving system.
Configured topics outside this list keep metadata only. See the
[message record](runtime.md#runlog) for their saved fields.

```yaml
runs:
  recording:
    sensor_ids: [front]
    representations: [raw, model_input]
    every_n_captures: 2
```

## Constructor filesystem parameters

Filesystem options follow constructor annotations for dataset, simulator, `AVStack`, attack, hook,
metric, observation pipeline, binding, corruption and nested constructors. `Path` or `os.PathLike`
annotations mark paths, including unions and sequence elements such as `str | Path` or `list[Path]`.

Relative values become absolute strings resolved beside the YAML file that contains them. A
leading `~` is expanded. Absolute paths preserve their location. URI strings containing `://`
are unchanged. Plain `str`, unannotated parameters, `Any`, `**kwargs` and untyped mappings do
not trigger path resolution. Unresolvable annotations mentioning Path are errors. Make their
names available at runtime. Other unresolved annotations are ignored.

Component YAML references and `runs.output` also resolve relative to their containing YAML file.

## Saved preparation compatibility

The saved identity includes dataset type/scenario IDs, simulator identity and each attack's
type, parameters and environment. Its preparation record includes the first run seed, split
counts/seed, merged offline environments, `runs.timing` and the driving history requirement.

Loading a preparation checks these values and recomputes saved units under the configured split.
A current metadata file must have the saved raw-byte hash, but its path may differ. A mismatch
raises `ConfigurationMismatchError`. Saved preparations require an offline report for every
completed attack, including attacks whose offline phase is a no-op.

Additional `seeds`, `treatments`, `repeats`, `recording`, `output`, `strict`, decision limits,
`split.execute` and evaluation settings may change. The first seed, partition and offline
environments must match. `run --prepared` writes the loaded preparation record, units, deployment
budget and metadata source unchanged into the new output. Driving-system options that change its
history affect compatibility through that declaration. Other `AVStack` and hook settings are not
compared. Current preparations require matching timing and driving history declarations. Legacy
preparations without these declarations require default timing and recheck saved starts with
`can_start_at` under the current driving history. `Attack` constructor options are compared as
written, including an explicitly written default versus an omitted option. Lists and tuples with
identical contents compare equally. Booleans, integers and floating-point numbers remain distinct,
so `true`, `1` and `1.0` are incompatible. The current hooks' setup ticks plus the attack's
deployment requirement must not exceed the saved deployment tick budget. Execution fails otherwise.

YAML declarations retain their constructor options. Direct Python attacks describe their inputs
through `configuration_parameters()`. An undescribable value produces a warning and marks
parameters unverified. Loading or evaluating that preparation is then rejected.

Reevaluation checks the saved identity and uses saved metadata and maps without accessing the
external metadata file or the dataset. Workflow details are in
[Experiments](../guides/experiments.md). Commands and exit codes are in [CLI](cli.md).
