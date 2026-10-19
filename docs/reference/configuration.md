# Configuration reference

> **Audience:** Users writing or editing experiment configurations.  
> **Outcome:** Look up every supported configuration key, its type, default and meaning.  
> **Prerequisites:** [Run experiments](../guides/experiments.md).


## File layout

An experiment configuration is one flat mapping, written as a YAML file for the CLI or passed as
a Python `dict` to `run_configured` and the setup functions. The complete supplied examples are
[`configs/carla_scenario.yaml`](../../configs/carla_scenario.yaml) (perception-stage phantom
injection) and [`configs/carla_patch_scenario.yaml`](../../configs/carla_patch_scenario.yaml)
(physical patch on a lead vehicle).

The top-level keys fall into two groups:

- **Experiment keys** describe the treatments and how they run: `frames`, `mode`, `treatment`,
  `output`, `stack`, `attacks`, `patches`, `defenses`, `interventions`, `runtime`,
  `clean_runtime`, `conditions` and `selection`.
- **Scene keys** are every other top-level key. They describe the world, for example `client`,
  `ego`, `npcs` and `lead` for CARLA, or `backend` for NuRec.

Setup passes only the scene keys to the simulator adapter. `scene_config(config)` returns a
detached copy of the configuration without the experiment keys. Setup also removes a top-level
`weather` key from that view and treats it as `conditions.world.weather`. The resulting scene view
is the argument of `backend_factory(scene_config)` and is stored as `ScenarioInstance.config`.
`ego.pipeline` is part of the scene view, and the modular stack reads it from the full
configuration.

```yaml
# Experiment keys
frames: 40
mode: pair
treatment: attack
conditions: {}
interventions: []

# Scene keys (CARLA)
client: {type: CarlaClient, connect_ip: 127.0.0.1, connect_port: 2000}  # abbreviated
ego: {type: CarlaMobileActor, spawn: 0}                                  # abbreviated
npcs: {count: 6, npc_type: vehicle}
```

Each key is read by one part of the framework:

| Reader | Keys |
|---|---|
| `avsectester run` | `frames`, `mode`, `treatment`, `backend`, `client`, `ego`, `stack`, `selection` and `runtime.hooks`, then everything `run_configured` reads |
| `run_configured` | `mode`, `treatment`, then the keys of the readers below |
| `scenario_from_config`, `source_from_config` | Scene keys, `weather`, `conditions`, `selection` |
| `prepare_from_config` | `attacks`, `patches`, `interventions`, `defenses` |
| `configured_runtime`, `configured_clean_runtime` | `runtime`, `clean_runtime` |
| `configured_stack` | `stack`, `ego.pipeline` |

Loading a configuration imports and executes the Python factories it names. Use configurations
and factory modules you trust.

## Experiment keys

| Key | Type | Default | Meaning | Validated by |
|---|---|---|---|---|
| `frames` | int, at least 0 | `40` | Driving steps per run. Only the CLI reads this key. `--frames` overrides it. Python callers pass `frames` to `run_configured`, which ignores the key | CLI, before the experiment starts |
| `mode` | `single` or `pair` | `pair` | `single` runs the selected treatment once. `pair` runs a clean baseline and the selected treatment from one origin. `--mode` and the `mode=` keyword override it | `run_configured` |
| `treatment` | `clean`, `attack`, `defense` or `attack-defense` | `attack` | The treatment to run. See [Treatments and modes](../guides/experiments.md#treatments-and-modes). `--treatment` and the `treatment=` keyword override it | `run_configured` |
| `output` | any | none | Reserved. Setup removes it from the scene view. No code reads it | None |
| `defenses` | none | none | Not supported. Its presence raises an error. Use [`interventions`](#interventions) with `kind: defense` | `prepare_from_config` |

The remaining experiment keys are described in [Driving system keys](#driving-system-keys),
[Intervention keys](#intervention-keys), [Condition keys](#condition-keys) and
[Selection keys](#selection-keys).

At the default `client.rate: 20.0`, 40 driving steps cover 2 seconds of simulated time. For
NuRec, each driving step lasts `backend.dt` seconds.

## CARLA scene keys

The scene is a CARLA scene when `backend` is omitted or `backend.type` is `CarlaSimulator`. With
the default backend factory, `backend` may then contain only `type`. CARLA settings belong in
`client`, `ego`, `npcs` and `lead`. `CarlaSimulator` reads these four keys and the weather
condition. It ignores other scene keys. The CLI requires `client` and `ego` to be mappings.

Most values are passed to avcarla, which builds the client and actors at each reset. Errors in
these values appear when the run starts. How the settings interact with seeds, spawning and
paired runs is explained in [Configure simulators](../guides/simulators.md#carla).

### client

The `client` mapping is passed to the avcarla `CarlaClient` constructor. AVSecTester reads and
completes some of its keys during pair preparation.

| Key | Type | Default | Meaning |
|---|---|---|---|
| `type` | string | required | `CarlaClient` |
| `connect_ip` | string | required | CARLA server host |
| `connect_port` | int | required | CARLA server RPC port |
| `traffic_manager_port` | int | required | Traffic Manager port used by NPC autopilot |
| `traffic_manager_seed` | int or `null` | required | Traffic Manager seed. `null` follows `client.seed`. Pair preparation also fills an omitted value from `client.seed` |
| `synchronous` | bool | required | Fixed-step synchronous mode. Must be `true` when `reset_world` is `true` |
| `rate` | float | required | Simulation steps per second. Each driving step advances the world by `1 / rate` seconds |
| `seed` | int or `null` | `null` | Seed for random scene choices: vehicle models, random spawns, destinations, traffic lights and LiDAR noise seeds. The CLI and pair preparation replace `null` with a generated seed. A Python single run without a seed is unseeded. See [Random choices and seeds](../guides/simulators.md#random-choices-and-seeds) |
| `map_name` | string or `null` | `null` | Map to load. `null` keeps the server's current map. Pair preparation records the loaded map name |
| `weather` | mapping or `null` | `null` | `carla.WeatherParameters` attribute values applied when the client connects. Pair preparation records the world's current weather here. A [weather condition](#condition-keys) is applied later and takes precedence |
| `traffic_lights` | mapping or `null` | `null` | Per-light settings keyed by OpenDRIVE ID, each with `state`, `green_time`, `yellow_time`, `red_time` and `frozen`. Pair preparation records the lights here so that both runs start with the same lights |
| `reset_world` | bool | `true` | Reload the world at every reset. `CarlaSimulator` sets `true` when the key is omitted. Pair preparation rejects `false` |
| `strict_spawn` | bool | `false` | Disable relocation of failed spawns in the first run. Replayed spawns are always strict. See [NPCs and spawn positions](../guides/simulators.md#npcs-and-spawn-positions) |
| `randomize_lights` | bool | `true` | Randomize traffic-light states with the client seed when `traffic_lights` is not set |
| `prob_light_green` | float | `0.35` | Probability of a green light when randomizing |
| `disable_static_actors` | bool | `true` | Unload the parked-vehicle map layer |
| `substepping`, `max_substep_delta_time`, `max_substeps` | bool, float, int | `true`, `0.01`, `10` | CARLA physics substepping settings |

### ego

The `ego` mapping is passed to the avcarla `CarlaMobileActor` constructor. `CarlaSimulator`
replaces `ego.pipeline` with an empty pipeline for the CARLA actor, because the driving system
runs outside it.

| Key | Type | Default | Meaning |
|---|---|---|---|
| `type` | string | required | `CarlaMobileActor` |
| `vehicle` | string or int | required | Blueprint filter such as `vehicle.tesla.model3`, a blueprint index, or `random`. `randint`, `vehicle` and `random-vehicle` also select a random model. Pair preparation treats an omitted value as `random` |
| `spawn` | int, `random`, `randint` or mapping | required | Map spawn-point index, a random unused index, or an avcarla transform configuration. Indices depend on the map |
| `autopilot` | bool | required | Keep `false` so that the driving system controls the ego |
| `destination` | `null`, int, `random` or `randint` | required | Destination spawn index, passed to avcarla. The supplied straight-driving planner does not use it |
| `sensors` | list of mappings | required | avcarla sensor specifications, each with `type` (such as `CarlaLidar` or `CarlaRgbCamera`), `name` and sensor attributes. Pair preparation sets a missing `source_ID` to `0` and derives a missing CarlaLidar `noise_seed` from `client.seed` |
| `pipeline` | mapping | required unless `stack` is set | avstack `ModularDrivingPipeline` configuration used by the modular stack. `--gpu` sets `pipeline.perception.gpu`. See [Pipeline configuration](../guides/driving-systems.md#pipeline-configuration) |
| `reference_to_spawn` | mapping | `{type: CarlaReferenceFrame}` | avcarla offset applied to the spawn point |
| `spawn_transform` | mapping | none | Recorded spawn pose written by the runner into `Trace.replay_scenario`. Do not set it in a configuration |

### npcs

`npcs` adds background vehicles. Omit it, or set it to `null`, `[]` or `{}`, for none. The
compact form is a mapping:

| Key | Type | Default | Meaning |
|---|---|---|---|
| `count` | int | required | Number of NPCs |
| `npc_type` | string | `vehicle` | Blueprint filter or random keyword for every NPC. `vehicle` selects a random model |
| `spawn_start` | int | `1` | Spawn index of the first NPC. NPC `i` uses `spawn_start + i` |

The list form gives each NPC its own mapping, passed to the avcarla `CarlaNpc` constructor:

| Key | Type | Default | Meaning |
|---|---|---|---|
| `type` | string | required | `CarlaNpc` |
| `spawn` | int, `random` or `randint` | required | Spawn index or a random unused index |
| `npc_type` | string | `random` | Blueprint filter or random keyword |
| `reference_to_spawn` | mapping | `{type: CarlaReferenceFrame}` | avcarla offset applied to the spawn point |

### lead

`lead` spawns one stationary vehicle directly ahead of the ego in every treatment. It is scene
content, for example the host of a physical patch. The run fails if the position is blocked.

| Key | Type | Default | Meaning |
|---|---|---|---|
| `gap` | float | `9.0` | Distance in meters ahead of the ego's spawn pose |
| `lateral` | float | `0.0` | Offset in meters along the ego's right direction |
| `vehicle` | string | `vehicle.tesla.model3` | Blueprint filter. The first match is used |

Selection with `sample_scene: true` sets `gap` and `lateral` for each candidate. See
[Selection keys](#selection-keys).

### weather

A top-level `weather` key is an alternative spelling of `conditions.world.weather`. Setup moves it
into the conditions before any backend is created. If both are set and differ, setup raises an
error. Prefer `conditions.world.weather`.

## NuRec backend keys

A NuRec scene is selected with `backend.type: NuRecSimulator`.
The default backend factory accepts only the keys below under `backend`. Any other key raises an
error when the backend is constructed. Usage is explained in [NuRec](../guides/simulators.md#nurec).

| Key | Type | Default | Meaning |
|---|---|---|---|
| `type` | string | required | `NuRecSimulator` |
| `scene` | any | `null` | Scene passed to the renderer at every reset. `NuRecRenderer` accepts a scene ID or a unique substring. Without one it uses its `scene_id` option, then the first available scene. `StubRenderer` ignores it |
| `renderer` | `StubRenderer`, `NuRecRenderer` or mapping | `StubRenderer` | Renderer selection. A mapping has `type` (`StubRenderer` or `NuRecRenderer`) and the renderer's options, listed below |
| `dynamics` | mapping | `{type: KinematicBicycle}` | Ego dynamics. `type` is `KinematicBicycle` (actuator commands) or `TrajectoryFollower` (trajectory commands). The other keys are that model's options, listed below |
| `ego0` | mapping | all zero | Initial ego pose: `x` and `y` in meters, `yaw` in radians, `speed` in m/s and `t` in seconds |
| `dt` | float | `0.05` | Seconds per driving step, for both physics and observations. See [Dynamics and time step](../guides/simulators.md#dynamics-and-time-step) |
| `camera_aliases` | mapping | `{}` | Native renderer camera name to the sensor ID used in `Observation.sensor_data`, for example `{camera_front: front}` |

The NuRec backend also receives the scene view, with the `backend` entries merged over it. A
top-level key such as `dt` is therefore also read. Keys under `backend` take precedence.

| Renderer or dynamics | Options and defaults |
|---|---|
| `renderer.type: StubRenderer` | `cameras` (`[camera_front]`), `height` (`480`), `width` (`640`). Returns black images and needs no service |
| `renderer.type: NuRecRenderer` | `endpoint` (`127.0.0.1:50051`), `scene_id` (`null`), `cameras` (`[camera_front_wide_120fov]`), `calibration` (`{}`), `image_quality` (`95.0`), `start_timestamp_us` (`null`) and `start_transform` (`null`). The last two override the recorded start time and ego pose |
| `dynamics.type: KinematicBicycle` | `wheelbase` (`2.8`), `max_accel` (`3.0`), `max_brake` (`8.0`), `max_steer` (`0.6`). Accepts actuator commands |
| `dynamics.type: TrajectoryFollower` | `decel` (`4.0`), the deceleration in m/s² without a plan. Accepts trajectory commands |

NuRec backends accept no world conditions. Configured selection with a NuRec backend raises an
error. Recorded NuRec selection uses a Python `Dataset`. See
[NuRec datasets](../guides/scenarios.md#nurec-datasets).

## Driving system keys

`stack` selects the driving system. Configured entry points build a new stack for every run.

| Form | Meaning |
|---|---|
| `stack` omitted or `null` | The supplied modular stack, built from `ego.pipeline` with perception outputs instrumented |
| `stack: {factory: ..., options: ...}` | A user driving system. `factory(**options)` must return an `AVStack` |
| `stack: {type: AlpamayoAVStack, options: ...}` | The supplied Alpamayo stack, constructed as `AlpamayoAVStack(**options)` |

| Key | Type | Default | Meaning |
|---|---|---|---|
| `stack.factory` | string `package.module:callable` | none | Importable driving-system factory |
| `stack.options` | mapping | `{}` | Keyword arguments for the factory or for `AlpamayoAVStack` |
| `stack.type` | string | none | Only `AlpamayoAVStack`. This form allows only `type` and `options` |

`AlpamayoAVStack` options are `checkpoint_path` (default: the `AVSECTESTER_ALPAMAYO_CHECKPOINT`
environment variable), `camera_ids` (`[camera_front_wide_120fov]`), `device` (`cuda:0`),
`context_length` (`4`) and `output_frequency_hz` (`10`). See
[Alpamayo](../guides/driving-systems.md#alpamayo). A custom factory is described in
[Use a custom driving system](../guides/driving-systems.md#use-a-custom-driving-system).

The stack is constructed when execution starts, before the first reset. Errors raise
`ConfigurationError`. The CLI's `--gpu` option applies only to the modular stack in CARLA scenes.
With `stack` set, configure the device in `stack.options`.

## Intervention keys

`attacks`, `patches`, `interventions` and `runtime` define the interventions. `clean_runtime`
optionally replaces the baseline. Which entries a treatment enables is described in
[Treatments and modes](../guides/experiments.md#treatments-and-modes). Disabled entries are
format-checked but not imported, constructed or prepared.

At a shared stage, hooks run in this order: `attacks` and `patches`, then `runtime.hooks`, then
`interventions` in definition order. Sensor conditions surround this chain. See
[Ordering at a shared stage](runtime.md#ordering-at-a-shared-stage).

### attacks

`attacks` is a list of native avstack hooks attached to modular pipeline components. Each entry
has exactly these two keys:

| Key | Type | Default | Meaning |
|---|---|---|---|
| `stage` | `perception`, `tracking`, `planning` or `control` | required | Component whose output the hook receives. The hook runs at `<stage>.post` |
| `hook` | mapping | required | avstack `HOOKS` configuration: `type` and the hook's arguments. A fresh hook is built for every run |

Each entry becomes one attack intervention named after `hook.type`, or `<stage>_attack` without
a type. The driving system must provide the component stage, as the modular stack does. See
[Phantom injection](../examples/phantom-injection.md) for the supplied `PhantomInjection` hook.

### patches

`patches` is a list of physical panels for CARLA. All entries form one attack intervention named
`PhysicalPatchIntervention`, which deploys the panels at `simulator.setup`. The backend must provide
CARLA setup ticks. Each panel uses one setup tick. See [Physical patch](../examples/physical-patch.md).

| Key | Type | Default | Meaning |
|---|---|---|---|
| `target` | string or int | `lead` | Actor carrying the panel: a backend name (`ego`, `lead`, `npc:<index>`), an actor ID, a selected role with one actor, or an indexed role such as `hosts[0]` |
| `prop` | string | `static.prop.ironplank` | CARLA blueprint filter for the panel |
| `location` | `[x, y, z]` | `[-2.3, 0.0, 1.0]` | Panel position in meters, in the target's CARLA frame (x forward, y right, z up) |
| `rotation` | `[pitch, yaw, roll]` | `[90.0, 0.0, 0.0]` | Panel rotation in degrees relative to the target |
| `texture` | mapping | `{pattern: checkerboard, size: 256}` | Panel image, described below |
| `emissive` | bool | `false` | `true` paints the texture as self-lit, unaffected by scene lighting. `false` uses a matte material lit by the scene |
| `type`, `gap`, `lead_vehicle` | any | none | Ignored |

| `texture` key | Type | Default | Meaning |
|---|---|---|---|
| `image` | string | none | Image file. When set, the file is loaded as RGBA and resized to `size` × `size` |
| `size` | int | `256` | Texture side in pixels |
| `squares` | int | `8` | Checkerboard squares per side |
| `color_a`, `color_b` | `[r, g, b]` | `[255, 0, 0]`, `[255, 255, 255]` | Checkerboard colors |
| `pattern` | string | `checkerboard` | Descriptive only. Every texture without `image` is a checkerboard |

Preparation checks only that each entry is a mapping. Unknown keys and invalid values raise an
error when the panel is deployed during the run.

### interventions

`interventions` is an ordered list of attack and defense methods. Each entry has these keys:

| Key | Type | Default | Meaning |
|---|---|---|---|
| `factory` | string `package.module:callable` | required | Called as `factory(**options)`. It must return a method with `hooks(artifacts)` and, unless `artifacts` is given, `prepare(scenario)` |
| `options` | mapping | `{}` | Keyword arguments for the factory |
| `kind` | `attack` or `defense` | `attack` | Decides which treatments enable the method |
| `name` | nonempty string | method class name | Name recorded in provenance |
| `artifacts` | any | none | Previously exported artifacts. They are passed to the method's `import_artifacts`, if defined, and preparation is skipped |

A handler factory is not an intervention factory. To configure an existing handler, wrap it in
an intervention method. See
[Package the attack as an intervention](../guides/attacks.md#package-the-attack-as-an-intervention)
and [Package and configure a defense](../guides/defenses.md#package-and-configure-a-defense).

```yaml
interventions:
  - factory: avsectester.attacks.adaptive:SpeedFeedbackAttack
    kind: attack
    name: speed-feedback
    options: {target_speed: 5.0, gain: 0.5, max_bias: 5.0}
```

### runtime

`runtime` configures raw hooks, an optional localizer and the handler random seed. It uses the
format of `Runtime.from_config`.

| Key | Type | Default | Meaning |
|---|---|---|---|
| `hooks` | list | `[]` | Ordered raw hooks. Configured treatments enable them only in `attack` and `attack-defense` |
| `localizer` | mapping | `null` | `factory` and optional `options` for a localizer, which returns a `StateEstimate`. Every treatment, including the default baseline, constructs its own instance |
| `seed` | int or `null` | `0` | Seed of the handlers' `context.rng`. Every run starts a fresh generator from it. `null` uses fresh entropy. It does not seed the simulator or models. See [Randomness](runtime.md#randomness) |

Each `hooks` entry has these keys:

| Key | Type | Default | Meaning |
|---|---|---|---|
| `stage` | nonempty string | required | Runtime stage. Availability is checked before the first reset. See [Stages](runtime.md#stages) |
| `factory` | string `package.module:callable` | none | Called as `factory(**options)`. It returns a handler or `Plugin` |
| `options` | mapping | `{}` | Keyword arguments for `factory` |
| `avstack` | mapping | none | Native avstack `HOOKS` configuration, rebuilt for every run. Stages ending in `.pre` use the hook's input convention, other stages its output convention. Cannot be combined with `options` |

Each hook requires exactly one of `factory` or `avstack`. A `kind` key is rejected, because raw
hooks are always attack handlers in configured treatments. Declare defenses under
[`interventions`](#interventions).

```yaml
runtime:
  seed: 0
  hooks:
    - stage: perception.post
      avstack: {type: PhantomInjection, target_xyz: [6.0, 0.0, -1.5]}
    - stage: observation
      factory: my_research.attacks:SpeedSpoofing  # user module
      options: {bias_mps: 5.0}
```

### clean_runtime

`clean_runtime` has the same format as `runtime`. When the key is present, it replaces the
default baseline in every run named `clean` or `baseline`, and its hooks run in those runs. A
Python `clean_runtime` argument takes precedence over the key. `clean_runtime: null` gives a
baseline with no hooks, no localizer and seed `0`.

Without `clean_runtime`, the baseline uses `runtime.localizer` and `runtime.seed` without any
hooks.

## Condition keys

`conditions` sets the operating environment shared by every treatment. Its use is explained in
[Apply operating conditions](../guides/conditions.md#configure-conditions).

| Key | Type | Default | Meaning |
|---|---|---|---|
| `world` | mapping | `{}` | World properties passed to the backend before the first reset. CARLA accepts only `weather`. NuRec accepts none |
| `world.weather` | string or mapping | none | CARLA weather: a `carla.WeatherParameters` preset name, such as `ClearNoon` or `HardRainNoon`, or a mapping of weather parameters |
| `sensors` | list | `[]` | Ordered sensor conditions |
| `seed` | int, at least 0 | `0` | Default seed for sensor conditions. It is independent of `runtime.seed` |

A sensor condition is either a built-in image corruption or a custom handler factory:

| Key | Type | Default | Meaning |
|---|---|---|---|
| `type` | string | required for built-ins | Corruption name. See [Built-in image corruptions](../guides/conditions.md#built-in-image-corruptions) |
| `severity` | number in `[0, 1]` | `0.5` | Corruption strength. Built-ins only |
| `camera` | nonempty string | first readable RGB camera | Sensor ID to corrupt. Built-ins only |
| `factory` | string `package.module:callable` | none | Custom handler factory, called as `factory(**options)` for every treatment. See [Custom sensor conditions](../guides/conditions.md#custom-sensor-conditions) |
| `options` | mapping | `{}` | Corruption arguments for built-ins, or factory arguments |
| `stage` | `sensors.post` or `observation` | `observation` | Stage of the condition handler |
| `order` | `before` or `after` | `after` | Position relative to all intervention hooks at that stage |
| `seed` | int, at least 0 | `conditions.seed` | Seed of this condition |

`stage`, `order` and `seed` apply to both forms. A custom entry accepts only `factory`, `options`,
`stage`, `order` and `seed`.

```yaml
conditions:
  seed: 12
  world:
    weather: {cloudiness: 70, precipitation: 20, fog_density: 15}
  sensors:
    - {type: GaussianNoise, severity: 0.2, camera: front}
```

## Selection keys

`selection` enables filter-based selection of the initial scene. Without it, the configuration
becomes one prepared scenario directly. Its use is explained in
[Selection in YAML](../guides/scenarios.md#selection-in-yaml).

| Key | Type | Default | Meaning |
|---|---|---|---|
| `selection.requirement` | mapping | required | Serialized `ScenarioRequirement`, as produced by `requirement_to_dict` |
| `selection.source` | mapping | `{}` | Options of the CARLA candidate source (`CarlaScenarioBuilder`) |

`selection` accepts no other keys. Configured selection is available for CARLA scenes. For
recorded datasets, pass `dataset=` to the Python setup functions.

The requirement mapping has these keys. Their meaning is explained in
[Compose a requirement](../guides/scenarios.md#compose-a-requirement).

| Key | Type | Default | Meaning |
|---|---|---|---|
| `name` | string | required | Requirement name |
| `description` | string | `""` | Free text |
| `target` | mapping or `null` | `null` | Single-target shorthand with `category` (`vehicle`), `camera` (`front`), `select` (`nearest_ahead`, `nearest` or `largest`) and `kind` (`object` or `placement`). Cannot be combined with `roles` |
| `constraints` | list | `[]` | Filters. Each entry has `kind` and the filter's fields. See [Filters](../guides/scenarios.md#filters) |
| `roles` | mapping | `{}` | Role name to `ids` (`null`), `count` (`1`), `category` (`vehicle`), `select` (`nearest_ahead`) and `camera` (`front`). The name `victim` is reserved. See [Roles and bindings](../guides/scenarios.md#roles-and-bindings) |
| `insertions` | list | `[]` | Inserted objects, described below |
| `window` | mapping | `{frames: 1}` | Initial window. `frames` is a positive integer. See [Initial windows and providers](../guides/scenarios.md#initial-windows-and-providers) |
| `max_bindings` | int, at least 1 | `1000` | Limit on evaluated role assignments |
| `camera` | string or `null` | `null` | Camera used by filters. `null` uses the target camera, then `front`, then the only camera |

Filter `kind` values are `All`, `Any`, `Not`, `InView`, `DistanceRange`, `ImageAreaFrac`,
`ViewpointRear`, `MinVisibility`, `EgoMoving`, `ClearLaneAhead` and any filter registered with
`register_constraint`. `All` and `Any` take a `filters` list. `Not` takes one `filter` mapping.

Each insertion entry has `id`, `asset`, `placement` and an optional `orientation`. Their meaning
is explained in [Specify inserted objects](../guides/scenarios.md#specify-inserted-objects).

| Key | Type | Default | Meaning |
|---|---|---|---|
| `id` | nonempty string | required | Unique insertion ID |
| `asset.kind` | string | required | `PlaneAsset`, the only serializable asset |
| `asset.width_m`, `asset.height_m` | float | required | Asset size in meters |
| `asset.texture` | string or nested list | required | Image path, or RGBA values as integers from 0 to 255 |
| `placement.kind` | string | required | `AttachedPlacement` or `WorldPlacement` |
| `placement.host` | string | required for `AttachedPlacement` | Logical ID of the host actor |
| `placement.anchor` | string | `center` | `center`, `front_center`, `rear_center`, `left_center`, `right_center`, `top_center` or `bottom_center` |
| `placement.offset_m` | `[x, y, z]` | `[0, 0, 0]` | Host-local offset in meters |
| `placement.position_m` | `[x, y, z]` | required for `WorldPlacement` | World position in meters |
| `orientation.mode` | string | `fixed_world` | `fixed_world`, `follow_host` or `face_victim` |
| `orientation.rotation_deg` | `[x, y, z]` | `[0, 0, 0]` | Euler angles in degrees. Must be zero for `face_victim` |
| `orientation.target_offset_m` | `[x, y, z]` | `[0, 0, 0]` | Victim-local aim point. Only for `face_victim` |

The `source` mapping accepts these `CarlaScenarioBuilder` options:

| Key | Type | Default | Meaning |
|---|---|---|---|
| `sample_scene` | bool | `false` | Sample the lead vehicle's `gap` and `lateral` for each candidate. Otherwise the configured scene is the only candidate |
| `samples` | int, at least 1 | `1` | Number of sampled candidates |
| `lateral_range` | `[min, max]` | `[0.0, 3.0]` | Range of the sampled lateral offset in meters |
| `min_gap` | float | `6.0` | Minimum sampled gap in meters. Gaps are sampled from the first `DistanceRange` filter's range, or from `min_gap` to 20 meters without one. The lower bound is never below `min_gap` |
| `seed` | int | `0` | Seed for candidate sampling |
| `ego_speed` | float | `0.0` | Must be `0`. Candidates start stationary |
| `analytic_preview` | bool | `false` | Evaluate filters on an approximate offline scene prediction, without a CARLA server |

The Python-only options `prepare_scene` and `prepare_context` take callables. Setup supplies the
base scene, conditions and candidate provider itself, so the source mapping cannot set them.

## Validation rules

Configured entry points check the configuration in a fixed order. The checks before execution
starts complete before the experiment's backend is constructed. Configured selection is the
exception: it opens its own candidate worlds during setup. All checks up to execution start
complete before the first reset.

| Phase | Checks | Error raised | CLI exit code |
|---|---|---|---|
| CLI loading | YAML syntax, top-level mapping, `mode` and `treatment` values, `--plot` in single mode, `backend` mapping, CARLA `client` and `ego` mappings, `ego.pipeline` or `stack`, `--gpu` use, `frames` | Printed as `Configuration error` | `2` |
| Experiment setup | `mode` and `treatment` values, `conditions` format and seeds, sensor condition options, conflicting `weather`, `selection` format, requirement fields and filter kinds, and selection itself | `ConfigurationError` | `2` |
| Intervention preparation | `defenses`, formats of `attacks`, `patches` and `interventions`, `kind` and `name`, factory paths, import and construction of enabled methods, method preparation | `ConfigurationError` | `2` |
| Runtime assembly | `runtime` and `clean_runtime` keys, hook entries, handler and localizer factories, and the [required kinds](../guides/experiments.md#treatments-and-modes) of the treatment | `ConfigurationError` | `2` |
| Execution start, before the first reset | Backend construction from `backend`, world conditions, stack construction, stage availability, command compatibility | `ScenarioConfigurationError`, `ConditionConfigurationError`, `ConfigurationError`, `UnsupportedStageError`, `CommandCompatibilityError` | `2` |
| During runs | Values passed to avcarla, avstack and renderers, server connections, spawns, CARLA weather preset names, physical patch keys, handler exceptions | The original exception | `1` |

All listed error classes are `ValueError` subclasses. Setup, preparation and assembly wrap
`ValueError`, `TypeError`, `KeyError` and `ImportError` in `ConfigurationError`. Other exceptions
raised by user factories propagate unchanged and are execution errors in the CLI.

The backend check applies to the default backend factory. A caller-supplied `backend_factory`,
including the CLI's CARLA factory, receives the scene view without that check. The CLI exit
codes are listed in [Exit codes](cli.md#exit-codes).
