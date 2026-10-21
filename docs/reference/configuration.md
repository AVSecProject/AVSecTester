# Configuration reference

> **Audience:** Experiment authors.
>
> **Outcome:** Construct experiments from YAML or Python.
>
> **Prerequisites:** [Architecture](../concepts/architecture.md).

## Experiment configuration

The same configuration drives CLI and Python. Dataset, simulator, driving system and attacks are
configured separately. Attack and AVStack specifications may be inline mappings or YAML file paths.
Relative filesystem arguments resolve against the YAML file containing that argument.

Every object constructor uses `type` and `options`. A name without a colon selects a built-in class.
`package.module:name` imports a user class or function. Such imports execute Python and must be
trusted. The same rule applies to nested objects, hooks and sensor transforms.

```yaml
dataset:
  type: CarlaDataset
  options:
    recipes:
      road:
        type: CarlaRecipe
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
              type: PrescribedMotion
              options:
                constant_control: {type: Control, options: {throttle: 0.35}}
            lead:
              type: PrescribedMotion
              options:
                constant_control: {type: Control, options: {brake: 1.0}}
simulator:
  type: CarlaSimulator
  options:
    client: {connect_ip: 127.0.0.1, connect_port: 2000, rate: 20, synchronous: true}
avstack: driving.yaml
attacks:
  phantom:
    type: PhantomInjectionAttack
    options: {decision_steps: 20, target_xyz: [6, 0, -0.7], max_segments: 1}
treatments:
  - {name: clean}
  - {name: attacked, attack_name: phantom}
seeds: [0]
evaluation:
  metrics: [{type: DrivingMetrics}, {type: DetectionMetrics}, {type: StopImpact}]
output: ../results/phantom
```

`driving.yaml` is a user-supplied driving specification; the complete supplied version is in
`configs/carla_phantom.yaml` at the repository root.

## Top-level keys

| Key | Meaning |
|---|---|
| `dataset` | Dataset object specification |
| `simulator` | Simulator object specification |
| `avstack` | Driving specification, inline or a file reference |
| `attacks` | Exactly one name mapped to an Attack specification or file reference |
| `treatments` | Named clean or single-attack runs with optional environment overrides |
| `seeds` | Nonempty unique nonnegative integers; default `[0]` |
| `decision_steps` | Optional positive upper bound on each selected segment's decisions |
| `environment` | Experiment base `{world, sensors}` settings |
| `hooks` | One stage-hook configuration applied to every treatment |
| `recording` | Sensor retention policy; state and commands remain recorded each decision |
| `evaluation` | Metrics and all-treatment pair comparisons |
| `output` | Required CLI output directory; optional for in-memory Python execution |
| `strict` | Stop on the first error when true; default false |

The attacks mapping must contain exactly one entry. Multiple attacks in one experiment are future
work; implement a composite Attack when several actions must be applied together. Empty and
multi-attack configurations fail loading with exit code 2. When treatments are omitted, loading
creates clean and attack treatments. Every treatment shares this attack's metadata and may enable
its attack_name and change environment settings. A clean treatment has no attack_name.

## Dataset and simulator

CARLA scene content is solely in `CarlaRecipe.scenario`. The simulator receives runtime client
settings, optional settling iterations and a depth-camera name. Recipe motion explicitly covers ego
and all other actors through `constant_control`, a control sequence or a state trajectory. A
trajectory contains the terminal state. Traffic Manager is not a prescribed motion source.

NuRec configures its dataset `time_step` and simulator `config.dt` to the same value, typically 0.1
s. Recorded trajectories are interpolated only inside their support. Changing the time step changes
segment positions. NuScenes uses RecordedReplaySimulator for open-loop playback.

See [Scenarios](../guides/scenarios.md) and [Simulators](../guides/simulators.md).

## Driving specification

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
Bindings expose components and scorers; they do not replace the AVStack. Optional avstack.vehicle_io
constructs VehicleIO from a {type, options} specification; avstack.command_modes asserts the
expected command modes and must match the AVStack declaration. These keys describe the configured
driving boundary, while options are constructor arguments. A driving configuration can be reused
across supported scenarios of one simulator.

## Attack specification

An attack specification has `type`, `options` and optional `environment`. Selection parameters,
input assets and optimization belong to the Attack constructor. The framework calls its metabuilder
for each scene, then its offline phase once. Selection, host ranking and score semantics belong
entirely to build_metadata; the framework only validates and saves AttackSegment records. It never
reruns selection after optimization.

```yaml
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
Attack in attack/implemented_attacks/<snake_case_class_name>/, exported by that folder's
__init__.py. Built-in attacks use the same discovery path as user attacks. MyAttack maps to
my_attack, MyNNAttack to my_nn_attack, XMLAttack to xml_attack and HTTP2Attack to http2_attack and
Attack2D to attack2_d. Continuous capitals form an acronym; the boundary before a following
capital-plus-lowercase word is split. Missing attacks report the available names. User code
elsewhere uses package.module:Class.

Assets, textures, placements, optimizers and compositor options also use type/options; lowercase
shortcuts and inferred placement types are rejected. Relative paths, including ImageTexture's image
option, SignAsset.face and PatchOptimization.init_texture, resolve against the containing
configuration file.

## Environment

The complete shape is `{world: {...}, sensors: [...]}`. Resolve experiment base, enabled Attack
environment and treatment overrides in that order. World dictionaries merge recursively; an
explicitly supplied sensor list replaces the preceding list. SimulatorRuntime.reset receives only
the world mapping. See the complete [environment merge
rules](../guides/environments.md#experiment-environment).

```yaml
environment:
  world: {weather: SoftRainNoon}
  sensors:
    - {type: GaussianNoise, options: {severity: 0.2}, camera: front}
```

CARLA supports the native keys declared by `CarlaSimulator.supported_world_settings`, including
weather, lighting, map_layers and road_friction; support is not limited to weather. Unsupported
world settings fail rather than becoming sensor approximations. See [Native CARLA
settings](../guides/environments.md#native-carla-settings).

## Hooks

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

## Recording

```yaml
recording:
  sensor_ids: [front]
  representations: [raw, model_input]
  every_n_steps: 2
```

Sensor payloads are sampled every N decisions. Empty sensor_ids retain no image payloads. Physical
states, commands, actions, outputs, available components and detections are recorded at each
decision. Choose payload retention before a run; reevaluation cannot recover omitted images.

## Evaluation

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

## Saved preparation compatibility

`PreparedExperiment.save(directory)` records dataset type and scenario list, simulator identity, and
each attack's type, parameters and environment. `PreparedExperiment.load(config, directory)` rejects
incompatible descriptions with ConfigurationMismatchError. The CLI maps that specific error to exit
code 2. Corrupt or unreadable saved files are result-reading failures (exit code 1). Output,
recording and evaluation can be changed for later runs.

Summaries are calculated only when saving or loading. Constructing or running a configuration does
not enumerate scenarios for this check or require parameter descriptions. YAML constructor options
retain their original description. Direct Python Attack objects use `configuration_parameters()`; if
that cannot describe inputs, saving warns and marks compatibility unverified. Loading such a
preparation fails explicitly. Type equality alone is insufficient.
