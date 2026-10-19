# Architecture

> **Audience:** Everyone. Read this before any guide.  
> **Outcome:** Understand the components, the attack workflow, resource ownership and supported paths.<br>
> **Prerequisites:** None.

## Overview

AVSecTester connects datasets, simulators, AV stacks and researcher-supplied attacks. It selects
attack segments, prepares artifacts, runs controlled treatments and records each driving decision.
Researchers can replace the attack, driving adapter, simulator or dataset through public interfaces.

The modular driving system uses [avstack](https://github.com/avstack-lab). The end-to-end path uses
NVIDIA Alpamayo. CARLA simulates worlds and traffic. NVIDIA NuRec renders reconstructed scenes,
with other actors following recorded trajectories. The framework runs its own driving loop.

## Components

### Scenario preparation

A `Dataset` supplies scenario descriptions, frame references, recorded `FrameData` or a scene
recipe, and `ScenarioStart` records. It owns no simulator. CARLA supplies recipes whose frame data
is generated into a preparation-owned cache. NuRec and nuScenes supply recorded metadata.
`FrameData.actors` uses stable IDs and canonical world poses, with velocity, category and full
(length, width, height) when available. nuScenes includes every annotation, not only actors visible
in the selected camera. Its legacy scene representation remains limited to camera-visible vehicles.

The attack's metabuilder, `build_metadata(ScenarioView)`, selects intervals and actor roles. The
experiment traverses scenarios and collects the resulting `AttackSegment` records. Selection
reasons include rejected and failed scenarios. Assets, hosts, placement and scores are optional.
An attack can select a segment using only its scenario identity and frame boundaries.

Guide: [Implement an attack](../guides/attacks.md#use-the-selected-scenario).

### Experiment

`Experiment.prepare` executes the fixed sequence of metadata selection and one offline phase per
attack. It closes preparation resources and returns a `PreparedExperiment`. Each attack's metadata
and artifacts can be saved and loaded without repeating selection or optimization.

`Experiment.execute` runs treatments grouped by attack, selected segment and seed. Treatments in
one group share the start and decision length. Every treatment gets a fresh simulator runtime,
AVStack and, when enabled, AttackRuntime. Each treatment enables at most one attack. Combinations
of attacks belong inside a user-defined attack.

Attack and driving configurations can be inline or referenced from other YAML files. Experiment
configuration selects the dataset, simulator, named attacks, treatments, seeds, environment and
recording. Environment layers resolve in order: experiment base, enabled attack, treatment override.
World settings retain their physical meaning. Unsupported world settings raise an error rather
than becoming image-space approximations.

Guide: [Experiments](../guides/experiments.md).

### Simulator

`Simulator` is reusable configuration and capability declarations. `create_runtime(seed)` creates
one `SimulatorRuntime` with a world, clock, sensors and native connections. The runtime initializes
from `ScenarioStart`, observes without advancing and advances when given a command. Its creator
closes it.

CARLA supports generated worlds and reactive traffic. Selected intermediate recipe starts require
explicit prescribed motion. NuRec renders camera views along the ego's simulated path while other
actors follow recorded tracks. `RecordedReplaySimulator` serves recorded frames for datasets such
as nuScenes. It is open-loop playback: commands do not change its recorded trajectory.

Guide: [Configure simulators](../guides/simulators.md).

### Vehicle I/O

Vehicle I/O processes captured sensors into model input and carries commands back to the simulator.
It can apply sensor transforms, localization and observation transforms. Its stages include
`sensors.post`, `localization.pre`, `localization.post`, `observation` and `command`.

Sensor-environment transforms act after attack sensor effects and before Vehicle I/O hooks. A
measurement attack reaches localization only when the selected localizer consumes that measurement.

Reference: [Runtime stages](../reference/runtime.md#stages).

### Driving system

An `AVStack` turns an `Observation` into a `Control`. The supplied modular adapter runs perception,
tracking, planning and control and produces actuator commands. Alpamayo produces a planned trajectory.
The simulator must support the command representation, sensor data and state format of the adapter.

`DrivingAdapter` holds a factory and explicit component/scorer bindings. It creates a fresh AVStack
for each run or closed-loop service evaluation. It is not a second driving runtime. Model weights
may be shared only behind a factory that preserves independent mutable driving histories.

Guide: [Configure driving systems](../guides/driving-systems.md).

### Attacks and defenses

An `Attack` defines three parts:

1. `build_metadata(scenario)` selects `AttackSegment` records, optionally with scores and ranks.
2. `run_offline(context)` returns reusable artifacts. Its default returns `assets` unchanged.
3. `create_runtime(segment, artifacts)` creates the online instance for one run.

`AttackRuntime` owns the attack's per-run history. It declares observation and physical-state
subscriptions and returns effects from `setup` and `step`. Effects change the world, a captured
sensor or an AVStack stage. Component-dependent callbacks can be bound methods of the same runtime.
The framework calls `stop` when the run ends.

Stage hooks remain available for custom processing and defense logic. The supplied concrete
attacks, defenses and CLI retain `Intervention`/`Runtime` compatibility orchestration. That path
uses the same underlying driving loop but does not automatically execute the three Attack methods.
**Not yet implemented.** A dedicated defense workflow and migration of concrete examples.

Guides: [Implement an attack](../guides/attacks.md), [Implement a defense](../guides/defenses.md).

### Optimization

The attack designs its offline objective and algorithm. `OfflineContext` provides all selected
metadata and the same `ClosedLoopServices` available to the metabuilder. Services supply sensor
captures, sensor views, attack-effect rendering, driving-component evaluation, scoring and candidate
rollout. Differentiable component/scorer capabilities are declared explicitly.

Candidate rollout uses `Attack.create_runtime`, just as formal execution does. Each candidate uses
an isolated session and fresh world, driving history and attack state. Online attacks receive no
service that can advance the active experiment. Sequential, effect-free sensor reads within one
session can reuse a scan runtime.

Existing PGD/NES patch tools remain separate concrete implementations.

**Not yet implemented.** Migration of these concrete optimizers to the complete Attack workflow.

Guide: [Optimize attack artifacts](../guides/optimization.md).

### Evaluation

`RunLog` stores decision records, physical pre/post states, model-visible inputs, produced/executed
commands, component outputs, attack-effect receipts and termination status. Persistent logs use
Parquet records and separate payload shards.

Existing evaluation utilities still accept compatibility traces and compare stops, component
outputs and robustness across conditions. The framework does not define one success criterion
for every attack.

**Not yet implemented.** Unified analysis directly from experiment RunLogs and configurable
success criteria. See [TODO](../development/TODO.md#driving-impact-metrics).

Guide: [Evaluate results](../guides/evaluation.md).

## One run, step by step

1. Create independent simulator and driving instances and initialize the selected `ScenarioStart`.
   Attack setup effects apply before the first capture.
2. Capture the current observation without advancing the world. Call `AttackRuntime.step` once.
3. Apply sensor attack effects, then sensor-environment transforms and Vehicle I/O processing.
4. Run the AVStack. At each targeted stage, apply AVStack attack effects before downstream use.
5. Record the produced command, apply command transforms and validate the executed command.
6. Apply world attack effects before advancing the simulator. Record the resulting physical state
   and effect receipts. Repeat for the requested decisions, then stop and close owned resources.

The final world state requires one frame beyond the last decision input. A segment with `N`
decisions therefore requires `N+1` frame records. The framework does not pad missing frames or
silently shorten a selected interval.

Reference: [Execution order](../reference/runtime.md#execution-order).

## Physical state and model-visible data

`WorldSnapshot` describes true ego and other-actor states. `Observation` carries the sensors,
estimated vehicle state, reported speed, calibration and clock available to the driving system.
Changing an observation never directly moves the world. World changes require simulator operations.

`OnlineContext` supplies independent snapshots of the results declared by an attack's subscriptions.
It has no simulator, AVStack or time-advancing session. Sensor effects receive a capture's SensorView,
AVStack effects receive their stage value, and world effects receive the live simulator when applied.
Stage-hook contexts retain their explicit native access for adapter-specific integrations.

Reference: [Read and write interfaces](../reference/runtime.md#read-and-write-interfaces).

## Supported combinations

| | Modular AVStack | Alpamayo |
|---|---|---|
| CARLA | Supplied path | Requires compatible camera inputs and trajectory-to-actuator conversion |
| NuRec | Requires compatible modules and sensor/state adapters | Supplied path with trajectory-following dynamics |

The interfaces do not automatically convert incompatible sensor, state or command formats.
See [Commands and compatibility](../guides/driving-systems.md#commands-and-compatibility).

## Built on avstack and AlpaSim

Git submodules under `third_party/` provide avstack-core (driving modules, geometry, sensors,
registries and hooks), lib-avstack-carla (CARLA actors and sensors) and avstack-api (dataset adapters).

The NuRec/Alpamayo path uses AlpaSim's `nre-ga` rendering service and `alpasim_driver` model adapter.
AVSecTester does not run AlpaSim's runtime, controller, physics or traffic services.

## Implementation status

| Component | Available | Remaining boundary |
|---|---|---|
| Dataset and selection | Stateless datasets, prepared CARLA frame data, attack-selected segments and reasons | Compatibility SceneGT and scenario-selection APIs remain |
| Experiment | Three configuration types, prepare/execute, saved metadata/artifacts, isolated treatments | CLI, concrete attack examples and legacy evaluation still use compatibility orchestration |
| Simulator | CARLA and NuRec adapters, sensor views, selected starts, supported world settings | NuRec traffic follows recorded tracks |
| Driving system | Modular AVStack and Alpamayo, component/scorer declarations | Alpamayo uses ideal trajectory-following dynamics |
| Attack | Fixed three-part workflow, per-run state and three effect types | Concrete attack migration and cross-platform patch example |
| Defense | Custom stage hooks and compatibility interventions | Dedicated defense workflow and reference defense algorithms |
| Optimization | Closed-loop service boundary, existing PGD/NES implementations | Concrete patch optimizer migration |
| Evaluation | RunLog and compatibility metrics/logging | Unified RunLog-based result analysis and configurable success criteria |

## Terminology

| Term | Meaning | Interface |
|---|---|---|
| Dataset | Recorded scenario data or generation recipes, without simulator ownership | `Dataset` |
| Frame reference | Position, native identifier and time of an actual frame | `FrameReference` |
| Frame data | Recorded/generated information at a frame, including stable actor states | `FrameData`, `ActorState` |
| Scenario start | Dataset information required to initialize a simulator at a frame | `ScenarioStart` |
| Metabuilder | Attack-defined selection performed once for each scenario | `Attack.build_metadata` |
| Attack segment | Selected scenario, decision interval and optional roles, placement and selection fields | `AttackSegment` |
| Metadata | All segments selected by an attack | `OfflineContext.metadata` |
| Asset | Original attack input, such as a patch texture | `Attack.assets` |
| Artifact | Offline return value reused during execution | `PreparedExperiment.artifacts` |
| Treatment | A named run enabling at most one attack, with explicit environment overrides | `TreatmentConfiguration` |
| Simulator | Reusable simulator configuration and capabilities | `Simulator` |
| Simulator runtime | One live world and its resources | `SimulatorRuntime` |
| Driving system / AV stack | The system under test, connected through an adapter | `AVStack`, `DrivingAdapter` |
| Vehicle I/O | Sensor, localization, observation and command processing | `VehicleIO` |
| Physical state | True ego and other-actor state | `WorldSnapshot` |
| Observation | Model-visible sensors, calibration, state estimate, speed and clock | `Observation` |
| Command | Actuator values or a planned trajectory | `Control` |
| Command mode | Command representation produced/accepted | `command_modes` |
| Attack runtime | One attack's online state and lifecycle | `AttackRuntime` |
| Effect | A change applied at the world, sensor or AVStack boundary | `WorldAttackEffect`, `SensorAttackEffect`, `AVStackAttackEffect` |
| Stage / hook / handler | A named processing point, its registration and the callback replacing its value | `HookSet`, `Hook`, `Plugin` |
| Environment | World settings and post-capture sensor transforms | `environment: {world, sensors}` |
| Run log | Decision records, events and payload references | `RunLog` |
| Recorded output | Named diagnostic value recorded by an attack or hook | `record_output`, `RecordedOutput` |
| Prepared scenario / match | Compatibility scenario factory and initial-selection result | `ScenarioInstance`, `ScenarioMatch` |
| Filter / requirement | Compatibility composable checks and role-selection specification | `Constraint`, `ScenarioRequirement` |
| Insertion | Explicit asset, placement and orientation used by concrete image attacks | `Insertion` |
| Intervention / Runtime | Compatibility preparation and ordered stage-hook integration | `Intervention`, `Runtime` |
| Trace / component outputs | Compatibility views used by existing evaluation | `Trace`, `ComponentTrace` |
