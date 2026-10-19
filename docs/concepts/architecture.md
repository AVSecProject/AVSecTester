# Architecture

> **Audience:** Everyone. Read this before any guide.  
> **Outcome:** Know the framework's components, how data flows between them, what is implemented and the terms used throughout the documentation.  
> **Prerequisites:** None.

## Overview

AVSecTester is a framework for testing attacks and defenses against autonomous-driving systems in
closed-loop simulation. Researchers supply the driving system, the scenario and the attack or
defense logic. The framework connects these parts, runs the same scenario under different
[treatments](#terminology), records what happened and compares the runs.

The framework does not reimplement simulators or driving models. The modular driving system is an
[avstack](https://github.com/avstack-lab) pipeline, the end-to-end driving system is NVIDIA's
Alpamayo model, and the simulators are CARLA and NVIDIA NuRec. AVSecTester provides the interfaces
between them, scenario preparation, attack and defense integration, and evaluation.

It does not define a universal attack-success criterion. The supplied metric scores induced and
suppressed stops. Experiments interpret results according to their own success definition.

## Components

### Scenario preparation

Scenario preparation produces the origin of an experiment: a [prepared scenario](#terminology).
A scenario comes from a dataset or a simulator scene. Setup can apply [filters](#terminology) to
select qualifying initial states, fix actor [role bindings](#terminology), declare
[insertions](#terminology) and attach [conditions](#terminology). Selection is optional. Without
selection, a configuration becomes one prepared scenario directly. Attack and defense preparation
can read the prepared scenario, for example to find the actor that carries a patch.

Guide: [Prepare scenarios](../guides/scenarios.md) and [Apply operating conditions](../guides/conditions.md).

### Simulator

The simulator owns the world: physical state, vehicle dynamics, other actors and sensor rendering.
It produces observations and executes commands. The supplied simulators are CARLA, with reactive
traffic, and NuRec, which renders reconstructed real-world scenes while other actors follow their
recorded trajectories. In code, a simulator is connected through a `WorldBackend` adapter.

Guide: [Configure simulators](../guides/simulators.md).

### Vehicle I/O

Vehicle I/O is the interface between the simulator and the driving system. Downward it carries
sensor data, the estimated vehicle state and the final model input. Upward it carries the
driving command. It makes no decisions, but attacks and defenses can act on every value that
passes through it. These values are exposed as the runtime stages `sensors.post`,
`localization.pre`, `localization.post`, `observation` and `command`.

Reference: [Runtime stages](../reference/runtime.md#stages).

### Driving system

The driving system is the system under test. It turns observations into commands. The supplied
driving systems are a modular avstack pipeline (perception, tracking, planning and control), which
outputs actuator commands, and Alpamayo, an end-to-end model that outputs a planned trajectory.
In code, a driving system is connected through an `AVStack` adapter.

Guide: [Configure driving systems](../guides/driving-systems.md).

### Attacks and defenses

Attacks and defenses share one interface. Each has two phases:

- **Preparation** runs before the experiment. It configures the method and can produce reusable
  [artifacts](#terminology), such as an optimized patch texture or a trained policy.
- **Execution** acts during each run through runtime [handlers](#terminology) registered at
  [stages](#terminology). An attack can act on the world, on Vehicle I/O or inside the driving
  system. A defense acts on Vehicle I/O or inside the driving system. Execution can be static or
  [adaptive](#terminology).

Treatments decide which attacks and defenses are enabled in each run.

Guides: [Implement an attack](../guides/attacks.md) and [Implement a defense](../guides/defenses.md).
Worked examples: [Examples](../examples/README.md).

### Optimization

Optimization searches for artifact parameters on behalf of preparation. Preparation supplies the
problem and data. The current interfaces optimize attack artifacts by rendering candidates in CARLA
or querying a driving-system component directly. Optimization returns the artifacts to preparation.

**Not yet implemented.** No optimization route scores closed-loop driving outcomes, and defenses
have no optimization interface.

Guide: [Optimize attack artifacts](../guides/optimization.md).

### Evaluation

Evaluation reads the [run records](#terminology) of an experiment: physical driving state,
executed commands, intervention outputs and provenance. The supplied tools compare a clean run
with a treated run, capture intermediate component outputs and summarize attack success across
corruption settings.

Guide: [Evaluate results](../guides/evaluation.md).

## One run, step by step

Every run uses the same loop, regardless of the simulator or driving system:

1. Handlers are initialized, and the simulator builds the world. World-setup handlers run.
2. The simulator captures sensor data. Vehicle I/O handlers process it into the model input.
3. The driving system decides a command. Handlers registered inside the driving system can
   replace component inputs and outputs.
4. Command handlers can replace the command. The simulator validates and executes it, then
   advances the world.
5. The physical state after the step is recorded. Handlers receive feedback.
6. Steps 2–5 repeat for the requested number of driving steps. Handlers are closed.

A command at one step affects only later observations. The exact order, including each stage, is
in [Execution order](../reference/runtime.md#execution-order).

## Physical state and model-visible data

The framework keeps two views of the ego vehicle separate:

- **Physical state** is the simulator's actual state. It is available to handlers as
  `context.ground_truth` and is what the run records contain.
- **Model-visible data** is what the driving system receives: sensor data, the estimated state,
  the reported speed and the clock in the `Observation`.

Changing model-visible data never moves the vehicle. A spoofed speed changes what the driving
system believes. Only the command it then issues changes the physical state. Changes to the
world itself, such as moving an actor or changing the weather, use explicit simulator operations.
See [Read and write interfaces](../reference/runtime.md#read-and-write-interfaces).

## Supported combinations

Simulators and driving systems compose only when their sensor data, state formats and command
types are compatible. The interfaces do not convert between incompatible formats.

| | Modular driving system | Alpamayo |
|---|---|---|
| **CARLA** | Supplied path. The CLI demo runs this combination | Requires a trajectory-to-actuator controller and compatible camera inputs |
| **NuRec** | Requires compatible modules and avstack sensor and state adapters | Supplied path, with trajectory-following dynamics |

See [Commands and compatibility](../guides/driving-systems.md#commands-and-compatibility).

## Built on avstack and AlpaSim

The avstack projects are included under `third_party/` as Git submodules:

- **avstack-core**: reconfigurable driving modules, geometry, sensors, configuration registries
  and hooks.
- **lib-avstack-carla** (`avcarla`): the CARLA 0.9.16 client, actors and sensors.
- **avstack-api**: KITTI, nuScenes and CARLA dataset adapters.

The NuRec and Alpamayo path uses two parts of NVIDIA [AlpaSim](https://github.com/NVlabs/alpasim):
the `nre-ga` rendering service, which serves reconstructed scenes over gRPC, and `alpasim_driver`,
which provides the Alpamayo model. AVSecTester runs its own driving loop and vehicle dynamics. It
does not run AlpaSim's runtime, controller, physics or traffic services.

## Implementation status

| Component | Implemented | Not yet implemented |
|---|---|---|
| Scenario preparation | Configured scenarios, filter-based selection for CARLA and recorded datasets, insertions, world and sensor conditions | — |
| Simulator | CARLA with reactive traffic. NuRec with recorded actor trajectories | Reactive traffic in NuRec |
| Driving system | Modular avstack pipeline. Alpamayo with ideal trajectory following | Trajectory-to-actuator dynamics for Alpamayo |
| Attacks | Shared preparation and execution interfaces. Static example attacks and one adaptive interface example | An actor-aware adaptive reference attack |
| Defenses | The same interfaces as attacks, and recorded defense outputs | Reference defenses and defense-output metrics |
| Optimization | PGD and NES over image-space and rendered patch textures | Optimization through the closed loop, and a shared interface for defenses |
| Evaluation | Stop-based driving-impact metric, component logging, robustness summaries | Selectable success criteria and a three-treatment defense comparison |

## Terminology

The documentation uses these terms consistently. Code names are given where a term corresponds to
a specific class or value.

**Experiments**

| Term | Meaning | Code |
|---|---|---|
| Experiment | Runs of one prepared scenario under shared conditions that differ only in their enabled interventions | `run_configured`, `run_prepared` |
| Run | One closed-loop execution from world reset for a requested number of driving steps | `run(...)` |
| Driving step | One decision cycle: process the observation, decide a command, execute it and advance the world. The `frames` argument counts driving steps | `frames` |
| Treatment | A named run within an experiment, defined by its enabled interventions. Configured treatments are `clean`, `attack`, `defense` and `attack-defense` (the combined treatment) | `TreatmentMode`, `runtimes` keys |
| Mode | `single` runs one treatment. `pair` runs a clean baseline and one selected treatment from the same origin | `ExecutionMode` |
| Baseline | The clean run used as the comparison reference. In a clean pair, the second clean run is named `baseline` | `"clean"`, `"baseline"` |

**Scenarios**

| Term | Meaning | Code |
|---|---|---|
| Prepared scenario | The origin of an experiment: a backend factory, an optional selection match, the scene configuration, conditions and provenance. "Scenario" means a prepared scenario unless stated otherwise | `ScenarioInstance` |
| Scenario source | An object that produces prepared scenarios, with or without selection | `ScenarioSource` |
| Selection | Optional evaluation of filters over candidate initial states, choosing qualifying scenarios and fixing role bindings | `ScenarioRequirement` |
| Filter | A check on a candidate initial state that returns pass, fail or unknown. Filters choose among states and never change them | `Constraint`, `FilterResult` |
| Requirement | The filters, roles, insertions and initial window used for selection | `ScenarioRequirement` |
| Role, binding | A named actor slot, such as `attacker`, and the fixed actor IDs assigned to it | `RoleSpec`, `binding_ids` |
| Match | The result of selection: selected scene, bindings, insertions and initial window. Handlers receive it as `context.case`, which is `None` without selection | `ScenarioMatch` |
| Insertion | An object with an explicit asset, placement and orientation, rendered into camera images | `Insertion` |
| Condition | An environment setting applied identically to every treatment of a scenario. A world condition is applied by the simulator before reset. A sensor condition is a handler at `sensors.post` or `observation` | `ScenarioConditions`, `SensorCondition` |

**Simulation and driving**

| Term | Meaning | Code |
|---|---|---|
| Simulator | The component that owns world state, dynamics and sensor rendering | — |
| Backend | The Python adapter that connects a simulator to the driving loop | `WorldBackend`, `CarlaSimulator`, `NuRecSimulator` |
| Driving system | The system under test, which turns observations into commands | — |
| Stack | The Python adapter that connects a driving system to the driving loop | `AVStack`, `ModularAVStack`, `AlpamayoAVStack` |
| Vehicle I/O | The interface between simulator and driving system: sensor data, state estimate, model input and command | Stages `sensors.post`, `localization.*`, `observation`, `command` |
| Observation | The model-visible input: sensor data, calibration, estimated state, reported speed and clock | `Observation` |
| Physical state | The simulator's actual ego state, independent of the observation | `WorldSnapshot`, `context.ground_truth` |
| Command | The driving system's output: actuator values or a planned trajectory | `Control` |
| Command mode | The command representation a backend accepts or a stack produces: `actuator` or `trajectory` | `command_modes` |
| Localizer | An optional component that estimates the vehicle state from sensor data | `Runtime(localizer=...)` |

**Attacks and defenses**

| Term | Meaning | Code |
|---|---|---|
| Intervention | An attack or a defense, with optional preparation and per-run execution | `Intervention`, `PreparedIntervention` |
| Kind | Whether an intervention is an `attack` or a `defense`. The kind decides which treatments enable it | `kind` |
| Preparation | Work done before runs that produces reusable artifacts, such as configuring, optimizing or training | `Intervention.prepare` |
| Artifact | A reusable result of preparation, such as a texture or a set of parameters | Return value of `prepare` |
| Execution | An intervention's per-run behavior, implemented by runtime handlers | `Intervention.hooks` |
| Runtime | The ordered handlers of one run, with an optional localizer and a random seed for handlers | `Runtime` |
| Stage | A named point in the driving loop where handlers receive a value and may replace it | `"observation"`, `"perception.post"`, ... |
| Hook | The registration of a handler at a stage | `Hook(stage, handler)` |
| Handler | A function or object called as `handler(value, context)` that returns the value passed downstream | Function, `Plugin` |
| Plugin | A handler with lifecycle methods for per-run state and feedback | `Plugin` |
| Static, adaptive | Execution properties. An adaptive handler changes its behavior based on information observed during the run. A static handler does not | — |

**Optimization and evaluation**

| Term | Meaning | Code |
|---|---|---|
| Optimization | A search for artifact parameters that evaluates candidates through rendering, a driving-system component or the closed loop | `avsectester.attacks.optim` |
| Run records | The physical state and executed command at each driving step, with intervention outputs and provenance | `Trace`, `FrameRecord` |
| Intervention output | A named diagnostic value recorded by a handler, such as a defense alarm | `context.record_output`, `RecordedOutput` |
| Provenance | Metadata describing how a run was produced: scenario source, treatment, conditions, seeds and prepared interventions | `Trace.provenance` |
| Component outputs | Intermediate driving-system outputs captured during a run | `ComponentTrace` |
| Verdict | The interpretation of a clean and attacked pair under the supplied stop-based criterion | `Impact.verdict` |
