# Interfaces

AVSecTester is two roles joined by a **pure data plane**. A **world backend** senses and actuates. An
**AV stack** (the driving system under test) turns observations into commands. Attack and defense
handlers modify supported runtime stages. Scenario preparation supplies the case and shared
operating conditions before that loop. Selection is optional.
Simulator and driving-model implementations come from avstack, avcarla and NVIDIA services,
connected through this repository's adapters.

## 0. The spine: data plane + interfaces

The data and base interfaces have no simulator or neural-model imports.

| Interface | Contract |
|---|---|
| `Observation` | Model-visible `sensor_data`, `calibration`, `vehicle_state`, `ego_speed`, `t` and `frame` |
| `Control` | `throttle`, `steer`, `brake` and optional `trajectory` |
| `WorldSnapshot` | Independent physical-state snapshot: `t`, `frame`, `vehicle_state`, `ego_speed` |
| `StateEstimate` | Localization output: `vehicle_state`, `ego_speed` |
| `WorldBackend` | `reset() -> Observation`, `step(control) -> Observation`, `ground_truth() -> WorldSnapshot` |
| `AVStack` | `reset(observation)` and `__call__(observation) -> Control` |
| `FrameRecord` | Physical post-step state and executed command telemetry |
| `Trace` | Driving records, named intervention outputs, replay settings and provenance |

`backend.copy_observation(observation)` detaches mutable model-visible data from backend state.
Changing the observation's state estimate does not move the physical ego. World changes are
explicit backend/native operations.
See the [read and write interfaces](INTERVENTIONS.md#read-and-write-interfaces) for runtime
measurement and truth access, and [world changes](INTERVENTIONS.md#world-changes-and-custom-adapters)
for backend-specific physical-state writes.

`run(backend, stack, frames, perturb=None, on_step=None, *, runtime=None) -> Trace` owns the
execution order. `record_run(..., runtime=...)` and `run_logged(..., runtime=...)` use the same loop.
The CARLA/modular convenience entry point `run_scenario(..., runtime=...)` forwards that runtime.

```text
reset handlers → reset world → process initial observation → stack.reset(processed input)

for each driving step:
    stack(processed input), including supported component hooks
    → command handlers → on_step → backend.step
    → record physical state → handler feedback
    → process next observation if another driving step remains

close handlers
```

Observation processing includes sensor hooks, optional localization, estimated-state hooks and
final input handlers. The initial input is processed once before stack initialization. `on_step`
receives copied observation and control values and cannot change the applied command through
those copies. Native CARLA captures remain borrowed as documented in
[INTERVENTIONS.md](INTERVENTIONS.md#truth-observations-and-native-resources). The caller closes
backend resources.

**Causal invariant:** control at *t* affects subsequent observations. Records use the backend's
physical state, even when the stack sees spoofed speed, pose or clock metadata.

See [INTERVENTIONS.md](INTERVENTIONS.md) for stage payloads, lifecycle, context, native access
and custom adapter requirements. `perturb(Observation) -> Observation` remains a shorthand for
final observation processing.

### Execution entry points

| Entry point | Use |
|---|---|
| `run(backend, stack, ..., runtime=...)` | One run with caller-owned backend resources |
| `run_pair(backend, stack_factory, ..., clean_runtime=..., attacked_runtime=...)` | Two runs from one origin, with a fresh stack per run |
| `run_prepared(scenario, stack_factory, ..., runtimes={name: runtime})` | Named treatments of a `ScenarioInstance`, including shared conditions and backend cleanup |
| `run_configured(config, ..., scenario=None, mode=..., treatment=...)` | Prepare enabled interventions and execute selected treatments. Use the supplied scenario or prepare one from the configuration |

`scenario_from_config(config)` returns a `ScenarioInstance`. Without selection it wraps one
configured case without opening a preview world, and its `match` is `None`. With selection it
prepares and checks candidate scenes. `scenario.match` is the selected `ScenarioMatch`, while
Runtime's `context.case` holds that optional match. Intervention preparation receives the complete
scenario. See [SCENARIOS.md](SCENARIOS.md#prepare-configured-scenarios) for the field contracts.

`run_prepared` runs once for one named treatment and shares the starting scene and actor setup
when several are supplied.
Its Runtime factories receive the backend before reset. Live camera/actor initialization belongs
in lifecycle callbacks or lazy transforms. Non-None Runtime cases are preserved. Otherwise the scenario match is supplied.
Stateful reused handlers must reset their state, or their factories must construct fresh handlers.
`run_prepared` returns results under the names supplied in `runtimes`. With `runtimes=None`, it
returns `{"clean": trace}`. `run_configured` adds preparation and treatment selection:

| Mode and treatment | Result keys |
|---|---|
| Single | The selected treatment: `clean`, `attack`, `defense` or `attack-defense` |
| Pair with attack, defense or combined treatment | `clean` and the selected treatment |
| Pair with clean treatment | `clean` and `baseline` |

These functions return recorder results without computing a metric, verdict or exit code.
To combine Python selection with configured methods:

```python
from avsectester.experiment import run_configured

# scenario is a selected ScenarioInstance. config defines compatible stack and intervention settings.
results = run_configured(config, frames=30, scenario=scenario, mode="pair", treatment="attack")
```

The supplied scenario retains its factory, match and conditions. Scene and selection settings
from the configuration are not reapplied, and the `backend_factory` and `provenance` arguments
do not override it. For custom
multi-treatment experiments, see [Runtime composition](INTERVENTIONS.md#single-and-paired-execution).
Direct `run` and `run_pair` callers apply scenario conditions themselves, as shown in
[lower-level execution](SCENARIOS.md#lower-level-execution).

## 1. Backend and stack compatibility

The shared loop supports different adapters, but does not convert arbitrary native sensor,
state or command formats. The current combinations are:

| | **ModularAVStack** | **AlpamayoAVStack** |
|---|---|---|
| **CarlaBackend** | Supplied phantom-injection driving demo (`avsectester run`) | Requires a trajectory-to-actuator controller and compatible camera inputs |
| **NuRecBackend** | Requires compatible modules and avstack sensor/state adapters | Supplied Alpamayo driving demo with `TrajectoryFollower` |

CARLA currently executes `throttle`, `brake` and `steer`, not `Control.trajectory`.
NuRec provides ndarray camera images and `EgoPose`, while the standard modular pipeline expects
avstack sensor wrappers and an avstack ego state. A common interface alone does not supply
these missing conversions. Native observation payloads also need appropriate codecs if
transported between processes.

Both adapters may declare `command_modes` as a `frozenset` of `actuator` and/or `trajectory`.
The runner checks declared compatibility before reset. A `command` hook can translate between
representations, so that path validates the final command after all handlers instead. Built-in
backends reject active fields they cannot consume, including actuator fields on a trajectory-only
follower. An undeclared custom adapter returns `None` and retains its own validation semantics.
Command compatibility does not supply camera, state or coordinate conversions.

A Python `stack_factory()` creates an `AVStack` for each treatment. Configured entry points also
accept an importable driving-system factory:

```yaml
stack:
  factory: research.driver:make_stack
  options:
    checkpoint: /path/to/model
```

The factory is called as `factory(**options)` and must return an `AVStack`. Without `stack`, configured CARLA runs
use `ego.pipeline`. The supplied end-to-end adapter is selected with `stack.type: AlpamayoAVStack`
and its constructor arguments under `stack.options`. For custom world adapters, provide
`backend_factory(scene_config)` to the Python setup/configured-run APIs. The observation and
command formats must agree regardless of how these factories are loaded.

### 1a. CarlaBackend (`simulators/carla.py`) + ModularAVStack (`stacks/modular.py`)

`CarlaBackend` owns an avcarla `CarlaClient` + ego (`CarlaMobileActor` with a no-op internal pipeline
— driving is external, done by the `AVStack`) + `CarlaNpc` traffic. `reset()` spawns actors and
captures a strict-spawn `replay_scenario`. `step(control)` applies the `Control` through CARLA
physics. `_observe()` reads sensors + ego into an `Observation`. `ModularAVStack` wraps an avstack
`ModularDrivingPipeline` (perception → tracking → planning → control) and returns a `Control`.

`run_scenario` prepares a CARLA case and executes one run. `run_scenario_pair` executes a
clean/attack comparison. The CLI selects single or paired execution and the enabled treatment.
`prepare_scenario(config)` resolves random choices **once**
(vehicle models, spawn indices, destinations and seeds, preserving explicit values) into an
in-memory config without mutating the input. A missing `client.seed` is generated once for the
pair. Clean runs from that configuration,
then attacked replays `clean.replay_scenario` — the actual successful spawn transforms. An experiment
uses a dedicated CARLA server. Treatments run sequentially on it, reloading the world into
synchronous fixed-step physics and rebuilding actors and the driving stack. The clean run may relocate vehicles when a spawn fails. The attacked run
reproduces the clean spawns strictly. NPC traffic keeps reacting to the ego after the shared start, so
an attack can change NPC trajectories as part of its driving consequence.

#### Configuring a YAML scenario

Start with the complete [example configuration](../configs/carla_scenario.yaml). Keep its
`type` fields and nested pipeline settings when editing individual parameters, then run:

```bash
avsectester run configs/carla_scenario.yaml --frames 40 --gpu 0
```

By default the command runs clean first, then attack. `--mode single` runs only the selected
treatment. `--treatment` accepts `clean`, `attack`, `defense` or `attack-defense`. Both options
override the YAML `mode` and `treatment` fields. `--frames` overrides YAML `frames` for each run.
`--gpu` overrides CARLA modular perception and does not select the CARLA server's GPU. Configure
other model devices in their stack options. The following YAML snippets are edits to the complete
example, not standalone files.

| Field | Meaning |
| --- | --- |
| `frames` | Simulation steps per run. 40 steps at `client.rate: 20.0` represent 2 seconds per run. |
| `mode`, `treatment` | `single` or `pair`, and `clean`, `attack`, `defense` or `attack-defense`. Defaults: `pair`, `attack`. |
| `selection` | Optional serialized requirement and source options. No selection means one configured case with `match=None`. |
| `conditions` | Shared world and sensor environment, applied to all treatments. See [SCENARIOS.md](SCENARIOS.md#scenario-conditions). |
| `stack` | Optional driving-system factory or supplied Alpamayo adapter configuration. Otherwise use `ego.pipeline`. |
| `client.connect_ip`, `connect_port` | Address of the running CARLA server. |
| `client.traffic_manager_port` | Traffic Manager used by NPC autopilot. |
| `client.synchronous`, `rate` | Use `true` for paired experiments. `rate` sets simulation steps per second. |
| `ego.vehicle`, `ego.spawn` | Vehicle blueprint ID and map spawn-point index, or `random` for either field. Spawn indices depend on the selected map. |
| `ego.autopilot` | Keep `false` to let the configured AV pipeline control the ego. |
| `ego.sensors` | Sensor settings. The example uses a LiDAR with `sensor_tick: 0.05` and `rotation_frequency: 20`. |
| `ego.pipeline` | Perception, tracking, planning and control modules. The default planner drives straight and brakes for obstacles. `target_speed` is in m/s, and `brake_distance` and `brake_corridor` are in meters. |
| `npcs` | Background vehicles, using the compact form or explicit list below. Omit it or use `[]` for none. |
| `attacks` | Native modular post-hook shorthand, enabled for attack and combined treatments. `PhantomInjection` uses `stage: perception`. |
| `patches` | Native physical panel shorthand, prepared as attack methods with a `world.setup` runtime plugin. |
| `interventions` | Ordered preparation factories with explicit `kind: attack` or `defense`, options and optional reusable artifacts. |
| `runtime` | Ordered hooks, optional localizer and handler RNG seed. Raw hooks belong to the attack treatment. Use typed `interventions` for mode-selectable defenses. |
| `clean_runtime` | Optional explicit baseline runtime. Otherwise it retains the configured localizer and handler seed without attack hooks. |

In the default `pair`/`attack` mode, an empty intervention configuration compares two baseline
runs and the CLI prints a notice. `single`/`clean` needs no intervention. Single attack runs and
defense/combined modes require the corresponding configured kinds. An explicit Python
`treated_runtime` replaces configured interventions, and the caller determines its role.
Repeated clean comparisons construct independent YAML baseline components. An explicit Python
`clean_runtime` is used in both runs and must reset any shared state correctly.

The CLI computes an attack verdict only for pairs containing an enabled attack. Defense-only
pairs report side effects, and clean/empty-attack pairs report baseline comparisons. See
[DOCKER.md](DOCKER.md#run-the-end-to-end-attack) for CLI outcome and error exit codes.


#### Fixed choices and random choices

Random choices are resolved **once per experiment pair**. Both runs use those choices, even when
each new command starts with a fresh seed. Explicit vehicle models and seeds are preserved. Explicit
spawn points are the first positions attempted, with retries as described below. Omitting a field
does not generally mean random: fields can have defaults or be required.

| Setting | Behavior across separate commands |
| --- | --- |
| `client.seed: 0` (the example default), or another fixed integer | Repeats seeded scene choices with the same configuration and map/blueprint candidates. This is not a guarantee of identical model outputs or complete driving traces. |
| `client.seed: null`, or omit `seed` | Generates a fresh seed for each pair and selects random items again. Individual choices can still repeat by chance. |
| A concrete value such as `ego.spawn: 0` | Always attempts this spawn point first, regardless of the seed. |

To select a new random ego model and spawn point for each command, edit these fields while keeping
the other client and ego settings:

```yaml
client:
  seed: null
  traffic_manager_seed: null
ego:
  vehicle: random
  spawn: random
```

An omitted or `null` `traffic_manager_seed` follows the resolved `client.seed`. The complete example
explicitly sets both seeds to `0`: changing only `seed` leaves the Traffic Manager seed fixed.
Similarly, an omitted or `null` LiDAR `noise_seed` is derived during preparation. An explicit
`noise_seed` is preserved. The CLI prints `[scenario] seed=...` before its first server operation,
including when preparation or execution later fails. Set `client.seed` to that integer to repeat
the seeded choices with the same remaining configuration and environment. This is separate from
`runtime.seed`, which initializes the handlers' `context.rng` and does not seed the simulator.

The seed does not randomly change every field. NPC count, map, weather, sensor specifications and
attack parameters are not automatically randomized. `map_name` can be set under `client`. Native world weather is set with
`conditions.world.weather` or legacy top-level `weather`. When omitted, those properties come
from the selected world's environment. Traffic
lights use the client's seeded `randomize_lights` behavior unless explicit light settings are given.

#### Configuring NPCs and spawn positions

The compact form attempts to create six NPCs at spawn indices 1 through 6. Here `vehicle` means a randomly
selected vehicle model, not a fixed model:

```yaml
npcs:
  count: 6
  npc_type: vehicle
  spawn_start: 1
```

To mix fixed and random NPC choices, replace the compact form with a list. Each entry creates one
NPC. `count` and `spawn_start` apply only to the compact form:

```yaml
npcs:
  - type: CarlaNpc
    npc_type: random
    spawn: random
  - type: CarlaNpc
    npc_type: vehicle.tesla.model3
    spawn: 3
```

Random spawn selection reserves explicit spawn indices first and selects unused indices for random
actors. Different indices do not guarantee that all vehicle geometries fit without collisions.
During clean setup, a failed spawn moves the requested position 3 meters along the vehicle heading
in the horizontal plane and tries again, up to 10 attempts in total. The runner records each
vehicle's actual position, orientation and model before advancing the simulation. After resetting
the world, the attacked run uses these recorded transforms in the same creation order, without
applying spawn offsets again. If a recorded spawn fails, the attacked run reports an error rather
than moving it. Separate pairs can therefore resolve different spawn positions.

`client.strict_spawn` defaults to `false` for the first run. Set it to `true` to disable clean
relocation as well. Recorded transforms are always replayed strictly, even if this setting is
`false`. If clean exhausts its retries, choose other indices or fewer vehicles.
`ego.destination` accepts `null`, a spawn index or `random`, but the default straight-driving
pipeline does not use it to navigate to a destination.

##### CARLA Python interface

```python
from avsectester.carla_runner import run_scenario, run_scenario_pair

clean, attacked = run_scenario_pair(config, frames=40)
# An independent clean run is also available:
single = run_scenario(config, frames=40)
```

`Trace.replay_scenario` is an in-memory configuration, not a generated file. Each actor's
`spawn_transform` contains native CARLA world coordinates (`location`: x/y/z in meters,
`rotation`: pitch/yaw/roll in degrees). It already includes `reference_to_spawn` and any retry
displacement. The runner produces this field. Users normally configure `spawn` instead.

### 1b. NuRecBackend (`avsectester/simulators/nurec.py`)

The backend advances an in-process ego within an NVIDIA **NuRec** reconstructed scene
(a 3D `.usdz` scene rendered per pose). Other actors follow recorded trajectories rather than
reacting to the ego. This adapter does not run the full AlpaSim simulation. It owns an `EgoPose`, dynamics
(`KinematicBicycle` for throttle/steer control, or `TrajectoryFollower` to track a rig-frame
`Control.trajectory`), and a pluggable `Renderer`:

- **`StubRenderer`** — deterministic black `HWC-uint8` frames. **no server or GPU**, so it runs the
  whole loop in CI and every `tests/core/test_nurec.py` case.
- **`NuRecRenderer`** — a `SensorsimService.render_rgb` gRPC call per camera and observation to an
  `nre-ga` renderer. The render pose is `world_rig @ rig_to_camera` (the camera extrinsic is composed
  in, matching AlpaSim's `construct_rgb_render_request`). The returned JPEG is decoded to `HWC-uint8`.

`step(control)` integrates dynamics → pose → `renderer.render(pose, camera)` → `Observation`.
The backend's `config["dt"]` is the physics and observation interval in seconds (default
`0.05`). It is independent of the policy's waypoint interval. For Alpamayo's 10 Hz output,
waypoints are 0.1 seconds apart. `TrajectoryFollower` interpolates from the current rig origin
to the first waypoint and between subsequent waypoints, so a 0.05-second physics step is valid.
Set `dt=0.1` for a simple loop aligned with the waypoint spacing. Do not rescale predicted
positions to change the simulation rate. The current runner invokes the stack once per step,
so `dt` also sets the policy-call rate rather than introducing a separate policy scheduler.
`checkpoint()`/`restore()` save and restore the local ego pose and frame counter.
They do not checkpoint a remote rendering service. Bringing up the
renderer + a scene is covered in [`SETUP.md`](SETUP.md). The runnable demo is
[`scripts/demos/nurec/alpamayo_nurec_demo.py`](../scripts/demos/nurec/alpamayo_nurec_demo.py).

Python scenario setup also accepts a `backend` mapping with these fields:

| Field | Value |
|---|---|
| `type` | `NuRecBackend` or `nurec` |
| `scene` | Scene identifier passed to the renderer during reset |
| `renderer` | `stub`, `nurec`, or a mapping with `type` and renderer constructor options |
| `dynamics` | Mapping with `type: actuator` (default) or `type: trajectory`, followed by the selected dynamics' constructor options |
| `ego0` | Initial `EgoPose` fields: `x`, `y`, `yaw`, `speed`, `t` |
| `dt` | Physics/observation interval in seconds, default `0.05` |
| `camera_aliases` | Mapping from native renderer camera names to model-visible sensor IDs, such as `{"camera_front": "front"}` |

For example, `scenario_from_config` accepts
`{"backend": {"type": "NuRecBackend", "renderer": "stub", "dynamics": {"type": "trajectory"}}}`.
`StubRenderer` names its default camera `camera_front`. Configure its `cameras` option or
`camera_aliases` when handlers and stacks expect a different sensor name.
Alpamayo requires trajectory-compatible dynamics, normally `dynamics.type: trajectory`.
Actuator dynamics with Alpamayo raises `CommandCompatibilityError` before reset unless the
Runtime supplies a command-conversion hook. Such a hook must return a compatible final command.
Use the [NuRec/Alpamayo demo instructions](SETUP.md#4-nurec--alpamayo-the-end-to-end-path)
for the real model and rendering service.

### 1c. AlpamayoAVStack (`avsectester/stacks/alpamayo.py`)

Wraps NVIDIA's real **Alpamayo-1.5-10B** end-to-end policy
(`alpasim_driver.models.alpamayo1_5_model.Alpamayo15Model`) as an `AVStack`. `__call__(obs)` builds a
`PredictionInput` (camera frames + ego speed + ego-pose history) → `model.predict()` →
`ModelPrediction.candidate_positions` (K×T×3 rig-frame waypoints) → `Control.trajectory`, which a
`TrajectoryFollower` backend then tracks. The model input contract requires
`context_length` camera frames (padded at startup) and ≥1.5 s of backward ego history (synthesized
constant-velocity). Model-facing timestamps carry an epoch offset while waypoints stay in the
backend's sim clock. Torch / `alpasim_driver` imports are **lazy**, so importing AVSecTester and
testing this adapter does not require the real model. Runtime dependencies and checkpoint
setup are described in [SETUP.md](SETUP.md#4-nurec--alpamayo-the-end-to-end-path).

### 1d. Models and supporting components

The repository uses different models for driving, auxiliary detection and image preparation.
They are not all stages of one pipeline:

| Component | Role and entry point | Participates in driving decisions? |
|---|---|---|
| Alpamayo-1.5-10B | End-to-end camera policy through `AlpamayoAVStack` | Yes, outputs a trajectory |
| PointPillars, CARLA vehicle weights | LiDAR perception in `configs/carla_scenario.yaml` through `ModularAVStack` | Yes, detections feed tracking, planning and control |
| Faster R-CNN, CARLA vehicle weights, MMDetection | `scripts/common/demo_common.py:build_detector` and patch optimization scripts | In `patch_driving_demo.py`, detections feed a rule-based braking policy. Other uses inspect or optimize detector response |
| Faster R-CNN ResNet-50 FPN, COCO weights, torchvision | `scripts/common/demo_common.py:build_coco_detector`, used by NuRec/nuScenes object demos | No, provides auxiliary detection scores only with `--eval` |
| PCTNet | `PCTNetHarmonizer` adjusts inserted image appearance | No, optional image harmonization |
| SAM, default `facebook/sam-vit-huge` | `scripts/preparation/extract_person_cutouts.py` prepares pedestrian RGBA assets | No, offline segmentation |
| NuRec through `nre-ga` | `NuRecRenderer` obtains images at the requested camera pose | No, scene rendering rather than a driving policy |

`AlpamayoAVStack(model=loaded_model, ...)` can create separate adapters around an already loaded
model, each with independent camera history and previous-plan state. Checkpoint loading is used
when no model is supplied. The supplied model must implement the AlpaSim driver's prediction
contract. Share it only when its own mutable state is safe for reuse.

COCO names the detector's pretraining dataset. Using these weights does not add a COCO scene
source, and the auxiliary detector's scores are not Alpamayo predictions or closed-loop outcomes.

The default modular driving configuration is
`PointPillars → BasicBoxTracker3D → ForwardCollisionPlanner → VehiclePIDController`.
The tracker, straight-ahead collision planner and PID controller are algorithmic components,
not additional end-to-end neural models. Demo-only `CruiseStack` holds a prescribed throttle,
while `CameraForwardCollisionStack` brakes using a detected box-area threshold.

See [SETUP.md](SETUP.md#5-model-dependencies-and-weights)
for loading, [IMAGE_ATTACKS.md](IMAGE_ATTACKS.md#auxiliary-models) for image-processing use, and
the [script index](../scripts/README.md) for each entry point's actual components.

## 2. Attack and defense

Use `Runtime` to register ordered handlers at supported stages. Attack and defense are optional
and use the same contract. Every returned replacement is passed to the next handler and
consumed downstream.

```python
from dataclasses import replace

from avsectester.backend import run
from avsectester.runtime import Hook, Runtime

def modify_estimate(estimate, context):
    return replace(estimate, ego_speed=20.0)

runtime = Runtime([Hook("localization.post", modify_estimate)], seed=0)
trace = run(backend, stack, frames=30, runtime=runtime)
```

The loop exposes captured observations, localization, final model input and executable commands.
Backend adapters expose world and rendering stages where supported. Stack adapters expose their
actual component inputs and outputs. Registration at an unavailable stage raises an error before
world reset. [INTERVENTIONS.md](INTERVENTIONS.md) specifies all stages, modifiable values,
resource ownership and stateful handler methods.

Wrap an existing avstack-registered hook with `NativeHook`:

```python
from avsectester.backend import run
from avsectester.stacks.modular import ModularAVStack
from avsectester.interventions import NativeHook
from avsectester.runtime import Hook, Runtime

stack = ModularAVStack(pipeline_config)
runtime = Runtime([Hook("perception.post", NativeHook({
    "type": "PhantomInjection", "target_xyz": (6.0, 0.0, -1.5),
    "obj_type": "Car", "score": 0.9,
}))])
trace = run(backend, stack, frames=30, runtime=runtime)
```

Here `pipeline_config` is the `ego.pipeline` mapping from the CARLA YAML. A native avstack
post-hook returns its output as `(output,)`. A runtime handler returns the output directly.
`NativeHook` adapts the tuple convention, creates a fresh native hook per run and preserves
component logging after modifications. Runtime bridges are removed when the run ends.
The YAML CLI accepts ordered factory hooks, native hooks and physical panels through the same
runtime. See [INTERVENTIONS.md](INTERVENTIONS.md#configuration-and-adapters) for configuration.

`PhantomInjection.target_xyz` is expressed in the source sensor's coordinates (forward, left, up).
It requires `detections.source_reference`, including when there are no detections. The perception
base class copies the input reference before inference and attaches it to the output before
post-hooks run, so later ego motion does not move an earlier phantom. Passthrough detector inputs
must supply `DataContainer(..., source_reference=reference)` explicitly. Missing source metadata
raises `ValueError`. Container copy, filter, mapping, addition and serialization retain the source
reference.

### Preparation, execution and outputs

`Intervention.prepare(scenario)` creates reusable artifacts. `Intervention.hooks(artifacts)`
constructs the execution hooks. This contract applies to offline attacks, adaptive attacks and
defenses. Preparation may use independently supplied models/renderers and does not require the
driving backend to be active. Existing `Runtime` and raw hook factories remain usable without
implementing an Intervention.

`prepare_intervention` returns a `PreparedIntervention`. Pass existing artifacts to skip preparation,
or save/load them using the method's explicit JSON export/import functions. `Treatment` preserves
the method order and builds a Runtime for selected attack/defense kinds. `PreparedIntervention.position`
places its hooks before or after the base Runtime's hooks, with default `after`.
`prepare_intervention(..., position="before")` selects the earlier position. Legacy convenience hooks
use `before` and precede raw `runtime.hooks`, followed by configured `interventions` in definition order.
The framework does not reorder defenses after attacks automatically.

| Treatment | Enabled configured methods |
|---|---|
| `clean` | Baseline Runtime only |
| `attack` | Attack methods and raw attack Runtime hooks |
| `defense` | Defense methods |
| `attack-defense` | Both kinds, in their configured order |

Baseline and treated runs share the scenario conditions. The configured localizer and handler
seed are retained unless the baseline is explicitly overridden. Explicit Python Runtime arguments
can replace configured interventions. Pair mode with a clean treatment compares two baseline runs.
`run_configured` returns named driving traces without computing a differential verdict in either
mode. See
[INTERVENTIONS.md](INTERVENTIONS.md) for preparation, artifact storage and lifecycle examples.

During a callback, `context.record_output(name, value)` saves an independent copy of a diagnostic
value. `Trace.outputs` contains these `InterventionOutput` records, with step, stage, sensor and
physical time/frame. Attacks, defenses and localization adapters use the same output interface.
This records alarms or scores without changing the driving command. It does not define a metric
for those values. `Trace.provenance` retains treatment and configured scenario/condition metadata.

## 3. Metric (clean vs attacked → verdict)

```python
from avsectester.metric import impact   # (clean: Trace, attacked: Trace) -> Impact
```

`Impact` answers the differential question — did the attack change the drive the clean run never had?
— via `induced_braking`, `induced_stop` (phantom family: stopped a car that was driving),
`suppressed_stop` / `overdrive_frames` (object-hiding family: kept the ego rolling where the clean run
braked), and `attack_succeeded`. The verdict is guarded by a **driving baseline**: `attack_succeeded`
is `clean_drove and (induced_stop or suppressed_stop)` — the clean run must actually have driven
(`clean_drove`, i.e. `clean.peak_speed >= baseline_speed`). If the clean ego never got moving (too few
frames, or stuck at the spawn) the result is **inconclusive**, not a success — "an already-stopped car
braking" does not establish driving impact.

`result.comparison(treated_label)` formats driving measurements without a verdict.
`str(result)` retains the attacked label and attack verdict. `plot_impact` accepts
`treated_label="attacked"` and `show_verdict=True` as keyword options. Set `show_verdict=False`
for defense side effects or baseline comparisons. The default title follows these options, and
an explicit `title` overrides it. Labels and display options do not change metric calculations.

## 4. Scenario preparation and selection

`scenario_from_config` and `source_from_config` prepare experiment inputs:

- `ConfiguredScenarioSource` yields one configured `ScenarioInstance` without preview or filters.
- `DatasetFilter` and `CarlaScenarioBuilder` select fixed actor bindings and insertion specifications.
- `Constraint.evaluate(context) -> FilterResult` returns `pass`, `fail` or `unknown`.
  `FilterContext` exposes normalized state, raw metadata and native provider APIs.
- `ScenarioRequirement` combines predicates across a prescribed `InitialWindow`.
- `ScenarioInstance.make_backend()` creates a fresh backend and configures its world conditions.
  The instance stores the raw callable as `factory`. `scenario.match` contains the optional
  selected scene, bindings and insertions. Use `dataclasses.replace` to replace its fields.

`ScenarioConditions` carries environment settings shared by all treatments. World conditions are
validated/configured before reset. Sensor conditions construct independent per-run handlers with
separate random streams. CARLA weather is applied before candidate selection. Custom filters can
inspect a copied conditioned measurement when their provider supplies observations. Original ground
truth and geometric visibility remain distinct from those measurements.

`scenarios.filters` defines predicates and combinations. `scenarios.requirement` performs role
assignment and window evaluation. Both selection and runtime insertion use the camera, texture
sampling and visibility code in `rendering/`. `InsertionGeometry` names the actor poses, victim pose
and world-to-camera transform passed to a custom insertion renderer.

Filters stop when preparation finishes. Runtime insertion uses the selected identities and never
chooses a replacement host. See [SCENARIOS.md](SCENARIOS.md) for configuration, custom filters,
placement, visibility, conditions and resource ownership.

## 5. What comes from avstack / AlpaSim

AVSecTester provides the shared interfaces, backend/stack adapters, attack integration, metrics,
scenario selection and visualization. The underlying simulator and model components include:

- **`ModularDrivingPipeline`** (`avstack.modules.pipeline`) — maps `(sensor_data, ego_state)` →
  control by running perception → tracking → planning → control. The modular counterpart to
  end-to-end / foundation-model stacks. Attacks/defenses attach as hooks on any stage.
- **`ForwardCollisionPlanner`** (`avstack.modules.planning.vehicle`) — drive straight, brake to a
  stop when a track occupies the forward corridor (body-frame check). The driving consequence a
  perception attack triggers.
- **`CarlaMobileActor`** (`avcarla`) — supplies ego state, sensors and `apply_control`. This
  backend gives it an empty internal pipeline. `ModularAVStack` runs the driving pipeline
  outside the actor, and the shared loop sends its commands back to the backend.
- **`nre-ga` NuRec renderer** and **`alpasim_driver` / Alpamayo-1.5** — NVIDIA
  [AlpaSim](https://github.com/NVlabs/alpasim) pieces that `NuRecRenderer` and `AlpamayoAVStack`
  wrap. See [`SETUP.md`](SETUP.md) for standing them up.
