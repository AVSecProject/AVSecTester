# Implement an attack

> **Audience:** Researchers implementing an attack.<br>
> **Outcome:** Select attack segments, prepare reusable artifacts and apply stateful effects during driving.<br>
> **Prerequisites:** [Architecture](../concepts/architecture.md) and the [core installation](../getting-started/installation.md#local-core-installation).

An `Attack` defines three parts, executed in a fixed order:

```text
build_metadata(scenario) → run_offline(context) → create_runtime(segment, artifacts)
```

The experiment calls `build_metadata` once per scenario, then `run_offline` once for the attack.
Each treated run gets its own `AttackRuntime` from `create_runtime`. Offline and online are
phases of the same attack. Either phase can be trivial.

The Python experiment API uses this workflow. The CLI and supplied concrete attack examples
still use the compatibility orchestration described under
[Package the attack as an intervention](#package-the-attack-as-an-intervention).

## Choose an attack surface

| Effect | Implement | Changes | When it applies |
|---|---|---|---|
| `WorldAttackEffect` | `apply(simulator_runtime, resources)` | Physical actors, environment or simulator state through supported simulator operations | Setup effects precede the first capture. Effects returned by `step` precede the next simulator step |
| `SensorAttackEffect` | `sensor` and `apply(data, view)` | One captured sensor payload, using the capture's `SensorView` | Current decision, before sensor-environment transforms and Vehicle I/O |
| `AVStackAttackEffect` | `stage` and `apply(value, context)` | A Vehicle I/O value, component input/output or command | Current decision, at the declared stage |

Only stages provided by the selected simulator, Vehicle I/O and AVStack can be targeted.
The stage and value tables are in [Runtime stages](../reference/runtime.md#stages).

A `SensorView` supplies the camera model, physical sensor pose, actor poses and available depth
for a capture. Camera models and actor poses live in `avsectester.geometry`. Image read/write
helpers live in `avsectester.closed_loop.datatypes`. The framework does not select an insertion
renderer: the attack implements its own sensor effect and can reuse it during selection,
offline optimization and deployment.

## Write your first attack

This attack adds a bias to the speed reported to the driving system. It selects a segment of an
explicit length, uses the original parameters as artifacts and returns an observation-stage effect.
The example defines the attack without requiring a server or model weights:

<!-- runnable -->
```python
from dataclasses import replace

from avsectester.attack import Attack, AttackRuntime, AttackSegment, AVStackAttackEffect


class SpeedRuntime(AttackRuntime):
    def __init__(self, bias):
        self.bias = bias

    def step(self, context):
        return (AVStackAttackEffect("observation", self.spoof),)

    def spoof(self, observation, context):
        context.record_output("speed.bias", self.bias)
        return replace(observation, ego_speed=observation.ego_speed + self.bias)


class SpeedSpoof(Attack):
    def __init__(self, decision_steps, bias=5.0):
        self.decision_steps = decision_steps
        self.assets = {"bias": bias}

    def build_metadata(self, scenario):
        frames = scenario.frames
        if len(frames) <= self.decision_steps:
            scenario.record_reason("Not enough frames for the requested decisions and terminal state")
            return []
        scenario.record_reason("Selected the first decision interval")
        return [AttackSegment(scenario.scenario_id, frames[0], frames[self.decision_steps])]

    def create_runtime(self, segment, artifacts):
        return SpeedRuntime(artifacts["bias"])
```

`run_offline` is not overridden because its default returns `self.assets` unchanged. Attacks
without assets can leave `assets=None`. An attack without online effects can use the default
`create_runtime`.

## Use the selected scenario

`build_metadata` receives a `ScenarioView` with:

- `scenario_id` and `frames`: the scenario identity and actual frame references.
- `frame_data(frame)`: recorded data, or prepared CARLA frame data, including stable actor IDs.
- `services.dataset.native(scenario_id)`: dataset-specific information.
- `services`: isolated sensor, rendering, component, scoring and rollout operations.
- `record_reason(text)`: explanations saved even when no segment is selected.

A segment describes decision inputs in `[start, stop)`. `stop` is the terminal frame, so
`decision_steps=N` needs `N+1` frame records. The framework rejects boundaries outside the
scenario and actor IDs absent at the start. It does not score or rank candidates on the attack's
behalf. Optional `selection_score` and zero-based `selection_rank` are supplied by the attack.
`user_data` carries attack-specific information.

An empty return value records the scenario as not selected. By default, an exception records a
failed selection and preparation continues with the next scenario. Configure `strict=True` to
stop immediately. Reasons reported before the exception remain available.

Use `roles={"hosts": ("actor-a", "actor-b")}` to bind multiple actors. `placement` can store the
selected placement specification. These values pass to `create_runtime` unchanged. Runtime
execution does not select another host or rerun `build_metadata`.

Use closed-loop services through a context manager:

```python
# Inside build_metadata. frame is one of scenario.frames.
from avsectester.closed_loop.services import SensorRequest

with scenario.services.session(scenario.seed) as session:
    sample = session.sensor_data(SensorRequest(scenario.scenario_id, frame, ("front",)))
```

Sensor requests without attack effects can scan consecutive frames using one session-owned
runtime. Rendering and rollout operations use independent runtimes. Native handles must not be
stored in segments or artifacts intended for later loading.

## Read inputs and state

`AttackRuntime.subscriptions` declares which snapshots the runtime needs:

| Subscription | Read through | Meaning |
|---|---|---|
| `ResultSubscription("observation")` | `context.observation` | Captured observation before this decision's attack sensor effects and Vehicle I/O |
| `ResultSubscription("physical_state")` | `context.physical_state` | True ego state and other actors from the simulator |

Unsubscribed fields are `None`. A snapshot can be inspected or edited locally without moving the
world. The framework creates `OnlineContext`, which includes `step`, `stage`, `sim_time` and
`frame`. It exposes no simulator, AVStack or time-advancing session. Use effects to apply changes.

Component outputs arrive directly as `AVStackAttackEffect.apply(value, context)`. Sensor effects
receive their current payload and `SensorView`. World effects receive the live simulator runtime
only when applied. These are distinct access points with distinct timing.

## Keep state and adapt

Initialize per-run history in the runtime constructor. All callbacks for that attack use the same
instance. Declare subscriptions only for the results you read:

<!-- runnable -->
```python
from avsectester.attack import AttackRuntime, AVStackAttackEffect, ResultSubscription


class StaleSpeed(AttackRuntime):
    subscriptions = (ResultSubscription("physical_state"),)

    def __init__(self, delay):
        self.delay = delay
        self.history = []

    def step(self, context):
        self.history.append(context.physical_state.ego_speed)
        return (AVStackAttackEffect("observation", self.spoof),)

    def spoof(self, observation, context):
        from dataclasses import replace

        if len(self.history) <= self.delay:
            return observation
        return replace(observation, ego_speed=self.history[-self.delay - 1])
```

The lifecycle is `setup → step ... → stop`. `setup` can return only world effects and runs before
the first capture. `step` is called once per decision. `stop` releases attack-owned state even
after failure. Return a fresh runtime for every candidate rollout and every formal run.

## Keep component decisions in one runtime

An `AVStackAttackEffect` can hold a bound method, such as `self.modify_detections`. When its
stage is reached, that method receives the component's actual value before downstream use.
It can update the same runtime's history and return a replacement. No second callback object
or separate attack history is required.

Use `context.record_output(name, value)` to save diagnostics in the decision's RunLog outputs.
The complete lifecycle and effect contract is in
[Online phase](../reference/runtime.md#online-phase).

## Attack localization

Modify a measurement with `SensorAttackEffect`, or target `sensors.post` with an
`AVStackAttackEffect`. A GNSS or IMU modification affects localization only when the configured
localizer consumes that measurement. Target `localization.post` to change the estimated state
directly. Target `observation` to change the final model input.

The physical camera pose and true world state stay independent of spoofed localization.
Configure the localization path through [Vehicle I/O](driving-systems.md#vehicle-io).

## Change the world

Subclass `WorldAttackEffect` and implement `apply(simulator_runtime, resources)`. Use the runtime's
supported actor/state operations or native API. Register cleanup callbacks on `resources` for
created objects. Setup effects are visible in the first capture. Effects returned by `step`
apply before the simulator advances and cannot change the capture already used for that decision.

`AttackRuntime.setup_ticks` declares the total tick budget required by setup effects. The
experiment uses a common deployment window for all treatments in a comparison group.

## Save and reuse artifacts

`Experiment.prepare` stores the offline return value in `prepared.artifacts[attack_name]`.
The `Attack` definition does not hold mutable artifacts. When an output
location is configured, each attack has its own directory:

```text
attacks/<name>/metadata.json
attacks/<name>/artifacts.pkl
attacks/<name>/manifest.json
```

The default `Attack.save_artifacts(artifacts, directory)` pickles the supplied artifacts and
records their content hash. `load_artifacts(directory)` restores and returns that value. Load
only trusted pickle files. Override both methods to use another format. The framework never reselects metadata after
offline optimization or requires optimized assets to retain their original dimensions.

`prepared.save(directory)` saves a complete prepared experiment. `PreparedExperiment.load(config,
directory)` restores its metadata and artifacts without calling `build_metadata` or `run_offline`.
The configuration is supplied again to construct fresh resources rather than saving live objects.
Metadata supports JSON values, paths, real numeric/boolean/Unicode NumPy arrays and the supported
segment, actor and placement records. Convert other `user_data` objects into these forms before
saving. Artifacts have a separate codec and can use other types supported by the attack's serializer.

## Configure and run the attack

Use `Experiment.prepare(config)` and `Experiment.execute(prepared)` from Python. The experiment
configuration supplies a dataset, simulator, reusable driving adapter, named attacks and treatments.
Attack and driving configurations can be inline or separate YAML files.

```python
from avsectester.experiment import Experiment

# config is an ExperimentConfiguration, a mapping, or a YAML path supplied by the caller.
prepared = Experiment.prepare(config)
results = Experiment.execute(prepared)
```

A treatment enables at most one attack. The clean treatment and treated runs within a segment
comparison use the same start and decision length, with fresh simulator, driving and attack
state. Different attacks select their own segment groups. Artifacts are prepared once per attack
and reused across groups and treatments. Environment layers are experiment base, attack settings
and explicit treatment overrides, in that order.

See [Experiments](experiments.md) for complete configuration and execution examples. The CLI
currently reads the compatibility configuration in the
[CLI reference](../reference/cli.md#avsectester-run).

## Package the attack as an intervention

The supplied concrete examples and CLI use `Intervention.prepare(scenario)` and
`Intervention.hooks(artifacts)` through `run_configured` and `run_prepared`. Their `Runtime` is a
compatibility form of `HookSet`. These APIs remain available for their existing callers.

This path does not call the three `Attack` methods automatically. Implement an `Attack` for the
Python experiment workflow above. Existing intervention examples are described individually in
[Examples](../examples/README.md). Their preparation format and JSON artifact codec remain in
[Preparation](../reference/python-api.md#preparation).

## Read the results

`Experiment.execute` returns named treatment results with a `RunLog` for each run. Logs record
true pre/post state, observed inputs, produced/executed commands, effect requests and receipts,
component outputs and termination status. Results are grouped by attack, segment and seed.

The existing evaluation utilities still accept compatibility traces. They do not define a
universal success criterion for arbitrary attacks. See [Evaluate results](evaluation.md).

## Next steps

- [Experiments](experiments.md): prepare, save, load and execute treatments.
- [Runtime reference](../reference/runtime.md): stages, effects and lifecycle contracts.
- [Configure simulators](simulators.md): simulator capabilities and native access.
- [Optimize attack artifacts](optimization.md): existing optimization algorithms and model access.
- [Examples](../examples/README.md): supplied concrete attack implementations.
