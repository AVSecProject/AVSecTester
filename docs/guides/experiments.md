# Run experiments

> **Audience:** Users who run experiments from the CLI or Python.  
> **Outcome:** Choose an entry point, select treatments, run single or paired experiments, and find their outputs.  
> **Prerequisites:** [Architecture](../concepts/architecture.md) and [Quickstart](../getting-started/quickstart.md).

## Prepare and execute attacks

Use `Experiment` for new Python experiments. Each attack owns its selection logic, optional
offline preparation and online behavior. The framework executes this fixed sequence:

```text
Dataset scenarios → Attack.build_metadata → Attack.run_offline → saved metadata and artifacts
                                                                   ↓
                         selected segment × seed × treatment → independent RunLog
```

```python
from avsectester.experiment import Experiment

prepared = Experiment.prepare("experiment.yaml")
results = Experiment.execute(prepared)

for group, logs in results.groups.items():
    print(group, {name: log.status for name, log in logs.items()})
```

The [configuration reference](../reference/configuration.md#experiment-configuration-python-api)
describes inline and file-based configurations. Python also accepts an
`ExperimentConfiguration` or a mapping containing configured `Dataset`, `Simulator`,
`DrivingAdapter` and `Attack` objects. Loading configuration does not load a driving model or
start a simulator. The adapter's factory constructs a fresh `AVStack` for each operation.

### Selection and offline preparation

`prepare` calls `build_metadata(view)` once per scenario for each configured attack, collects its
`AttackSegment` records, then calls that attack's `run_offline(context)` once with all of them.
Use `view.record_reason(text)` to explain acceptance or rejection. Each decision appears in
`prepared.selections`, including scenarios with no selected segments and scenarios that failed.
Preparation uses the first configured seed. The metabuilder does not inherit the experiment's
base environment: it requests its own environment explicitly through the services. Preparation
checks segment identity, actual frame boundaries and actor role bindings at the start. It does
not add filters or choose candidates on the attack's behalf.

Selection errors are recorded and preparation continues by default. Set `strict: true` to raise
on the first error. An error in offline preparation stops preparation, since no artifacts are
available for that attack. Opened service sessions are closed on success and failure. The returned
object contains configuration, metadata and artifacts, without live service sessions.

Artifacts are shared by all treatments of that attack. They may have different geometry from the
original assets. The framework does not select again or reassess selection conditions after offline
preparation. Attacks without assets use the same workflow, with `None` as the default artifacts.
See [Write an attack](attacks.md) for the extension interfaces.

### Treatments and independent runs

Each treatment enables at most one attack and may override environment settings. Its
`attack_name` references an entry in the configured attacks. Treatments with no attack name are
clean baselines. Multiple configured attacks select and prepare independently.
For each attack's selected segment and each seed, `execute` runs its treatments and all configured
clean baselines from the same start, with the same number of decisions. It never combines attacks.
Write a composite `Attack` if one run needs several attack behaviors.

Every formal run creates a fresh simulator runtime, `AVStack` and, when enabled, `AttackRuntime`.
Resources are closed at the end of the run. CARLA resolves actual spawn positions once per
comparison group and replays them for each treatment. The shared deployment window accounts for
the largest setup tick budget in the group. `Attack.create_runtime` and configured hook
constructors must create independent mutable state for each run.
A Python `HookSet` supplied directly retains its caller-owned handlers, which must implement
the documented reset lifecycle. Its configuration shell is copied for each run and uses the
experiment seed, as do sensor-environment transforms. The supplied `HookSet` is not mutated.
Before a seed's treatments start, the framework checks that every segment role still exists at
that seed's start. Missing actors fail the run; selection is not repeated.

`decision_steps=N` needs `N+1` actual dataset frames. The segment's `stop` identifies the terminal
state, and is excluded from the decision inputs. An optional experiment-level `decision_steps`
caps execution explicitly. Both selected and executed lengths are recorded. Missing terminal
frames are never padded or silently shortened.

Environment settings merge in the order experiment → enabled attack → treatment. Sensor
corruptions run after sensor attack effects and before hooks. See
[Environment settings](conditions.md#experiment-environment).

`results.groups` maps `(attack_name, segment_index, seed)` to treatment logs. Failed formal runs
retain their partial log and error in `results.failures`. Set `strict: true` to raise instead of
continuing after an execution error. Selection failures remain available as
`results.selection_errors`. A treatment construction error is recorded against that treatment.
If a shared start cannot be resolved, the group's runnable treatments receive failed logs and
execution continues with the next group.

### Save and resume

Set `output` in the configuration to save preparation and each run automatically, or save
preparation explicitly:

```python
from avsectester.experiment import PreparedExperiment

prepared.save("results/prepared")
restored = PreparedExperiment.load("experiment.yaml", "results/prepared")
results = Experiment.execute(restored)
```

`PreparedExperiment.load` reads metadata and artifacts without invoking `build_metadata` or
`run_offline`.
It takes the configuration again, rather than serializing factories, model instances or SDK
handles. Summaries are computed only when saving or loading; configuration construction does not enumerate
scenarios or inspect attack parameters. Saved summaries identify the dataset, scenario list,
simulator and attack definitions. Loading rejects incompatible configuration; the supplied dataset
must also contain the saved frame references. If parameter description is unavailable, saving warns
and marks the description as unverified; loading requires a complete, verified description. See
[Preparation identity](../reference/python-api.md#experiment-preparation-and-execution).

```text
OUTPUT/
  prepared.json
  attacks/ATTACK_NAME/
    metadata.json
    artifacts.pkl
    manifest.json
  runs/ATTACK_NAME/segment-00000/seed-0/TREATMENT_NAME/
    manifest.json
    ... RunLog tables and payloads ...
```

`Attack.save_artifacts(artifacts, directory)` and `load_artifacts(directory)` can override the
default pickle storage. Artifacts belong to the prepared experiment, not to mutable Attack
state. The default manifest records the artifact content hash. Load pickle artifacts only from trusted sources.
Metadata accepts JSON values, frame references, supported placement records and NumPy arrays.
Convert custom metadata objects to data before saving. A run directory must be new, so use a new
`output` location when repeating a saved experiment. With `output: null`, results remain in memory.

`Experiment.prepare` needs at least one attack to supply selected segments. For an individual
clean run, use `ClosedLoopRunner` and an explicit `RunRequest`.

## Existing CLI and configured runners

The CLI and the entry points below still accept the existing configuration format. They do not
read the `Experiment` schema above. Their Python imports are under `legacy_experiment` while
these entry points remain in use.

### Choose an entry point

These entry points run one [prepared scenario](../concepts/architecture.md#terminology) under one or
more [treatments](../concepts/architecture.md#terminology) that differ in their enabled
legacy interventions. Each follows the same
workflow: prepare the scenario, prepare the enabled interventions, run each treatment from the
same origin with a fresh driving system, and evaluate the recorded runs. The entry points differ
in how much of this workflow they do for you:

| Entry point | Use it when | It does for you |
|---|---|---|
| `avsectester run CONFIG` | You have a YAML configuration and want printed summaries, a verdict and an exit code | Everything below, plus output lines, the optional plot and the exit code |
| `run_configured(config, frames, ...)` | You configure the experiment as a mapping from Python, or combine a selected scenario with configured interventions | Scenario setup, intervention preparation, treatment selection and execution. Returns traces |
| `run_prepared(scenario, stack_factory, frames, runtimes=...)` | You build the runtimes yourself, or need treatments other than `single` and `pair`, such as three treatments | Backend construction and teardown, scenario conditions, one shared origin |
| `run_pair(backend, ...)`, `run(backend, ...)` | You construct and own the backend yourself | Execution only. Scenario conditions are not applied |

The CLI calls `run_configured`, which calls `run_prepared`, which runs every treatment with the
same loop as `run`. Signatures, options and result keys are in
[Driving loop and runners](../reference/python-api.md#driving-loop-and-runners). For CARLA,
`run_scenario` and `run_scenario_pair` also assemble the backend and the modular driving system.
See [Python entry points](simulators.md#python-entry-points).

### Experiment configuration

`avsectester run` and `run_configured` read one flat mapping. The supplied
[`configs/carla_scenario.yaml`](../../configs/carla_scenario.yaml) is a complete example. Its
top-level keys fall into these groups:

| Group | Keys | Reference |
|---|---|---|
| Execution | `frames`, `mode`, `treatment` | [Experiment keys](../reference/configuration.md#cli-experiment-keys) |
| Scene | `client`, `ego`, `npcs`, `lead` for CARLA. `backend` for NuRec | [CARLA scene keys](../reference/configuration.md#carla-scene-keys), [NuRec backend keys](../reference/configuration.md#nurec-backend-keys) |
| Driving system | `stack`, or `ego.pipeline` for the modular driving system | [Driving system keys](../reference/configuration.md#driving-system-keys) |
| Interventions | `attacks`, `patches`, `interventions`, `runtime`, `clean_runtime` | [Intervention keys](../reference/configuration.md#intervention-keys) |
| Conditions | `conditions` | [Condition keys](../reference/configuration.md#condition-keys) |
| Selection | `selection` | [Selection keys](../reference/configuration.md#selection-keys) |

Scene keys go to the simulator adapter. The other groups configure the experiment. See
[File layout](../reference/configuration.md#file-layout). Configured entry points check the
configuration before the first reset, in the order given in
[Validation rules](../reference/configuration.md#validation-rules). Values passed on to the
simulator and the models are checked only when the runs start. Loading a configuration imports
the Python factories it names, so use configurations you trust.

### Treatments and modes

The mode decides how many runs an experiment has:

- `single` runs the selected treatment once.
- `pair` runs a clean baseline and then the selected treatment. Both runs start from the same
  initial scene, each with a fresh driving system.

The treatment decides which configured interventions are enabled:

| Treatment | Enabled interventions |
|---|---|
| `clean` | None. The run uses the baseline runtime |
| `attack` | `interventions` entries of kind `attack`, the convenience `attacks` and `patches` entries, and raw `runtime.hooks` |
| `defense` | `interventions` entries of kind `defense` |
| `attack-defense` | Both kinds, in their configured order |

The default is `pair` with `attack`. `--mode` and `--treatment`, or the `mode=` and `treatment=`
keywords of `run_configured`, override the `mode` and `treatment` keys.

In the combined treatment (`attack-defense`), hooks at a shared stage run in configured order:
`attacks` and `patches`, then `runtime.hooks`, then `interventions` in definition order. The
framework does not move defenses after attacks. Place a defense after the attack it should see in
the `interventions` list. See [Ordering at a shared stage](../reference/runtime.md#ordering-at-a-shared-stage)
and [Order defenses relative to attacks and conditions](defenses.md#order-defenses-relative-to-attacks-and-conditions).

These rules decide how a configured experiment is assembled:

- **Preparation.** Enabled interventions are prepared once per experiment. Each run builds fresh
  handlers from the prepared artifacts. Disabled entries are format-checked but not imported,
  constructed or prepared.
- **Required kinds.** A non-clean treatment needs an enabled intervention of each kind it names.
  Otherwise the experiment stops before any run with a `ConfigurationError` such as
  `No enabled intervention configured for ['defense']`. Raw `runtime.hooks` always count as
  attacks. Declare defenses under `interventions` with `kind: defense`.
- **Empty attack pair.** A `pair` with treatment `attack` and no attack hooks runs as a comparison
  of two baselines instead of failing. The CLI prints a notice and labels the second run
  `baseline`. A `single` attack run without an attack is a configuration error.
- **Baseline runtime.** The default clean runtime has no hooks. It keeps `runtime.seed` and builds
  its own instance of `runtime.localizer`. A configured `clean_runtime` replaces it, and a Python
  `clean_runtime=` argument takes precedence over both. A `clean` pair runs two baselines, named
  `clean` and `baseline`. A Python `clean_runtime` object is used in both of these runs, so it
  must reset its own state.
- **Explicit treated runtime.** A Python `treated_runtime=` replaces all configured
  interventions: `attacks`, `patches`, `runtime.hooks` and `interventions`. Nothing is prepared,
  the required kinds are not checked and the provenance lists no interventions. The result keeps
  the treatment name, so the caller decides what the runtime represents. Without a
  `clean_runtime`, the default baseline shares that runtime's localizer object, seed and `case`. A
  stateful localizer must then reset itself, or you supply an independent `clean_runtime`.
- **Conditions.** World and sensor conditions apply identically to every treatment. See
  [Order relative to attacks and defenses](conditions.md#order-relative-to-attacks-and-defenses).

Comparing clean with `attack-defense` measures the attack's remaining impact. Measuring a
defense's effectiveness also needs an attack-only run from the same origin. Run such experiments
from Python, as in [Custom treatments](#custom-treatments).

**Not yet implemented.** Configured and CLI runs cannot compare clean, attack and attack-defense
treatments from one origin.

### Run from the command line

`avsectester run` runs a configured experiment and prints its results:

```bash
avsectester run configs/carla_scenario.yaml --mode pair --treatment attack --frames 40
```

On top of `run_configured`, the CLI generates and prints the CARLA scene seed, prints a summary of
each run, computes the verdict for a pair with an enabled attack, optionally saves a plot, and
reports the outcome as its exit code. Options, output lines and exit codes are listed in the
[CLI reference](../reference/cli.md). A first run is walked through in the
[Quickstart](../getting-started/quickstart.md#run-a-carla-comparison).

### Run from Python

The Python entry points return traces and compute no verdict. Evaluate the traces as described in
[Evaluate results](evaluation.md). The examples in this section run offline in the core
installation, with NuRec and the stub renderer.

#### Configured runs

`run_configured(config, frames, ...)` runs a configured experiment and returns a
`dict[str, Trace]` keyed by treatment name. This example configures the supplied
`SpeedFeedbackAttack`, which inflates the model-visible speed, and passes a minimal driving system
as `stack_factory`:

<!-- runnable -->
```python
from avsectester.closed_loop.avstack import AVStack
from avsectester.legacy_experiment import run_configured
from avsectester.closed_loop.datatypes import Control, Observation


class CruiseControl(AVStack):
    """Accelerate until the reported speed reaches 5 m/s, then hold it."""

    @property
    def command_modes(self) -> frozenset[str]:
        return frozenset({"actuator"})

    def __call__(self, observation: Observation) -> Control:
        return Control(throttle=0.5) if observation.ego_speed < 5.0 else Control()


config = {
    "backend": {"type": "NuRecSimulator", "renderer": "StubRenderer", "dt": 0.1},
    "interventions": [{
        "factory": "avsectester.attacks.adaptive:SpeedFeedbackAttack",
        "kind": "attack",
        "name": "speed-feedback",
        "options": {"target_speed": 2.0, "gain": 0.5, "max_bias": 5.0},
    }],
}

results = run_configured(config, 40, mode="pair", treatment="attack", stack_factory=CruiseControl)
for name, trace in results.items():
    names = [item["name"] for item in trace.provenance["interventions"]]
    print(f"{name}: final_speed={trace.final_speed:.2f} interventions={names}")
```

```text
clean: final_speed=5.10 interventions=[]
attack: final_speed=0.90 interventions=['speed-feedback']
```

The attacked ego believes it drives faster than it does, so it stops accelerating early.
`frames` is an argument here. `run_configured` ignores the `frames` key, which only the CLI
reads. Without `stack_factory`, the driving system comes from `stack` or `ego.pipeline`. A
`run_fn` passed to `run_configured` must return a `Trace`. A configured NuRec run with Alpamayo
is shown in [Backend configuration](simulators.md#backend-configuration).

To run configured interventions on a scenario you selected in Python, call
`run_configured(config, frames, scenario=scenario)`. The scenario's backend factory, match and
conditions are used unchanged, and the configuration's scene, selection and condition settings are
not reapplied. See
[Use a selected scenario in an experiment](scenarios.md#use-a-selected-scenario-in-an-experiment).

#### Custom treatments

For treatments that `run_configured` does not offer, build the runtimes from the configuration
and pass them to `run_prepared`. This example adds a toy defense to the previous configuration and
runs three treatments: clean, attack and attack-defense. The defense must be importable, so save
it as `my_defenses.py` next to your script:

<!-- runnable -->
```python
from dataclasses import replace

from avsectester.preparation import Intervention
from avsectester.closed_loop.hooks import Hook


class ReportedSpeedClamp(Intervention):
    """A toy defense: cap the model-visible speed and record each correction."""

    def __init__(self, max_speed: float) -> None:
        self.max_speed = max_speed

    def hooks(self, artifacts):
        def clamp(observation, context):
            if observation.ego_speed <= self.max_speed:
                return observation
            context.record_output("speed_clamped", observation.ego_speed)
            return replace(observation, ego_speed=self.max_speed)

        return [Hook("observation", clamp)]
```

Then continue the configured-run example:

```python
from avsectester.interventions import (
    configured_clean_runtime, configured_runtime, prepare_from_config,
)
from avsectester.runner import run_prepared
from avsectester.scenarios import scenario_from_config

config["interventions"].append({
    "factory": "my_defenses:ReportedSpeedClamp",  # my_defenses.py must be importable
    "kind": "defense",
    "options": {"max_speed": 3.0},
})
scenario = scenario_from_config(config)
prepared = prepare_from_config(config, scenario, kinds={"attack", "defense"})
results = run_prepared(scenario, CruiseControl, 40, runtimes={
    "clean": configured_clean_runtime(config, case=scenario.match),
    "attack": configured_runtime(config, instance=scenario, prepared=prepared, kinds={"attack"}),
    "attack-defense": configured_runtime(
        config, instance=scenario, prepared=prepared, kinds={"attack", "defense"},
    ),
})
for name, trace in results.items():
    print(f"{name}: final_speed={trace.final_speed:.2f} outputs={len(trace.outputs)}")
```

```text
clean: final_speed=5.10 outputs=0
attack: final_speed=0.90 outputs=0
attack-defense: final_speed=6.00 outputs=36
```

`prepare_from_config` prepares both interventions once. Each `configured_runtime` call builds
fresh handlers from those artifacts for the selected kinds. Without `prepared=`, it prepares the
interventions again. `run_prepared` returns results under the names in `runtimes`, applies the
scenario's conditions to every runtime and starts all three runs from one origin. The toy
defense restores driving but overcorrects: the defended ego exceeds the 5 m/s that the clean run
holds. The attack-only run is the reference that shows how much of the attack the defense removes.

#### Direct runs and resource ownership

`run` and `run_pair` take a backend that you construct:

- `run(backend, stack, frames, runtime=...)` executes one run. The caller owns the backend and
  closes it. See the [offline quickstart example](../getting-started/quickstart.md#run-offline-without-a-gpu).
- `run_pair(backend, stack_factory, frames, clean_runtime=..., attacked_runtime=...)` calls
  `backend.prepare_clean_attack_pair()` once, so that both runs start from the same initial scene,
  and runs the clean and then the attacked runtime, each with a fresh stack from
  `stack_factory()`. It closes the backend on success or failure. Pass `close_backend=False` to
  keep ownership.
- `run_prepared(scenario, ...)` creates its backend with `scenario.make_backend()` and always
  closes it.

Each runtime argument accepts a `Runtime`, `None` for a run without handlers, or a factory
`backend -> Runtime`. A factory is called after pair preparation and before the first reset, so
it cannot see live actors. Resolve those in world hooks. `run_fn` replaces `run`, for example with
`run_logged` to capture [component outputs](evaluation.md#component-outputs). It must accept
`runtime=`. The runners return its results and compute no metric.

Before preparing the origin, `run_pair` builds the first stack and checks the stage availability
and command compatibility of every supplied runtime against it. Factory-built runtimes are checked
as soon as they are built, so both treatments pass before the clean run starts. Later runs use
fresh stacks, so `stack_factory` must return stacks with the same capabilities. Each run checks
its actual stack again before reset.

This example continues the configured-run example. It reuses one NuRec backend for two pairs
and passes the attacked runtime as a factory:

```python
from dataclasses import replace

from avsectester.runner import run_pair
from avsectester.closed_loop.hooks import Hook, Runtime
from avsectester.closed_loop.simulators.nurec import NuRecSimulator, StubRenderer


backend = NuRecSimulator({"dt": 0.1}, renderer=StubRenderer()).create_runtime()
try:
    for bias in (1.0, 3.0):
        def spoof_speed(observation, context):  # report `bias` m/s more than estimated
            return replace(observation, ego_speed=observation.ego_speed + bias)

        clean, attacked = run_pair(
            backend, CruiseControl, 40,
            attacked_runtime=lambda _backend: Runtime([Hook("observation", spoof_speed)]),
            close_backend=False,  # the caller reuses and closes the backend
        )
        print(f"bias={bias} clean={clean.final_speed:.2f} attacked={attacked.final_speed:.2f}")
finally:
    backend.close()
```

```text
bias=1.0 clean=5.10 attacked=4.05
bias=3.0 clean=5.10 attacked=2.10
```

`run` and `run_pair` do not apply a scenario's world or sensor conditions. To use them with a
prepared scenario, see [Lower-level execution](conditions.md#lower-level-execution), or use
`run_prepared`. What the shared origin covers for each simulator is described in
[Paired runs on one server](simulators.md#paired-runs-on-one-server).

### Reproducibility

Treatments of a configured experiment share the initial scene, conditions and seeds. Separate
experiments repeat only what their seeds fix:

- **Scene.** CARLA resolves random scene choices once per pair from `client.seed`. A fixed seed
  repeats them across commands. The CLI prints the seed it used, and paired CARLA traces store the
  resolved scene in `trace.replay_scenario`. See
  [Random choices and seeds](simulators.md#random-choices-and-seeds).
- **Handlers.** `runtime.seed` (default `0`) seeds the handlers' `context.rng`, afresh for every
  run. It does not seed the simulator or the models. See
  [Randomness](../reference/runtime.md#randomness).
- **Sensor conditions.** Sensor conditions use their own seeds and draw the same values in every
  treatment. See [Randomness across treatments](conditions.md#randomness-across-treatments).

Model inference and simulator physics can still differ between runs. A fixed seed repeats the
starting point, not the complete driving trace. To reproduce a run later, keep its configuration
together with its provenance, described below.

### Outputs and provenance

Each run produces a `Trace` with:

- `records`: the physical state and executed command at each driving step.
- `outputs`: diagnostic values recorded by handlers, such as defense alarms.
- `provenance`: how the run was produced. The runners record the treatment name, runtime seed and
  stages, the scenario's provenance and conditions, and, for `run_configured`, the prepared
  interventions. The CLI adds `config_path`.
- `replay_scenario`: the resolved CARLA scene of the run, or `None` for other simulators.

Traces are held in memory. The CLI prints a summary of each trace and saves only the optional
plot, so save what you need from Python. How to read each part, compute the verdict and capture
component outputs is described in [Evaluate results](evaluation.md), in particular
[Driving traces](evaluation.md#driving-traces) and
[Intervention outputs and provenance](evaluation.md#intervention-outputs-and-provenance). The
keys each runner writes are listed in
[Provenance written by runners](../reference/python-api.md#provenance-written-by-runners).

## Next steps

- [Prepare scenarios](scenarios.md): configure, select and insert objects into scenarios.
- [Implement an attack](attacks.md) and [Implement a defense](defenses.md): write the
  interventions that treatments enable.
- [Apply operating conditions](conditions.md): world and sensor conditions shared by treatments.
- [Evaluate results](evaluation.md): traces, verdicts, outputs and robustness summaries.
- [CLI reference](../reference/cli.md) and [Configuration reference](../reference/configuration.md).
