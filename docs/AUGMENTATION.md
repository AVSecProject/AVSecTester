# Scenario conditions and attack robustness

Augmentation applies controlled weather, lighting and sensor corruptions to examine how an
attack behaves across corruption settings. The evaluation harness compares clean and attacked
runs under each setting. `ScenarioConditions` holds the shared world and sensor environment.
The robustness grid adds image corruptions to that environment.

## Apply corruptions as scenario conditions

Attach operating conditions to a `ScenarioInstance`. The prepared runner constructs a fresh
condition handler for every treatment and applies the same settings and relative-step seeds.
Attack and defense handlers use the ordinary Runtime interface:

```python
from dataclasses import replace
from avsectester.conditions import ScenarioConditions, SensorCondition
from avsectester.perturbations.image import AugmentationPipeline, GaussianNoise
from avsectester.runner import run_prepared
from avsectester.runtime import Hook, Runtime

# scenario is a ScenarioInstance, attack(value, context) is an observation handler.
noise = AugmentationPipeline([GaussianNoise(severity=0.2)], seed=7)
conditions = ScenarioConditions(
    world=scenario.conditions.world,
    sensors=scenario.conditions.sensors + (SensorCondition.image(noise, camera="front"),),
    seed=scenario.conditions.seed,
)
conditioned = replace(scenario, conditions=conditions)
traces = run_prepared(conditioned, stack_factory, frames=30, runtimes={
    "clean": Runtime(),
    "attack": Runtime([Hook("observation", attack)]),
})
```

`SensorCondition` defaults to `stage="observation", order="after"`: corruption runs after all
observation attacks and defenses, including inserted attack pixels. Set `order="before"` to
corrupt the measurement before all interventions at that same stage. Hooks at different stages
retain the pipeline's stage order. Conditions have independent random streams, so an attack
consuming `context.rng` does not change condition noise.
Custom sensor-condition factories can implement the same handler/Plugin interface. World conditions
are applied before reset. See [SCENARIOS.md](SCENARIOS.md#scenario-conditions) for Python and YAML
configuration, selection measurements and supported backend properties.

For direct `run` or `run_pair` calls, build the backend with `scenario.make_backend()` and wrap
each Runtime with `scenario.conditions.apply_runtime(runtime)`. Otherwise sensor conditions
are not applied. See the [lower-level execution example](SCENARIOS.md#lower-level-execution).

For a single run, `sensor_augmentation(pipeline)` creates a one-argument observation transform,
and `run(..., perturb=compose(attack, augment))` composes such transforms. `compose(a, b)` applies
`a` then `b`, through the same Runtime chain. `sensor_augmentation` seeds the pipeline from
absolute `observation.frame`, so it does not guarantee matching noise across resets. Use
`SensorCondition.image(...)` or YAML `conditions.sensors` for paired experiments.

## Two tiers of fidelity

1. **Image-space corruptions** — `avsectester.perturbations.image`, backend-agnostic. Each `Corruption` is an
   `apply(rgb, rng) -> rgb` on the camera image. An `AugmentationPipeline` chains several operators.
   Attach it with `SensorCondition.image(...)` or configure YAML `conditions.sensors`. It works on
   camera images from CARLA, NuRec reconstructed-scene rendering (`NuRecBackend`), and recorded-frame
   replay (`RecordedFrameBackend`, used by nuScenes). NuRec generates new views from the ego pose,
   while recorded-frame replay returns captured images.
2. **CARLA world-weather** — `carla.py`. `conditions.world.weather` (or legacy `weather`) sets native
   `carla.WeatherParameters` (cloud, precipitation, sun altitude, fog, wetness) at `reset`, so the whole
   scene, including native physical actors, is rendered under that weather. Post-render image
   insertions do not automatically receive native weather effects. This option is CARLA-only.

## Operators (`avsectester.perturbations.image`)

Every operator takes a `severity` in `[0, 1]` (0 ≈ identity, 1 = strong) and a
`numpy.random.Generator`. `SensorCondition.image` supplies matching seeds at corresponding
driving steps in clean and attacked runs.

| category  | operators                                                             |
|-----------|-----------------------------------------------------------------------|
| weather   | `Fog`, `Rain`, `Snow`                                                 |
| lighting  | `Brightness`, `LowLight`, `Contrast`, `Gamma`, `ColorTemp`           |
| sensor    | `GaussianNoise`, `ShotNoise`, `MotionBlur`, `DefocusBlur`, `JPEGCompression`, `ChromaticAberration` |

`CORRUPTIONS` is a name-to-class registry. `common_corruptions(severity)` returns one pipeline
per built-in operator. This is the framework's default corruption grid, not an implementation
of a published benchmark.

### Backends: hand-rolled (default) or Albumentations (optional)

The default operators use NumPy and OpenCV without a learned model. An optional
**Albumentations** backend (`AlbumentationsCorruption`,
`albumentations_corruptions(severity)`, `augment` extra) that wraps `A.RandomFog/RandomRain/RandomSnow/
GaussNoise/ISONoise/MotionBlur/Defocus/ImageCompression` behind the same `Corruption` interface, seeding
`A.Compose` from the supplied RNG. With `SensorCondition.image`, that RNG is seeded by the
pipeline seed and relative driving step. The `augment` extra
selects classic Albumentations (`>=1.4,<2`). See [SETUP.md](SETUP.md) for the dependency set.

## Randomness across treatments

Each comparison holds the corruption settings fixed. `AugmentationPipeline.apply(rgb, frame)`
seeds its RNG from `(seed, frame)`. `SensorCondition.image` supplies relative `context.step` as
that frame index, because absolute simulator frame IDs can differ after reset. Corresponding
steps with the same image dimensions use the same random draws. Their output images can differ
because the attack or subsequent driving changes the input.

Custom sensor handlers receive independent condition RNG streams. Use `context.rng` and reset
run-local state to preserve matching random choices. The framework does not seed arbitrary
external libraries or reset state hidden in a user closure.

## Robustness evaluation

`evaluate_robustness(source, req, attack_for, stack, frames, ...)` runs the grid and returns a
`RobustnessReport` (ASR and resilience per corruption setting). `stack` is a factory
(`Callable[[], AVStack]`) so each run gets a fresh stack. Given a scenario source, requirement,
attack method and stack factory:

```python
from avsectester.evaluation.robustness import evaluate_robustness

report = evaluate_robustness(source, req, method, stack_factory, frames=30, severity=0.5)
print(report.summary())
```

| Parameter | Meaning |
|---|---|
| `conditions` | Iterable of `AugmentationPipeline`, not `ScenarioConditions`. `None` uses `common_corruptions(severity)`. Supply `albumentations_corruptions(...)` for that implementation, or `[]` for the baseline only |
| `severity` | Strength of the default corruption grid, default `0.5` |
| `limit` | Maximum number of accepted scenarios from the source, or `None` for all |
| `run_fn` | Per-treatment recorder accepting `runtime` and returning a driving `Trace`. Defaults to the shared loop |

The grid always includes a no-extra-corruption baseline, labelled `clean`, followed by the
specified corruption pipelines. Existing `ScenarioConditions` still apply to that baseline.
Every grid entry contains its own clean and attacked treatment. For each pair the harness uses
the shared prepared-run execution and cleanup. Added image corruptions follow all observation
interventions.

`method` may be an `Intervention` or a `PreparedIntervention` of kind `attack`. An Intervention
prepares once per selected scenario, and its artifacts are reused across conditions. Execution
handlers are constructed separately for each attacked run. Passing a PreparedIntervention reuses
its supplied artifacts explicitly, including across scenarios. Preparation is never inferred from
an artifact filename or a previous run. See [INTERVENTIONS.md](INTERVENTIONS.md#preparation-and-execution).

Construct `source` and `req` using [scenario selection](SCENARIOS.md#select-a-case). For a source
from `source_from_config`, pass `req=None` to use its configured requirement, or to run its
unfiltered scenario. An independently constructed `DatasetFilter` or `CarlaScenarioBuilder`
receives the requirement explicitly as in the example above.
Source filters run before this grid, not on every corrupted observation. To filter by conditioned
measurements, configure [conditions on the source](SCENARIOS.md#conditions-during-selection)
before selecting its cases. Changing conditions after selection does not recheck filters.

The legacy `attack_for(match, backend)` callback also remains supported. It is called once per pair,
after world conditions are configured and before origin preparation or backend reset. It returns
a Runtime, an observation transform, or
None to skip. The default clean runtime shares this Runtime's localizer and seed without its hooks.
The framework invokes `reset(context)` only when the component implements it. Stateful localizers
shared by the legacy callback must implement reset. For a closure without reset, construct
independent clean/attacked localizers in `run_prepared` and build the grid in the caller. The legacy
callback has no separate baseline factory. Insertion adapters may now be constructed before reset,
and resolve current camera/actor information during execution. Preserve `match.binding_ids` and
`match.insertions` rather than selecting a new target.

`RobustnessReport.rows` retains each attempt's status and reason:

- `success` / `failure`: a valid driving baseline, scored by `metric.impact`.
- `skipped`: `attack_for(match, backend)` returned `None`.
- `inconclusive`: the clean run did not establish a driving baseline.

**ASR per corruption setting** is `successes / (successes + failures)`. Skipped and inconclusive attempts are
reported separately and excluded from the denominator. With no valid pairs, ASR is `NaN` and the
summary displays `N/A`. Direct callers of `report.record()` must pass a status string, not a boolean.

**Resilience** is `ASR(corruption) / ASR(clean)`, capped at 1. Here `ASR(clean)` refers to the
grid's no-extra-corruption baseline, not to a clean treatment's success rate.
It is `NaN` if either setting has no valid pairs, and 0 when the valid baseline ASR is 0.
These are the current implementation's conventions.
The current success criterion covers induced or suppressed stops. Experiment-specific criteria remain
tracked in [TODO.md](TODO.md#driving-impact-metrics), so this is not a universal definition of
attack success.

## Fidelity and extensions

- Image-space weather is a *2-D approximation*: fog is uniform airlight (no per-pixel depth), rain/snow
  are procedural overlays. It is a stress test of the *perception+attack* pipeline, not a physically
  exact renderer. Use the CARLA world tier when physical light transport matters.
- `Corruption.apply` consumes an RGB camera image. LiDAR, radar and navigation conditions can
  use custom `SensorCondition` factories operating on their native payloads. They do not use
  the image-corruption registry, and no built-in algorithms for those modalities are supplied.
