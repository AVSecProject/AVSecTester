# Scenario augmentation — perturbing the world to test attack robustness

Augmentation applies controlled weather, lighting and sensor corruptions to examine how an
attack behaves across conditions. The evaluation harness compares clean and attacked runs
under the same corruption settings.

## Register attacks and corruptions in Runtime

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

`SensorCondition` defaults to `stage="observation", order="after"`: corruption affects the
inserted attack pixels. Set `order="before"` to corrupt the measurement before interventions at
that same stage. Hooks at different stages retain the pipeline's stage order. Conditions have
independent random streams, so an attack consuming `context.rng` does not change condition noise.
Custom sensor-condition factories can implement the same handler/Plugin interface. World conditions
are applied before reset. See [SCENARIOS.md](SCENARIOS.md#scenario-conditions) for Python and YAML
configuration, selection measurements and supported backend properties.

Direct callers can also register ordered `observation` hooks in Runtime and use `run_pair`.
For a single run, `run(..., perturb=compose(attack, augment))` remains a shorthand for
one-argument observation transforms. `compose(a, b)` applies `a` then `b`. The shorthand is
adapted into the same runtime hook chain.

## Two tiers of fidelity

1. **Image-space corruptions** — `avsectester.perturbations.image`, backend-agnostic. Each `Corruption` is an
   `apply(rgb, rng) -> rgb` on the camera image; an `AugmentationPipeline` chains several and
   `sensor_augmentation(pipeline)` lifts it to an observation transform. Because it operates on pixels it works on
   camera images from CARLA, NuRec reconstructed-scene rendering (`NuRecBackend`), and recorded-frame
   replay (`RecordedFrameBackend`, used by nuScenes). NuRec generates new views from the ego pose,
   while recorded-frame replay returns captured images.
2. **CARLA world-weather** — `carla.py`. `conditions.world.weather` (or legacy `weather`) sets native
   `carla.WeatherParameters` (cloud, precipitation, sun altitude, fog, wetness) at `reset`, so the whole
   scene, including native physical actors, is rendered under that weather. Post-render image
   insertions do not automatically receive native weather effects. This option is CARLA-only.

## Operators (`avsectester.perturbations.image`)

Every operator takes a `severity` in `[0, 1]` (0 ≈ identity, 1 = strong) and is driven by a seeded
`numpy.random.Generator`, so runs are reproducible and — crucially — the *same* corruption can be applied
to the clean and attacked runs of a pair.

| category  | operators                                                             |
|-----------|-----------------------------------------------------------------------|
| weather   | `Fog`, `Rain`, `Snow`                                                 |
| lighting  | `Brightness`, `LowLight`, `Contrast`, `Gamma`, `ColorTemp`           |
| sensor    | `GaussianNoise`, `ShotNoise`, `MotionBlur`, `DefocusBlur`, `JPEGCompression`, `ChromaticAberration` |

`CORRUPTIONS` is a name-to-class registry. `common_corruptions(severity)` returns one pipeline
per built-in operator. This is the framework's default condition set, not an implementation
of a published benchmark.

### Backends: hand-rolled (default) or Albumentations (optional)

The default operators use NumPy and OpenCV without a learned model. An optional
**Albumentations** backend (`AlbumentationsCorruption`,
`albumentations_corruptions(severity)`, `augment` extra) that wraps `A.RandomFog/RandomRain/RandomSnow/
GaussNoise/ISONoise/MotionBlur/Defocus/ImageCompression` behind the same `Corruption` interface, seeding
`A.Compose` from our own RNG so the `(seed, frame)` pairing still holds. The `augment` extra
selects classic Albumentations (`>=1.4,<2`). See [SETUP.md](SETUP.md) for the dependency set.

## Determinism across the pair (the subtle bit)

Robustness is measured by *diffing* a clean run and an attacked run **under the same condition**. If the
fog pattern or noise realisation differed between the two, the diff would conflate the attack with the
corruption. `AugmentationPipeline.apply(rgb, frame)` seeds its RNG from `(seed, frame)`. The condition handler uses the same pipeline seed and relative driving step, since absolute
simulator frame IDs can differ after reset. Corresponding steps therefore use the same random draws,
even when their input images differ because of the attack or subsequent driving.

## Robustness evaluation protocol (implemented: `avsectester.evaluation.robustness`)

`evaluate_robustness(source, req, attack_for, stack, frames, ...)` runs the grid and returns a
`RobustnessReport` (ASR + resilience per condition). `stack` is a **factory** (`Callable[[], AVStack]`)
so each run gets a fresh box. Given a scenario source, requirement, attack method and stack factory:

```python
from avsectester.evaluation.robustness import evaluate_robustness

report = evaluate_robustness(source, req, method, stack_factory, frames=30, severity=0.5)
print(report.summary())
```

The harness includes a no-extra-corruption baseline plus the corruption suite. Existing scenario
conditions still apply to that baseline. For each scenario and condition it delegates execution and
teardown to `runner.run_prepared`. The added image corruption follows observation attacks.

`method` may be an `Intervention` or a `PreparedIntervention` of kind `attack`. An Intervention
prepares once per selected scenario, and its artifacts are reused across conditions. Execution
handlers are constructed separately for each attacked run. Passing a PreparedIntervention reuses
its supplied artifacts explicitly, including across scenarios. Preparation is never inferred from
an artifact filename or a previous run. See [INTERVENTIONS.md](INTERVENTIONS.md#preparation-and-execution).

Construct `source` and `req` using [scenario selection](SCENARIOS.md#select-a-case).
Source filters run before this grid, not on every corrupted observation. To filter by conditioned
measurements, configure conditions on the source before selecting its cases.

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

**ASR per condition** is `successes / (successes + failures)`. Skipped and inconclusive attempts are
reported separately and excluded from the denominator. With no valid pairs, ASR is `NaN` and the
summary displays `N/A`. Direct callers of `report.record()` must pass a status string, not a boolean.

**Resilience** is `ASR(corruption) / ASR(clean)`, capped at 1. It is `NaN` if either condition has no
valid pairs, and 0 when the valid baseline ASR is 0. These are the current implementation's conventions.
The current success criterion covers induced or suppressed stops. Experiment-specific criteria remain
a TODO in the harness, so this is not a universal definition of attack success.

## Fidelity and extensions

- Image-space weather is a *2-D approximation*: fog is uniform airlight (no per-pixel depth), rain/snow
  are procedural overlays. It is a stress test of the *perception+attack* pipeline, not a physically
  exact renderer. Use the CARLA world tier when physical light transport matters.
- Corruptions act on the **camera** stream. Lidar/other-modality corruptions are a later addition (a
  `Corruption` on those payloads slots into the same registry).
