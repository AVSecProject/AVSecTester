# Apply operating conditions

> **Audience:** Users who test attacks or defenses under weather, lighting or sensor degradation.  
> **Outcome:** Configure world and sensor conditions shared by every treatment, and control their order and randomness.  
> **Prerequisites:** [Architecture](../concepts/architecture.md) and [Prepare scenarios](scenarios.md).

A [condition](../concepts/architecture.md#terminology) sets the operating environment of a prepared
scenario, such as rain, fog, low light or camera noise. Every
[treatment](../concepts/architecture.md#terminology) of the scenario runs under the same
conditions, so a comparison between treatments measures only the enabled interventions.

## Conditions and filters

Conditions and filters both mention weather, but they do different things:

| | Condition | Filter |
|---|---|---|
| Purpose | Imposes an environment on every treatment | Chooses among candidate initial states |
| When it acts | During every run, and optionally during selection | Only during selection, never inside a run |
| Effect | Changes the world (world condition) or the sensor data (sensor condition) | Never changes a state. Returns pass, fail or unknown |
| Code | `ScenarioConditions`, `SensorCondition` | `Constraint`, `FilterResult` |

A recorded night or rain label is a property of the recording. Select it with a filter (see
[Filters](scenarios.md#filters)). A condition adds an effect on top of whatever the scene already
contains. A setting that differs between treatments, such as a weather change made only in the
attacked run, is an intervention. Write it as an attack (see
[Change the world](attacks.md#change-the-world)).

## Configure conditions

In an experiment configuration, conditions live under the top-level `conditions` key:

```yaml
conditions:
  seed: 12
  world:
    weather: {cloudiness: 70, precipitation: 20, fog_density: 15}
  sensors:
    - {type: GaussianNoise, severity: 0.2}
    - {type: LowLight, severity: 0.4, stage: sensors.post, order: before}
```

`world` holds world conditions for the simulator. `sensors` is an ordered list of sensor
conditions. A built-in entry corrupts the camera named by `camera`, a key of
`Observation.sensor_data`, or the first readable RGB camera when `camera` is omitted. `seed` is the
default seed of the sensor conditions. Every key, type and default is in
[Condition keys](../reference/configuration.md#condition-keys). `scenario_from_config` and
`run_configured` read this section, and `scenario.provenance["conditions"]` records the result.

In Python, build a `ScenarioConditions` and attach it to a prepared scenario with
`dataclasses.replace`. `SensorCondition.image(pipeline, camera=...)` wraps an
`AugmentationPipeline` of image corruptions. This example runs offline with the NuRec stub
renderer, whose camera is named `camera_front`:

<!-- runnable -->
```python
from dataclasses import replace

from avsectester.closed_loop.avstack import AVStack
from avsectester.conditions import ScenarioConditions, SensorCondition
from avsectester.perturbations.image import AugmentationPipeline, Fog, GaussianNoise
from avsectester.closed_loop.datatypes import Control
from avsectester.runner import run_prepared
from avsectester.scenarios import scenario_from_config


class Cruise(AVStack):
    """A minimal driving system that always applies light throttle."""

    def __call__(self, observation):
        return Control(throttle=0.3)


scenario = scenario_from_config({"backend": {"type": "NuRecSimulator", "renderer": "StubRenderer"}})
haze = AugmentationPipeline([Fog(severity=0.3), GaussianNoise(severity=0.2)], seed=7)
conditions = ScenarioConditions(
    world=scenario.conditions.world,                # Keep any configured world conditions.
    sensors=scenario.conditions.sensors + (SensorCondition.image(haze, camera="camera_front"),),
    seed=scenario.conditions.seed,
)
conditioned = replace(scenario, conditions=conditions)

results = run_prepared(conditioned, Cruise, frames=10)
sensor = results["clean"].provenance["conditions"]["sensors"][0]
print(sensor["camera"], [c["type"] for c in sensor["corruptions"]], sensor["seed"])
# camera_front ['Fog', 'GaussianNoise'] 7
```

`run_prepared` and `run_configured` apply both kinds of condition to every treatment. Each trace
records the conditions in `trace.provenance["conditions"]`, which is how you verify what a run
used. `ScenarioConditions` is also exported by `avsectester.scenarios`. The complete signatures
are in [Conditions](../reference/python-api.md#conditions).

## World conditions

A world condition changes the simulated world itself. The scenario passes `conditions.world` to
the backend's `configure_conditions` method before the first reset, and the backend applies it at
every reset. All treatments therefore render the same weather, including on the physical actors
and objects in the scene.

Only CARLA accepts a world condition, and only `weather`. The built-in NuRec and recorded-frame
(`RecordedReplaySimulator`, used by nuScenes) backends accept none. CARLA weather is either a
`carla.WeatherParameters` preset name, such as `weather: HardRainNoon`, or a mapping of CARLA's own
parameter names, such as `cloudiness`, `precipitation`, `sun_altitude_angle`, `fog_density` and
`wetness`. `CARLA_WEATHER_PRESETS` in `avsectester.closed_loop.simulators.carla` lists presets from
`ClearNoon` to `HardRainSunset`, and any other CARLA preset also works. An unknown preset name
raises `ValueError` at reset. A top-level `weather` key is another spelling of
`conditions.world.weather` (see [weather](../reference/configuration.md#weather)).

A backend that cannot apply a world condition rejects it instead of ignoring it.
`scenario.make_backend()` raises `ConditionConfigurationError` and closes the backend. For example,
NuRec raises `NuRecSimulator does not support world conditions: ['weather']`. Reconstructed and
recorded scenes cannot change their lighting or weather. Use sensor conditions on them instead, and
see [Fidelity limits](#fidelity-limits).

A custom simulator adapter supports world conditions by overriding
`configure_conditions(world)`. It must apply the settings it supports at every subsequent reset and
raise `ValueError` for the rest. See
[Write a custom simulator adapter](simulators.md#write-a-custom-simulator-adapter).

## Sensor conditions

A sensor condition is a handler that changes sensor data on its way to the driving system. It runs
at the `sensors.post` or `observation` stage and receives an `Observation`, like any handler at
those stages (see [Stages](../reference/runtime.md#stages)). It never changes the physical world.

`SensorCondition(factory, stage="observation", order="after", seed=None)` holds a factory, not a
handler. `run_prepared` calls the factory once for every treatment, so each treatment gets an
independent handler with its own state. `stage` is `"sensors.post"` or `"observation"`. `order`
places the handler `"before"` or `"after"` all intervention hooks at that stage. `seed=None` uses
`ScenarioConditions.seed`.

`SensorCondition.image(pipeline, camera=None, stage=..., order=...)` builds an image condition. It
copies the pipeline, so later changes to your pipeline object have no effect. The pipeline's own
`seed` becomes the condition seed, and `ScenarioConditions.seed` does not override it. With
`camera=None`, the first readable RGB camera is corrupted. A named camera that is missing raises
`ValueError` during the run.

Image corruptions work on camera images from CARLA, from NuRec rendering and from recorded-frame
replay. LiDAR, radar and other modalities need a custom sensor condition.

### Built-in image corruptions

The operators are in `avsectester.perturbations.image`. Each one is a `Corruption` with a
`severity` in `[0, 1]`, where 0 is close to the identity and 1 is strong. Values outside the range
are clipped in Python and rejected in YAML. The operators use NumPy and, where noted, OpenCV
(install the `rendering` extra, see
[Installation](../getting-started/installation.md#local-core-installation)). None uses a learned
model.

| Category | Operator | Effect |
|---|---|---|
| Weather | `Fog` | Uniform haze: blends toward a bright airlight and reduces contrast. No depth |
| Weather | `Rain` | Darkens the image and adds motion-blurred streaks. OpenCV |
| Weather | `Snow` | Brightens slightly and adds blurred bright flakes. OpenCV |
| Lighting | `Brightness` | Multiplicative brightening |
| Lighting | `LowLight` | Darkens the image and adds read noise |
| Lighting | `Contrast` | Pulls pixel values toward the image mean |
| Lighting | `Gamma` | Darkens mid-tones with a gamma curve |
| Lighting | `ColorTemp` | Colour cast. Option `warm` (default `True`); `False` gives a cool cast |
| Sensor | `GaussianNoise` | Additive Gaussian noise |
| Sensor | `ShotNoise` | Signal-dependent Poisson noise |
| Sensor | `MotionBlur` | Directional blur at a random angle. OpenCV |
| Sensor | `DefocusBlur` | Gaussian blur. OpenCV |
| Sensor | `JPEGCompression` | JPEG encoding at a quality that falls with severity. OpenCV |
| Sensor | `ChromaticAberration` | Shifts the red and blue channels apart |

`CORRUPTIONS` maps these names to their classes. A YAML `type` must be one of these names, and
`options` passes extra constructor arguments, such as `{warm: false}` for `ColorTemp`.
`AugmentationPipeline(corruptions, seed=0)` applies several operators in order.
`common_corruptions(severity)` returns one single-operator pipeline per entry of `CORRUPTIONS`.
It is the framework's default corruption grid, not an implementation of a published benchmark.

**Albumentations backend.** The optional `augment` extra installs Albumentations 1.x
(`albumentations>=1.4,<2`). `albumentations_corruptions(severity)` then returns one pipeline per
operator below, and `albumentations_available()` reports whether the package can be imported.
Without it, `albumentations_corruptions` raises `RuntimeError`.

| Category | Label and Albumentations transform |
|---|---|
| Weather | `Fog` (`RandomFog`), `Rain` (`RandomRain`), `Snow` (`RandomSnow`) |
| Sensor | `GaussNoise` (`GaussNoise`), `ISONoise` (`ISONoise`), `MotionBlur` (`MotionBlur`), `DefocusBlur` (`Defocus`), `JPEGCompression` (`ImageCompression`) |

Pipelines are named `alb:<label>@<severity>`. Each operator is an `AlbumentationsCorruption` that
maps `severity` to the transform's parameters and seeds Albumentations from the pipeline's
generator, so it follows the same seeding as the built-in operators. To wrap another transform,
pass `AlbumentationsCorruption(build, severity)`, where `build(severity)` returns an
Albumentations transform. These operators are not in `CORRUPTIONS`, so use them from Python with
`SensorCondition.image`, not from YAML.

### Custom sensor conditions

Write a custom sensor condition as a handler `handler(observation, context)` or as a `Plugin`, and
return the replaced `Observation`. This one delivers the previous camera image at random, as a
delayed camera would:

<!-- runnable -->
```python
from avsectester.perturbations.image import read_camera_rgb, write_camera_rgb
from avsectester.closed_loop.hooks import Plugin


class StaleCamera(Plugin):
    """Repeat the previous camera image with the given probability."""

    def __init__(self, probability=0.2, camera=None):
        self.probability, self.camera = probability, camera

    def reset(self, context):
        self.previous = None              # Run-local state, cleared for every treatment.

    def transform(self, observation, context):
        found = read_camera_rgb(observation, self.camera)
        if found is None:
            return observation
        key, image = found
        stale = context.rng.random() < self.probability
        previous, self.previous = self.previous, image
        if stale and previous is not None:
            return write_camera_rgb(observation, key, previous)
        return observation
```

In Python, attach it with a factory that builds a new instance on every call, such as
`SensorCondition(lambda: StaleCamera(0.2), stage="sensors.post")`. In YAML, name an importable
`module:callable`, which is called as `factory(**options)` for every treatment:

```yaml
conditions:
  seed: 3
  sensors:
    - factory: my_package.conditions:StaleCamera
      options: {probability: 0.2}
      stage: sensors.post
```

Follow three rules so that every treatment receives the same condition:

- **Return a fresh handler from the factory.** A factory that returns one shared object leaks state
  from one treatment into the next.
- **Draw random numbers only from `context.rng`.** The condition receives its own generator, which
  is independent of the attack and defense generator. Random numbers from another source differ
  between treatments.
- **Clear run-local state in `reset`.** `reset(context)` runs before each run, and `context.rng`
  there is seeded from the condition seed. `after_step` and `close` are also forwarded (see
  [Plugin lifecycle](../reference/runtime.md#plugin-lifecycle)).

## Order relative to attacks and defenses

Effects reach the driving system in this order:

1. World conditions and world-changing attacks act in the simulator, so the captured sensor data
   already contains them.
2. Handlers at `sensors.post` run before handlers at `observation`, whatever their `order`.
3. At one stage, sensor conditions with `order: before` run first, then all attack and defense
   hooks in their configured order, then sensor conditions with `order: after`.

The complete rule, including the `perturb` shorthand and intervention `position`, is in
[Ordering at a shared stage](../reference/runtime.md#ordering-at-a-shared-stage).

| Condition placement | Observation attack | Observation defense | Driving system |
|---|---|---|---|
| `observation`, `after` (default) | Changes the clean image | Sees the attacked image without the condition | Sees the attacked image with the condition |
| `observation`, `before` | Changes the conditioned image | Sees the conditioned, attacked image | Same as the defense |
| `sensors.post`, either order | Changes the conditioned image | Sees the conditioned, attacked image | Same as the defense |

The default keeps the attack-then-corruption order, so inserted attack pixels are degraded like
the rest of the image. As a consequence, an `observation` defense inspects an uncorrupted image
while the driving system receives a corrupted one. To evaluate a defense on the corrupted input,
set `order: before` or use `stage: sensors.post`. See
[Order defenses relative to attacks and conditions](defenses.md#order-defenses-relative-to-attacks-and-conditions).

## Randomness across treatments

Every treatment receives the same random draws from a condition at corresponding driving steps.
A condition handler's generator is seeded from the condition seed and the relative driving step,
`context.step`, not from simulator frame IDs, which can differ between runs.
`SensorCondition.image` passes the driving step to `AugmentationPipeline.apply(rgb, frame)`, which
seeds from `(seed, frame)`. Condition generators are independent of `Runtime.seed`, so an attack
that draws from `context.rng` does not change the condition's noise (see
[Randomness](../reference/runtime.md#randomness)). Equal draws do not mean equal images. Once an
attack changes the input or the vehicle drives differently, the corrupted images differ too.

**`sensor_augmentation` and `compose` do not guarantee matching randomness.**
`sensor_augmentation(pipeline, camera=None)` turns a pipeline into a one-argument observation
function, and `compose(a, b)` applies `a` then `b`. They are meant for a single direct run, such as
`run(backend, stack, frames, perturb=compose(attack, sensor_augmentation(pipeline)))`.
`sensor_augmentation` seeds from the absolute `observation.frame`, which can differ between runs.
For comparisons between treatments, use `SensorCondition.image` or YAML `conditions.sensors`.

The framework does not seed external libraries or reset state hidden in a closure. For other
sources of randomness, see [Reproducibility](experiments.md#reproducibility).

## Conditions during selection

A scenario source can apply conditions while it selects, so that filters see the same environment
as the runs:

```python
from avsectester.perturbations.image import AugmentationPipeline, GaussianNoise
from avsectester.scenarios import (
    CarlaScenarioBuilder, DatasetFilter, ScenarioConditions, SensorCondition,
)

# config is a CARLA scene configuration and dataset is a Dataset, both supplied by the caller.
weather = ScenarioConditions(world={"weather": {"precipitation": 20}})
carla_source = CarlaScenarioBuilder(base_scenario=config, conditions=weather)

noise = AugmentationPipeline([GaussianNoise(severity=0.2)], seed=12)
sensor_conditions = ScenarioConditions(sensors=(SensorCondition.image(noise, camera="front"),))
recorded_source = DatasetFilter(dataset, conditions=sensor_conditions)
```

- **CARLA applies world conditions before preparing candidates.** `CarlaScenarioBuilder` sets the
  weather on each candidate world before filters run, and accepts only `weather`. Without a
  `conditions` argument, it uses `conditions` and a top-level `weather` from `base_scenario`.
- **Recorded sources accept only sensor conditions.** `DatasetFilter` defaults to no conditions and
  raises `ValueError` for a nonempty `world`, because a recording cannot change its weather.
- **Configured selection uses the configured conditions.** With a `selection` key,
  `scenario_from_config` passes the configuration's `conditions` to its source.
- **Selected scenarios keep their conditions.** Every run of an accepted scenario uses them.

Sensor conditions give custom filters a conditioned measurement. The provider supplies the raw
`context.native["observation"]` at the frame and time of `context.scene`. The source applies the
sensor conditions to a copy and stores it in `context.native["conditioned_observation"]`, and
`context.native["conditions"]` describes the conditions. A filter reads the measurement with
`ScenarioConditions.selection_observation(context)`, which raises `ValueError` when the provider
supplies no observation:

<!-- runnable -->
```python
from dataclasses import dataclass

from avsectester.scenarios import Constraint, FilterResult, ScenarioConditions


@dataclass
class MinBrightness(Constraint):
    min_mean: float               # Minimum mean pixel value of the conditioned front camera.

    def evaluate(self, context):
        try:
            measurement = ScenarioConditions.selection_observation(context)
        except ValueError as error:
            return FilterResult("unknown", str(error))
        mean = float(measurement.sensor_data["front"].mean())
        return FilterResult.from_bool(mean >= self.min_mean, f"Mean brightness {mean:.1f}")
```

The CARLA provider supplies aligned observations. Metadata-only dataset contexts, including the
standard NuRec context, do not. They need a custom `prepare_context` or dataset adapter (see
[Support another dataset or provider](scenarios.md#support-another-dataset-or-provider)), and the
filter above reports `unknown` without one. Normalized ground truth, raw observations and
geometric [visibility](scenarios.md#visibility) stay separate. Image-space fog does not lower a
geometric visibility fraction.

Over a multi-frame initial window, one condition handler persists across the frames of the
window. A custom source does the same with `conditions.selection_scope()`, a context manager whose
returned function it calls as `prepare(context, frame_index)` for each frame.
`conditions.selection_context(context)` handles a single frame. A custom source that supports
world conditions must configure its preview world before computing filter inputs.

Replacing `scenario.conditions` with `dataclasses.replace` does not rerun filters. If the new
conditions must take part in the selection decision, select again with them on the source.

## Lower-level execution

`run_prepared` and `run_configured` apply conditions automatically. When you assemble a run from
`run`, `run_pair` or `run_logged`, apply them yourself:

- `scenario.make_backend()` configures world conditions. `scenario.factory()` does not.
- `scenario.conditions.apply_runtime(runtime)` returns a new `Runtime` with fresh sensor-condition
  handlers. Call it once for each run. Without it, no sensor condition runs.
- Direct runs do not set the match. Pass `case=scenario.match` when handlers need the selected
  roles and insertions.
- Direct runs do not add `conditions` to `trace.provenance`. Store `scenario.provenance` with your
  results if you need that record.

```python
from avsectester.runner import run_pair
from avsectester.closed_loop.hooks import Runtime

# scenario, stack_factory and attack_runtime (a Runtime) are supplied by the caller.
backend = scenario.make_backend()
try:
    clean_runtime = scenario.conditions.apply_runtime(Runtime(case=scenario.match))
    treated_runtime = scenario.conditions.apply_runtime(attack_runtime)
    clean, attacked = run_pair(
        backend, stack_factory, frames=30,
        clean_runtime=clean_runtime, attacked_runtime=treated_runtime,
        close_backend=False,
    )
finally:
    backend.close()
```

For a single run, pass one wrapped runtime: `run(backend, stack, frames, runtime=...)`. A
`perturb` function given to `run` runs before every runtime hook at `observation`, including
conditions with `order: before`. Resource ownership in direct runs is described in
[Direct runs and resource ownership](experiments.md#direct-runs-and-resource-ownership).

## Fidelity limits

- **Image corruptions are 2-D approximations.** `Fog` is a uniform airlight without depth. Rain and
  snow are procedural overlays. They stress the perception of the driving system and the attack.
  They are not physically correct renderings. Use CARLA weather when light transport matters.
- **CARLA weather covers only what CARLA renders.** Native actors and world-level patches receive
  the weather. Images inserted after rendering do not, unless their compositing reproduces it.
- **Image corruptions need an RGB camera.** LiDAR, radar and navigation conditions need custom
  `SensorCondition` factories that work on the native payloads. No built-in algorithms are
  supplied for those modalities.
- **Other effects are not built-in conditions.** Traffic density, latency and similar effects need
  condition-aware providers or custom handlers.

## Next steps

- [Robustness across corruption settings](evaluation.md#robustness-across-corruption-settings) to
  measure attack success under a grid of corruptions.
- [Condition keys](../reference/configuration.md#condition-keys) for every configuration key.
- [Order defenses relative to attacks and conditions](defenses.md#order-defenses-relative-to-attacks-and-conditions)
  to evaluate defenses under conditions.
- [Conditions](../reference/python-api.md#conditions) and
  [Image corruption API](../reference/python-api.md#image-corruption-api) for signatures.
