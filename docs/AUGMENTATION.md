# Scenario augmentation — perturbing the world to test attack robustness

## The problem

An attack that succeeds on one clean rendering is not necessarily a *robust* attack. A physical patch
that hides a vehicle in clear noon light may wash out in fog, break under a night-time colour cast, or
survive sensor noise but not motion blur. To claim an attack is real we must show it survives the
**nuisance variation** a deployed AV actually sees: weather, lighting, and sensor/optics degradation.

So alongside the *attack* transform we need an *augmentation* transform — a controlled corruption of the
scene — and a protocol that measures how the attack's success degrades as the augmentation gets harsher.

## Where it plugs in — the same seam as an attack

The sim↔stack contract carries exactly one perturbable stream: the `Observation`. `run(backend, stack,
frames, perturb)` applies `perturb: Observation -> Observation` before the box sees each frame. **An
attack is a `perturb`; an augmentation is also a `perturb`.** They therefore compose:

```
clean      :  run(backend, stack, N)                         # no perturb
attacked   :  run(backend, stack, N, perturb=attack)         # patch only
robustness :  run(backend, stack, N, perturb=compose(attack, augment))  # patch, then corruption
```

`compose(a, b)` applies `a` then `b`, so `compose(attack, augment)` inserts the patch and *then* corrupts
the frame — the patch is fogged/blurred/noised along with the rest of the scene, as it would be in
reality.

## Two tiers of fidelity

1. **Image-space corruptions** — `augment.py`, backend-agnostic. Each `Corruption` is an
   `apply(rgb, rng) -> rgb` on the camera image; an `AugmentationPipeline` chains several and
   `sensor_augmentation(pipeline)` lifts it to a `perturb`. Because it operates on pixels it works on
   camera images from CARLA, NuRec reconstructed-scene rendering (`NuRecBackend`), and recorded-frame
   replay (`RecordedFrameBackend`, used by nuScenes). NuRec generates new views from the ego pose,
   while recorded-frame replay returns captured images.
2. **CARLA world-weather** — `carla.py`. A `weather` key in the scenario config sets native
   `carla.WeatherParameters` (cloud, precipitation, sun altitude, fog, wetness) at `reset`, so the whole
   scene — including a *world-level* physical patch — is re-rendered under that weather. Higher fidelity
   (real light transport, wet-road reflections), CARLA-only. The image-space tier is the portable
   workhorse; the world tier is the faithful option when running in CARLA.

## Operators (`augment.py`)

Every operator takes a `severity` in `[0, 1]` (0 ≈ identity, 1 = strong) and is driven by a seeded
`numpy.random.Generator`, so runs are reproducible and — crucially — the *same* corruption can be applied
to the clean and attacked runs of a pair.

| category  | operators                                                             |
|-----------|-----------------------------------------------------------------------|
| weather   | `Fog`, `Rain`, `Snow`                                                 |
| lighting  | `Brightness`, `LowLight`, `Contrast`, `Gamma`, `ColorTemp`           |
| sensor    | `GaussianNoise`, `ShotNoise`, `MotionBlur`, `DefocusBlur`, `JPEGCompression`, `ChromaticAberration` |

`CORRUPTIONS` is a name→class registry; `common_corruptions(severity)` returns a standard suite (one of
each) for benchmarking — the AV analogue of ImageNet-C's common-corruptions set, plus driving weather.

### Backends: hand-rolled (default) or Albumentations (optional)

The operators above are **zero-dependency** (numpy + cv2) and are the default. For battle-tested
implementations there is an optional **Albumentations** backend (`AlbumentationsCorruption`,
`albumentations_corruptions(severity)`, `augment` extra) that wraps `A.RandomFog/RandomRain/RandomSnow/
GaussNoise/ISONoise/MotionBlur/Defocus/ImageCompression` behind the same `Corruption` interface, seeding
`A.Compose` from our own RNG so the `(seed, frame)` pairing still holds. It is **classic albumentations**
(`>=1.4,<2`, MIT) — AlbumentationsX/2.x needs numpy≥2, which this stack's numpy<2 / torch-2.1 pin
excludes (see [`SETUP.md`](SETUP.md)); the classic API (`import albumentations as A`, params
`blur_limit`/`radius`/`color_shift`) is otherwise identical.

## Determinism across the pair (the subtle bit)

Robustness is measured by *diffing* a clean run and an attacked run **under the same condition**. If the
fog pattern or noise realisation differed between the two, the diff would conflate the attack with the
corruption. `AugmentationPipeline.apply(rgb, frame)` seeds its RNG from `(seed, frame)`. The harness
uses the same pipeline seed and aligns frames relative to each run's first observation, since absolute
simulator frame IDs can differ after reset. Corresponding steps therefore use the same random draws,
even when their input images differ because of the attack or subsequent driving.

## Robustness evaluation protocol (implemented: `avsectester.evaluation.robustness`)

`evaluate_robustness(source, req, attack_for, stack, frames, ...)` runs the grid and returns a
`RobustnessReport` (ASR + resilience per condition). `stack` is a **factory** (`Callable[[], AVStack]`)
so each run gets a fresh box. Given a scenario source, requirement, attack factory and stack factory:

```python
from avsectester.evaluation.robustness import evaluate_robustness

report = evaluate_robustness(source, req, attack_for, stack, frames, severity=0.5)
print(report.summary())
```

The harness includes a no-corruption baseline plus the corruption suite. For each scenario and
condition it creates a backend, calls `prepare_clean_attack_pair()`, runs clean and attacked with
aligned corruption, and closes the backend. The attacked run applies the attack before corruption.

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

## Non-goals / approximations (kept honest)

- Image-space weather is a *2-D approximation*: fog is uniform airlight (no per-pixel depth), rain/snow
  are procedural overlays. It is a stress test of the *perception+attack* pipeline, not a physically
  exact renderer. Use the CARLA world tier when physical light transport matters.
- Corruptions act on the **camera** stream. Lidar/other-modality corruptions are a later addition (a
  `Corruption` on those payloads slots into the same registry).
