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
   **every** backend — CARLA, in-process NuRec, and dataset replay (nuScenes/nuRec `RecordedFrameBackend`)
   — so robustness can be measured on real recorded frames, not only in simulation.
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
`albumentations_corruptions(severity)`) that wraps `A.RandomFog/RandomRain/RandomSnow/GaussNoise/ISONoise/
MotionBlur/Defocus/ImageCompression` behind the same `Corruption` interface — determinism preserved by
seeding `A.Compose` from our own RNG, so the `(seed, frame)` pairing still holds. Install with the
`augment` extra.

**Backend — classic albumentations (`albumentations>=1.4,<2`, MIT).** Installed via the `augment` extra.
It imports as `import albumentations as A`; the builder param names target the 1.4.x API (`blur_limit`,
`radius`, `color_shift`, …). The `dependencies` branch stack is **numpy<2** (torch 2.1.0+cu121, the newest
combo OpenMMLab supports without version-cap patching), so we use classic albumentations rather than
AlbumentationsX/2.x — those require numpy≥2, which would force torch≥2.4 → mmcv 2.2.0 → breaks the
`mmdet3d<mmcv2.2` cap. Classic albumentations exposes the same transforms and API, so the adapter is
unchanged.

## Determinism across the pair (the subtle bit)

Robustness is measured by *diffing* a clean run and an attacked run **under the same condition**. If the
fog pattern or noise realisation differed between the two, the diff would conflate the attack with the
corruption. So `AugmentationPipeline.apply(rgb, frame)` seeds its RNG from `(seed, frame)` — deterministic
in the frame index — and the harness gives the clean and attacked runs the *same* pipeline `seed`. The
corruption is then pixel-identical across the pair, and `metric.impact` measures only the attack.

## Robustness evaluation protocol (implemented: `avsectester.evaluation.robustness`)

This protocol is realised by `evaluate_robustness(source, req, attack_for, stack, frames, ...)`, which
returns a `RobustnessReport` (ASR + resilience per condition). The sketch below is what it does:


```python
from avsectester.simulators.augment import common_corruptions, sensor_augmentation, compose

for cor in common_corruptions(severity):                 # or sweep severities per corruption
    aug   = sensor_augmentation(cor, camera="front")     # same seed for both runs
    clean    = run(backend, stack, N, perturb=aug)                 # corruption only
    attacked = run(backend, stack, N, perturb=compose(attack, aug))# attack under the corruption
    verdict  = impact(clean, attacked)                            # did the attack still fire?
report = aggregate_over(corruptions, severities)          # attack-success-rate vs condition
```

The scenario layer yields *which* scenes to run; this augmentation layer yields *under which conditions*;
the eval harness runs the grid and reports an **attack robustness curve** — success rate as a function of
corruption type and severity.

## Non-goals / approximations (kept honest)

- Image-space weather is a *2-D approximation*: fog is uniform airlight (no per-pixel depth), rain/snow
  are procedural overlays. It is a stress test of the *perception+attack* pipeline, not a physically
  exact renderer. Use the CARLA world tier when physical light transport matters.
- Corruptions act on the **camera** stream. Lidar/other-modality corruptions are a later addition (a
  `Corruption` on those payloads slots into the same registry).
