# Experiment environments

> **Audience:** Experiment authors studying world or sensor conditions.
>
> **Outcome:** Configure environmental treatments and explicit selection conditions.
>
> **Prerequisites:** [Configuration](../reference/configuration.md).

## Experiment environment

An environment separates world settings, applied by the simulator at reset, from sensor
corruptions, applied to captured model inputs. The merge order is `runs.environment`, the
enabled `Attack.environment` and the treatment environment. World mappings merge recursively; an
explicitly provided sensor list replaces the preceding list.

```yaml
runs:
  environment:
    sensors:
      - {type: GaussianNoise, options: {severity: 0.2}, camera: front}
  treatments:
    - {name: clean}
    - name: heavy_noise
      environment:
        sensors:
          - {type: GaussianNoise, options: {severity: 0.7}, camera: front}
```

The YAML keys and defaults are defined in
[environment configuration](../reference/configuration.md#environment).

## Selection and the offline phase

Metabuilders do not inherit the experiment environment. Selection requests must supply any
desired environment explicitly. `OfflineContext.environment` merges `runs.environment` with the
attack's environment; pass it to sensor reads, renders or candidate rollouts when those
conditions are required.

Changing an environment does not reselect segments. The attack owns eligibility decisions; see
[selection and metadata](attacks.md).

## Sensor processing

Sensor corruptions change model-visible captures without changing physical state. The
[runtime sequence](../reference/runtime.md#one-decision) places them between sensor attack
actions and the observation pipeline.

Camera selection follows [Environment configuration](../reference/configuration.md#environment).

Camera transport preserves RGB uint8. Non-camera writes preserve the payload representation
and dtype; neither path infers sensor type from array shape. See
[data transport](../reference/python-api.md#avsectesterdata_types).

## Supplied corruptions

Built-in image corruptions approximate weather, lighting and sensor effects in image space.
`Fog` does not integrate metric scene depth, and procedural rain is not physical rainfall. They
are separate from native world settings. The
[corruption API](../reference/python-api.md#avsectesterclosed_loopsensor_corruptions) lists the
classes and parameters.

This complete configuration uses the repository's memory components and a built-in corruption.
It loads without a simulator service, model weights or GPU.

<!-- runnable -->
```yaml
dataset: {type: "tests.fakes:MemoryDataset"}
simulator: {type: "tests.fakes:MemorySimulator"}
avstack: {type: "tests.fakes:ConstantAVStack"}
attacks:
  probe: {type: "tests.fakes:MinimalAttack"}
runs:
  environment:
    sensors:
      - type: GaussianNoise
        options: {severity: 0.2}
        camera: front
```

`AlbumentationsCorruption` selects an Albumentations transform through its transform option and
requires the augment extra. It uses the same sensor-environment mechanism; see
[constructor options](../reference/configuration.md#environment).

## Custom sensor corruptions

Implement apply(rgb, rng), or a callable with that signature, returning uint8 RGB with the same
shape. Select it with a constructor path such as `my_project.corruptions:DimImage`.
Filesystem constructor parameters follow
[path resolution](../reference/configuration.md#constructor-filesystem-parameters).

<!-- runnable -->
```python
import numpy as np

class DimImage:
    def __init__(self, factor=0.8):
        self.factor = factor

    def apply(self, rgb, rng):
        return np.clip(rgb.astype(float) * self.factor, 0, 255).astype(np.uint8)

image = np.full((4, 5, 3), 100, dtype=np.uint8)
assert DimImage().apply(image, np.random.default_rng(0))[0, 0, 0] == 80
```

## Native world settings

World-setting support and keys are defined in
[Configuration](../reference/configuration.md#environment).

Conditions changed by a world attack action during execution are recorded as actions, rather
than as static environment settings. Native simulator behavior is described
in [Simulators](simulators.md).

**Not yet implemented.** Reference evaluations that train and test artifacts under different
conditions; see [Environment transfer](../development/TODO.md#environment-transfer).

## Randomness

Sensor corruptions use a fresh NumPy generator keyed by
`(run_seed, absolute_dataset_position, corruption_index)`. Past and current captures at the same
dataset position therefore use the same corruption randomness. Hooks share a run-scoped NumPy
generator initialized from the hook seed. Registration and execution order determine its
consumption; shared callback handlers retain their declared lifecycle.

## Next steps

Read [Experiments](experiments.md#environment-only-robustness-experiments),
[Simulators](simulators.md) and [Runtime](../reference/runtime.md).
