# Experiment environments

> **Audience:** Experiment authors studying world or sensor conditions.
>
> **Outcome:** Configure explicit environmental treatments with reproducible sensor noise.
>
> **Prerequisites:** [Configuration](../reference/configuration.md).

## Experiment environment

An environment has two independent parts: `world`, applied by the simulator at reset, and `sensors`,
applied to captured model inputs. Settings merge in this order: experiment base, enabled
Attack.environment, treatment overrides. World mappings merge recursively; an explicitly provided
sensor list replaces the preceding list.

```yaml
environment:
  world: {weather: SoftRainNoon}
  sensors:
    - {type: GaussianNoise, options: {severity: 0.2}, camera: front}
treatments:
  - {name: clean}
  - name: heavy_noise
    environment:
      sensors:
        - {type: GaussianNoise, options: {severity: 0.7}, camera: front}
```

Metabuilders do not inherit the experiment base environment. They must request any alternative
environment explicitly. The formal runner receives the resolved environment and passes only its
world mapping to SimulatorRuntime.reset.

## Sensor processing order

The simulator captures the world, sensor attack actions change the selected samples, the sensor
environment transforms them, and VehicleIO applies sensor hooks, localization and observation hooks.
Images retained as raw and model_input document this distinction. Sensor transforms never change
physical world state.

Supply `camera` explicitly when observations contain multiple sensors. If omitted, the first
sensor_data entry is chosen; it must be readable RGB data. A LiDAR-first observation fails rather
than silently searching for a camera.

Randomness derives from run seed, absolute dataset timeline position and transform index. Offline
sensor requests and online decisions at the same position receive matching noise. The relative
decision index and native simulator frame number are different quantities.

## Built-in image corruptions

Built-in transforms live in `closed_loop/sensor_transforms.py`. Severity is in [0, 1].

| Family | Classes |
|---|---|
| Weather approximation | Fog, Rain, Snow |
| Lighting | Brightness, LowLight, Contrast, Gamma, ColorTemp |
| Sensor and optics | GaussianNoise, ShotNoise, MotionBlur, DefocusBlur, JPEGCompression, ChromaticAberration |

These are image-space approximations. Fog does not integrate metric scene depth, and procedural rain
does not reproduce physical rainfall. World settings remain separate.

`AlbumentationsCorruption` selects an Albumentations transform through its options and uses the
augment extra. It is configured as a sensor transform, not a parallel augmentation pipeline. The
transform option is required and must name a supported transform; the recorded transform label
includes that name. For example:

<!-- runnable -->
```yaml
environment:
  sensors:
    - type: AlbumentationsCorruption
      options: {transform: Fog, severity: 0.5}
      camera: front
```

## Custom sensor environments

Implement `apply(rgb, rng) -> rgb` or a callable with that signature. Return uint8 RGB with the same
shape. Select the constructor using `type: package.module:Name` and `options`.

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

## Native CARLA settings

CarlaSimulator declares supported world setting names and validates values before a world is
created. Its keys are weather, lighting, map_layers and road_friction. See its
`supported_world_settings` declaration for the authoritative keys. Actor-dependent checks still
require initialization. Use a dedicated server: each formal run reloads the world.

Unsupported settings fail explicitly. Do not substitute an image transform for native weather
without specifying it under sensors. A world action can change the world during a run, but its
timing and application are then recorded as attack actions rather than static environment.

## Next steps

Read [Experiments](experiments.md#environment-only-robustness-experiments),
[Simulators](simulators.md) and [Runtime](../reference/runtime.md).
