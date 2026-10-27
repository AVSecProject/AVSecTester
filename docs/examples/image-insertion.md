# Image object insertion

> **Audience:** Users inserting an asset into captured camera images.
>
> **Outcome:** Run `ObjectInsertionAttack` on a selected simulator configuration.
>
> **Prerequisites:** [Installation](../getting-started/installation.md) and the selected
> dataset.

## Configurations

```bash
docker compose exec carla-experiment avsectester run configs/carla_object_insertion.yaml
docker compose run --rm nurec-experiment run configs/nurec_object_insertion.yaml
docker compose exec carla-experiment avsectester run configs/nuscenes_object_insertion.yaml
```

The NuRec configuration selects the following asset and placement:

```yaml
type: ObjectInsertionAttack
options:
  decision_steps: 20
  max_segments: 1
  assets:
    type: avsectester.implemented_attacks.object_insertion_attack.assets:SignAsset
  sensor: camera_front_wide_120fov
  placement:
    type: avsectester.attack.insertion.placement:WorldPlacement
    options: {position_m: [28.0, -3.0, 0.0]}
  orientation:
    type: avsectester.attack.insertion.placement:Orientation
    options: {mode: face_ego}
  placement_relative_to_ego_origin: true
```

The relative placement is resolved once from the initial ego actor/rig origin and then remains
fixed in world coordinates. Asset origins, sizes and placement constraints are defined in the
[Python API](../reference/python-api.md).

## Offline phase

The attack selects eligible camera views without changing their inputs. Its offline phase
retains the configured asset. `Attack` authoring responsibilities are in
[Writing attacks](../guides/attacks.md).

## Online phase

Each runtime composites the asset into the selected captured image. This does not create a
collidable actor; [Simulators](../guides/simulators.md) describes insertion and occlusion
behavior. nuScenes uses recorded playback rather than a reactive driving world.

This attack modifies only the selected camera image.
**Not yet implemented.** Insertion into LiDAR or other non-camera sensors; see
[Non-camera insertion](../development/TODO.md#inserted-objects-in-non-camera-sensors).

## Harmonization and custom assets

Use `ExposureHarmonizer` for differentiable exposure matching. `Harmonizer` formulas, nominal
textures, shared deployment/optimization calculations and custom interfaces are in the
[Python API](../reference/python-api.md).

This self-contained example constructs an insertion action with a custom plane:

<!-- runnable -->
```python
import numpy as np
from avsectester.attack.insertion import Insertion, InsertionAttackAction, PatchCompositor, PlaneAsset
from avsectester.attack.insertion.placement import WorldPlacement
from avsectester.rendering import ExposureHarmonizer

asset = PlaneAsset(np.full((16, 32, 4), 255, dtype=np.uint8), width_m=1.0, height_m=0.5)
insertion = Insertion("banner", asset, WorldPlacement((10.0, 0.0, 1.5)))
action = InsertionAttackAction(
    "front", insertion, PatchCompositor(ExposureHarmonizer(), soften=0.0))
assert action.insertion.asset.planes()[0].texture.shape == (16, 32, 4)
```

Return the action from an attack runtime as described in [Writing attacks](../guides/attacks.md).

## Prepare person assets

```bash
docker compose run --rm --entrypoint python nurec-experiment \
  -m scripts.preparation.extract_person_cutouts --help
```

The script lists its dataset and segmentation-model inputs. Asset preparation is separate from
experiment execution; [Installation](../getting-started/installation.md) owns prerequisites.

## Evidence

The NuRec configuration records the front-wide camera's `model_input` representation.
Its default metrics are `DrivingMetrics` and `TrajectoryDeviation`.
The CARLA configuration defaults to `DrivingMetrics` and `StopImpact`.
Inspect their evidence with [Evaluation](../guides/evaluation.md).

## Next steps

Read [Datasets](../guides/datasets.md), [Optimization](../guides/optimization.md)
and [Evaluation](../guides/evaluation.md).
