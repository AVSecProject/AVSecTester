# Physical patch

> **Audience:** Users deploying or optimizing a texture patch.
>
> **Outcome:** Run a selected CARLA or NuRec `PatchAttack` configuration.
>
> **Prerequisites:** [Installation](../getting-started/installation.md) for the chosen
> simulator.

## Configuration

```bash
docker compose exec carla-experiment avsectester run configs/carla_patch.yaml
docker compose exec carla-experiment avsectester run configs/carla_patch_optimized.yaml
docker compose run --rm nurec-experiment run configs/nurec_patch.yaml
```

Each configuration selects and prepares its own segments. The NuRec configuration includes:

```yaml
type: PatchAttack
options:
  decision_steps: 20
  max_windows: 1
  sensor: camera_front_wide_120fov
  width_m: 1.0
  height_m: 0.5
  max_distance: 30.0
  candidate_hosts: ['15']
  hosts_per_window: 1
  assets:
    type: avsectester.rendering.textures:CheckerboardTexture
    options: {size: 256, cells: 8}
```

`Texture` and placement parameters are listed in the [Python API](../reference/python-api.md). A
Python texture object can provide RGBA pixels directly:

<!-- runnable -->
```python
import numpy as np
from avsectester.implemented_attacks.patch_attack import PatchAttack

class SolidTexture:
    def rgba(self):
        return np.full((32, 64, 4), (255, 64, 0, 255), dtype=np.uint8)

attack = PatchAttack(decision_steps=5, assets=SolidTexture())
```

## Offline phase

Selection ranks candidate hosts within each eligible window. Without optimization, the input
texture becomes the artifact. An optimization option prepares a texture using the selected
training segments and configured scorer. See [Optimization](../guides/optimization.md).

## Online phase

The runtime receives the explicit texture artifact. CARLA defaults to `static.prop.mesh` with
the packaged IronPlank asset specified by `DEFAULT_PROP_MESH_PATH` in the
[panel API](../reference/python-api.md#avsectesterimplemented_attackspatch_attackonline).
Set `PatchAttack.prop_mesh_path` to select another mesh. Other prop blueprints require
`prop_mesh_path=None`. The installation orientation is calibrated for the default mesh and is
not guaranteed for other meshes or props. The default mesh's native UV mapping crops the right
edge of the texture.
NuRec composites into the selected camera capture. Their geometry and occlusion behavior are
described in [Simulators](../guides/simulators.md).
The CARLA mesh requires the mounted AdditionalMaps pack on both maps, including `Town10HD_Opt`.
See [CARLA map packs](../getting-started/installation.md#carla-map-packs).

Scaling a CARLA prop mesh to arbitrary requested panel dimensions is **Not yet implemented.**
See [CARLA physical patches](../development/TODO.md#attacks).

## Evidence

Use the recorded camera input, deployment receipts and driving metrics described in
[Evaluation](../guides/evaluation.md). The [example overview](README.md) explains the
distinction between workflow execution and measured attack impact.

The `carla_patch_optimized.yaml` configuration uses a white-box attack (algorithm used: `PGD`)
through the `mmdet.rpn_objectness` scorer bound by `modular_driving_adapter`.

## Next steps

Read [Image insertion](image-insertion.md) and [Optimization](../guides/optimization.md).
