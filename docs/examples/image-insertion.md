# Image object insertion

> **Audience:** Users inserting objects into captured images.
>
> **Outcome:** Reuse one attack on CARLA, NuRec and nuScenes playback.
>
> **Prerequisites:** [Attacks](../guides/attacks.md) and the target dataset/environment.

## Configurations

```bash
avsectester run configs/carla_object_insertion.yaml
avsectester run configs/nurec_object_insertion.yaml
avsectester run configs/nuscenes_object_insertion.yaml
```

ObjectInsertionAttack is exported by avsectester.attack.implemented_attacks.object_insertion_attack.
Required parameters are decision_steps and placement. The assets type/options selects SignAsset,
PersonAsset or TrafficLightAsset; Python callers may provide a PlanarAsset. Other options include
sensor, orientation, min_visible_fraction, max_segments and keep_tail. Asset lengths are in metres
and carry the _m suffix, for example SignAsset.width_m and mount_height_m. TrafficLightAsset uses
texture_height_pixels for image resolution; head_count and gap_fraction set the signal layout.
SignAsset.post_count controls the number of support posts.

```yaml
type: ObjectInsertionAttack
options:
  decision_steps: 10
  assets: {type: SignAsset, options: {}}
  sensor: front
  placement:
    type: WorldPlacement
    options:
      position_m: [20.0, 0.0, 1.5]
```

Use the complete supplied configurations as the source for supported placement options in your
scene. Position is in canonical world metres, not screen pixels. With
placement_relative_to_ego_origin=true, an explicit WorldPlacement position is interpreted in the
first true ego actor/rig origin frame, transformed once and saved as a fixed world placement. This
is the actor origin rather than its box center. In CARLA the actor origin is at ground level, so a
roadside asset with a ground-level local origin uses z=0. It does not follow the moving ego.

The nuScenes configuration uses the supported Faster R-CNN/nuScenes detector pair. Its recorded
sensor/render path is covered offline; real dataset/model execution remains a separate validation.

## Geometry and rendering

Shared camera projection and canonical actor poses live in geometry/. Placement types live in the
core attack/placement.py module. Reusable assets, compositing, occlusion and harmonization live in
the attack/insertion/ framework tool; object-specific assets live beside ObjectInsertionAttack.

The simulator provides a SensorView: RGB data, camera model, camera-from-world transform, actor and
ego cuboids, time and optional depth. The attack resolves placement and projects through the actual
pinhole or f-theta camera model. It uses measured depth when available and actor cuboids for
supported occlusion estimates. Visibility and full-silhouette denominators remain attack-specific
evidence; missing evidence is not assumed visible.

Selection reads the clean view and insertion evidence without applying effects. Offline candidate
rendering and online execution invoke the same InsertionAttackEffect. Undefined online visibility
leaves the input unchanged and records that no rendering occurred. Image insertion changes only
selected captured data. It does not create a collidable world object. nuScenes remains open-loop
recorded playback.

## Harmonization

PatchCompositor and optional harmonizers live in attack/insertion. Classic harmonization is
lightweight; PCTNet requires its model dependencies and weights. strict=True raises
loading/inference errors; permissive fallback must be explicitly chosen for the experiment.
Harmonizer is a public extension point: implement __call__(composite_rgb, mask, background_rgb).
Configure compositor as {type: PatchCompositor, options: {soften: 0.0, harmonizer: {type:
package.module:MyHarmonizer, options: {}}}}. Built-ins are ClassicHarmonizer and PCTNetHarmonizer.
See [PCTNet installation](../getting-started/installation.md#pctnet-harmonizer).

## Use insertion tools in your attack

Import public utilities directly from avsectester.attack.insertion. A custom PlanarAsset implements
planes(), returning PlaneSurface records with local TL, TR, BR, BL corners in metres and RGBA
textures. PlaneAsset provides a single rectangular YZ plane. Combine it with AttachedPlacement or
WorldPlacement from avsectester.attack.placement; placement is a core data type.

<!-- runnable -->
```python
import numpy as np
from avsectester.attack.insertion import (
    Harmonizer, Insertion, InsertionAttackEffect, PatchCompositor, PlaneAsset,
)
from avsectester.attack.placement import WorldPlacement

class PreserveColors(Harmonizer):
    def __call__(self, composite_rgb, mask, background_rgb):
        return composite_rgb.copy()

asset = PlaneAsset(np.full((16, 32, 4), 255, dtype=np.uint8), width_m=1.0, height_m=0.5)
insertion = Insertion("banner", asset, WorldPlacement((10.0, 0.0, 1.5)))
effect = InsertionAttackEffect(
    "front", insertion, PatchCompositor(PreserveColors(), soften=0.5))
assert effect.insertion.asset.planes()[0].texture.shape == (16, 32, 4)
```

Return such an effect from your AttackRuntime.step. Each effect resolves its placement from the
simulator's canonical actor poses. For custom placement policies, calculate a WorldPlacement or
AttachedPlacement in your metabuilder and save it in the segment; the simulator never receives
attack metadata.

Visibility and compositing can also be used separately without starting an attack:

```python
from avsectester.attack.insertion import CuboidVisibilityEstimator, render_resolved

# Requires a SensorView view and the configured effect above.
resolved = effect.resolve(view)
evidence = CuboidVisibilityEstimator().estimate(
    resolved, view.camera, view.cam_from_world, occluders=view.actors,
    camera_name=view.sensor)
if evidence.visibility is not None:
    composite, mask = render_resolved(
        view.data, view.camera, view.cam_from_world, resolved, evidence,
        compositor=effect.compositor)
```

Use DepthVisibilityEstimator with aligned depth when available. Missing visibility is explicit and
must not be treated as full visibility. To select a user harmonizer from YAML, set
compositor.options.harmonizer.type to package.module:PreserveColors with its constructor options.

## Prepare person assets

```bash
python -m scripts.preparation.extract_person_cutouts --help
```

This preparation script extracts RGBA person cutouts using SAM. Check its required data/model
arguments before running it. Asset extraction is independent of the closed-loop experiment.

## Evidence and outputs

Enable sensor recording before the run. Export camera views or overlays from saved RunLog using
evaluation.visualization. DetectionMetrics compares canonical driving-system detections with
physical truth; it does not evaluate an inserted image object as a newly spawned true actor. A
separate user metric may assess inserted-object recognition if that is the objective.

## Next steps

Read [Scenarios](../guides/scenarios.md), [Optimization](../guides/optimization.md) and
[Evaluation](../guides/evaluation.md).
