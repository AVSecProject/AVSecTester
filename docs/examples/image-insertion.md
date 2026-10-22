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

ObjectInsertionAttack is exported by avsectester.implemented_attacks.object_insertion_attack.
Required parameters are decision_steps and placement. The assets type/options selects SignAsset,
PersonAsset or TrafficLightAsset; Python callers may provide a PlanarAsset. Other options include
sensor, orientation, min_visible_fraction, max_segments and keep_tail. Asset lengths are in metres
and carry the _m suffix, for example SignAsset.width_m and mount_height_m. TrafficLightAsset uses
texture_height_pixels for image resolution; head_count and gap_fraction set the signal layout.
SignAsset.post_count is a nonnegative integer controlling the support posts; its default is 1.
Set post_count=0 for a panel without posts.

```yaml
type: ObjectInsertionAttack
options:
  decision_steps: 10
  assets:
    type: avsectester.implemented_attacks.object_insertion_attack.assets:SignAsset
    options: {}
  sensor: front
  placement:
    type: avsectester.attack.insertion.placement:WorldPlacement
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
attack/insertion/placement.py tool module. Reusable insertion assets, compositing and visibility
live in attack/insertion/; shared PlaneSurface, sampling and harmonizers live in rendering/.
Object-specific
assets live beside ObjectInsertionAttack.

geometry/projection.py supplies plane_sampling_grid, project_box and visible_plane_mask for users
who need calibrated sampling, box rectangles or a depth-visible plane mask independently. They
receive camera models and world transforms directly, support pinhole and f-theta lenses, and do
not depend on SensorView or attack metadata.
geometry/cameras.py also provides the shared NEAR_Z threshold and camera_ray_mask for rejecting
nonfinite, backward or out-of-lens rays. DepthVisibilityEstimator restricts plane-depth queries to
each projected face's image bounds; pixels outside those bounds are False in the returned mask.

The simulator provides a SensorView: RGB data, camera model, camera-from-world transform, actor and
ego cuboids, time and optional depth. The attack resolves placement and projects through the actual
pinhole or f-theta camera model. It uses measured depth when available and actor cuboids for
supported occlusion estimates. Visibility and full-silhouette denominators remain attack-specific
evidence; missing evidence is not assumed visible.
Without depth, inserted world objects in SensorView.objects also occlude through their
alpha-aware faces. Visibility sources and receipt details are listed in the
[Python API](../reference/python-api.md) and [RunLog](../reference/runtime.md#runlog) references.

Selection reads the clean view and insertion evidence without applying actions. Offline candidate
rendering and online execution invoke the same InsertionAttackAction. Undefined online visibility
leaves the input unchanged and records that no rendering occurred. Image insertion changes only
selected captured data. It does not create a collidable world object. nuScenes remains open-loop
recorded playback.

## Harmonization

PatchCompositor remains in attack/insertion, while harmonizers live in rendering/.
PatchCompositor defaults to harmonizer=None and soften=0.0: inserted pixels receive no
harmonization. Choose one of these three forms:

- Omit harmonizer, or pass None in Python, to keep the default without harmonization.
- Supply ClassicHarmonizer for lightweight color/blending adjustments, or PCTNetHarmonizer for
  learned harmonization. PCTNet requires dependencies and weights; strict=True raises loading or
  inference errors instead of falling back.
- Subclass Harmonizer and implement __call__(composite_rgb, mask, background_rgb). Reference that
  implementation as package.module:Class in YAML.

For example, explicitly enable the framework's classic harmonizer:

```yaml
compositor:
  type: avsectester.attack.insertion.compositor:PatchCompositor
  options:
    soften: 0.0
    harmonizer:
      type: avsectester.rendering.harmonizers:ClassicHarmonizer
      options: {}
```

Use avsectester.rendering.harmonizers:PCTNetHarmonizer for the learned implementation, supplying
libcom_directory as the libcom repository root, or
package.module:MyHarmonizer for user code. See [PCTNet
installation](../getting-started/installation.md#pctnet-harmonizer).

## Use insertion tools in your attack

Import public utilities directly from avsectester.attack.insertion. A custom PlanarAsset implements
planes(), returning PlaneSurface records with local TL, TR, BR, BL corners in metres and RGBA
textures. PlaneAsset provides a single rectangular YZ plane. Combine it with AttachedPlacement or
WorldPlacement from avsectester.attack.insertion.placement; placement belongs to the insertion
tools.

<!-- runnable -->
```python
import numpy as np
from avsectester.attack.insertion import Insertion, InsertionAttackAction, PatchCompositor, PlaneAsset
from avsectester.attack.insertion.placement import WorldPlacement
from avsectester.rendering import Harmonizer

class PreserveColors(Harmonizer):
    def __call__(self, composite_rgb, mask, background_rgb):
        return composite_rgb.copy()

asset = PlaneAsset(np.full((16, 32, 4), 255, dtype=np.uint8), width_m=1.0, height_m=0.5)
insertion = Insertion("banner", asset, WorldPlacement((10.0, 0.0, 1.5)))
action = InsertionAttackAction(
    "front", insertion, PatchCompositor(PreserveColors(), soften=0.5))
assert action.insertion.asset.planes()[0].texture.shape == (16, 32, 4)
```

Return such an action from your AttackRuntime.step. Each action resolves its placement from the
simulator's canonical actor poses. For custom placement policies, calculate a WorldPlacement or
AttachedPlacement in your metabuilder, then save its parameters as JSON-like user_data rather
than storing the placement object itself. Reconstruct it in the attack's online phase; the
simulator never receives attack metadata.

Visibility and compositing can also be used separately without starting an attack:

```python
from avsectester.attack.insertion import CuboidVisibilityEstimator, render_resolved

# Requires a SensorView view and the configured action above.
resolved = action.resolve(view)
evidence = CuboidVisibilityEstimator().estimate(
    resolved, view.camera, view.cam_from_world, occluders=view.actors,
    camera_name=view.sensor, world_objects=tuple(view.objects.values()))
if evidence.visibility is not None:
    composite, mask = render_resolved(
        view.data, view.camera, view.cam_from_world, resolved, evidence,
        compositor=action.compositor)
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
