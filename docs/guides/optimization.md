# Optimize attack artifacts

> **Audience:** Attack authors implementing an offline objective.
>
> **Outcome:** Reuse closed-loop services and optimizer implementations without changing deployment.
>
> **Prerequisites:** [Attacks](attacks.md).

## Offline optimization

Attack.run_offline receives the selected metadata and ClosedLoopServices. It runs once per attack
and returns explicit artifacts. The user chooses the objective, candidate sampling and optimization
algorithm. The framework neither supplies selection criteria nor reselects after optimization.

Within a service session, the attack can evaluate driving components, request scorer outputs, render
its SensorAttackAction on a sensor view, or execute complete candidate rollouts. Gradient
requirements are checked against declared capabilities. No path silently detaches and continues.

Candidate rollouts use the same Attack.create_runtime and artifacts deployment as formal runs. World
state, attack state and driving history are fresh for each candidate. Never optimize by advancing
the active formal run.

## Optimization interfaces

Implementations live in `avsectester/attack/optimization/`.

| Interface | Responsibility |
|---|---|
| DataSource.sample(batch) | Select optimization samples |
| AdversarialSample | Image, target and user sample context |
| Perturbation | Initialize, apply, project and export a candidate |
| Scorer.target_score(image, target) | Produce an objective score |
| OptimizationObjective.loss(score) | Convert score to optimization loss |
| GradientOptimizer.run(...) | Execute the algorithm and return OptimizationResult |

PGD is in whitebox.py; NES is in blackbox.py. HideObject minimizes a target score and SpoofObject
maximizes it. The algorithm interfaces accept the sample provider, perturbation, scorer and
objective rather than a fixed simulator. SessionScorer connects arbitrary image scoring to a
closed-loop session. Patch-specific SessionDataSource and PatchOptimization live beside PatchAttack
in implemented_attacks/patch_attack/, not in the generic optimization tool. Driving-model scorers
live with the driving adapters, not in framework attack orchestration.

## Use the optimization tools

The public package exports DataSource, AdversarialSample, Perturbation, Scorer,
OptimizationObjective, HideObject, SpoofObject, GradientOptimizer, OptimizationResult, PGD, NES,
PatchPerturbation, LinfImage and SessionScorer. A user Attack supplies its own sampling logic; the
tool does not choose hosts, scenes, placements or simulator deployment origins. TargetSpec is the
shared image-target record imported from avsectester.data_types.

This offline example optimizes a bounded additive image perturbation with PGD. It uses a simple
local scorer so it needs no simulator or model weights:

<!-- runnable -->
```python
import torch
from avsectester.attack.optimization import (
    AdversarialSample, DataSource, HideObject, LinfImage, PGD, Scorer,
)
from avsectester.data_types import TargetSpec

class FixedImageSource(DataSource):
    def sample(self, batch):
        return [AdversarialSample(torch.full((3, 8, 8), 0.5), TargetSpec((0, 0, 8, 8)))
                for _ in range(batch)]

class MeanIntensityScorer(Scorer):
    def target_score(self, image, target):
        return image.mean()

perturbation = LinfImage(epsilon=0.1, image_hw=(8, 8), device="cpu")
result = PGD(steps=2, batch=1, step_size=0.01).run(
    FixedImageSource(), perturbation, MeanIntensityScorer(), HideObject())
assert result.delta.shape == (3, 8, 8)
assert torch.max(torch.abs(result.delta)).item() <= 0.1
```

LinfImage projects an additive delta into an L-infinity ball and clips the resulting image to [0,
1]. PatchPerturbation renders its texture into each sample and exports an RGBA texture. A sample's
context supplies mask, the visible region, and either plane_sampling or sampling_grid:

```python
from avsectester.geometry.projection import project_box
from avsectester.rendering.surfaces import plane_sampling, prepare_surface

# Requires calibrated camera geometry, the patch's world-space PlaneSurface and a host ActorState.
sampler = prepare_surface(surface, camera, cam_from_world)
sampling = plane_sampling(sampler, camera, sampler.image_bounds(camera))
target_box = project_box(camera, cam_from_world, host.box_transform(), host.extent)
```

plane_sampling is the framework renderer's footprint geometry. PatchPerturbation filters the patch
with the same prefiltered sampler that renders it online, so optimized and deployed pixels match
and gradients reach every texel, also for distant patches. sampling_grid is an image-sized
grid_sample grid measured from a native renderer, such as a CARLA prop, that samples the texture
itself. Selecting the host and score region remains the attack's responsibility.

For a driving model, replace the scorer with SessionScorer inside the attack's service session:

```python
from avsectester.attack.optimization import PGD, HideObject, SessionScorer

# Requires OfflineContext context, a custom source and perturbation.
with context.services.session(context.seed) as session:
    scorer = SessionScorer(session, "detector", require_grad=True,
                           input_format="CHW float tensor [0,1]", device="cuda:0")
    result = PGD(steps=10).run(source, perturbation, scorer, HideObject())
```

Use a declared differentiable scorer for PGD. For NES, supply the same DataSource, Perturbation,
Scorer and OptimizationObjective contracts, with numpy candidates and a scorer that does not require
gradients. Its Perturbation.apply may render a candidate through closed-loop services; the generic
algorithm owns no rendering logic. NES options population_size and learning_rate control its query
batch and update size. HideObject suppresses a target score; SpoofObject maximizes it. Objectives
and data sources belong to the user's attack, while the same algorithms remain reusable.

## Configure a physical patch optimization

```yaml
type: PatchAttack
options:
  decision_steps: 20
  assets: ../assets/patch.png
  candidate_hosts: [lead]
  hosts_per_window: 1
  sensor: camera-0-0
  width_m: 1.0
  height_m: 0.5
  optimization:
    type: avsectester.implemented_attacks.patch_attack.offline:PatchOptimization
    options:
      algorithm:
        type: avsectester.attack.optimization.whitebox:PGD
        options: {steps: 10, step_size: 0.02, seed: 0}
      scorer_id: detector
      size: [64, 64]
      device: cuda:0
```

The named scorer must be exposed through avstack.bindings. PGD needs differentiable inputs; NES can
use score queries without gradients. PGD uses calibrated plane ray sampling and an occlusion mask as
a differentiable surrogate. NES uses candidate actions with session.render. Optional
validate_rollout checks the exported candidate in a rollout with the merged experiment/attack
environment and shared deployment budget. It validates completed status and decision count, then
logs a summary including final speed; it does not save that rollout log or claim an attack
impact. The session uses the selection seed; the algorithm seed controls optimizer randomness
independently. Empty metadata returns the original texture without attempting optimization. NES
accepts init_texture for a warm start. PGD rejects init_texture rather than ignoring it. Physical
deployment samples only origins accepted by session.can_start_at(position), which uses the
framework's deployment and capture tick requirements. An unsupported origin fails explicitly; the
attack does not calculate a separate window formula. By default each sample targets the host's
projected bounding box. An explicit target_box overrides it; this objective region is distinct from
patch placement. Save the returned artifact and verify its action receipts and recorded input rather
than assuming a completed optimizer implies successful deployment.

## Rendering and limitations

The simulator supplies SensorView evidence. Concrete attack rendering resolves host/world placement
and uses the view's pinhole or f-theta camera model. Selection reads clean geometry and visibility
evidence. Offline candidate rendering and online execution share SensorAttackAction deployment. A
physical CARLA panel uses WorldAttackAction instead.

Optimization algorithms can use different approximations and objective regions. Passing execution
checks establishes that the workflow runs; it does not establish attack impact.
Validate texture sampling, occlusion, distortion, score regions and deployed artifacts for the
scientific objective. CARLA's physical prop retains its native mesh dimensions; width_m/height_m do
not rescale it. CARLA sampling uses the native rendered prop region, while PGD remains a
differentiable surrogate. Its black/white emissive probes preserve the native visible footprint and
occlusion mask, but texture sampling maps into that region's image bounding rectangle rather than
recovering native mesh UV coordinates. This does not guarantee exact native appearance. Remaining
geometry work is tracked in [TODO](../development/TODO.md#physical-patch-geometry).

## Optimize a world object's texture

World objects inserted with InsertObject are composited by the simulator at every capture.
SensorView.composite_plan plans that composition for a view captured without the objects, such as
a clean `ClosedLoopSession.render(RenderRequest(...)).view`. The plan fixes geometry, coverage and
occlusion; `plan.composite` is the same blend as a differentiable function of surface colours.
Rounding its result reproduces the deployed capture, so an optimized texture is the one deployed.

<!-- runnable -->
```python
import numpy as np
import torch

from avsectester.closed_loop.simulators.base import SensorView
from avsectester.data_types import ActorState
from avsectester.geometry.cameras import BODY_FROM_OPTICAL, PinholeCamera
from avsectester.geometry.poses import make_pose, rotation_matrix
from avsectester.rendering import ObjectAsset, PlacedObject

# In an attack, use a clean ClosedLoopSession.render(...).view instead of this synthetic view.
camera = PinholeCamera(np.array([[300., 0., 160.], [0., 300., 120.], [0., 0., 1.]]), 320, 240)
view = SensorView("front", np.full((240, 320, 3), 90, np.uint8), camera,
                  np.linalg.inv(BODY_FROM_OPTICAL), {}, ActorState(np.eye(4)), 0.0, 0)
face = np.full((64, 64, 4), 255, np.uint8)
asset = ObjectAsset.from_panel(face, category="sign", width_m=0.9, mount_height_m=1.0)
sign = PlacedObject("sign", asset, make_pose(rotation_matrix([0., 0., 180.]), [8., 0., -1.5]))

plan = view.composite_plan([sign])
colors = torch.rand(3, 64, 64, requires_grad=True)
background = torch.as_tensor(view.data).permute(2, 0, 1).float() / 255
image = plan.composite(background, {("sign", 0): colors})
image.mean().backward()
assert colors.grad.abs().sum() > 0
```

The surface index refers to `asset.surfaces`; from_panel lists optional posts before the face.
The asset's alpha keeps the face shape, so only colours are optimized. To deploy, write the
optimized colours into an RGBA texture with the original alpha, build the ObjectAsset and return
InsertObject. Harmonizers are not differentiable: plan the objects without them, or account for
their effect separately. The objective, model inputs and update rule belong to the attack.

## User-defined objectives

A custom Attack can compute any objective from component values or a candidate RunLog, including
DrivingMetrics. A component scorer must declare inputs, outputs, codecs, direction and gradient
support. It is acceptable to use an expensive rollout objective; the user owns the optimization
budget. The framework does not implement an alternating dependency/workflow engine.

## Next steps

Read [Physical patch](../examples/physical-patch.md), [Driving-system
bindings](driving-systems.md#components-and-scorers-for-offline-work) and
[Evaluation](evaluation.md).
