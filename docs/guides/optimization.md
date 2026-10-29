# Optimize attack artifacts

> **Audience:** Attack authors implementing an offline objective.
>
> **Outcome:** Reuse scoring, rendering and optimization tools, and report prepared artifacts.
>
> **Prerequisites:** [Attacks](attacks.md).

## Optimization interfaces

An attack supplies its sample selection, perturbation and objective. Algorithms do not choose
scenes, hosts or placement rules.

| Interface | Responsibility |
|---|---|
| `DataSource` | Map-style dataset: `__len__` and `__getitem__` provide `AdversarialSample` records |
| `Perturbation` | Initialize, apply, project, export and measure the smoothness of a candidate |
| `Scorer` | `target_scores(images, targets)` returns one score per input |
| `OptimizationObjective` | Convert scores into losses |
| `GradientOptimizer` | Run the algorithm and return `OptimizationResult` |

`HideObject` minimizes a score. `SpoofObject` maximizes it. `HideBelowThreshold` minimizes only
the part of a score above a decision threshold minus a margin, for targets where one detection
decides the outcome. Objectives operate elementwise on score batches. Scores can be detector
confidence or a physical quantity such as distance. An optional `monitor(index, delta)` observes
the initial variable at index 0, then each completed PGD epoch or NES step. `PGD` requires a
differentiable score, while `NES` uses score queries. `PGD`'s `tv_weight` weights
`Perturbation.smoothness`, which defaults to total variation of the whole variable. A perturbation
that realizes only part of it overrides the method. A `SessionScorer` connects either path to a
declared `DrivingAdapter` binding. [Driving-system bindings](driving-systems.md) describes
model integration. The [Python API](../reference/python-api.md) lists signatures and
optimizer constraints.

This self-contained example uses a local scorer and needs no model weights:

<!-- runnable -->
```python
import torch
from avsectester.attack.optimization import (
    AdversarialSample, DataSource, HideObject, LinfImage, PGD, Scorer,
)
from avsectester.data_types import TargetSpec

class FixedImageSource(DataSource):
    def __len__(self):
        return 1

    def __getitem__(self, index):
        if index != 0:
            raise IndexError(index)
        return AdversarialSample(torch.full((3, 8, 8), 0.5), TargetSpec((0, 0, 8, 8)))

class MeanIntensityScorer(Scorer):
    def target_scores(self, images, targets):
        if len(images) != len(targets):
            raise ValueError("Image and target batch lengths must match")
        return images.mean(dim=(1, 2, 3))

perturbation = LinfImage(epsilon=0.1, image_hw=(8, 8), device="cpu")
result = PGD(epochs=2, batch_size=1, step_size=0.01).run(
    FixedImageSource(), perturbation, MeanIntensityScorer(), HideObject())
assert result.delta.shape == (3, 8, 8)
assert torch.max(torch.abs(result.delta)).item() <= 0.1
```

`LinfImage` applies a bounded additive image change. `PatchPerturbation` renders texture colors
using the geometry supplied in each sample. The sample's target region belongs to the objective.
It is independent of the placement rule.

`PGD` shuffles samples using `seed` and retains the final smaller batch. Images in one batch
must have equal dimensions. Random augmentations are redrawn in `DataSource.__getitem__`.
`OptimizationResult.history` records each epoch's mean sample objective, excluding TV.

`NES` uses `steps` because it optimizes one seeded dataset sample rather than traversing the
dataset. It scores candidate perturbations in groups of at most `batch_size` per query.

## Use a driving-system scorer

Create a session inside the offline phase, select a declared scorer, and keep that session open
for calibration, updates and evaluation:

```python
from avsectester.attack.optimization import HideObject, PGD, SessionScorer

# Requires OfflineContext context, a user DataSource and a Perturbation.
with context.services.session(context.seed) as session:
    scorer = SessionScorer(session, "rule_based.detection_confidence", require_grad=True,
                           input_format="BCHW float tensor [0,1]", device="cpu")
    result = PGD(epochs=10, batch_size=4).run(source, perturbation, scorer, HideObject())
```

Scorer declarations follow
[Driving-system bindings](driving-systems.md#components-and-scorers-for-offline-work).
Use [candidate rollouts](attacks.md#offline-phase) when the score depends on a planned trajectory or
physical response.

## Optimize a world object's texture

A captured `SensorView.composition` retains the background and the composition plan used for
that capture. Reuse its plan when optimizing colors. In geometry occlusion mode, a clean view
can also build a plan through `composite_plan`. The [Simulator guide](simulators.md) describes
that capability.

Differentiable harmonizers use the same floating-point computation during optimization and
deployment, followed by uint8 rounding for deployment. This example checks that equivalence and
backpropagates through `ExposureHarmonizer`:

<!-- runnable -->
```python
import numpy as np
import torch

from avsectester.closed_loop.simulators.base import SensorView
from avsectester.data_types import ActorState
from avsectester.geometry.cameras import BODY_FROM_OPTICAL, PinholeCamera
from avsectester.geometry.poses import make_pose, rotation_matrix
from avsectester.rendering import ExposureHarmonizer, ObjectAsset, PlacedObject

# In an attack, use a clean ClosedLoopSession.render(...).view instead of this synthetic view.
camera = PinholeCamera(np.array([[300., 0., 160.], [0., 300., 120.], [0., 0., 1.]]), 320, 240)
view = SensorView("front", np.full((240, 320, 3), 90, np.uint8), camera,
                  np.linalg.inv(BODY_FROM_OPTICAL), {}, ActorState(np.eye(4)), 0.0, 0)
face = np.full((64, 64, 4), 255, np.uint8)
asset = ObjectAsset.from_panel(face, category="sign", width_m=0.9, mount_height_m=1.0)
sign = PlacedObject(
    "sign", asset, make_pose(rotation_matrix([0., 0., 180.]), [8., 0., -1.5]),
    harmonizer=ExposureHarmonizer(reference="under", gain_range=(0.5, 0.5)),
)

plan = view.composite_plan([sign])
colors = (torch.as_tensor(face[..., :3]).permute(2, 0, 1).float() / 255).requires_grad_()
background = torch.as_tensor(view.data).permute(2, 0, 1).float() / 255
image = plan.composite(background, {("sign", 0): colors})
deployed = plan.apply(view.data).image
rounded = image.detach().permute(1, 2, 0).mul(255).round().to(torch.uint8).numpy()
np.testing.assert_array_equal(deployed, rounded)
image.mean().backward()
assert colors.grad.abs().sum() > 0
```

Keep the asset's alpha when exporting colors. A surface's `nominal_texture` provides an
unchanged appearance reference when scene statistics should not depend on the optimized texture.
`Harmonizer` formulas, parameters and non-differentiable restrictions are described in
the [rendering API](../reference/python-api.md#avsectesterrenderingharmonizers).

## Offline reports

`report.json` contains `status`, `values`, `series`, `tables` and `images`. Values retain their
unit and description. Series retain coordinates, axis labels, units and description. Tables
retain rows, column descriptions and CSV filenames. Images retain descriptions and PNG
filenames. CSV column order is the union of row keys in first-appearance order. Auxiliary files
live below `files/`.

The offline HTML page expands tables with at most 20 rows and eight columns. Larger tables
are collapsible.

`context.report` collects measurements independently of returned artifacts. Each setter replaces
an item with the same name and kind. Include units, descriptions and table-column explanations
so the saved report is understandable without opening the implementation:

```python
report = context.report
report.set_value("training_segments", len(context.metadata), unit="segments",
                 description="Segments used by this offline phase.")
report.set_series("confidence", objective_history, x_label="iteration",
                  y_label="Mean confidence", y_unit="ratio (0–1)",
                  description="Confidence measured before each update.")
report.set_table(
    "samples", [{"frame": 12, "distance_m": 20.0}],
    description="Captured training frames.",
    columns={"frame": "Recorded frame index.", "distance_m": "Distance to the line (m)."},
)
report.set_image("initial_texture", initial_texture, description="Initial texture.")
```

Read structured entries through `report.values[name].value`, `report.tables[name].rows` and
`report.images[name].pixels`. `report.directory` is the workspace for additional files. It is
`None` for in-memory output. The
[`OfflineReport` API](../reference/python-api.md#avsectesterattackoffline_report) defines
supported values and name validation.

The framework persists partial reports on failure and restores completed reports during resume.
[Experiments](experiments.md) defines checkpoint behavior. The
[CLI reference](../reference/cli.md) describes output locations and the `evaluate` rebuild.

## Supplied optimization paths

`PatchOptimization` connects `PatchAttack` to driving scorers and candidate rendering.
`DetectorPatchOptimization` and `RolloutPatchOptimization` configure the stop-sign white-box and
black-box attacks. Their supplied settings and uses are in
[Physical patch](../examples/physical-patch.md) and
[Stop-sign patch](../examples/stop-sign-patch.md).

## Next steps

Read [Driving-system bindings](driving-systems.md), [Evaluation](evaluation.md)
and [Examples](../examples/README.md).
