# Optimize attack artifacts

> **Audience:** Researchers who optimize attack artifacts such as patch textures.  
> **Outcome:** Understand where optimization fits, use the current optimization interfaces and connect an optimized artifact to attack preparation.  
> **Prerequisites:** [Architecture](../concepts/architecture.md) and [Implement an attack](attacks.md).


## Where optimization fits

[Optimization](../concepts/architecture.md#terminology) searches for the parameters of an
[artifact](../concepts/architecture.md#terminology), such as a patch texture. It works on behalf
of [preparation](../concepts/architecture.md#terminology): preparation supplies the problem and the
data, optimization evaluates candidates and returns the optimized artifact, and preparation
returns that artifact to the experiment. Execution then deploys it in every run of the attack
treatments. No runner, CLI command or configuration key starts an optimization directly. It runs
inside `Intervention.prepare` or in a standalone script.

The architecture names three ways to evaluate a candidate. Only the first two exist today:

| Evaluation route | How a candidate is scored | Current support |
|---|---|---|
| Render through the simulator | Paint the candidate into the world, render the camera frame and score it | `CarlaRenderPerturbation` with `NES` and `MMDetInferenceScorer`. No supplied script uses this route |
| Query a driving-system component | Apply the candidate to a captured image and score it with a perception model outside the driving loop | `PatchPerturbation` or `LinfImage` with `PGD` and `MMDetRPNObjectnessScorer`. Both supplied scripts use this route |
| Roll out in the closed loop | Run driving steps with the candidate deployed and score the driving outcome | Not available. See [Current limitations](#current-limitations) |

The supplied scripts also use CARLA around the second route: to capture the clean frame and, in
`optimize_patch_physical.py`, to measure where the panel renders and to re-detect after
deployment. The rendered frame is not inside the optimization loop.

## Current optimization interfaces

`avsectester.attacks.optim` splits an optimization into a data source, a threat model, model
access, a goal and an algorithm. Each part is a separate interface, so an algorithm works with
every compatible threat model and scorer (see [Valid combinations](#valid-combinations)):

```text
DataSource.sample(batch)          -> [AdvSample(x, target, context)]
Perturbation.apply(sample, delta) -> adversarial input
Perturbation.project(delta)       -> delta inside the feasible set
Scorer.target_score(image, target) -> target confidence in [0, 1]
AttackObjective.loss(score)       -> scalar to minimize
GradientAttack.run(data, perturbation, scorer, objective) -> AttackResult
```

In the torch implementations, images and `delta` are RGB floats in `[0, 1]`, channels first
`(C, H, W)`, on the scorer's device. The rendering route uses NumPy instead. For every objective,
a lower loss means a more successful attack.

### Interfaces

The interfaces are exported by `avsectester.attacks.optim` and import without torch.

| Name | Signature | Contract |
|---|---|---|
| `TargetSpec` | `TargetSpec(box: tuple[float, float, float, float], label: str \| None = None)` | The attacked object as an image-space `xyxy` box in pixels. This class differs from the scenario `TargetSpec` |
| `AdvSample` | `AdvSample(x, target: TargetSpec, context: dict = {})` | One clean input. `context` carries threat-model data, such as a patch `placement` |
| `DataSource` | `sample(batch: int) -> list[AdvSample]` | Returns `batch` samples, with any per-sample randomness drawn again |
| `Perturbation` | `init()`, `apply(sample, delta)`, `project(delta)`, `export(delta) -> np.ndarray \| None` | Owns the variable `delta`, how it enters the input and its feasible set. `export` returns a deployable form, or `None` by default |
| `Scorer` | `differentiable: bool = True`, `target_score(image, target)` | Confidence that the target is detected. Differentiable scorers return a scalar tensor. Others return a detached value |
| `AttackObjective` | `loss(score)` | `HideObject` returns `score`. `SpoofObject` returns `1 - score` |
| `GradientAttack` | `run(data, perturbation, scorer, objective) -> AttackResult` | The algorithm. It calls only the four interfaces above |
| `AttackResult` | `AttackResult(delta, export, history: list[float], final_score: float)` | The optimized `delta`, `perturbation.export(delta)`, the loss per step and a final score |

### Supplied implementations

| Class | Module | Constructor | Behavior |
|---|---|---|---|
| `FrameSource` | `.data` | `FrameSource(samples, brightness=(0.8, 1.2), contrast=(0.8, 1.2), seed=0)` | Draws samples with replacement and sets random `brightness` and `contrast` in each sample's context. This is the expectation over transformation. Raises `ValueError` for an empty list |
| `PatchPerturbation` | `.perturbations` | `PatchPerturbation(size=(128, 128), device="cuda:0", init="random", seed=0)` | `delta` is a `(3, Hp, Wp)` texture. `apply` warps it with the 3x3 homography `context["placement"]` (patch pixels to image pixels), applies the context's brightness and contrast to the patch and composites it. `project` clamps to `[0, 1]`. `export` returns `(Hp, Wp, 4)` opaque `uint8` RGBA. Any `init` other than `"random"` starts from gray |
| `LinfImage` | `.perturbations` | `LinfImage(epsilon=8/255, image_hw=(600, 800), device="cuda:0")` | Adds `delta` to the whole image and clamps to `[0, 1]`. `project` clamps `delta` to `[-epsilon, epsilon]`. `export` returns `None` |
| `CarlaRenderPerturbation` | `.render` | `CarlaRenderPerturbation(paint, render, size=(64, 64), init_texture=None, seed=0)` | NumPy `delta`. `apply` calls `paint(rgba)` with the `(H, W, 4)` `uint8` texture, then returns `render()`. It ignores `sample.x`. `init_texture` is an optional `(3, H, W)` warm start in `[0, 1]`. `export` returns RGBA `uint8` |
| `MMDetRPNObjectnessScorer` | `.scorers` | `MMDetRPNObjectnessScorer(model, strides=(4, 8, 16, 32, 64))` | Differentiable. The mean RPN objectness of a two-stage MMDetection model over anchor cells whose centers lie in the target box. Returns 0 when no cell lies in the box |
| `MMDetInferenceScorer` | `.scorers` | `MMDetInferenceScorer(model)` | Not differentiable. Runs `inference_detector` on an `(H, W, 3)` `uint8` RGB frame and returns the highest score of a detection whose box center lies in the target box, or 0.0 |
| `PGD` | `.attacks` | `PGD(steps=200, step_size=4/255, batch=4, tv_weight=0.0, seed=0, verbose=False, log_every=25)` | Signed-gradient descent averaged over a batch, plus `tv_weight` times total variation. Raises `ValueError` for a scorer that is not differentiable. Seeds torch's global generator. `history` holds the loss per step, including the TV term. `final_score` is the mean confidence over a fresh batch |
| `NES` | `.blackbox` | `NES(steps=40, popsize=20, sigma=0.08, lr=0.03, seed=0, verbose=False, log_every=5)` | Gradient-free. Draws one sample once and estimates the gradient from `popsize // 2` antithetic pairs. `history` starts with the initial loss. `final_score` is the last loss, which is the confidence only for `HideObject` |

Two placement helpers in `avsectester.attacks.optim.geometry` build `placement` homographies:
`homography_4pt(src, dst)` maps four points to four points, and
`rear_panel_homography(box, patch_hw, wfrac=0.7, hfrac=0.5, ycenter=0.55)` maps the patch onto a
centered rectangle inside a 2D box.

The MMDetection scorers take the raw model. In the scripts this is the `model` attribute of
avstack's `MMDetObjectDetector2D`, the CARLA-trained Faster R-CNN described in
[Models used in this repository](driving-systems.md#models-used-in-this-repository).

### Valid combinations

| Algorithm | Perturbation | Scorer |
|---|---|---|
| `PGD` | `PatchPerturbation` or `LinfImage`, or another torch perturbation that is differentiable in `delta` | `MMDetRPNObjectnessScorer`, or another scorer with `differentiable = True` that returns a tensor |
| `NES` | `CarlaRenderPerturbation`, or another perturbation whose `delta` is a NumPy array | `MMDetInferenceScorer`, or another scorer whose result converts to `float` |

`PGD` checks `scorer.differentiable`. It does not check the perturbation, so a NumPy perturbation
fails during the gradient step. `NES` checks neither. Torch perturbations fail inside `NES`, and
`MMDetRPNObjectnessScorer` cannot score the NumPy frames that `CarlaRenderPerturbation` returns.

### Combine the pieces

This example runs offline. A brightness scorer stands in for a detector, so the effect is easy to
check: hiding the target darkens the patch.

<!-- runnable -->
```python
import torch

from avsectester.attacks.optim.attacks import PGD
from avsectester.attacks.optim.data import FrameSource
from avsectester.attacks.optim.geometry import rear_panel_homography
from avsectester.attacks.optim.interface import AdvSample, HideObject, Scorer, TargetSpec
from avsectester.attacks.optim.perturbations import PatchPerturbation


class BoxBrightness(Scorer):
    """A differentiable stand-in for a detector: mean brightness inside the target box."""

    differentiable = True

    def target_score(self, image, target):
        x0, y0, x1, y1 = (int(v) for v in target.box)
        return image[:, y0:y1, x0:x1].mean()


box = (40.0, 30.0, 120.0, 90.0)  # Target box, xyxy image pixels.
frame = torch.full((3, 120, 160), 0.6)  # Clean RGB image, (C, H, W) in [0, 1].
placement = torch.as_tensor(rear_panel_homography(box, (32, 32))).float()
data = FrameSource([AdvSample(x=frame, target=TargetSpec(box, label="car"),
                              context={"placement": placement})])

result = PGD(steps=30, step_size=0.05, batch=2, tv_weight=0.02).run(
    data,
    PatchPerturbation(size=(32, 32), device="cpu", init="gray"),
    BoxBrightness(),
    HideObject(),
)
print(round(result.history[0], 3), round(result.history[-1], 3), round(result.final_score, 3))
print(result.export.shape, result.export.dtype)
```

```text
0.591 0.428 0.404
(32, 32, 4) uint8
```

The loss falls, and `export` is the RGBA texture to deploy. To attack a detector, build the frame
from a camera image, replace `BoxBrightness()` with `MMDetRPNObjectnessScorer(model)` and create
the frame and the perturbation on the model's device, as the supplied scripts do.

For the rendering route, `paint` and `render` connect the perturbation to a live panel. In
`optimize_patch_physical.py`, the equivalent of `paint` applies the texture with
`world.apply_color_texture_to_object` to the panel's painted object, and `render` ticks the world
and returns the camera frame:

```python
# Continues the previous example. paint, render and model are supplied by the caller.
# CARLA and MMDetection are required.
from avsectester.attacks.optim.blackbox import NES
from avsectester.attacks.optim.render import CarlaRenderPerturbation
from avsectester.attacks.optim.scorers import MMDetInferenceScorer

data = FrameSource([AdvSample(x=None, target=TargetSpec(box))])
perturbation = CarlaRenderPerturbation(paint, render, size=(64, 64))
nes_result = NES(steps=40, popsize=20).run(
    data, perturbation, MMDetInferenceScorer(model), HideObject()
)
```

Every loss evaluation paints and renders once, so a run renders
`1 + steps * (2 * (popsize // 2) + 1)` frames: 841 with the defaults. The PGD texture from the
previous example can serve as a warm start: `init_texture=result.delta.cpu().numpy()`.

## Use optimization in attack preparation

### Optimize inside `prepare`

Run the optimizer in `Intervention.prepare` and return the exported texture as the artifact. The
artifact must be reusable without optimizing again, so save the texture to a file and return its
path. The following intervention extends `PhysicalPatchIntervention`, whose `hooks` already
deploy the artifact through `PhysicalPatchPlugin` at `world.setup`:

<!-- runnable -->
```python
from pathlib import Path

from PIL import Image

from avsectester.attacks.optim.attacks import PGD
from avsectester.attacks.optim.interface import HideObject
from avsectester.attacks.optim.perturbations import PatchPerturbation
from avsectester.attacks.patch.physical_patch import PhysicalPatchIntervention


class OptimizedPatchIntervention(PhysicalPatchIntervention):
    """Optimize a panel texture during preparation, then deploy it as a CARLA panel."""

    def __init__(self, patches, problem, texture_path, *, size=128, device="cuda:0", pgd=None):
        super().__init__(patches)
        self.problem = problem  # problem(scenario) -> (DataSource, Scorer)
        self.texture_path = Path(texture_path)
        self.size = size
        self.device = device
        self.pgd = pgd or PGD(steps=300, step_size=6 / 255, batch=4, tv_weight=0.02)

    def prepare(self, scenario):
        patches = super().prepare(scenario)  # Validates the panel configuration.
        data, scorer = self.problem(scenario)
        perturbation = PatchPerturbation(size=(self.size, self.size), device=self.device)
        result = self.pgd.run(data, perturbation, scorer, HideObject())
        self.texture_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(result.export).save(self.texture_path)
        for patch in patches:
            patch["texture"] = {"image": str(self.texture_path), "size": self.size}
            patch["emissive"] = True
        return patches
```

`problem` receives the complete prepared [scenario](../concepts/architecture.md#terminology). It
can capture frames from `scenario.make_backend()`, take the target from `scenario.match` and load
the scorer's model. The artifact is the panel list with the texture path, which the default
artifact codec stores as JSON. `emissive: true` keeps the optimized colors independent of scene
lighting. Remove it to deploy a lit panel.

Prepare once and run the treatments with the prepared intervention:

```python
from avsectester.preparation import Treatment, prepare_intervention
from avsectester.runner import run_prepared
from avsectester.runtime import Runtime

# scenario is a CARLA prepared scenario, and stack_factory, problem and the panel list patches
# are supplied by the caller.
method = OptimizedPatchIntervention(patches, problem, "tmp/optimized_patch/texture.png")
prepared = prepare_intervention(method, scenario, name="optimized-patch")
prepared.save("tmp/optimized_patch/prepared.json")
results = run_prepared(
    scenario, stack_factory, 20,
    runtimes={"clean": Runtime(), "attack": Treatment((prepared,)).runtime()},
)
```

The preparation part of this example was run offline with a synthetic problem. The CARLA run was
checked against the API only. `PreparedIntervention.load(method, path)` restores the record
without optimizing again. The record stores the texture path, not the image, so keep the file.
See [Save and reuse artifacts](attacks.md#save-and-reuse-artifacts) and
[Artifact codec](../reference/python-api.md#artifact-codec). Panel keys and target resolution are
described in [Physical patch](../examples/physical-patch.md).

To reuse a saved texture in a configured run, reference the file from a
[`patches`](../reference/configuration.md#patches) entry with
`texture: {image: <path>, size: <pixels>}`.

### Deploy through image insertion

A texture can also be inserted into camera images instead of the world. `result.export` is a valid
`PlaneAsset` texture, and `PlaneAsset` also accepts the saved file path:

```python
from avsectester.insertion import AttachedPlacement, Insertion, Orientation, PlaneAsset

# result is the AttackResult of a PatchPerturbation optimization.
patch = Insertion(
    id="optimized_patch",
    asset=PlaneAsset(result.export, width_m=0.6, height_m=0.4),
    placement=AttachedPlacement(host="attacker", anchor="rear_center", offset_m=(-0.05, 0.0, 0.15)),
    orientation=Orientation("follow_host", rotation_deg=(0, 0, 180)),
)
```

Insertions are specified in [Specify inserted objects](scenarios.md#specify-inserted-objects) and
fed to the driving system as described in
[Insert into the model input](../examples/image-insertion.md#insert-into-the-model-input).

### How the supplied scripts do it

The two scripts in `scripts/optimization/` combine scene setup, optimization, deployment and
validation in one process. They run outside the intervention interface: they spawn their own ego
and lead vehicles, and nothing calls them from `prepare`. Both need a CARLA server at
`127.0.0.1:2000`, avstack with MMDetection and a GPU (`--gpu`, default 1).

| Step | `optimize_patch.py` | `optimize_patch_physical.py` |
|---|---|---|
| Problem | Captures one frame with a lead vehicle 6 m ahead. The highest-scoring detection gives the target box | The same scene and target box. Spawns an emissive `PhysicalPatch` on the lead vehicle |
| Placement | `rear_panel_homography(box, (patch, patch))` | Paints the panel green, renders and takes the bounding rectangle of green pixels. `homography_4pt` maps the patch onto that rectangle |
| Optimization | `FrameSource`, `PatchPerturbation`, `MMDetRPNObjectnessScorer`, `HideObject`, `PGD(step_size=6/255, batch=4, tv_weight=0.02)` with `--steps` (default 300) | The same pieces with `--steps` (default 250) |
| Validation | Pastes the texture digitally and reruns the full detector | Paints the texture on the panel, renders and reruns the full detector |

The texture files, outputs and the commands that deploy them are listed in
[Physical patch](../examples/physical-patch.md). To move a script's problem into preparation, put
its capture and detector setup into the `problem` function above.

## Current limitations

- **Not yet implemented.** No evaluation route scores candidates by rolling them out in the closed
  loop, so no objective can use commands or driving outcomes.
- **Not yet implemented.** The supplied scorers and `TargetSpec` cover camera images and 2D target
  boxes only; no scorer reads other sensors or later driving-system outputs.
- **Not yet implemented.** The interfaces are specific to attacks, with object-hiding and spoofing
  objectives; defenses have no optimization interface.
- **Not yet implemented.** Optimization placement does not use the insertion geometry that
  deployment uses: the scripts place patches from a detector box or an axis-aligned rendered
  footprint. See [TODO](../development/TODO.md#unify-insertion-geometry).
- `NES` optimizes against one sample and applies no transformation sampling. No supplied script
  runs the rendering route.
- `PGD` seeds torch's global random generator.

## Next steps

- [Physical patch](../examples/physical-patch.md) to deploy an optimized texture in CARLA.
- [Package the attack as an intervention](attacks.md#package-the-attack-as-an-intervention) for
  the preparation interface.
- [Evaluate results](evaluation.md) to measure the driving impact of the deployed artifact.
- [Implementation status](../concepts/architecture.md#implementation-status) for planned
  optimization routes.
