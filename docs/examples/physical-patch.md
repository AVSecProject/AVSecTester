# Physical patch

> **Audience:** Users studying inserted patches and offline optimization.
>
> **Outcome:** Deploy one patch Attack across CARLA and NuRec.
>
> **Prerequisites:** [Attacks](../guides/attacks.md) and the selected simulator environment.

## Configuration

```bash
avsectester run configs/carla_patch.yaml
avsectester run configs/carla_patch_optimized.yaml
avsectester run configs/nurec_patch.yaml
```

PatchAttack is defined in avsectester.implemented_attacks.patch_attack.attack. The fixed and
optimized CARLA variants are separate configurations, each with one attack and clean/attacked
treatments on its selected metadata. NuRec uses the same class for image synthesis.

Required decision_steps is user-defined. Options include assets, candidate_hosts, hosts_per_window,
sensor, width_m, height_m, anchor, offset_m, orientation, max_distance, min_visible_fraction,
max_windows, keep_tail, optimization, prop and emissive. The assets option accepts a uint8 RGB/RGBA
array, an image path, or any Texture object with rgba(). None selects CheckerboardTexture.
load_rgba_image validates the supplied texture; invalid inputs do not select a fallback. YAML
assets can be a direct image path; CheckerboardTexture and user Texture constructors use a full
module:Class type/options specification. Unknown types and misspelled options fail.

A custom texture can return its pixels directly; no image-wrapper class is required:

<!-- runnable -->
```python
import numpy as np
from avsectester.implemented_attacks.patch_attack import PatchAttack

class SolidTexture:
    def rgba(self):
        return np.full((32, 64, 4), (255, 64, 0, 255), dtype=np.uint8)

attack = PatchAttack(decision_steps=5, assets=SolidTexture())
```

## Selection and deployment

The metabuilder scores eligible candidate_hosts independently within each decision window, then
keeps the user-requested hosts_per_window best hosts. Omitting candidate_hosts considers non-ego
actors. It reads clean SensorView geometry/evidence without deploying the attack for selection.
Ranking is local to hosts in each window, not a ranking of windows. It returns AttackSegment records
with host IDs, placement parameters and scores in user_data. Visibility thresholds and segment
lengths are attack parameters, not framework defaults. anchor and offset_m describe a host-local
mounting rule without naming a placeholder actor. AttachedPlacement is created only when a real
candidate host is selected.

A CARLA physical panel is a WorldAttackAction, deployed during the shared initialization window.
NuRec uses the patch's SensorAttackAction on the current capture. These are distinct physical
models: an image composite does not exert physical forces or alter other sensor modalities.

The CARLA prop mesh has its blueprint's native dimensions. width_m and height_m configure the ideal
planar asset used for image synthesis and visibility; they do not scale that native mesh. Do not
assume an arbitrary requested plane size is physically reproduced. Check the chosen prop, its
mounting and retained images in CARLA. CARLA optimization obtains the actual native prop region
through calibration renders. PGD then uses a differentiable sampling surrogate; emissive deployment
limits lighting differences but does not establish exact native material or texture equivalence.

## Offline optimization

Without optimization, assets pass through unchanged. Optional PGD or NES uses the selected metadata
and configured scorer to return a texture artifact. Formal execution and candidate rollouts receive
that explicit artifact through create_runtime. Selection is not rerun afterward. See
[Optimization](../guides/optimization.md) for the option structure and scorer requirements.

## Evidence

Record front raw/model_input images to inspect visible texture. Read action receipts for actual
deployment, artifact hashes for identity and driving/detection metrics for attack impacts. Geometric
visibility and successful execution alone do not guarantee attack impact.

## Next steps

Read [Image insertion](image-insertion.md) and [Evaluation](../guides/evaluation.md).
