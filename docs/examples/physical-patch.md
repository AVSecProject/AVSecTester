# Physical patch

> **Audience:** Researchers who attack with physical objects in the simulated world.  
> **Outcome:** Understand how a textured panel is prepared, deployed in CARLA at world setup, and bound to a selected vehicle.  
> **Prerequisites:** [Architecture](../concepts/architecture.md) and [Implement an attack](../guides/attacks.md).

**What it does.** The attack spawns a flat prop (`static.prop.ironplank` by default) rigidly
attached to a target vehicle and paints a texture onto it. CARLA then renders the panel in every
camera with scene perspective, lighting and occlusion, starting with the first observation.
By default the panel has a matte material that scene lighting affects. `emissive: true` also
paints the texture into the emissive channel, so scene lighting does not dim it. This mode is not
physically realistic, but it keeps an optimized texture's appearance.

**Preparation.** A texture comes from an image file or from the built-in checkerboard. Two scripts
optimize a texture offline with PGD against the CARLA-trained Faster R-CNN detector, to hide the
lead vehicle. Both connect to a CARLA server at `127.0.0.1:2000` and write to `tmp/patch_optim/`:

| Script | Method | Texture |
|---|---|---|
| [`optimize_patch.py`](../../scripts/optimization/optimize_patch.py) | Pastes the patch digitally onto a captured frame, placed from the detected box | `patch_texture.png` |
| [`optimize_patch_physical.py`](../../scripts/optimization/optimize_patch_physical.py) | Measures the footprint of a spawned emissive panel, optimizes for that region, then redeploys and re-detects | `phys_texture.png` |

```bash
python -m scripts.optimization.optimize_patch_physical --steps 250 --gpu 1
```

These scripts run outside the intervention interface. `PhysicalPatchIntervention.prepare` only
validates the panel configuration, without creating simulator actors, and returns a copy as the
artifact. See [Optimize attack artifacts](../guides/optimization.md) for the optimization interfaces.

**Not yet implemented.** Optimization placement derived from the same insertion geometry that
deployment uses.

**Execution.** `PhysicalPatchPlugin` runs at `simulator.setup`, before the first observation, and
changes the world. For each configured panel it resolves the target, spawns the prop at the
configured offset in the target's frame, and uses one backend setup tick to register the prop's
paintable mesh name. Each run therefore needs one setup tick per panel. In a comparison, every
treatment advances the same number of setup ticks, including treatments without panels. See
[World setup ticks](../reference/runtime.md#world-setup-ticks). The plugin requires a backend with
CARLA setup ticks. It removes its panels when the run ends, including after a failed deployment.

`target` defaults to `lead` and is resolved in this order:

1. A role from `context.case.binding_ids` that holds exactly one actor, for example `attacker`.
2. An indexed role alias, for example `hosts[0]`. A role with several actors requires an index and
   raises `ValueError` without one.
3. A backend logical name: `ego`, `lead` (when the scene configures `lead`) or `npc:<index>`.
4. The CARLA actor ID of one of those backend-managed actors.

An unresolved target raises `ValueError`. Roles come from the [match](../concepts/architecture.md#terminology)
of a selected scenario. See [Roles and bindings](../guides/scenarios.md#roles-and-bindings).

**Configure and run.** The [`patches`](../reference/configuration.md#patches) list in
`configs/carla_patch_scenario.yaml` prepares this plugin as an attack, enabled in the `attack` and
`attack-defense` treatments. The scene also spawns a stationary `lead` vehicle in every treatment,
so the panel is the only difference between the runs:

```yaml
lead:
  gap: 9.0
  vehicle: vehicle.tesla.model3

patches:
  - target: lead
    prop: static.prop.ironplank
    location: [-2.3, 0.0, 1.0]   # Offset in the target's frame, in metres.
    rotation: [90.0, 0.0, 0.0]   # Pitch, yaw, roll in degrees. Stands the plank upright.
    texture: {pattern: checkerboard, size: 256, squares: 8, color_a: [255, 0, 0], color_b: [255, 255, 255]}
```

Use `texture: {image: <path>, size: <pixels>}` for an optimized or supplied image. Without
`image`, the texture is a checkerboard built from `size`, `squares`, `color_a` and `color_b`. The
`pattern` key documents the choice but is not read. `avsectester run configs/carla_patch_scenario.yaml`
runs the configured pair. The driving system in that file detects objects from lidar, so the
panel's texture does not reach its perception. To view the panel in camera images, run
[`carla_patch_demo.py`](../../scripts/demos/carla/carla_patch_demo.py), which drives a fixed-throttle
`CruiseStack` with the attack enabled and overlays the CARLA-trained Faster R-CNN detections unless
`--no-detect` is given:

```bash
python -m scripts.demos.carla.carla_patch_demo --frames 16 --gap 6 \
    --texture tmp/patch_optim/phys_texture.png
```

With `--texture`, the demo deploys the image as an emissive panel. It writes frames to
`tmp/carla_patch/seq/`, `patch_filmstrip.png` and `patch.gif`. From Python, register the plugin
directly:

```python
import yaml

from avsectester.attacks.patch.physical_patch import PhysicalPatchPlugin
from avsectester.experiment import configured_stack
from avsectester.runner import run_prepared
from avsectester.closed_loop.hooks import Hook, Runtime
from avsectester.scenarios.setup import scenario_from_config

config = yaml.safe_load(open("configs/carla_patch_scenario.yaml"))
scenario = scenario_from_config(config)  # The scene view ignores the patches list.
attack = Runtime([Hook("simulator.setup", PhysicalPatchPlugin(config["patches"]))])
results = run_prepared(
    scenario, lambda: configured_stack(config), 20,
    runtimes={"clean": Runtime(), "attack": attack},
)
```

To composite a patch into the camera input instead of spawning a panel, see
[Image insertion](image-insertion.md) and `patch_driving_demo.py`.

**Code.** `avsectester/attacks/patch/physical_patch.py` (`PhysicalPatch`, `PhysicalPatchPlugin`,
`PhysicalPatchIntervention`).
