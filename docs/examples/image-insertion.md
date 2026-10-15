# Image insertion

> **Audience:** Researchers who insert objects or patches into camera images.  
> **Outcome:** Understand how payloads are built, inserted into the model input, harmonized and inspected with the demo scripts.  
> **Prerequisites:** [Architecture](../concepts/architecture.md) and [Implement an attack](../guides/attacks.md).

**What it does.** Image insertion renders an object into camera images before they reach the
driving system. The object can be an ordinary STOP sign, pedestrian image or traffic-signal board.
The attack comes from placing that visual cue where it does not belong, so an optimized texture is
not required. No physical actor is spawned.

**Preparation.** Build a payload asset and give it an explicit placement and orientation as an
[insertion](../concepts/architecture.md#terminology). Selection is optional. A selection requirement
can declare insertions, bind their host roles and apply visibility filters. Person payloads can be
extracted beforehand from nuScenes images.

**Execution.** An insertion adapter runs at `observation` and returns an `Observation` with the
camera image replaced. It acts on Vehicle I/O. At each driving step it resolves each insertion
against current actor poses, applies visibility evidence, composites the surfaces and optionally
harmonizes their appearance. It never reruns selection filters.

**Configure and run.** No configuration key renders insertions into the model input. Connect an
adapter in a Python runtime factory, as shown in
[Insert into the model input](#insert-into-the-model-input). The demo scripts are listed in
[Demo scripts and outputs](#demo-scripts-and-outputs).

**Code.** `avsectester/attacks/object_insertion/` (payloads), `avsectester/insertion.py`
(insertion types), `avsectester/rendering/` (camera geometry, surface sampling, visibility,
compositing and harmonization), `avsectester/simulators/carla.py` (`insertion_perturbation`),
`avsectester/scenarios/datasets/nurec.py` (`NuRecDataset.insertion_renderer`) and
`avsectester/perturbations/image.py` (`frame_perturbation`).

## Payloads

| Payload | Module | Asset |
|---|---|---|
| STOP sign face and post | `avsectester/attacks/object_insertion/sign_spoof.py` | `SignAsset` |
| Pedestrian cut-out or printed poster | `avsectester/attacks/object_insertion/person_poster.py` | `standee`, `billboard` |
| Printed traffic-signal board | `avsectester/attacks/object_insertion/traffic_light.py` | `roadside_rig` with `traffic_lights_rgba` |
| Patch texture or sign face | `avsectester/insertion.py` | `PlaneAsset` |

Payload constructors return local assets made of textured planes. A `SignAsset` stands on the
ground point beneath its face, faces local +X and places the face's bottom edge at `mount_height`.
Placement and orientation are given separately by `Insertion`. This example runs offline:

<!-- runnable -->
```python
from avsectester.attacks.object_insertion.sign_spoof import SignAsset
from avsectester.attacks.object_insertion.traffic_light import roadside_rig, traffic_lights_rgba
from avsectester.insertion import (
    AttachedPlacement, Insertion, Orientation, PlaneAsset, WorldPlacement,
)

# A roadside STOP sign at a fixed world position, facing back along -x.
sign = Insertion(
    "stop_sign", SignAsset(width=0.9, mount_height=1.5),
    WorldPlacement((20, -3, 0)), Orientation("fixed_world", (0, 0, 180)),
)
# A red traffic-signal board on a post.
lights = Insertion(
    "lights", roadside_rig(traffic_lights_rgba(n=3, lit="red")),
    WorldPlacement((26, -6, 0)), Orientation("fixed_world", (0, 0, 180)),
)
# A sign face, without a post, on the rear of the actor bound to the role "attacker".
face = Insertion(
    "rear_face", PlaneAsset(SignAsset().face, 0.6, 0.6),
    AttachedPlacement("attacker", "rear_center", (-0.03, 0, 0)),
    Orientation("follow_host", (0, 0, 180)),
)
```

The STOP face is bundled in `avsectester/assets/signs/`. Its source and license are recorded in
[SOURCES.md](../../avsectester/assets/signs/SOURCES.md). Person payloads take a caller-supplied RGBA
image. [`extract_person_cutouts.py`](../../scripts/preparation/extract_person_cutouts.py) segments
annotated nuScenes pedestrians with SAM. `--sam` accepts a model ID or a local model directory.
The cut-outs are derived dataset assets and are not bundled. The script records their nuScenes
source and license in `index.json`:

```bash
python -m scripts.preparation.extract_person_cutouts --nuscenes /path/to/nuscenes \
    --out /path/to/pedestrians
```

Placement anchors, orientation modes and coordinate conventions are described in
[Specify inserted objects](../guides/scenarios.md#specify-inserted-objects). Builder signatures and
asset geometry are in [Insertion specification](../reference/python-api.md#insertion-specification).

## Insert into the model input

The rendered insertion must change the observation that the driving system receives. Register the
adapter at `observation` through `ObservationTransform`, or pass it as the `perturb` shorthand,
which becomes the same handler chain. A visualization callback changes only displayed or saved
images and is not an input attack.

| Simulator | Adapter | Visibility evidence |
|---|---|---|
| CARLA | `insertion_perturbation(backend, insertions, ...)` with a `CarlaSelectionBackend` | Aligned depth from the backend's depth camera |
| NuRec | `NuRecDataset.insertion_renderer(...)` wrapped by `frame_perturbation` | Labeled actor cuboids |

Use `run_prepared` with a fresh compatible driving system per treatment. It prepares the shared
origin and closes the backend, including after failed runs. Runtime factories receive the backend
before reset, so camera- and actor-dependent setup must happen lazily. Both supplied adapters read
the camera and actor poses only when they process an image.

For a scenario selected by `CarlaScenarioBuilder` with `CarlaSelectionBackend`:

```python
from avsectester.interventions import ObservationTransform
from avsectester.runner import run_prepared
from avsectester.runtime import Hook, Runtime
from avsectester.simulators.carla import insertion_perturbation

# scenario and stack_factory are supplied by the caller.
def attacked_runtime(backend):
    perturb = insertion_perturbation(
        backend, scenario.match.insertions, bindings=scenario.match.binding_ids,
        camera=scenario.match.camera,
    )
    return Runtime([Hook("observation", ObservationTransform(perturb))], case=scenario.match)

results = run_prepared(
    scenario, stack_factory, frames=30,
    runtimes={"clean": Runtime(), "attack": attacked_runtime},
)
clean, attacked = results["clean"], results["attack"]
```

The CARLA adapter checks that the depth, the camera image and the insertion geometry come from the
same simulator frame, and raises an error otherwise. Bound actors missing from the current frame
raise `KeyError`. The logical host names `lead` and `npc:<index>` remain valid across paired
resets. The adapter reads the camera calibration on every frame, so a reset that recreates the
sensors never reuses an old calibration.

For a scenario selected from the `NuRecDataset` instance named `dataset`, use this factory instead:

<!-- runnable -->
```python
from avsectester.perturbations.image import frame_perturbation

# dataset and scenario are supplied by the caller.
def attacked_runtime(backend):
    insert = dataset.insertion_renderer(
        scenario.match.scene, backend, scenario.match.insertions,
        bindings=scenario.match.binding_ids,
    )
    perturb = frame_perturbation(insert, camera=dataset.camera)
    return Runtime([Hook("observation", ObservationTransform(perturb))], case=scenario.match)
```

`NuRecDataset.insertion_renderer` can be constructed before reset and does not need a connected
rendering service. It starts from the recorded camera intrinsics and reads the loaded camera, actor
time and rendering pose when it processes each image. Geometry follows the bound actor IDs and
each treatment's physical ego pose.

`NuRecDataset.make_backend(scene)` uses actuator dynamics. Alpamayo outputs trajectory commands,
so replace the backend factory of the prepared scenario with one that uses `TrajectoryFollower`:

```python
from dataclasses import replace

from avsectester.simulators.nurec import TrajectoryFollower

# alpamayo_stack_factory is supplied by the caller.
def trajectory_backend():
    backend = dataset.make_backend(scenario.match.scene)
    backend.dynamics = TrajectoryFollower()
    return backend

scenario = replace(scenario, factory=trajectory_backend)
results = run_prepared(
    scenario, alpamayo_stack_factory, frames=30,
    runtimes={"clean": Runtime(), "attack": attacked_runtime},
)
```

The follower executes the planned trajectory ideally. It adds no reactive traffic and no actuator
limits. See [Commands and compatibility](../guides/driving-systems.md#commands-and-compatibility).

**Not yet implemented.** Trajectory-to-actuator dynamics for Alpamayo. See
[TODO](../development/TODO.md#vehicle-dynamics).

Both adapters place insertions from the effective camera rendering pose and scene time. The victim
pose is the physical ego pose. Spoofing the model-visible position, orientation or timestamp
therefore does not move the camera or a host-bound insertion. When a `render.pre` handler changes
the NuRec render request, the insertion follows that changed viewpoint and scene time. For other
cameras or simulators, see [Insertion rendering](../reference/python-api.md#insertion-rendering).
To capture component outputs or sweep corruption settings over a selected scenario, see
[Component outputs](../guides/evaluation.md#component-outputs) and
[Robustness across corruption settings](../guides/evaluation.md#robustness-across-corruption-settings).

## Harmonization

Harmonization adjusts the inserted object's appearance to the surrounding image. The insertion
pipeline is: resolve placement, sample surfaces, apply visibility, alpha composite, optionally
harmonize, then pass the image to the model input. It runs only for image insertions with an
enabled compositor, never for physical CARLA panels.

Pass a harmonizer to an adapter through `PatchCompositor`. Both supplied adapters and
`InsertionRenderer` accept `compositor=`:

<!-- runnable -->
```python
from avsectester.rendering.compositor import PatchCompositor
from avsectester.rendering.harmonizers import PCTNetHarmonizer

compositor = PatchCompositor(PCTNetHarmonizer(strict=True), soften=0.6, harmonize=False)
# Pass compositor=compositor to insertion_perturbation or dataset.insertion_renderer.

compositor.harmonize = True   # Enable for subsequent frames.
compositor.harmonize = False  # Disable without replacing the adapter or harmonizer.
```

- Without a compositor, the texture's colors are preserved.
- `PatchCompositor()` without a harmonizer selects `ClassicHarmonizer`, not a plain alpha paste.
- With `harmonize=False`, the harmonizer is not called and rendering does not load PCTNet weights.
  Alpha compositing and softening still apply.
- `soften` is a Gaussian blur sigma in pixels. It changes appearance inside the visible mask only.
  It never paints over foreground occluders or changes the visibility fraction.
- `PCTNetHarmonizer(strict=True)` propagates loading and inference errors. The default
  `strict=False` falls back to `ClassicHarmonizer` on errors.

The demo scripts select the method with `--harmonizer`:

| Value | Compositor |
|---|---|
| `none` | Harmonization disabled |
| `classic` | `ClassicHarmonizer()`: color transfer and Poisson blending |
| `chroma` | `ClassicHarmonizer(preserve_chroma=True, blend="feather")`: lightness only, hue preserved |
| `libcom` | `PCTNetHarmonizer(strict=True)`: learned color transformation |

Choose color transfer carefully when the payload's hue carries meaning. Classic color transfer can
turn a red STOP sign toward the road's color. Harmonization changes appearance, not geometry or
visibility. It uses local background color as a lighting proxy: a dark vehicle body can dim the
object too much, and self-emitting traffic lights or an unlit poster at night are not modeled.
Cast shadows, reflections and retro-reflection are not simulated. The complete parameter list is
in [Compositing and harmonization](../reference/python-api.md#compositing-and-harmonization).

## Demo scripts and outputs

The CARLA scripts need a client that matches the server version and a dedicated world. The NuRec
scripts need a running `nre-ga` renderer and the matching USDZ. See
[Installation](../getting-started/installation.md).

| Script | What it shows | Models | Default output |
|---|---|---|---|
| [`carla_insertion_demo.py`](../../scripts/demos/carla/carla_insertion_demo.py) | A rear patch in the three orientation modes and a world-fixed sign, with prescribed CARLA motion and depth visibility | None | `tmp/carla-insertions/` |
| [`nurec_insertion_demo.py`](../../scripts/demos/nurec/nurec_insertion_demo.py) | Three host-bound patches and a world-fixed sign at recorded poses, with cuboid visibility | None | `tmp/nurec-insertion/` |
| [`nurec_patch_demo.py`](../../scripts/demos/nurec/nurec_patch_demo.py) | The same host-bound patches without the sign. `--texture` replaces the diagnostic textures | None | `tmp/nurec-insertion/` |
| [`nurec_object_demo.py`](../../scripts/demos/nurec/nurec_object_demo.py) | Clean and inserted-object NuRec drives at constant speed | PCTNet for `libcom`, which is in the default list. COCO Faster R-CNN with `--eval` | `tmp/nurec_<object>/` |
| [`nuscenes_object_demo.py`](../../scripts/demos/nuscenes/nuscenes_object_demo.py) | STOP sign, standee and billboard in recorded nuScenes photographs, without a driving loop | PCTNet by default. COCO Faster R-CNN with `--eval` | `tmp/nuscenes_objects/` |
| [`alpamayo_attack_demo.py`](../../scripts/demos/nurec/alpamayo_attack_demo.py) | Clean and attacked Alpamayo drives with a roadside insertion | Alpamayo. PCTNet by default | `tmp/alpamayo_<object>/` |
| [`patch_driving_demo.py`](../../scripts/demos/carla/patch_driving_demo.py) | Clean and patched CARLA drives with a patch composited onto the lead vehicle's rear | CARLA-trained Faster R-CNN feeding a box-area braking rule. PCTNet for `libcom` | `tmp/patch_driving/` |

The attachment demos (`carla_insertion_demo.py`, `nurec_insertion_demo.py` and
`nurec_patch_demo.py`) are for inspecting attachment and visibility. They save model-input frames,
annotated comparisons, a GIF and per-frame placement and visibility in `measurements.json`. They run
no driving policy and keep diagnostic colors without harmonization. NuRec `--host` is a track ID in
the USDZ and is never reselected. `--world-position X Y Z` sets the sign's world position.

```bash
python -m scripts.demos.carla.carla_insertion_demo --port 2300 --frames 30 --output tmp/carla-insertions

python -m scripts.demos.nurec.nurec_insertion_demo --usdz /path/to/scene.usdz \
    --endpoint 127.0.0.1:50051 --host 15 --frames 30 --out tmp/nurec-insertion
```

`nurec_object_demo.py` accepts `--object stop`, `standee`, `billboard` or `trafficlights`. The
pedestrian variants require `--asset`. `--mode roadside` places the object at a fixed world
position. `--mode vehicle` requires `--host ID`, a vehicle track present at the initial frame, and
installs the face on that vehicle's `rear_center` with local yaw 180° and a small outward offset.
Its width is a fraction of the host's width and its height keeps the payload's aspect ratio.
A missing host raises an error instead of switching to another vehicle. For other dimensions or
orientations, build an `Insertion` directly. Roadside placement uses these options:

| Option | Meaning |
|---|---|
| `--x`, `--y` | Absolute coordinates in the NuRec scene frame, in metres |
| `--yaw` | Surface orientation in radians |
| `--size` | Sign or board width, or standee height, in metres |
| `--mount` | Height of the bottom edge above the ground, in metres |
| `--ground-z` | Ground height in scene coordinates, supplied by the caller |

`--usdz` supplies the scene metadata. The optional `--scene` must identify the same scene. Both
drives start at the first recorded camera timestamp. The ego starts at the planar origin and
advances at `dt=0.1 s`. The metadata clock drives the recorded actor poses, while the camera follows
the simulated ego pose. `--harmonizer` takes several values and defaults to
`classic chroma libcom`. Each `<mode>_<harmonizer>/` directory contains `side_by_side.gif`,
`filmstrip.png`, `zoom_XXXX.png` crops and attacked frames under `seq/`. `--eval` writes
`perception_eval.json` and `perception_<mode>.png` at the output root. It finds the object from
clean and attacked pixel differences, not from the visibility estimate. Placement does not depend
on this detector. Its scores are not measurements of the driving system under test, and selection
does not use them.

```bash
python -m scripts.demos.nurec.nurec_object_demo --usdz /path/to/scene.usdz \
    --endpoint 127.0.0.1:50051 --object stop --mode roadside vehicle --host 15 \
    --harmonizer none classic chroma libcom --frames 50 --eval

python -m scripts.demos.nuscenes.nuscenes_object_demo --nuscenes /path/to/nuscenes \
    --asset /path/to/person.png --n 8 --harmonizer libcom --eval
```

`nuscenes_object_demo.py` accepts a candidate position only if the bounding box of its image
footprint lies inside the image and clear of every annotated 2D box. It does not use scene depth.

`alpamayo_attack_demo.py` runs clean and attacked drives from the same metadata start time, with
a roadside insertion and `TrajectoryFollower` dynamics. It accepts the roadside options above,
`--harmonizer`, `--soften` and `--checkpoint`. `--gpu` selects Alpamayo's device and `--harm-gpu`
selects PCTNet's device:

```bash
python -m scripts.demos.nurec.alpamayo_attack_demo --usdz /path/to/scene.usdz \
    --endpoint 127.0.0.1:50051 --object stop --harmonizer libcom --frames 30 --gpu 1 --harm-gpu 0
```

It writes `speed.png`, `side_by_side.gif`, the frames of each drive and `trace.json`. `trace.json`
contains `object`, `at` (the insertion position), `harmonizer`, `verdict` (the
[driving-impact](../guides/evaluation.md#driving-impact-metric) result) and the step lists `clean`
and `attacked`. Each step contains:

- `frame`: zero-based step index.
- `input_t`: timestamp of the observation used to choose the command.
- `reasoning`: the model's explanation, when available.
- `t`, `speed`: physical state after executing the command.
- `throttle`, `brake`, `steer`, `n_detections`: the remaining `FrameRecord` fields.

A step is an input-to-outcome transition, so `t` and `input_t` differ. Trajectory commands leave
the actuator fields and the detection count at their defaults.

`patch_driving_demo.py` composites the patch with `insertion_perturbation` and runs a camera-based
braking rule on the result. It writes `impact.png`, `clean_filmstrip.png`, `patched_filmstrip.png`,
`clean.gif` and `patched.gif`. `--harmonizer` accepts `none`, `classic` and `libcom`.
[`patch_hide_probe.py`](../../scripts/validation/patch_hide_probe.py) compares detector confidence
across patch sizes and harmonizers on one CARLA frame:

```bash
python -m scripts.demos.carla.patch_driving_demo --frames 40 \
    --texture tmp/patch_optim/phys_texture.png --harmonizer libcom
```

The model roles are listed in
[Models used in this repository](../guides/driving-systems.md#models-used-in-this-repository).

## Rendering scope

The shared insertion renderer supports opaque planar and multi-surface assets. A texel belongs to
the silhouette when its alpha is greater than 127. Surfaces are two-sided. Translucent light
transport and native mesh rendering require additional renderer support.

CARLA insertion visibility uses calibrated scene depth. NuRec visibility is estimated from labeled
actor cuboids, which can miss unannotated occluders and cannot establish exact mesh visibility.
The low-level primitives `warp_patch` and `render_plane` acquire no scene visibility. Use
`InsertionRenderer` with visibility evidence when foreground occlusion matters. See
[Visibility](../guides/scenarios.md#visibility) and the estimators in
[Visibility](../reference/python-api.md#visibility).

Explicit coordinates are not checked against a drivable-area or sidewalk map. The framework does
not infer a physically valid installation surface or move an insertion.
