# Image attacks

Image attacks choose a payload and a placement. The simulation layer projects and composites the
payload into camera observations before they reach the driving stack. Shared camera geometry,
surface sampling, visibility and harmonization live in `avsectester/rendering/`.
`avsectester/simulators/patch_insertion.py` supplies the compositor and observation/view wrappers.

Object-insertion attacks can use an ordinary sign, pedestrian image or signal board. The attack
comes from placing that visual cue where it does not belong. An optimized adversarial texture is
not required.

## Payloads

| Payload | Module |
|---|---|
| Patch textures and CARLA physical panels | `attacks/patch/physical_patch.py` |
| STOP sign face and post | `attacks/object_insertion/sign_spoof.py` |
| Pedestrian cutout or printed poster | `attacks/object_insertion/person_poster.py` |
| Printed traffic signal board | `attacks/object_insertion/traffic_light.py` |

The STOP face is included in `avsectester/assets/signs/`. Its source and license are recorded in
[SOURCES.md](../avsectester/assets/signs/SOURCES.md). Person payloads take a caller-supplied RGBA
asset. `scripts/preparation/extract_person_cutouts.py` can extract them from annotated nuScenes images using SAM.
These cutouts are derived dataset assets and are not bundled with the repository. The extraction
script records their nuScenes source and license in `index.json`.

CARLA physical panels are a separate world-level insertion path. The image-space renderers below
modify camera observations without spawning a physical actor.

## Placement and model input

Use `Insertion` for stable host-bound or fixed-world placement. Each object has an independent
asset, local offset and orientation. The three orientation modes are `follow_host`, `fixed_world`
and `face_victim`. See [SCENARIOS.md](SCENARIOS.md) for the coordinate convention, configuration
examples, initial-scene selection and visibility providers.

`InsertionRenderer` takes resolved actor poses and camera calibration, projects the surfaces,
applies visibility masks, and optionally harmonizes their appearance. Connect its output at the `observation` runtime stage, or through the
`perturb(Observation)` shorthand, so the driving model receives the inserted image. A visualization callback
alone changes only displayed/saved images and must not be used as the input attack.
See [INTERVENTIONS.md](INTERVENTIONS.md) for lifecycle, handler ordering and world/render stages.

Geometry callbacks return `InsertionGeometry(actors, victim, cam_from_world)`. Built-in adapters
use the effective rendering viewpoint and scene time, independently of model-visible localization
or clock changes. NuRec `render.pre` request changes are reflected in insertion geometry.
See [custom rendering adapters](SCENARIOS.md#custom-rendering-adapters) for coordinate conventions,
camera protocols and visibility evidence. The built-in estimators and compositor share texture
prefiltering and alpha samples, so visibility clips the same silhouette that is rendered.

The rendering primitives are:

- `render_plane` projects a world-space textured rectangle through pinhole or f-theta calibration.
- `render_resolved` composites a resolved asset with surface depth and optional visibility evidence.
- `warp_patch` is a low-level image-space primitive for caller-supplied quads. Runtime insertion demos use
  explicit 3D insertions instead of deriving attachment positions from detector boxes.

For stable attachment and visibility filtering, follow the complete
[selection and execution example](SCENARIOS.md#select-a-case). The payload-composition examples
below demonstrate additional objects and appearance options with the stated placement method.

## Build a payload

Payload constructors return local assets. Placement is specified separately:

```python
from avsectester.attacks.object_insertion.sign_spoof import SignAsset
from avsectester.insertion import Insertion, Orientation, WorldPlacement

sign = Insertion(
    "stop_sign",
    SignAsset(width=0.9, mount_height=1.5),
    WorldPlacement((20, -3, 0)),
    Orientation("fixed_world", (0, 0, 180)),
)
```

`SignAsset` includes the STOP face and a post by default. Its origin is the ground point beneath
the face, local +X is the face normal, and `mount_height` locates the face's bottom edge.
`standee(person, height=1.75)` returns a cut-out without posts.
`billboard(person, width=1.4, mount_height=0.6)` returns a printed board on two posts.
`roadside_rig(face, width=2.4, mount_height=2.2)` returns a traffic-light board on a post.
All implement the same `planes() -> Sequence[PlaneSurface]` contract. To install only a face on
a vehicle, use `PlaneAsset` with `AttachedPlacement`, as in
[SCENARIOS.md](SCENARIOS.md#specify-inserted-objects).

## Auxiliary models

Model roles across the framework are listed in
[INTERFACE.md](INTERFACE.md#1d-models-and-supporting-components). Image-attack scripts use them at
specific points:

- `nurec_object_demo.py` and `nuscenes_object_demo.py` load COCO-pretrained Faster R-CNN only
  with `--eval`, to score the inserted object's class. Placement does not depend on this detector.
- CARLA patch probes and optimization use the separate CARLA-trained MMDetection Faster R-CNN.
  In `patch_driving_demo.py`, its detections also drive the script's rule-based braking policy.
- `--harmonizer libcom` selects PCTNet in the object-composition demos. It modifies the inserted
  appearance, not the driving policy. `none`, `classic` and `chroma` do not load a learned
  harmonization model.
- `extract_person_cutouts.py` runs SAM during asset preparation. Insertion scripts load the saved
  RGBA image and do not rerun SAM. `--sam` accepts a model ID or a local model directory.

The attachment/visibility demos use neither a learned driving policy nor an auxiliary detector.
`alpamayo_attack_demo.py` uses Alpamayo for driving and the selected harmonizer for insertion,
without the COCO detector's evaluation stage. Dependencies and weight locations are described in
[SETUP.md](SETUP.md#5-model-dependencies-and-weights).

## Harmonization

Import harmonizers from `avsectester.rendering.harmonizers`.

| Harmonizer | Behavior |
|---|---|
| `InsertionRenderer(..., compositor=None)` | Preserve the texture's colors |
| `PatchCompositor(harmonize=False)` | Skip harmonization while keeping alpha compositing and any configured softening |
| `ClassicHarmonizer()` | Color transfer and Poisson blending |
| `ClassicHarmonizer(preserve_chroma=True, blend="feather")` | Adjust lightness while preserving hue |
| `PCTNetHarmonizer(strict=True)` | Learned color transformation, with loading/inference errors propagated |

Pass a harmonizer through `PatchCompositor(harmonizer)` to `InsertionRenderer`. Calling
`PatchCompositor()` without a harmonizer selects `ClassicHarmonizer`, not a plain alpha paste.

`harmonize` is a boolean switch, enabled by default for a supplied compositor. Set it at
construction or change it before the next frame:

```python
from avsectester.rendering.harmonizers import PCTNetHarmonizer
from avsectester.simulators.patch_insertion import InsertionRenderer, PatchCompositor

compositor = PatchCompositor(PCTNetHarmonizer(strict=True), harmonize=False)
renderer = InsertionRenderer(insertions, camera, geometry, compositor=compositor)

compositor.harmonize = True   # Enable for subsequent frames.
compositor.harmonize = False  # Disable without replacing the renderer or harmonizer.
```

When disabled, the harmonizer is not called and PCTNet weights are not loaded by rendering.
The insertion pipeline is `resolve placement → sample surfaces → apply visibility → alpha
composite → optional harmonization → model input`. Harmonization does not run automatically
for all attacks. It only runs for image insertions with an enabled compositor.

The object-composition demos and `patch_driving_demo.py` expose the same choice through
`--harmonizer none`. Select `classic`, `chroma` where supported, or `libcom` to enable the
corresponding method. The motion/visibility attachment demos preserve diagnostic colors.

The compositor accepts `soften` as a nonnegative Gaussian blur sigma in image pixels:
`PatchCompositor(harmonizer, soften=0.6)`. Softening changes appearance within the geometric
visible mask. It never paints over foreground occluders or changes the visibility fraction.

Choose color transfer carefully when the payload's hue carries meaning. Harmonization changes
appearance, not geometry or the visibility denominator. Estimated lighting does not provide cast
shadows, reflections, retro-reflection or physically simulated illumination.
Local background color is only a lighting proxy. A dark vehicle body can cause excessive dimming,
and this method does not model self-emitting traffic lights or an unlit poster at night.

The object demos use `PCTNetHarmonizer(strict=True)` for `--harmonizer libcom`, so loading or
inference errors stop the experiment. Direct callers using the default `strict=False` receive a
classic-harmonizer fallback on errors.

## Run an insertion demo

Start the required simulator or NuRec renderer as described in [SETUP.md](SETUP.md).
CARLA scripts require a client matching the server version and a dedicated world.

| Script | What it demonstrates | Motion and visibility |
|---|---|---|
| `carla_insertion_demo.py` | Host attachments with three orientation modes and a world-fixed sign | Prescribed CARLA motion, aligned depth visibility |
| `nurec_insertion_demo.py` | Stable recorded host attachments and a world-fixed sign | Recorded poses, labeled-cuboid visibility estimate |
| `nurec_patch_demo.py` | Patch-only variant of the NuRec insertion demo | Same host binding and visibility method |

These are the entry points for inspecting attachment and visibility. They modify observation
images but do not run a driving policy. For clean/attacked driving runs, use the selected-case
adapters in [SCENARIOS.md](SCENARIOS.md#feed-insertions-to-the-driving-model).

```bash
python -m scripts.demos.carla.carla_insertion_demo --port 2300 --frames 30 --output tmp/carla-insertions

python -m scripts.demos.nurec.nurec_insertion_demo --usdz /path/to/scene.usdz \
    --endpoint 127.0.0.1:50051 --host 15 --frames 30 --out tmp/nurec-insertion

python -m scripts.demos.nurec.nurec_patch_demo --usdz /path/to/scene.usdz \
    --endpoint 127.0.0.1:50051 --host 15 --texture /path/to/patch.png --frames 30
```

The demos save model-input frames, annotated comparisons,
a GIF animation, and per-frame placement/visibility measurements. NuRec `--host` is the track
ID in the supplied USDZ. `--world-position X Y Z` sets the sign's explicit world position.

## Payload composition examples

`nurec_object_demo.py` demonstrates STOP signs, pedestrian standees/posters and printed traffic
signals with selectable harmonizers. Both placements use the common insertion renderer and
known-cuboid visibility estimate. Roadside mode uses a fixed world position. Vehicle mode
installs the face on the rear of an explicitly selected recorded vehicle.
`nuscenes_object_demo.py` places payloads in recorded photographs, without a closed-loop drive.
Its candidate spots are checked against annotated 2D boxes, not exact scene depth.
These scripts inspect payload appearance and do not exercise case selection.

```bash
python -m scripts.preparation.extract_person_cutouts --nuscenes /path/to/nuscenes \
    --out /path/to/pedestrians

python -m scripts.demos.nurec.nurec_object_demo --usdz /path/to/scene.usdz --endpoint 127.0.0.1:50051 --object stop \
    --mode roadside vehicle --host 15 --harmonizer none classic chroma libcom --frames 50 --eval

python -m scripts.demos.nurec.nurec_object_demo --usdz /path/to/scene.usdz --endpoint 127.0.0.1:50051 --object billboard \
    --asset /path/to/person.png --mode roadside --eval

python -m scripts.demos.nuscenes.nuscenes_object_demo --nuscenes /path/to/nuscenes \
    --asset /path/to/person.png --n 8 --harmonizer libcom --eval
```

`nurec_object_demo.py` accepts `--object stop`, `standee`, `billboard` or `trafficlights`.
The pedestrian variants require `--asset`. Roadside placement uses these options:

| Option | Meaning |
|---|---|
| `--x`, `--y` | Absolute coordinates in the NuRec scene frame, in metres |
| `--yaw` | Surface orientation in radians |
| `--size` | Sign/board width or standee height in metres |
| `--mount` | Bottom edge height above the specified ground in metres |
| `--ground-z` | Ground height in scene coordinates, supplied by the caller |

`--usdz` supplies the matching scene metadata. Optional `--scene` must identify that same scene.
Both runs start at the first recorded camera timestamp. The constant-speed demo keeps its
planar ego origin at `(0, 0, 0)` and advances at `dt=0.1 s`. The metadata clock drives actor poses,
while the camera follows the actual simulated ego pose.

`--mode vehicle` requires `--host ID`, a vehicle track present at the initial frame. The attachment
uses `rear_center`, local yaw 180°, and a small outward offset. Width is a fraction of the host's
3D width, with height preserving the payload's aspect ratio. The same host and local mount are
used throughout the sequence. A missing host raises an error instead of switching to another car.
For explicit metric dimensions, offsets or other orientations, use `Insertion` directly or the
attachment demo above.

The output root defaults to `tmp/nurec_<object>` and can be changed with `--out`. Each
`<mode>_<harmonizer>/` directory contains `side_by_side.gif`, `filmstrip.png`, available
`zoom_XXXX.png` crops and attacked frames under `seq/`.

`--eval` runs a COCO Faster R-CNN detector and writes `perception_eval.json` and
`perception_<mode>.png` at the output root. This demo identifies evaluation regions from
clean/attacked pixel differences, not from the geometric visibility provider. Its detector scores
are separate from closed-loop driving impact and are not measurements from the stack under test.
The geometric selection API does not use detector success as a prerequisite.

## Driving evaluation output

`scripts/demos/nurec/alpamayo_attack_demo.py` runs clean and attacked driving experiments with the same
metadata start time and world-fixed insertion path. For example:

```bash
python -m scripts.demos.nurec.alpamayo_attack_demo --usdz /path/to/scene.usdz \
    --endpoint 127.0.0.1:50051 --object stop --harmonizer libcom --frames 30 --gpu 1 --harm-gpu 0
```

Its `trace.json` contains both step sequences and their impact verdict. Each step retains:

- `frame`: zero-based step index.
- `input_t`: timestamp of the observation used to choose the control.
- `reasoning`: the model's explanation, when available.
- `t`, `speed`: state after executing the control.
- `throttle`, `brake`, `steer`, `n_detections`: fields from `Trace.records`.

A row represents an input-to-outcome transition. `t` and `input_t` have different meanings.
Trajectory-based models can leave actuator fields and detection counts at their defaults.

## Rendering scope

The shared insertion renderer supports opaque planar and multi-surface assets. Surface silhouette
membership uses alpha greater than 127. Surfaces are two-sided. General translucent transport and
native mesh rendering require additional renderer support.

CARLA insertion visibility uses calibrated scene depth. NuRec uses a labeled-cuboid estimate,
which can miss unannotated occluders and cannot establish exact mesh visibility. Low-level
`warp_patch` and `render_plane` do not acquire scene visibility. Use `InsertionRenderer` with
evidence when foreground occlusion matters.

Explicit coordinates are not checked against a drivable-area or sidewalk map. The framework does
not infer a physically valid installation surface or automatically move the user's insertion.
