# Image attacks

Image attacks choose a payload and a placement. The simulation layer projects and composites the
payload into camera observations before they reach the driving stack. Shared rendering lives in
`avsectester/simulators/patch_insertion.py`.

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
asset. `scripts/extract_person_cutouts.py` can extract them from annotated nuScenes images using SAM.
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
applies visibility masks, and optionally harmonizes their appearance. Its output is connected to
`perturb(Observation)` so the driving model receives the inserted image. `composite_view` is for
visualization only and must not be used as the input attack.

The rendering primitives are:

- `render_plane` projects a world-space textured rectangle through pinhole or f-theta calibration.
- `render_resolved` composites a resolved asset with surface depth and optional visibility evidence.
- `detector_quad` and `warp_patch` implement image-space insertion from a detected box. This
  path does not bind a stable actor identity. `hold_quad` retains the last image location across short
  detection gaps, rather than tracking the actor in 3D.

For stable attachment and visibility filtering, follow the complete
[selection and execution example](SCENARIOS.md#select-a-case). The payload-composition examples
below demonstrate additional objects and appearance options with the stated placement method.

## Auxiliary models

Model roles across the framework are listed in
[INTERFACE.md](INTERFACE.md#1d-models-and-supporting-components). Image-attack scripts use them at
specific points:

- `nurec_object_demo.py --mode vehicle` uses a COCO-pretrained Faster R-CNN to locate vehicle boxes.
  Adding `--eval` uses the detector to score the inserted object's class. Roadside mode without
  `--eval` needs no detector. `nuscenes_object_demo.py` also loads this detector only with `--eval`.
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

| Harmonizer | Behavior |
|---|---|
| `InsertionRenderer(..., compositor=None)` | Preserve the texture's colors |
| `ClassicHarmonizer()` | Color transfer and Poisson blending |
| `ClassicHarmonizer(preserve_chroma=True, blend="feather")` | Adjust lightness while preserving hue |
| `PCTNetHarmonizer(strict=True)` | Learned color transformation, with loading/inference errors propagated |

Pass a harmonizer through `PatchCompositor(harmonizer)` to `InsertionRenderer`. Calling
`PatchCompositor()` without a harmonizer selects `ClassicHarmonizer`, not a plain alpha paste.

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
python scripts/carla_insertion_demo.py --port 2300 --frames 30 --output tmp/carla-insertions

python scripts/nurec_insertion_demo.py --usdz /path/to/scene.usdz \
    --endpoint 127.0.0.1:50051 --host 15 --frames 30 --out tmp/nurec-insertion

python scripts/nurec_patch_demo.py --usdz /path/to/scene.usdz \
    --endpoint 127.0.0.1:50051 --host 15 --texture /path/to/patch.png --frames 30
```

These demos inspect geometry and visibility using prescribed motion or recorded trajectories.
They do not run a driving-policy evaluation. They save model-input frames, annotated comparisons,
a GIF animation, and per-frame placement/visibility measurements. NuRec `--host` is the track
ID in the supplied USDZ. `--world-position X Y Z` sets the sign's explicit world position.

## Payload composition examples

`nurec_object_demo.py` demonstrates STOP signs, pedestrian standees/posters and printed traffic
signals with selectable harmonizers. Its roadside mode uses fixed world planes. Its vehicle
mode uses detector-derived image quads, so it cannot guarantee a stable host or depth occlusion.
`nuscenes_object_demo.py` places payloads in recorded photographs, without a closed-loop drive.
Use these scripts to inspect payload appearance. They do not exercise case selection.

```bash
python scripts/extract_person_cutouts.py --nuscenes /path/to/nuscenes \
    --out /path/to/pedestrians

python scripts/nurec_object_demo.py --endpoint 127.0.0.1:50051 --object stop \
    --mode roadside vehicle --harmonizer none classic chroma libcom --frames 50 --eval

python scripts/nurec_object_demo.py --endpoint 127.0.0.1:50051 --object billboard \
    --asset /path/to/person.png --mode roadside --eval

python scripts/nuscenes_object_demo.py --nuscenes /path/to/nuscenes \
    --asset /path/to/person.png --n 8 --harmonizer libcom --eval
```

`nurec_object_demo.py` accepts `--object stop`, `standee`, `billboard` or `trafficlights`.
The pedestrian variants require `--asset`. Roadside placement uses these options:

| Option | Meaning |
|---|---|
| `--x`, `--y` | Metres relative to the starting pose, X forward and Y left |
| `--yaw` | Surface orientation in radians |
| `--size` | Sign/board width or standee height in metres |
| `--mount` | Bottom edge height above the specified ground in metres |
| `--ground-z` | Ground height in scene coordinates, supplied by the caller |

`--mode vehicle` uses a detected image-space quad. `pick="lane"` prefers boxes spanning the
image centre column, and `hold_quad` retains coordinates during short detection gaps. These
helpers do not track vehicle identity. Use host-bound `Insertion` for stable attachment.

The output root defaults to `tmp/nurec_<object>` and can be changed with `--out`. Each
`<mode>_<harmonizer>/` directory contains `side_by_side.gif`, `filmstrip.png`, available
`zoom_XXXX.png` crops and attacked frames under `seq/`.

`--eval` runs a COCO Faster R-CNN detector and writes `perception_eval.json` and
`perception_<mode>.png` at the output root. This demo identifies evaluation regions from
clean/attacked pixel differences, not from the geometric visibility provider. Its detector scores
are separate from closed-loop driving impact and are not measurements from the stack under test.
The geometric selection API does not use detector success as a prerequisite.

## Driving evaluation output

`scripts/alpamayo_attack_demo.py` runs clean and attacked driving experiments. Its `trace.json`
contains both step sequences and their impact verdict. Each step retains:

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
which can miss unannotated occluders and cannot establish exact mesh visibility. Image-space
warping and `apply_planes` do not automatically acquire scene visibility. Use
`InsertionRenderer` with evidence when foreground occlusion matters.

Explicit coordinates are not checked against a drivable-area or sidewalk map. The framework does
not infer a physically valid installation surface or automatically move the user's insertion.
