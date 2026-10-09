# Scripts

Run scripts from the repository root in the environment described in [SETUP.md](../docs/SETUP.md).
Python entry points use `python -m scripts.<group>.<module>` so shared imports and repository-relative
configuration paths resolve consistently. These tools are part of the checkout, not the installed
`avsectester` wheel. Shell tools are invoked by their paths.

Use a matching CARLA client/server for CARLA scripts, and a running `nre-ga` service for rendered
NuRec examples. Models are loaded only by the entry points that use them. See the
[model overview](../docs/INTERFACE.md#1d-models-and-supporting-components) and
[weight-loading guide](../docs/SETUP.md#5-model-dependencies-and-weights).

## Directory layout

| Directory | Purpose |
|---|---|
| `demos/carla/` | CARLA driving, physical panels and image insertions |
| `demos/nurec/` | NuRec insertions and Alpamayo driving |
| `demos/nuscenes/` | Insertions into recorded nuScenes images |
| `preparation/` | Model download and reusable asset extraction |
| `optimization/` | Existing patch optimization workflows |
| `validation/` | CARLA visibility and detector probes |
| `visualization/` | Scene-label and corruption figures |
| `common/` | Shared utilities imported by the entry points |

## Driving and insertion examples

| Entry point | Function | Driving policy and other learned models |
|---|---|---|
| `avsectester run configs/carla_scenario.yaml` | Paired clean/phantom-attacked CARLA drive | Modular stack with CARLA-trained PointPillars, tracking, collision planning and PID control |
| [alpamayo_nurec_demo.py](demos/nurec/alpamayo_nurec_demo.py) | Drive in a rendered NuRec scene | Alpamayo-1.5-10B |
| [alpamayo_attack_demo.py](demos/nurec/alpamayo_attack_demo.py) | Compare clean and object-inserted driving, saving traces, reasoning and imagery | Alpamayo, plus PCTNet when `--harmonizer libcom` is selected. No COCO detector |
| [carla_insertion_demo.py](demos/carla/carla_insertion_demo.py) | Inspect three attachment orientations, a world-fixed sign and depth visibility | No learned driving policy or detector, prescribed actor motion |
| [nurec_insertion_demo.py](demos/nurec/nurec_insertion_demo.py) | Inspect recorded host attachments, a world-fixed sign and cuboid visibility estimates | No learned driving policy or detector, recorded poses and NuRec rendering |
| [nurec_patch_demo.py](demos/nurec/nurec_patch_demo.py) | Patch-only version of the NuRec insertion example | Same components as `nurec_insertion_demo.py` |
| [carla_patch_demo.py](demos/carla/carla_patch_demo.py) | Render a physical textured panel attached to the lead vehicle | Fixed-throttle `CruiseStack`. CARLA-trained Faster R-CNN for the overlay unless `--no-detect` |
| [patch_driving_demo.py](demos/carla/patch_driving_demo.py) | Compare camera-based braking with and without a composited patch | CARLA-trained Faster R-CNN feeding a box-area braking rule, optional PCTNet |

For host binding, initial-case selection and paired execution through the Python API, use
[SCENARIOS.md](../docs/SCENARIOS.md). Geometry demos generate images/GIFs and measurements for
inspection. They do not measure the response of an autonomous-driving policy.

```bash
python -m scripts.demos.carla.carla_insertion_demo --port 2300 --frames 30 --output tmp/carla-insertions
python -m scripts.demos.nurec.nurec_insertion_demo --usdz /path/to/scene.usdz \
    --endpoint 127.0.0.1:50051 --host 15 --frames 30 --out tmp/nurec-insertion
```

Run Alpamayo examples in the AlpaSim driver environment, with this repository importable:

```bash
python -m scripts.demos.nurec.alpamayo_nurec_demo 8 --endpoint 127.0.0.1:50051 --gpu 1 --save-frames
```

`--stub` replaces NuRec rendering with black frames. It still loads Alpamayo and needs its model
weights. Current checkpoint-path behavior is documented in
[SETUP.md](../docs/SETUP.md#sam-assets-and-alpamayo-checkpoints).

## Payload composition and detector evaluation

| Script | Function | Models |
|---|---|---|
| [nurec_object_demo.py](demos/nurec/nurec_object_demo.py) | Insert a STOP sign, standee, billboard or traffic-signal board in NuRec frames | COCO Faster R-CNN only with `--eval`, optional PCTNet |
| [nuscenes_object_demo.py](demos/nuscenes/nuscenes_object_demo.py) | Insert objects into recorded nuScenes `CAM_FRONT` photographs | COCO Faster R-CNN only with `--eval`, optional PCTNet |
| [patch_hide_probe.py](validation/patch_hide_probe.py) | Inspect detector response to different patch sizes and harmonization choices on a CARLA image | CARLA-trained Faster R-CNN and PCTNet |

`nurec_object_demo.py` requires the matching `--usdz`. Roadside mode uses fixed world insertions.
Vehicle mode requires a stable `--host` track ID and updates the rear attachment from the host's
current 3D pose. Both use known-cuboid visibility estimates and perturb the stack's camera input.
The sequence uses constant-speed `CruiseStack`, while `alpamayo_attack_demo.py` uses Alpamayo
and trajectory-following dynamics. Both clean and attacked runs share the metadata start timestamp.
`nuscenes_object_demo.py` processes recorded images and checks candidate positions against
annotated 2D boxes. It has no driving loop or exact depth visibility.

```bash
python -m scripts.demos.nurec.nurec_object_demo --usdz /path/to/scene.usdz \
    --endpoint 127.0.0.1:50051 --mode roadside vehicle --host 15 \
    --harmonizer none --frames 30

python -m scripts.demos.nurec.alpamayo_attack_demo --usdz /path/to/scene.usdz \
    --endpoint 127.0.0.1:50051 --object stop --frames 30 --gpu 1 --harm-gpu 0
```

`patch_driving_demo.py` and `patch_hide_probe.py` use a depth-equipped CARLA backend for composited
rear attachments. Clean and attacked driving runs use the same prepared scene and pinhole RGB
camera settings. `carla_patch_demo.py` and `optimize_patch_physical.py` retain native world panels,
including physical lighting and their configured local installation poses.
For composited objects and `patch_driving_demo.py`, use `--harmonizer none` to disable appearance
harmonization. This preserves insertion geometry and visibility. `classic` and `libcom` select
classic and PCTNet harmonization respectively. Direct API callers use
`PatchCompositor(..., harmonize=False)` and may toggle `compositor.harmonize` between frames.

COCO names the auxiliary detector's pretraining dataset. It does not provide scenes or drive the
vehicle. See [IMAGE_ATTACKS.md](../docs/IMAGE_ATTACKS.md#payload-composition-examples) for options,
placement conventions and outputs.

## Model and asset preparation

| Script | Function | Models or data |
|---|---|---|
| [fetch_models.sh](preparation/fetch_models.sh) | Download perception configurations/checkpoints and create avstack model-path links | CARLA PointPillars, Faster R-CNN, Cascade R-CNN variants and KITTI PointPillars |
| [extract_person_cutouts.py](preparation/extract_person_cutouts.py) | Segment annotated pedestrians into reusable RGBA assets | SAM, default `facebook/sam-vit-huge`, on nuScenes images |
| [optimize_patch.py](optimization/optimize_patch.py) | Optimize a patch with a digital projection surrogate | PGD against a CARLA-trained Faster R-CNN objectness scorer |
| [optimize_patch_physical.py](optimization/optimize_patch_physical.py) | Optimize a patch and inspect its detector response after physical CARLA rendering | Same detector family and PGD, plus native CARLA patch deployment |

PGD is an optimization algorithm, not an additional learned model.
Person cutouts are generated outside the repository and retain their source-dataset provenance.

## Geometry and visualization utilities

| Script | Function | Learned models |
|---|---|---|
| [validate_carla_visibility.py](validation/validate_carla_visibility.py) | Compare target reference silhouettes, masks and depth in a CARLA road scene | None |
| [visualize_scene_labels.py](visualization/visualize_scene_labels.py) | Overlay dataset ground-truth geometry on recorded camera frames | None |
| [visualize_augmentations.py](visualization/visualize_augmentations.py) | Show corruption operators applied to a recorded frame | None |

The two `visualize_*` scripts currently contain local dataset-path constants. Set those to your
available dataset before running. `demo_common.py` supplies shared detector factories, insertion payload builders and
`CruiseStack` to the examples, and is not a standalone entry point.

## CARLA CLI output

```bash
./scripts/preparation/fetch_models.sh
avsectester run configs/carla_scenario.yaml --frames 40 --gpu 1 --plot results/impact.png
```

Start the server using [DOCKER.md](../docs/DOCKER.md) or [SETUP.md](../docs/SETUP.md), and select an
available inference device with `--gpu`. `--frames` controls steps per run. `--plot` saves a driving
impact figure and requires the `viz` extra. No particular attack result is guaranteed.

The CLI prints driving statistics and an impact verdict. Exit codes are `0` for attack success,
`1` for failure under the current criterion and `2` for an inconclusive driving baseline. See
[INTERFACE.md](../docs/INTERFACE.md#3-metric-clean-vs-attacked--verdict) for that criterion and
[tests/README.md](../tests/README.md) for offline and opt-in simulator tests.
