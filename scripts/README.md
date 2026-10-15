# Scripts

Run scripts from the repository root in the environment described in
[Installation](../docs/getting-started/installation.md).
Python entry points use `python -m scripts.<group>.<module>` so shared imports and repository-relative
configuration paths resolve consistently. Run these tools from the checkout. Shell tools are
invoked by their paths.

Use a matching CARLA client/server for CARLA scripts, and a running `nre-ga` service for rendered
NuRec examples. Models are loaded only by the entry points that use them. See the
[model overview](../docs/guides/driving-systems.md#models-used-in-this-repository) and
[model weights](../docs/getting-started/installation.md#model-weights-and-assets). The
[Examples](../docs/examples/README.md) page explains how the supplied attacks use the framework.

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
| [alpamayo_attack_demo.py](demos/nurec/alpamayo_attack_demo.py) | Compare clean and object-inserted driving, saving traces, reasoning and imagery | Alpamayo and PCTNet by default. No COCO detector |
| [carla_insertion_demo.py](demos/carla/carla_insertion_demo.py) | Inspect three attachment orientations, a world-fixed sign and depth visibility | No learned driving policy or detector, prescribed actor motion |
| [nurec_insertion_demo.py](demos/nurec/nurec_insertion_demo.py) | Inspect recorded host attachments, a world-fixed sign and cuboid visibility estimates | No learned driving policy or detector, recorded poses and NuRec rendering |
| [nurec_patch_demo.py](demos/nurec/nurec_patch_demo.py) | Patch-only version of the NuRec insertion example | Same components as `nurec_insertion_demo.py` |
| [carla_patch_demo.py](demos/carla/carla_patch_demo.py) | Render a physical textured panel attached to the lead vehicle | Fixed-throttle `CruiseStack`. CARLA-trained Faster R-CNN for the overlay unless `--no-detect` |
| [patch_driving_demo.py](demos/carla/patch_driving_demo.py) | Compare camera-based braking with and without a composited patch | CARLA-trained Faster R-CNN feeding a box-area braking rule, optional PCTNet |

For host binding, scenario selection and paired execution through the Python API, see
[Prepare scenarios](../docs/guides/scenarios.md). Geometry demos generate images, GIFs and
measurements for inspection. They do not measure the response of an autonomous-driving policy.

```bash
python -m scripts.demos.carla.carla_insertion_demo --port 2300 --frames 30 --output tmp/carla-insertions
python -m scripts.demos.nurec.nurec_insertion_demo --usdz /path/to/scene.usdz \
    --endpoint 127.0.0.1:50051 --host 15 --frames 30 --out tmp/nurec-insertion
```

Run Alpamayo examples in the AlpaSim driver environment, with this repository importable:

```bash
python -m scripts.demos.nurec.alpamayo_nurec_demo 8 --endpoint 127.0.0.1:50051 --gpu 1 --save-frames \
    --checkpoint /path/to/Alpamayo-1.5-10B
```

Both Alpamayo entry points accept `--checkpoint` or the `AVSECTESTER_ALPAMAYO_CHECKPOINT`
environment variable. `--stub` replaces NuRec rendering with black frames. It still loads Alpamayo and needs its model
weights. Checkpoint resolution is documented in
[Alpamayo checkpoint](../docs/getting-started/installation.md#alpamayo-checkpoint).

## Payload composition and detector evaluation

| Script | Function | Models |
|---|---|---|
| [nurec_object_demo.py](demos/nurec/nurec_object_demo.py) | Insert a STOP sign, standee, billboard or traffic-signal board in NuRec frames | COCO Faster R-CNN only with `--eval`. Compares classic, chroma and PCTNet harmonization by default |
| [nuscenes_object_demo.py](demos/nuscenes/nuscenes_object_demo.py) | Insert objects into recorded nuScenes `CAM_FRONT` photographs | COCO Faster R-CNN only with `--eval`, PCTNet by default |
| [patch_hide_probe.py](validation/patch_hide_probe.py) | Inspect detector response to different patch sizes and harmonization choices on a CARLA image | CARLA-trained Faster R-CNN and PCTNet |

`nurec_object_demo.py` requires the matching `--usdz`. Roadside mode uses fixed world insertions.
Vehicle mode requires a stable `--host` track ID and updates the rear attachment from the host's
current 3D pose. Both use known-cuboid visibility estimates and perturb the stack's camera input.
The sequence uses constant-speed `CruiseStack`, while `alpamayo_attack_demo.py` uses Alpamayo
and trajectory-following dynamics. Both clean and attacked runs share the metadata start timestamp. Driving examples use prepared
scenarios and the shared execution loop. Geometry inspection utilities can operate directly
on renderers without a driving experiment.
`nuscenes_object_demo.py` processes recorded images and checks candidate positions against
annotated 2D boxes. It has no driving loop or exact depth visibility.

```bash
python -m scripts.demos.nurec.nurec_object_demo --usdz /path/to/scene.usdz \
    --endpoint 127.0.0.1:50051 --mode roadside vehicle --host 15 \
    --harmonizer none --frames 30

python -m scripts.demos.nurec.alpamayo_attack_demo --usdz /path/to/scene.usdz \
    --endpoint 127.0.0.1:50051 --object stop --frames 30 --gpu 1 --harm-gpu 0
```

Harmonizer choices, output files and placement options for these scripts are described in
[Demo scripts and outputs](../docs/examples/image-insertion.md#demo-scripts-and-outputs) and
[Harmonization](../docs/examples/image-insertion.md#harmonization). COCO names the auxiliary detector's
pretraining dataset. It does not provide scenes or drive the vehicle.

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

Start the CARLA server as described in [Installation](../docs/getting-started/installation.md#docker)
or [Start a CARLA server](../docs/getting-started/installation.md#start-a-carla-server). Options,
printed output and exit codes are listed in the [CLI reference](../docs/reference/cli.md). NuRec
with Alpamayo has no CLI path. Use the NuRec demo scripts in the
[driver environment](../docs/getting-started/installation.md#nurec-and-alpamayo-environment).
