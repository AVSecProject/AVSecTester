# Stop-sign patch attack

> **Audience:** Users studying a physical patch on reconstructed stop signs.
>
> **Outcome:** Select stop approaches, prepare fixed or optimized patches, run closed-loop
> treatments and evaluate the saved physical trajectories.
>
> **Prerequisites:** [Attacks](../guides/attacks.md), the NuRec renderer and a local Alpamayo model.

## Configuration

Run either version from the repository root after setting its local data, model and device paths:

```bash
avsectester run configs/nurec_stop_sign_patch_2601.yaml
avsectester run configs/nurec_stop_sign_patch_2604.yaml
```

The configurations contain 21 scenes from 26.01 and 17 scenes from 26.04. The duplicated
07981e6a scene appears only in 26.01. Paths are relative to the configuration file. The supplied
paths expect archives at data/nurec/SCENE/SCENE.usdz, Alpamayo at models/Alpamayo-1.5-10B and
an existing Faster R-CNN COCO_V1 checkpoint at
models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth. The attack uses local detector weights.

StopSignPatchAttack lives in avsectester.implemented_attacks.stop_sign_patch_attack.attack.
Each configuration contains ten attack choices: the unpatched replica as a control, and white,
black and Gaussian-noise initial patches, each deployed directly, optimized with detector PGD, or
optimized with rollout NES. A minimal unoptimized attack specification is:

```yaml
type: StopSignPatchAttack
options:
  initial_patch: white
  offset_m: 0.10
  width_scale: 1.1
```

Detector PGD and seeded Alpamayo rollouts share one preparation process. Seeded Alpamayo enables
deterministic PyTorch algorithms, which need CUBLAS_WORKSPACE_CONFIG before the first GPU matrix
product; the framework sets it when configuration loading imports the adapter. See
[Driving systems](../guides/driving-systems.md#alpamayo).

When the NuRec image runs a source checkout through PYTHONPATH and has no installed
avsectester console executable, invoke the same public CLI as
python -m avsectester.cli run CONFIG (or evaluate CONFIG).


## Selection and placement

The metabuilder uses the semantic stop sign and its associated stop-sign stop line, the recorded
route, ego position, heading and speed. It selects at most one segment per scene. The approach
must come from the entering side, face the sign, pass within 3 m of the stop line, and contain a
recorded speed below 0.5 m/s within 15 m before it. The segment starts at the earliest frame
within approach_m (default 50 m) of the line that leaves the 15 real past observations required
by the driving adapter, and includes 20 hold decisions. If that start is closer to the line than
min_approach_m (default 20 m), the approach is rejected: a recording that begins near the line
gives the replica no distance in which to be seen. Scene versions are distinguished by
configurations.

The runtime inserts one stop_sign_replica object: the canonical MUTCD R1-1 face
(avsectester.rendering.STOP_SIGN_FACE) without a post, covering the reconstructed sign. The patch
replaces only the face pixels selected by a binary mask in face-texture coordinates, so it never
covers the legend. Masks are 512 by 512, like the face, and must lie inside its opaque octagon. The
default rp2_bars mask covers bars above and below STOP; face_interior is an alternative packaged
mask.

The default face_orientation="upright" projects the annotated positive X face normal onto the
horizontal plane and normalizes it. The replica's up axis is world positive Z, and its left axis
is cross(up, normal), giving a right-handed orientation. A nearly vertical annotated normal is
rejected because its horizontal direction is undefined. face_orientation="map" instead keeps
the annotated rotation, including its tilt.

The default face_size="mutcd_standard" selects the MUTCD R1-1 face width nearest the annotated
sign height from 0.762, 0.914 and 1.219 m (30, 36 and 48 inches). face_size="map_width" instead
uses the annotated width. The selected width is multiplied by width_scale, which defaults to
1.1; panel height follows the texture aspect ratio. A height mismatch warning is emitted only
with map_width, because mutcd_standard deliberately chooses a standard size.

The face center starts at the annotated center and moves offset_m along the selected
orientation's positive X normal; offset_m defaults to 0.10 m. The object's bottom-center origin
is then half the panel height below that face center along its positive Z axis. Online deployment
and both optimization paths share replica_placement. The attack keeps the annotated center and
yaw without scene-specific correction. Annotation position and yaw errors can leave the original
sign visible around the replica's edges at close range; this is a data limitation. Occlusion is
handled by the simulator's occlusion_tolerance_m rather than by adjusting attack placement.

All ten supplied attack options use ExposureHarmonizer(reference="under"), written once as a
YAML anchor. It compares the original reconstructed sign's luminance under the replica's visible
pixels with the unpatched replica's, because the replica panel's nominal_texture is the canonical
face. The
resulting gain describes scene lighting only: every patch, including an optimized one, receives
the gain of the unpatched control, and a patch cannot brighten or darken the rest of the sign. The
same harmonizer instance is passed to InsertObject for online deployment and white-box sampling.
Differentiable composition therefore includes the same exposure and feather calculation during
optimization and deployment. World insertion changes camera composition; it does not insert
returns into LiDAR.

## Data split and treatments

Both supplied configurations use the framework split:

```yaml
split:
  train_segments: 4
  seed: 0
  execute: [test]
```

All ten attacks select their metadata before the framework forms complete segment units. Identical
segments, including user_data, share a unit across attacks. Four units are assigned to train by the
seeded permutation; the remainder are test units. No validation units are requested. Only training
segments reach offline optimization. StopSignPatchAttack does not use validation_metadata.

Closed-loop execution runs only test units. Each unit and seed has one clean treatment and the ten
applicable attack treatments in one comparison group; all treatment pairs are evaluated. A clean
run is shared by attacks selecting identical metadata. Different user_data means a different unit,
even for the same window, so images can overlap between train and test. Sharing clean across those
distinct units and conditional transfer are not implemented. See
[Data split](../reference/configuration.md#data-split).

## Initial patches and preparation

initial_patch accepts none, white, black or gaussian_noise. none deploys the unpatched replica
and cannot be optimized; it separates the effect of the replica from the effect of the patch.
White and black set every mask pixel to 255 or 0. Gaussian noise samples independent RGB values
from clip(0.5 + noise_std * N(0, 1), 0, 1), then rounds to uint8. noise_std defaults to 0.25 and
noise_seed to 0. Every texture keeps the face's alpha and its colours outside the mask; deployed
artifacts that change anything else are rejected.

Experiment.prepare performs all selection, unit splitting and training-only optimization. With
optimization=None it returns the
initial texture. DetectorPatchOptimization and RolloutPatchOptimization are immutable options
objects in the attack's offline module. Preparation alone can be invoked through the public API:

```python
from avsectester.experiment import Experiment, load_configuration

configuration = load_configuration("configs/nurec_stop_sign_patch_2601.yaml")
prepared = Experiment.prepare(configuration)  # Also saved to runs.output.
```

DetectorPatchOptimization uses a stop-sign detector, PGD and HideObject. It renders the replica
using replica_placement, skips frames with too few visible replica pixels, and projects the map
sign face for the target box. The sample range starts at speed squared divided by twice the
braking deceleration and ends at the farthest frame where the detector finds the unpatched replica
above the configured threshold. Held-out scores compare the reconstructed sign without insertion
(clean), the unpatched replica, the initial patch and the optimized patch. Time-uniform samples alternate
between PGD training and held-out frames within the selected training segments. These held-out
frames are an internal detector diagnostic, separate from the framework validation split. The
supplied settings use all four framework training segments (max_training_segments=None), eight
samples per segment, 20 PGD updates, batch 1 and step size 4/255. Outputs include calibration and sample tables,
PGD scores and curve, held-out scores and detection rates, and initial and optimized textures.

RolloutPatchOptimization uses NES and single-decision candidate rollouts. It scores the selected
produced command by interpolating its forward X displacement at a 3 s horizon, then maximizes the
mean across configured starts. Every query saves its one-step RunLog; rollouts.csv preserves the
query identity, forward distance, completion status and errors. Each query constructs an independent
driving system and can reload model weights.

The rollout budget is:

```text
number of starts * (1 + steps * (population_size + 1))
```

RolloutPatchOptimization uses at most max_training_segments training segments, default 1. This
limits cost; it does not assign data to training or testing. DetectorPatchOptimization defaults to
max_training_segments=None, using all available training segments. Both options accept a positive
integer limit or None, and neither reads validation_metadata.

The supplied YAML uses max_training_segments=1, start_distances_m=[40, 30, 20], steps=1 and
population_size=4: three starts times six evaluations, or 18 rollouts per NES attack.
The three initial patches each have their own optimization output directory. Optimization strength
and driving impact must be measured; successful preparation alone establishes neither.

## Resume

```bash
avsectester run configs/nurec_stop_sign_patch_2601.yaml --resume
```

Resume validates the saved preparation inputs and reuses the saved units and parts. Completed
attacks load their saved artifacts, and only unfinished offline phases run again. Matching completed
closed-loop logs are read without rerunning them; missing, failed or incomplete runs are retried
individually. Without --resume, a nonempty output directory is rejected. --resume cannot be combined
with --prepared. Python provides Experiment.prepare(configuration, resume=True) followed by
Experiment.execute(prepared, resume=True).

Resume checkpoints whole attacks, not individual PGD updates or NES queries. An unfinished offline
phase starts from its beginning. If output_directory/rollouts is nonempty, the black-box optimizer
renames it to rollouts-incomplete-<UTC timestamp> in the same output directory, then starts a fresh
rollouts directory. The previous query logs are preserved rather than deleted or merged, and
optimization.json records the archive path in archived_rollouts (or null when no archive was
needed). Matching completed executions are retained only when their full expected metadata still
agrees, including segment, part, environment, decision counts and deployment budget. See [Resume interrupted work](../guides/experiments.md#resume-interrupted-work).

## One renderer for both scene versions

Use a single directory of absolute archive symlinks with one file per scene ID. Deduplicate
07981e6a by keeping its 26.01 archive. For example, for a local sample_set containing both releases:

```bash
DATA=/absolute/path/to/sample_set
mkdir -p results/nre-scenes results/nre-probes
python - "$DATA" "$PWD/results/nre-scenes" <<'PY'
from pathlib import Path
import sys
source, destination = map(Path, sys.argv[1:])
archives = {}
for archive in sorted(source.glob("*_release/*/*.usdz")):
    archives.setdefault(archive.stem, archive.resolve())
for scene, archive in archives.items():
    (destination / f"{scene}.usdz").symlink_to(archive)
PY
nvidia-smi
# Set NVIDIA_VISIBLE_DEVICES to an idle card before starting the renderer.
docker run -d --rm --name nre --network host \
  -e NVIDIA_VISIBLE_DEVICES=0 -e HOME=/tmp \
  -v "$DATA":"$DATA":ro \
  -v "$PWD/results/nre-scenes":/mnt/nre-data:ro \
  -v "$PWD/results/nre-probes":/mnt/nre-probes:ro \
  --entrypoint /app/internal/scripts/pycena/runtime/pycena_nrm_full \
  nvcr.io/nvidia/nre/nre-ga:26.04 \
  serve-grpc --host=0.0.0.0 --port=50051 \
  '--artifact-glob=/mnt/nre-data/*.usdz' --cache-size=2 --enable-editing-actors
```

Wait for the renderer to log Serving on before running experiments. Both client configurations
use results/nre-probes; mount that same host directory into the renderer at /mnt/nre-probes.
Rendered probes determine scene occlusion, and visible_pixels counts visible replica pixels.
The configurations use depth=none; ego ground contact still uses mesh in 26.01 and mesh_ground
in 26.04. They use map navigation, ego_only physics updates and AlpaSimController. Alpamayo
receives four consecutive 10 Hz image frames through frame_subsample=1.
Stop the renderer as soon as all requested runs are finished:

```bash
docker stop nre
docker ps -a --filter name=nre
```

## Evaluation and visualization

```bash
avsectester evaluate configs/nurec_stop_sign_patch_2601.yaml
python -m scripts.visualization.plot_stop_line_approaches configs/nurec_stop_sign_patch_2601.yaml
```

Evaluation writes evaluation/ under runs.output. StopLineMetrics reads stop_line_id from each
segment and looks up that line in the saved scenario map. It uses horizontal distance to the
stop-line polyline with each endpoint extended by 3 m, assumes the start is before the line,
and interpolates crossing speed between physical samples. ReasoningPhraseRate counts the
case-insensitive phrase stop sign in recorded policy reasoning. DrivingMetrics supplies geometric
collision and other driving measures. Missing evidence remains unavailable with a reason.

The plotting command reads saved ExperimentResult records and writes one speed-versus-signed-stop-line-
distance plot per scene into stop_line_plots/. Use --output DIR to choose another destination.
It performs no model inference and calculates no scalar evaluation metrics.

The supplied run command executes test-unit treatments and evaluates their saved logs, with
summaries separated by part. To perform only preparation, call Experiment.prepare without
Experiment.execute. Runs use runs/segment-xxxxx/seed-s/treatment/; old v5/v6 output layouts cannot
be loaded or reevaluated with this version.

## Next steps

Read [Evaluation](../guides/evaluation.md) and the
[Python API](../reference/python-api.md#avsectesterimplemented_attacksstop_sign_patch_attackattack).
