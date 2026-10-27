# Stop-sign patch attack

> **Audience:** Users studying a physical patch on reconstructed stop signs.
>
> **Outcome:** Select stop approaches, prepare fixed or optimized patches, run closed-loop
> treatments and evaluate the saved physical trajectories.
>
> **Prerequisites:** [Attacks](../guides/attacks.md), the NuRec renderer and local driving-model weights.

## Configuration

Use the modular configurations for detector PGD (white-box) and the Alpamayo configurations for
rollout NES (black-box). Run from the repository root after setting data, model and device paths:

```bash
avsectester run configs/nurec_stop_sign_patch_modular_2601.yaml
avsectester run configs/nurec_stop_sign_patch_alpamayo_2601.yaml
```

Each driving-system variant has a 2601 and 2604 configuration. NuRecDataset enumerates archives
under data/nurec/26.01_release or data/nurec/26.04_release. Paths are relative to the configuration
file. Alpamayo uses models/Alpamayo-1.5-10B; the modular stack uses an existing Faster R-CNN
COCO_V1 checkpoint at models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth.

StopSignPatchAttack lives in avsectester.implemented_attacks.stop_sign_patch_attack.attack.
Each configuration contains seven attacks: the unpatched replica control, three directly deployed
patches (white, black and Gaussian noise), and three optimized versions. Modular configurations
optimize with PGD; Alpamayo configurations optimize with NES. Treatments explicitly list clean
and the seven attacks in that order. A minimal unoptimized attack specification is:

```yaml
type: StopSignPatchAttack
options:
  initial_patch: white
  offset_m: 0.10
  width_scale: 1.1
```

Both optimization paths run inside Experiment.prepare. Seeded Alpamayo enables
deterministic PyTorch algorithms, which need CUBLAS_WORKSPACE_CONFIG before the first GPU matrix
product; the framework sets it when configuration loading imports the adapter. See
[Driving systems](../guides/driving-systems.md#alpamayo).

When the NuRec image runs a source checkout through PYTHONPATH and has no installed
avsectester console executable, invoke the same public CLI as
python -m avsectester.cli run CONFIG (or evaluate CONFIG).


## Optional reactive traffic

Reactive traffic is off by default; add `simulator.options.traffic: {type: avsectester.closed_loop.simulators.nurec:AlpaSimTraffic}` to enable it.
Start `avsectester trafficsim serve CONFIG --gpu N --models DIR` alongside NRE for execution and offline rollouts, then stop both services.
Traffic requires 0.1 s steps and 1.5 s history; `metadata.load` shares selected windows between modes.
See [Reactive traffic](../guides/simulators.md#reactive-traffic) for filtering and actor overrides.

## Run with the NuRec rule-based driving system

Use configs/nurec_stop_sign_patch_modular_2601.yaml or
configs/nurec_stop_sign_patch_modular_2604.yaml to replace Alpamayo with RuleBasedAVStack.
Their dataset, simulator and split settings match the Alpamayo examples, while driving
uses a COCO stop-sign detector, calibrated monocular range, bearing EKF sign tracking and a
distance-based stop/hold/resume planner with IDM free-road cruising.
The binding exposes the driving detector itself as the differentiable scorer; the attack does not
construct a separate detector. The stack composes a
CocoStopSignDetector with a default StopSignTracker and StopSignPlanner; device is injected into
the nested detector from the top-level setting. These configurations omit ReasoningPhraseRate because this policy produces no
reasoning text.

```bash
avsectester nre serve configs/nurec_stop_sign_patch_modular_2601.yaml --gpu 0
avsectester run configs/nurec_stop_sign_patch_modular_2601.yaml
avsectester nre stop configs/nurec_stop_sign_patch_modular_2601.yaml
```

To compare driving systems on exactly the same units, first export metadata.save with one
configuration, then set metadata.load in the other. Loading revalidates the frames, required
user_data and deployment budget for the new system, and split still applies. This reuses selection
without reusing optimization artifacts. See [Reuse the same segments](#reuse-the-same-segments) and
[NuRec rule-based driving system](../guides/driving-systems.md#nurec-rule-based-driving-system).
The planner stops stop_before_sign_m before the detected sign along the route. The geometric
basis of its 5.5 m default and its lateral sign filter are described in the driving-system guide.
Ranging assumes a known sign height, so errors in that assumption scale the estimated distance.
The policy handles stop signs only; it does not avoid other vehicles or obstacles.

## Selection and placement

The metabuilder uses the semantic stop sign and its associated stop-sign stop line, the recorded
route, ego position, heading and speed. It selects at most one segment per scene. The approach
must come from the entering side, face the sign, pass within 3 m of the stop line, and contain a
recorded speed below 0.5 m/s within 15 m before it. The segment starts at the earliest frame
within approach_m (default 50 m) of the line that leaves the recorded history required by the
driving adapter (15 observations for Alpamayo; none for RuleBasedAVStack), and includes 20 hold decisions. If that start is closer to the line than
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

All seven supplied attack options use ExposureHarmonizer(reference="under"), written once as a
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

Both supplied configurations list every archive in the version's release directory through
NuRecDataset dataroot, place the split directly after the dataset, and use the framework split:

```yaml
split:
  train_segments: 5
  seed: 0
  execute: [test]
```

All seven attacks select their metadata before the framework forms complete segment units. Identical
segments, including user_data, share a unit across attacks. Five units are assigned to train by the
seeded permutation; the remainder are test units. No validation units are requested. Only training
segments reach offline optimization. StopSignPatchAttack does not use validation_metadata.

Closed-loop execution runs only test units. Each unit and seed has one clean treatment and the seven
applicable attack treatments in one comparison group; all treatment pairs are evaluated. A clean
run is shared by attacks selecting identical metadata. Different user_data means a different unit,
even for the same window, so images can overlap between train and test. Sharing clean across those
distinct units and conditional transfer are not implemented. See
[Data split](../reference/configuration.md#data-split).

## Reuse the same segments

Export a selection independently of optimization artifacts by adding:

```yaml
metadata:
  save: ../results/metadata/stop_sign_2601.json
```

For a later experiment, use metadata.load with that path instead of save. All configured attacks
then receive every saved segment, skip selection and share the same clean comparison groups.
StopSignPatchAttack validates the required sign, stop-line and route user_data before preparation.
The dataset type, frame windows, shared deployment budget and simulator start must also match.
split still determines training and test units; optimization runs only on training segments.

An existing metadata export is accepted only if its content is identical. Preparation resumes
from preparation-progress.json and --prepared reuse do not run selection or write metadata.save.
Compatibility checks use the file's raw-byte SHA256, so a byte-identical file may move to another
path. Evaluation reads saved segments and the saved hash, without requiring the metadata file.
For a custom subset, use load_metadata and save_metadata from avsectester.experiment in a user
script; pass the original dataset_type when saving. See
[Metadata](../reference/configuration.md#metadata) for the format and compatibility rules.

## Initial patches and preparation

initial_patch accepts none, white, black or gaussian_noise. none deploys the unpatched replica
and cannot be optimized; it separates the effect of the replica from the effect of the patch. The
supplied configurations name this control attack and its treatment no_patch.
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

configuration = load_configuration("configs/nurec_stop_sign_patch_alpamayo_2601.yaml")
prepared = Experiment.prepare(configuration)  # Also saved to runs.output.
```

DetectorPatchOptimization uses the victim detector through avstack.bindings, PGD and HideObject.
rule_based_driving_adapter declares rule_based.stop_sign_confidence, its sensor_id and the driving
detector's decision_threshold. A SessionScorer uses that binding for calibration, PGD updates and
held-out scoring. scorer_id=None selects the unique image-differentiable scorer; set an explicit
scorer_id if the adapter offers more than one. The optimization device inherits the top-level
device; camera and decision threshold come from the scorer description. It renders the replica
using replica_placement, skips frames with too few visible replica pixels, and projects the map
sign face for the target box. The sample range starts at speed squared divided by twice the
braking deceleration and ends at the farthest frame where the detector finds the unpatched replica
at or above the driving detector's decision threshold. braking_deceleration_mps2 (default 3.4)
is an attacker sampling assumption for the near distance, not a planner setting. Held-out scores compare the reconstructed sign without insertion
(clean), the unpatched replica, the initial patch and the optimized patch. Time-uniform samples alternate
between PGD training and held-out frames within the selected training segments. These held-out
frames are an internal detector diagnostic, separate from the framework validation split. The
supplied settings use all five framework training segments (max_training_segments=None), eight
samples per segment, 200 PGD updates, batch 4 and step size 4/255. The offline report includes calibration and sample tables, the pgd_confidence series,
held-out scores and detection rates, and initial and optimized textures. Calibration has one row per training segment, and calibration_probes has one row per capture
with explicitly named evidence columns. heldout_summary compares the four texture variants;
heldout_scores and detection_rate_vs_distance use one row per frame and per 10 m bin,
respectively, with the variants in separate columns.

RolloutPatchOptimization uses NES and single-decision candidate rollouts. It scores the selected
produced command by interpolating its forward X displacement at a 3 s horizon, then maximizes the
mean across configured starts. With persistent output, every query saves its one-step RunLog under
offline/<attack>/files/rollouts; the rollouts table preserves query identity, forward distance,
completion status, errors and paths relative to offline/<attack>/files/ (for example rollouts/0003). The scalar rollout_count records the number
of queries and planned_forward_m records the per-step mean planned forward distance. Without persistent output, query logs
are not written. Each query constructs an independent driving system and can reload model weights. All
start_distances_m must lie in the attack's [min_approach_m, approach_m] interval.

The rollout budget is:

```text
number of starts per segment * training segments * (1 + steps * (population_size + 1))
```

Both RolloutPatchOptimization and DetectorPatchOptimization default to
max_training_segments=None, using all available training segments. A positive integer limits
cost; it does not assign data to training or testing. Neither reads validation_metadata.

The supplied YAML uses all five training segments, start_distances_m=[40, 30, 20], steps=40 and
population_size=20. Each black-box attack therefore executes
3 × 5 × (1 + 40 × 21) = 12,615 single-decision candidate rollouts.
Each attack has an offline/<attack>/ report directory, including the unoptimized no_patch control.
The framework owns these paths; optimization options have no output_directory setting.
offline_report.html combines all attacks' values, curves, tables and images. It includes
initial_texture, the complete replica face, even for attacks without optimization. Structured data lives in each report.json,
with CSV tables and PNG images beside it. evaluation_report.html presents the closed-loop results.
Optimization strength and driving impact must be measured; successful preparation alone
establishes neither.

## Resume

```bash
avsectester run configs/nurec_stop_sign_patch_alpamayo_2601.yaml --resume
```

Resume validates the saved preparation inputs and reuses the saved units and parts. Completed
attacks load their saved artifacts, and only unfinished offline phases run again. Matching completed
closed-loop logs are read without rerunning them; missing, failed or incomplete runs are retried
individually. Without --resume, a nonempty output directory is rejected. --resume cannot be combined
with --prepared. Python provides Experiment.prepare(configuration, resume=True) followed by
Experiment.execute(prepared, resume=True).

Resume checkpoints whole attacks, not individual PGD updates or NES queries. An unfinished offline
phase starts from its beginning. The framework first archives the whole existing
offline/<attack>/ directory to offline-incomplete/<attack>-<UTC timestamp>/, preserving its
partial failed report and query logs. The new attempt starts with a fresh report directory.
Matching completed executions are retained only when their full expected metadata still
agrees, including segment, part, environment, decision counts and deployment budget. See [Resume interrupted work](../guides/experiments.md#resume-interrupted-work).

## NRE service

Start NRE from the same configuration used for the experiment. The command resolves the dataset's
archives and mounts each file read-only, including the probe directory configured on the renderer.

```bash
nvidia-smi
avsectester nre serve configs/nurec_stop_sign_patch_alpamayo_2601.yaml --gpu 0
avsectester nre stop configs/nurec_stop_sign_patch_alpamayo_2601.yaml
```

The `--gpu` option selects the host GPU for NRE. The YAML `device` selects devices for constructors
such as the driving model and detector; it does not select the NRE GPU. Choose available resources
before starting either process. Use the 2604 configuration when serving its dataset instead.

`serve` creates the client probe directory and waits for "Serving on". Its default container name
is `avsectester-nre-<port>`; `stop` stops and removes that container. Use `--name` consistently to
select a different name. To print the exact command without creating files or using Docker:

```bash
avsectester nre serve configs/nurec_stop_sign_patch_alpamayo_2601.yaml --gpu 0 --dry-run
```

Every resolved archive filename must be unique. A configuration that includes both releases of the
same archive is rejected instead of silently choosing one; configure the intended scene version.
No symlink staging directory is required. See [NRE commands](../reference/cli.md#nre) for startup
timeouts, image/cache options and cleanup.

The supplied configurations share results/nre-probes with NRE at /mnt/nre-probes. Rendered probes
determine scene occlusion, and visible_pixels counts visible replica pixels. The configurations use
depth=none and ground_mesh=mesh_ground (see
[Ground contact](../guides/simulators.md#ground-contact)). They use map
navigation, ego_only physics updates and AlpaSimController. Alpamayo receives four consecutive 10 Hz
image frames through frame_subsample=1.

## Evaluation and visualization

```bash
avsectester evaluate configs/nurec_stop_sign_patch_alpamayo_2601.yaml
python -m scripts.visualization.plot_stop_line_approaches configs/nurec_stop_sign_patch_alpamayo_2601.yaml
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


## Add a replica-specific evaluation metric

ReplicaVisibility is an example of an attack-specific custom Metric, implemented in
[stop_sign_patch_attack/metrics.py](../../avsectester/implemented_attacks/stop_sign_patch_attack/metrics.py).
Subclass Metric and implement evaluate(group): iterate over group.runs and return a mapping from
each treatment to its field/value mapping. Read only saved RunLog evidence and return
UnavailableMetricValue with a reason when the evidence is absent; custom metrics do not need a
framework type-registry entry. Also implement describe() to return MetricDescription with a summary
and a MetricField for every output. For ReplicaVisibility, the visible_steps field has unit
"decision steps" and explains the configured pixel threshold, camera and missing-replica case.

This metric reads each step's object_visibility using REPLICA_OBJECT_ID, shared with online
insertion and detector sampling. visible_steps counts decisions with at least min_visible_pixels
(default 100) in camera_id. The default camera_id=None selects the unique camera with replica
visibility evidence across the comparison group. Multiple recorded cameras require an explicit
camera_id; a missing entry for the chosen camera counts as invisible. Empty runs and treatments
that never recorded the replica, including clean, are unavailable rather than zero.

Configure a custom metric with its complete module path and optional constructor parameters:

```yaml
evaluation:
  metrics:
    - type: StopLineMetrics
    - type: avsectester.implemented_attacks.stop_sign_patch_attack.metrics:ReplicaVisibility
      options:
        camera_id: camera_front_wide_120fov
        min_visible_pixels: 100
```

The supplied stop-sign configurations include ReplicaVisibility. Its field appears alongside the
built-in metrics in evaluation/report.json, evaluation/runs.csv and evaluation_report.html.
The page displays the custom metric's own meaning and unit alongside built-in definitions.


The offline values use measurement names: final_confidence is the final detector confidence
(lower is better for the attack), while final_planned_forward_m is the final mean planned forward
distance (higher is better for the attack). pgd_confidence shows the batch confidence before each
update; planned_forward_m shows NES steps with step 0 representing the initial texture. Initial
and optimized images both show the complete replica face for direct comparison. Every offline
table includes column explanations and units. The two root-level HTML pages link to one another;
their numeric display uses four decimal places and their JSON/CSV links retain full precision.
