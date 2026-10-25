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
Each configuration contains nine attack choices: white, black and Gaussian-noise initial patches,
each deployed directly, optimized with detector PGD, or optimized with rollout NES. A minimal
unoptimized attack specification is:

```yaml
type: StopSignPatchAttack
options:
  initial_patch: white
  offset_m: 0.10
  width_scale: 1.0
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
recorded speed below 0.5 m/s within 15 m before it. The default approach distance is 50 m and the
segment includes 20 hold decisions. Selection also leaves the 15 real past observations required
by the driving adapter available at the start. Scene versions are distinguished by configurations.

The patch canvas is a binary mask, 512 by 512 pixels for the supplied masks. Only mask pixels
have alpha 255; everything else is transparent. The runtime inserts one stop_sign_patch object,
so the real reconstructed sign remains visible outside the patch. The default rp2_bars mask
covers upper and lower bars; face_interior is an alternative packaged mask.

The panel width is the map sign width times width_scale, which defaults to 1.0. Its face center
is aligned with the map sign face center and moved 10 cm toward approaching vehicles along the
sign's positive X normal. This offset produces parallax as the viewpoint changes. Map pose and
size errors can shift or resize the patch relative to the reconstructed sign; the same placement
rule is used in all scenes. A height mismatch produces a warning rather than a scene-specific
correction. World insertion changes camera composition; it does not insert patch returns into LiDAR.

## Initial patches and preparation

initial_patch accepts white, black or gaussian_noise. White and black set every opaque pixel to
255 or 0. Gaussian noise samples independent RGB values from clip(0.5 + noise_std * N(0, 1), 0, 1),
then rounds to uint8. noise_std defaults to 0.25 and noise_seed to 0. All initial textures have zero
RGB outside the mask and alpha equal to the mask times 255.

Experiment.prepare performs selection and all optimization. With optimization=None it returns the
initial texture. DetectorPatchOptimization and RolloutPatchOptimization are immutable options
objects in the attack's offline module. Preparation alone can be invoked through the public API:

```python
from avsectester.experiment import Experiment, load_configuration

configuration = load_configuration("configs/nurec_stop_sign_patch_2601.yaml")
prepared = Experiment.prepare(configuration)  # Also saved to runs.output.
```

DetectorPatchOptimization uses a stop-sign detector, PGD and HideObject. It renders patches using
patch_placement, skips frames with too few visible patch pixels, and projects the map sign face
for the target box. The clean detector score uses the reconstructed background without a patch.
The sample range starts at speed squared divided by twice the braking deceleration and ends at
the farthest clean detection above the configured threshold. Time-uniform samples alternate
between training and held-out frames. The supplied settings use four scenes, eight samples per
scene, 20 PGD updates, batch 1 and step size 4/255. Outputs include calibration and sample tables,
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

The supplied YAML uses one scene, start_distances_m=[40, 30, 20], steps=1 and
population_size=4: three starts times six evaluations, or 18 rollouts per NES attack.
The three initial patches each have their own optimization output directory. Optimization strength
and driving impact must be measured; successful preparation alone establishes neither.

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
Rendered probes determine scene occlusion, and visible_pixels counts visible opaque patch pixels.
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

Each attack selects its own metadata and forms its own comparison groups. The clean treatment
therefore runs again in every attack group, even when segments coincide. Comparing different
attacks within one group remains a framework TODO. The supplied run command executes complete
treatments; to perform only optimization, use Experiment.prepare without Experiment.execute.

## Next steps

Read [Evaluation](../guides/evaluation.md) and the
[Python API](../reference/python-api.md#avsectesterimplemented_attacksstop_sign_patch_attackattack).
