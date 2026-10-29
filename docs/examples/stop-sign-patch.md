# Stop-sign patch attack

> **Audience:** Users studying stop-sign patches in reconstructed NuRec scenes.
>
> **Outcome:** Prepare fixed or optimized replica textures, run test treatments and inspect
> reports.
>
> **Prerequisites:** [Installation](../getting-started/installation.md), NuRec scene archives
> and the driving-system weights referenced by the chosen configuration.

## Choose a configuration

The optimized treatments use the following attack type and algorithm. Each configuration also
includes unoptimized controls.

| Configuration | Driving system | Attack type | Algorithm used |
|---|---|---|---|
| [nurec_stop_sign_patch_modular_2601.yaml](../../configs/nurec_stop_sign_patch_modular_2601.yaml) | `RuleBasedAVStack` | white-box attack | `PGD` |
| [nurec_stop_sign_patch_modular_2604.yaml](../../configs/nurec_stop_sign_patch_modular_2604.yaml) | `RuleBasedAVStack` | white-box attack | `PGD` |
| [nurec_stop_sign_patch_alpamayo_2601.yaml](../../configs/nurec_stop_sign_patch_alpamayo_2601.yaml) | `AlpamayoAVStack` | black-box attack | `NES` |
| [nurec_stop_sign_patch_alpamayo_2604.yaml](../../configs/nurec_stop_sign_patch_alpamayo_2604.yaml) | `AlpamayoAVStack` | black-box attack | `NES` |

Run from the repository root after configuring data and weights:

```bash
docker compose run --rm nurec-experiment run configs/nurec_stop_sign_patch_modular_2601.yaml
docker compose run --rm nurec-experiment run configs/nurec_stop_sign_patch_alpamayo_2601.yaml
```

Service management follows the
[CLI service lifecycle](../reference/cli.md#automatic-local-services).
See [Examples](README.md#configurations) for the complete supplied configuration list.

The `modular` configurations use `RuleBasedAVStack`. The suffixes `_2601` and `_2604` refer to
NuRec 26.01 and 26.04, stored under `data/nurec/26.01_release` and `data/nurec/26.04_release`.
See [Installation](../getting-started/installation.md) for weights.

## Treatments

The modular configurations contain the detector-optimized treatments. The Alpamayo
configurations contain the rollout-optimized treatments. Both include the same controls.

| Treatment | Initial patch | Optimized | Attack type |
|---|---|---|---|
| `clean` | Unpatched replica | No | Baseline |
| `white` | White | No | Fixed patch |
| `black` | Black | No | Fixed patch |
| `noise` | Low-frequency noise | No | Fixed patch |
| `white_detector_pgd` | White | Yes | white-box attack |
| `black_detector_pgd` | Black | Yes | white-box attack |
| `noise_detector_pgd` | Low-frequency noise | Yes | white-box attack |
| `white_rollout_nes` | White | Yes | black-box attack |
| `black_rollout_nes` | Black | Yes | black-box attack |
| `noise_rollout_nes` | Low-frequency noise | Yes | black-box attack |

Each attack has an offline phase and an online phase. With no optimization, the offline phase
returns the initial texture. The online phase inserts that texture through the same placement
path used during optimization. The `clean` treatment runs the `no_patch` attack, which uses
`initial_patch: none` and cannot be optimized, so every treatment shows the same inserted
replica and differs only in its patch.

`low_frequency_noise` draws independent color values on a grid of one value per
`noise_cell_texels` face texels, default 32, and upsamples them bicubically. Its color patches stay
visible at the small image sizes of an approaching camera, where per-texel noise averages to gray.

## Selection and placement

The selector uses the mapped stop sign, its stop-sign stop line and the recorded ego route. It
chooses at most one approach per scene, from the entering side and facing the sign. The route
must pass within 3 m of the line and contain a recorded speed below 0.5 m/s within 15 m
before it.

The Alpamayo configurations set `approach_m: 100` and `min_approach_m: 60`. The modular
configurations use the defaults of 50 m and 20 m. The selector waits for required recorded
history and includes `hold_steps` hold decisions, with a default of 20. A start closer than
`min_approach_m` is rejected. Constructor defaults are in the
[stop-sign attack API](../reference/python-api.md#avsectesterimplemented_attacksstop_sign_patch_attackattack).

The runtime inserts a post-free canonical MUTCD R1-1 replica face over the reconstructed sign.
The `mask_path` binary mask selects which face-texture colors form the patch. The default
`rp2_bars.png` places bars above and below STOP. `face_interior.png` is the packaged
alternative. Artifacts keep the face alpha and all colors outside the mask.

The default placement is upright with a standard-size face. Map modes are alternatives:

| Option | Behavior |
|---|---|
| `face_orientation="upright"` | Use the horizontal projection of the annotated face normal and world +Z as up |
| `face_orientation="map"` | Preserve the annotated rotation, including tilt |
| `face_size="mutcd_standard"` | Select the width closest to annotated height from 0.762, 0.914 and 1.219 m |
| `face_size="map_width"` | Use the annotated width |
| `width_scale=1.1` | Multiply the selected face width. Height follows the texture aspect ratio |
| `offset_m=0.10` | Move the replica face center along the selected forward normal |

The annotation supplies the center and yaw. Position or yaw errors can expose the reconstructed
sign around the replica's edges at close range. The attack does not correct those annotations.
The simulator's `occlusion_tolerance_m` handles occlusion.
See [Simulators](../guides/simulators.md).

The configurations use `ExposureHarmonizer(reference="under")`. Its gain is estimated from the
reconstructed sign and the unpatched nominal replica, then applied to each patch. The same
differentiable calculation is used for optimization and deployment. Formulas and constraints are
in the [harmonizer API](../reference/python-api.md#avsectesterrenderingharmonizers).

## Training and test segments

All attacks select metadata before the shared split is formed. The supplied settings are:

```yaml
split:
  train_segments: 5
  seed: 0
  execute: [test]
```

This needs at least six selected approaches: five training units and at least one test unit. See
[split configuration](../reference/configuration.md#data-split) for unit identity, allocation
and training input rules.

## Prepare the white-box attack

The white-box attack uses the driving system's YOLOX detector through `avstack.bindings`.
Algorithm used: `PGD`. `rule_based_driving_adapter` declares `rule_based.detection_confidence`,
the camera and decision threshold. A `SessionScorer` reuses it for calibration and optimization.
The [driving-system guide](../guides/driving-systems.md) describes the binding behavior.

The driving system stops once the detector fires, so the patch must suppress every detection
until the vehicle can no longer stop before the line. Training therefore covers recorded frames by
route distance to the stop line, from the farthest frame where the unpatched replica reaches the
decision threshold down to `v² / (2 × braking_deceleration_mps2)`. `v` is the recorded speed at the
segment start. `braking_deceleration_mps2` is the driving system's strongest braking and defaults
to 6 m/s², the `RuleBasedAVStack` planner limit. The sign-face projection defines the target box.
Frames below `min_visible_pixels` are skipped. Its default is 100 visible replica pixels.

The configurations use all five training segments, every eligible frame (`samples_per_scene: all`),
`epochs: 100`, `batch_size: 16`, `tv_weight: 0.3` and step size 4/255.
An integer `samples_per_scene` of at least four instead selects that many time-uniform frames
per segment. Every selected frame is a training sample.
Validation, when needed, uses separate segments from the
[data split](../reference/configuration.md#data-split).

The objective is `HideBelowThreshold`: `max(0, s − (τ − m))` for detector confidence `s`, the
driving system's decision threshold `τ` and `confidence_margin` `m`, default 0.1. One detection is
enough for the driving system to stop, so frames already below `τ − m` stop contributing and the
updates concentrate on frames that would still be detected. The margin absorbs differences
between recorded training views and closed-loop views.

`PGD`'s `tv_weight` adds a smoothness term to the detector loss, so the patch keeps structure that
survives printing and camera sampling. The term is the mean Charbonnier RGB distance,
`sqrt(||c_p - c_q||² + ε²) - ε` with ε = 0.001, over vertically and horizontally adjacent texels
that are both inside the mask. Pairs across the patch outline are excluded, so a uniform patch
has zero smoothness cost against the red face.

The offline report contains calibration and sample tables, the `pgd_objective` series, the final
`patch_smoothness`, and initial/optimized replica textures. `pgd_objective` records the mean
training hinge objective per epoch, excluding the smoothness term, at epochs 1 through
`epochs`. After optimization every training frame is scored for the clean reconstruction,
unpatched replica, initial patch and optimized patch. `detected_frame_fraction` and
`max_confidence` summarize the optimized patch, and detection rates are binned by route distance
to the stop line. The configurations set `progress_interval: 10` to score the current patch on
every training frame every 10 epochs and after the last epoch. The option defaults to 1.
The `training_progress` table and its series record the full-set hinge loss, detected fraction
and detected fraction per 10 m bin. Epoch 0 is the initial patch.
[Optimization tools](../guides/optimization.md) explains report usage.

## Prepare the black-box attack

The black-box attack minimizes the mean stop-command distance over retained training segments.
Algorithm used: `NES`.

A stop command is a decision whose timestamped planned trajectory has an adjacent-waypoint
horizontal speed below `stop_speed_mps`, default 0.5 m/s. Stop-command distance is the horizontal
distance from the ego origin to the sign-face center at the first such decision, before passing
the sign. Only decisions with `(ego_position − sign_center) · sign_x > 0` count, where `sign_x`
is the first column of `sign_pose`. A rollout that passes the sign without a stop command
scores 0. A rollout whose decisions run out before either event scores its remaining distance to
the sign after the last decision, so ending early is not rewarded. The `outcome` column records
which case applies.

Calibration evaluates the unpatched replica once per training segment, in metadata order. A
segment is retained only if it issues a stop command at step `min_stop_command_step` or later,
default 5, before passing the sign. Segments with an earlier command, no command before passing
or decisions that run out first are excluded with a reason. Calibration stops after
`max_training_segments` retained segments, or uses every eligible training segment when this
limit is `None`. Calibration raises `ValueError` if no segment is retained.

Each candidate starts at each retained segment's recorded origin and runs for at most
`segment.decision_steps` decisions. It ends when the first stop command is issued or the ego
passes the sign. The scalar score is the mean stop-command distance across those segments.

The search variables are RGB offsets on a grid with one cell per `noise_cell_texels` face
texels, default 32. Bicubic upsampling maps these offsets to the texture. Only masked RGB
changes, and alpha and colors outside the mask stay fixed. No smoothness term is added.

The supplied configurations retain at most two training segments, set `stop_speed_mps: 0.5`
and `min_stop_command_step: 5`, and use `steps: 10`, `population_size: 8`, `sigma: 0.15`,
`learning_rate: 0.1`, `batch_size: 1` and `seed: 0`. The candidate query budget is:

```text
retained training segments × (1 + steps × (population_size + 1))
```

Calibration adds one rollout per examined segment.

The offline report contains a `calibration` table with selection reasons, a `rollouts` table
with per-query distances and run details, the `stop_command_distance_m` series, and initial and
optimized textures. `final_stop_command_distance_m` is the final texture's evaluated mean
distance. No additional rollout is used for that value. The
[attack API](../reference/python-api.md#avsectesterimplemented_attacksstop_sign_patch_attackoffline)
describes the optimization options.

## Reuse segments and resume

Use `metadata.save` to export selected units, or `metadata.load` to share a user-created
selection. These settings and content-hash compatibility are defined by
[Metadata configuration](../reference/configuration.md). Saving a selection does not reuse
optimized textures across driving systems.

Resume interrupted work with:

```bash
docker compose run --rm nurec-experiment run configs/nurec_stop_sign_patch_modular_2601.yaml --resume
```

Completed attack offline phases and matching completed runs are reused. Unfinished work follows
[the experiment resume workflow](../guides/experiments.md).

## Run from Python

To run the workflow from Python, open `local_services` around preparation and execution, or
start NRE with `nre serve` first. Save this module as `my_module.py` in the repository root,
which is the container's working directory:

```python
from avsectester.closed_loop.simulators.nurec.server import local_services
from avsectester.experiment import Experiment, evaluate, load_configuration

def run(configuration_path, nre_gpu):
    configuration = load_configuration(configuration_path)
    Experiment.validate(configuration)
    with local_services(configuration.dataset, configuration.simulator,
                        nre_gpu=nre_gpu, log=print):
        prepared = Experiment.prepare(configuration)
        result = Experiment.execute(prepared)
        return evaluate(result, configuration)

if __name__ == "__main__":
    run("configs/nurec_stop_sign_patch_modular_2601.yaml", nre_gpu=None)
```

See [manual NRE commands](../reference/cli.md#nre) for reusable services. Run a Python module
through the same Compose environment:

```bash
docker compose run --rm --entrypoint python nurec-experiment -m my_module
```

## Optional reactive traffic

Add `simulator.options.traffic` with type
`avsectester.closed_loop.simulators.nurec:AlpaSimTraffic` and pass
`--trafficsim-models models/alpasim-trafficsim` to `run`. It manages the local traffic service
alongside NRE. The [Simulator guide](../guides/simulators.md) defines behavior and restrictions.
Prepare the traffic image and weights following
[Installation](../getting-started/installation.md#reactive-traffic-weights-and-image).

## Evaluate and plot

Evaluate saved runs and generate stop-line plots:

```bash
docker compose run --rm nurec-experiment evaluate configs/nurec_stop_sign_patch_modular_2601.yaml
docker compose run --rm --entrypoint python nurec-experiment -m scripts.visualization.plot_stop_line_approaches configs/nurec_stop_sign_patch_modular_2601.yaml
```

The [CLI reference](../reference/cli.md) defines the HTML and structured-output locations.
[Evaluation](../guides/evaluation.md) defines the stop-line, driving and reasoning metrics.
`RuleBasedAVStack` has no reasoning text, so its configuration omits `ReasoningPhraseRate`.

### Custom metric example

`ReplicaVisibility` demonstrates an importable `Metric` outside the built-in metric registry.
Its `evaluate` reads saved capture evidence and returns `visible_steps`. `describe` supplies the
field's meaning and unit. A treatment without a replica has an unavailable value. A decision
counts as visible when the replica reaches `min_visible_pixels` in at least one selected camera.
By default `camera_ids` selects every camera that recorded the replica.

Reference the class by its full path, as in the supplied YAML:

```yaml
evaluation:
  metrics:
    - type: avsectester.implemented_attacks.stop_sign_patch_attack.metrics:ReplicaVisibility
```

For a user metric, implement the same public methods in your own package and use its
`package.module:Class` path. See the [metric authoring guide](../guides/evaluation.md) for that
interface, and the
[supplied implementation](../../avsectester/implemented_attacks/stop_sign_patch_attack/metrics.py)
for a concrete example.

## Next steps

Read [Experiment workflows](../guides/experiments.md), [Evaluation](../guides/evaluation.md) and
the [attack-impact guidance](README.md#attack-impact).
