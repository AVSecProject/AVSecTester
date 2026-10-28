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

The modular configurations contain the detector-optimized treatments; the Alpamayo
configurations contain the rollout-optimized treatments. Both include the same controls.

| Treatment | Initial patch | Optimized | Attack type |
|---|---|---|---|
| `clean` | Unpatched replica | No | Baseline |
| `white` | White | No | Fixed patch |
| `black` | Black | No | Fixed patch |
| `noise` | Gaussian noise | No | Fixed patch |
| `white_detector_pgd` | White | Yes | white-box attack |
| `black_detector_pgd` | Black | Yes | white-box attack |
| `noise_detector_pgd` | Gaussian noise | Yes | white-box attack |
| `white_rollout_nes` | White | Yes | black-box attack |
| `black_rollout_nes` | Black | Yes | black-box attack |
| `noise_rollout_nes` | Gaussian noise | Yes | black-box attack |

Each attack has an offline phase and an online phase. With no optimization, the offline phase
returns the initial texture. The online phase inserts that texture through the same placement
path used during optimization. The `clean` treatment runs the `no_patch` attack, which uses
`initial_patch: none` and cannot be optimized, so every treatment shows the same inserted
replica and differs only in its patch.

## Selection and placement

The selector uses the mapped stop sign, its stop-sign stop line and the recorded ego route. It
chooses at most one approach per scene, from the entering side and facing the sign. The route
must pass within 3 m of the line and contain a recorded speed below 0.5 m/s within 15 m
before it.

`approach_m` defaults to 50 m before the line. The selector waits for required recorded history
and includes `hold_steps` hold decisions, with a default of 20.
A start closer than `min_approach_m`,
default 20 m, is rejected. Constructor defaults are in the
[stop-sign attack API](../reference/python-api.md#avsectesterimplemented_attacksstop_sign_patch_attackattack).

The runtime inserts a post-free canonical MUTCD R1-1 replica face over the reconstructed sign.
The `mask_path` binary mask selects which face-texture colors form the patch. The default
`rp2_bars.png` places bars above and below STOP; `face_interior.png` is the packaged
alternative. Artifacts keep the face alpha and all colors outside the mask.

The default placement is upright with a standard-size face; map modes are alternatives:

| Option | Behavior |
|---|---|
| `face_orientation="upright"` | Use the horizontal projection of the annotated face normal and world +Z as up |
| `face_orientation="map"` | Preserve the annotated rotation, including tilt |
| `face_size="mutcd_standard"` | Select the width closest to annotated height from 0.762, 0.914 and 1.219 m |
| `face_size="map_width"` | Use the annotated width |
| `width_scale=1.1` | Multiply the selected face width; height follows the texture aspect ratio |
| `offset_m=0.10` | Move the replica face center along the selected forward normal |

The annotation supplies the center and yaw. Position or yaw errors can expose the reconstructed
sign around the replica's edges at close range; the attack does not correct those annotations.
The simulator's `occlusion_tolerance_m` handles occlusion;
see [Simulators](../guides/simulators.md).

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

The white-box attack uses the driving system's own detector through `avstack.bindings`;
algorithm used: `PGD`. `rule_based_driving_adapter` declares the scorer, camera and decision
threshold. A `SessionScorer` reuses that binding for calibration, updates and held-out scoring.
The [driving-system guide](../guides/driving-systems.md) describes the binding behavior.

The sample interval starts at `v² / (2 × braking_deceleration_mps2)` and ends at the farthest
eligible frame where the unpatched replica reaches the driving detector's decision threshold.
The braking assumption defaults to 3.4 m/s² and only controls sampling. The sign-face projection
defines the target box. Frames below `min_visible_pixels` are skipped; its default is 100
visible replica pixels.

The configurations use all five training segments, eight samples per segment, 200 updates, batch
4 and step size 4/255. Time-uniform samples alternate between update frames and held-out frames
within each training segment. These held-out frames are a detector diagnostic, separate from the
framework's validation split.

The offline report contains calibration, sample and held-out tables, the `pgd_confidence`
series, and initial/optimized replica textures. Held-out diagnostics compare clean
reconstruction, unpatched replica, initial patch and optimized patch.
[Optimization tools](../guides/optimization.md) explains report usage.

## Prepare the black-box attack

The black-box attack scores the mean planned forward distance from the driving command;
algorithm used: `NES`. Every candidate runs one decision at each configured start. The
`horizon_s` objective uses the command's timestamped trajectory, without extrapolating
beyond it.

The supplied settings use all five training segments, starts 40, 30 and 20 m before the stop
line, a 3 s horizon, 40 updates, population size 20, sigma 0.08 and learning rate 0.03. Query
budget is:

```text
starts × training segments × (1 + steps × (population_size + 1))
= 3 × 5 × (1 + 40 × 21)
= 12,615 single-decision candidate rollouts per optimized attack
```

The offline report records starts, candidate runs, planned-distance history and
initial/optimized textures. Use the
[attack API](../reference/python-api.md#avsectesterimplemented_attacksstop_sign_patch_attackoffline)
for the cost-limiting `max_training_segments` option.

## Reuse segments and resume

Use `metadata.save` to export selected units, or `metadata.load` to share a user-created
selection. These settings and content-hash compatibility are defined by
[Metadata configuration](../reference/configuration.md). Saving a selection does not reuse
optimized textures across driving systems.

Resume interrupted work with:

```bash
docker compose run --rm nurec-experiment run configs/nurec_stop_sign_patch_modular_2601.yaml --resume
```

Completed attack offline phases and matching completed runs are reused; unfinished work follows
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
alongside NRE; the [Simulator guide](../guides/simulators.md) defines behavior and restrictions.
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
Its `evaluate` reads saved capture evidence and returns `visible_steps`; `describe` supplies the
field's meaning and unit. A treatment without a replica has an unavailable value. The default
camera is inferred from unique recorded replica evidence; set `camera_id` when multiple cameras
record it.

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
