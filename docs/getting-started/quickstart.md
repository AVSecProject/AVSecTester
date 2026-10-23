# Quickstart

> **Audience:** Users running their first experiment.
>
> **Outcome:** Run a comparison and inspect saved results.
>
> **Prerequisites:** [Installation](installation.md), including services and model weights.

## Run a CARLA comparison

Configure the dedicated CARLA server address and GPU in `configs/carla_phantom.yaml`. World content
and prescribed motion belong to the dataset recipe; server connection and timing belong to the
simulator. The perception device is `avstack.options.pipeline_cfg.perception.gpu`.

```bash
docker compose up -d --build
docker compose exec avsectester avsectester run configs/carla_phantom.yaml
```

The phantom experiment selects segments, prepares its artifacts once, then runs clean and attacked
treatments from each selected start. The terminal displays metric summaries. It also writes
metadata, artifacts, Parquet logs and an evaluation report under runs.output. An attack
that changes nothing can still complete successfully: exit codes report execution, not scientific
success.

A configuration contains one or more attacks, each selecting its own segments. Treatments without
attack_name run on every attack's segments; all treatment pairs within one attack's segment and seed
are compared, with optional world/sensor environments. Combine actions within one Attack when they
must act together in a run.

## Read the output

Open `evaluation/report.json` for per-group metrics and summaries, `evaluation/runs.csv` for one row
per run, and `evaluation/plots/` for speed and brake comparisons. `runs/` contains the underlying
RunLog records. Failed runs and selection errors remain distinguishable from completed runs; missing
values are reported with reasons where a metric cannot be calculated.

## Save experiment outputs

Compose mounts model weights but does not mount the checkout or results. Copy results before
recreating its container:

```bash
docker compose cp avsectester:/app/results/carla_phantom ./results/carla_phantom
```

To retain outputs directly on the host, use a suitable bind mount and configure its container path
as runs.output. See [Devices, mounts and rebuilds](installation.md#devices-mounts-and-rebuilds).

## Reuse preparation

Set a fresh runs.output directory in the experiment configuration and reference the previous saved
preparation:

```bash
avsectester run configs/carla_phantom.yaml --prepared results/carla_phantom
```

The framework verifies dataset/scenario, simulator and attack compatibility, the first seed and
the offline environments before executing. It does not rerun the metabuilder or offline
optimization. It refuses to reuse a preparation whose attack parameters were not describable and
therefore could not be verified. See
[Saved preparation compatibility](../reference/configuration.md#saved-preparation-compatibility).

## Reevaluate saved runs

Edit only the desired `evaluation` settings, then run:

```bash
avsectester evaluate configs/carla_phantom.yaml
```

The command recalculates metrics and plots from the configured output. No simulator or new inference
runs are involved. Keep runs.output pointing to the results to analyze.

## Supplied configurations

| Configuration | Purpose |
|---|---|
| `configs/carla_phantom.yaml` | Phantom perception output with ModularAVStack |
| `configs/carla_patch.yaml` | Fixed physical patch comparison |
| `configs/carla_patch_optimized.yaml` | Optimized physical patch comparison |
| `configs/carla_object_insertion.yaml` | Image object insertion with aligned camera depth |
| `configs/nurec_speed_feedback.yaml` | Ego-speed feedback consumed by Alpamayo |
| `configs/nurec_object_insertion.yaml` | Image insertion with Alpamayo |
| `configs/nurec_patch.yaml` | Patch image synthesis with Alpamayo |
| `configs/nuscenes_object_insertion.yaml` | Recorded insertion setup; real dataset/model run unverified |

Declared filesystem parameters are relative to the configuration file containing them; see
[path resolution](../reference/configuration.md#constructor-filesystem-parameters). Set dataset roots,
renderer endpoint, checkpoint paths and device options for your environment. NuRec uses the
AlpaSim driver
interpreter; from the checkout, use `python -m avsectester.cli run CONFIG.yaml` with that
interpreter.

## Next steps

Read [Experiments](../guides/experiments.md), [Configuration](../reference/configuration.md) and
[Attack examples](../examples/README.md).
