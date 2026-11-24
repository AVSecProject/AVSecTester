# CLI reference

> **Audience:** Experiment operators.
>
> **Outcome:** Look up commands, options, output files and exit codes.
>
> **Prerequisites:** [Installation](../getting-started/installation.md).

## Commands

Use the configured Compose experiment service. NuRec commands are:

```bash
docker compose run --rm nurec-experiment version
docker compose run --rm nurec-experiment run CONFIG.yaml
docker compose run --rm nurec-experiment run CONFIG.yaml --prepared DIRECTORY
docker compose run --rm nurec-experiment run CONFIG.yaml --resume
docker compose run --rm nurec-experiment evaluate CONFIG.yaml
docker compose run --rm nurec-experiment nre serve CONFIG.yaml --gpu N
docker compose run --rm nurec-experiment nre stop CONFIG.yaml
docker compose run --rm nurec-experiment trafficsim serve CONFIG.yaml --gpu N --models models/alpasim-trafficsim
docker compose run --rm nurec-experiment trafficsim stop CONFIG.yaml
```

For an already started CARLA experiment service, use
`docker compose exec carla-experiment avsectester COMMAND`.
The [quickstart](../getting-started/quickstart.md) provides
first-run examples.

## Run

`run CONFIG.yaml` loads configuration, prepares artifacts, executes treatments and evaluates
saved evidence. It requires a nonempty attack mapping and `runs.output`. The
[configuration reference](configuration.md) describes YAML keys, defaults and
preparation compatibility.

| Option | Default | Meaning |
|---|---|---|
| `--prepared DIRECTORY` | omitted | Load a complete preparation into a fresh output directory |
| `--resume` | `false` | Continue preparation and execution in `runs.output` |
| `--nre-gpu N` | `AVSECTESTER_HOST_GPU` | GPU for an automatically started NRE |
| `--nre-name NAME` | avsectester-nre-PORT | Name of an automatically started NRE container |
| `--trafficsim-gpu N` | `AVSECTESTER_HOST_GPU` | GPU for automatically started traffic |
| `--trafficsim-models DIR` | omitted | Local model directory required when starting traffic |
| `--trafficsim-name NAME` | avsectester-trafficsim-PORT | Name of an automatically started traffic container |
| `--service-timeout SECONDS` | 600 | Positive service startup timeout |

The output path must be a directory. A nonempty directory requires `--resume`. `--prepared` and
`--resume` are mutually exclusive.

### Preparation reuse

`--prepared` maps to loading a complete preparation. `--resume` continues saved preparation and
execution. See [Experiments](../guides/experiments.md#resume-interrupted-work) for validation,
resume and persistence semantics.

### Automatic local services

With reactive traffic enabled, provide its local weights when a service may need starting:

```bash
docker compose run --rm nurec-experiment run CONFIG.yaml --trafficsim-models models/alpasim-trafficsim
```

`run` checks the configured NuRec renderer and traffic endpoints after configuration, output and
preparation checks. A running service must contain every configured dataset scene. A compatible
service is reused and left running. A scene mismatch is an error. When `run` starts services, it
uses their default images and an NRE cache size of two. Use the service commands to select an image
explicitly.

Missing services can start only on local endpoints: `127.0.0.1`, `localhost`, `0.0.0.0` or `::1`.
Starting traffic requires `--trafficsim-models`. GPU selection uses the explicit service option
or `AVSECTESTER_HOST_GPU`, which the NuRec Compose service sets from `NUREC_GPU`. The value must
be one nonnegative host GPU index. This selects the physical device independently of the model's
YAML `device` option.

Python callers start required services themselves, for example with
[`local_services`](python-api.md#avsectesterclosed_loopsimulatorsnurecserver).

Managed services cover preparation, execution and evaluation. Startup, reuse and shutdown each
print a progress line. On completion, exceptions or Ctrl-C, only containers started by this
command are stopped and removed, in reverse startup order. Cleanup errors are attached to an
active exception. After a completed experiment they are reported without replacing its result.
A killed command can leave its exited container behind. The next start removes it, following the
[NRE rules](#nre) for existing containers.

NRE startup creates its client asset directory. Docker mounts each archive read-only at
`/mnt/nre-data/<filename>` and mounts the asset directory read-only at the same absolute client
and server path by default. An explicit `server_asset_directory` sets a different mount path.
Archive filenames must be unique. The Compose host-path and Docker access setup is described
in [Installation](../getting-started/installation.md).

The optional manual service commands below support reuse across multiple experiments.

## Evaluate

`evaluate CONFIG.yaml` reads `runs.output`, recomputes metrics and replaces evaluation outputs.
It leaves run logs and preparation unchanged. It uses saved segments, maps and the metadata
hash. The dataset, external metadata file, model inference and simulator services are
not needed.

`Metric` configuration may change before reevaluation, subject to the saved
[compatibility rules](configuration.md#saved-preparation-compatibility). Corrupt individual run
logs are recorded as failures while other runs remain available. Saved directories for removed
or unexpected treatments are reported and skipped.

Both `run` and `evaluate` print `offline_report.html` and `evaluation_report.html` paths.
Evaluate rebuilds both pages from saved files. See [Evaluation](../guides/evaluation.md) for
terminal summaries, statistics and unavailable values.

## NRE

```text
nre serve CONFIG --gpu N [--name NAME] [--image IMAGE] [--cache-size 2] [--timeout 600] [--dry-run]
```
loads the configuration, starts Docker and waits for `Serving on`.

| Option | Default |
|---|---|
| `--gpu N` | required host GPU index |
| `--name NAME` | avsectester-nre-PORT |
| `--image IMAGE` | `nvcr.io/nvidia/nre/nre-ga:26.04` |
| `--cache-size K` | 2 |
| `--timeout SECONDS` | 600 |
| `--dry-run` | `false` |

The configuration must contain `NuRecDataset` and `NuRecRenderer` with a local endpoint. This
command does not validate experiment environments. Archive and asset mounts follow the
[automatic service rules](#automatic-local-services).

```bash
docker compose run --rm nurec-experiment nre serve configs/nurec_stop_sign_patch_alpamayo_2601.yaml --gpu N
docker compose run --rm nurec-experiment nre serve configs/nurec_stop_sign_patch_alpamayo_2601.yaml --gpu N --dry-run
docker compose run --rm nurec-experiment nre stop configs/nurec_stop_sign_patch_alpamayo_2601.yaml
```

`--dry-run` prints a shell-quoted Docker command without creating directories or containers and
does not require Docker on PATH. A normal launch requires Docker. An exited container with the
chosen name that an earlier AVSecTester start by the same user created is removed first. Any
other container with that name is left untouched and reported as an error. Successful startup
prints endpoint, GPU, scene count, container name and the corresponding stop command.

Startup failure or timeout includes the last 50 log lines and removes the newly created
container. The container is created without Docker's `--rm` so startup diagnostics
remain readable.

`nre stop CONFIG [--name NAME]` derives the default name from the endpoint without enumerating
dataset archives, then stops and removes that container. A missing container is an error.

## Trafficsim

```text
trafficsim serve CONFIG --gpu N --models DIR [--name NAME] [--image IMAGE] [--timeout 600] [--dry-run]
```
requires `NuRecDataset` and a configured `AlpaSimTraffic` with a local endpoint.

| Option | Default |
|---|---|
| `--gpu N` | required host GPU index |
| `--models DIR` | required model directory |
| `--name NAME` | avsectester-trafficsim-PORT |
| `--image IMAGE` | `avsectester:trafficsim` |
| `--timeout SECONDS` | 600 |
| `--dry-run` | `false` |

```bash
docker compose run --rm nurec-experiment trafficsim serve CONFIG.yaml --gpu N --models models/alpasim-trafficsim
docker compose run --rm nurec-experiment trafficsim stop CONFIG.yaml
```

Archives are mounted read-only under `/mnt/nre-data` and models at `/mnt/trafficsim-models`. The
server uses `prediction_steps=15`, `OMP_NUM_THREADS=1` and `MKL_NUM_THREADS=1`. Readiness
requires the CATK startup log followed by a successful metadata RPC. Startup errors, dry-run and
existing container handling follow the [NRE rules](#nre).

`trafficsim stop CONFIG [--name NAME]` derives the name without enumerating datasets, then stops
and removes that container. Use the same explicit name for serve and stop. Reactive traffic
behavior is described in [Simulators](../guides/simulators.md#reactive-traffic).

## Interruptions

Ctrl-C exits 130, cleans up owned containers and prints the output directory when available.
Partial reports may remain. `evaluate` regenerates them without simulation.
Checkpoint and retry behavior follow
[Resume interrupted work](../guides/experiments.md#resume-interrupted-work).

## Output layout

```text
<output>/
  preparation-progress.json
  prepared.json
  attacks/<attack>/metadata.json
  attacks/<attack>/artifacts/artifacts.pkl
  attacks/<attack>/artifacts/manifest.json
  attacks/<attack>/offline_report/report.json
  attacks/<attack>/offline_report/<table>.csv
  attacks/<attack>/offline_report/<image-or-series>.png
  attacks/<attack>/offline_report/files/
  attacks/<attack>/incomplete_offline_reports/<UTC timestamp>/
  offline_report.html
  maps/<scenario>.json
  runs/segment-<NNNNN>/seed-<S>/<treatment>/
  evaluation/report.json
  evaluation/runs.csv
  evaluation/treatment_summary.csv
  evaluation/pair_summary.csv
  evaluation/plots/
  evaluation_report.html
```

Repeated experiments add `repeat-R` between the seed and treatment directories and add
`repeat: R` to run metadata, with zero-based repeat numbers. Single runs omit both.
Repeated experiments write `evaluation/repeat_pair.csv` even when it has no data rows.

Each attack directory holds the selected segments in `metadata.json`, its artifacts in
`artifacts/` and its offline report in `offline_report/`. The `artifacts/` directory is created
only when the offline phase returns a value other than `None`. A failed or interrupted offline
phase moves its partial report to `incomplete_offline_reports/` before the phase runs again.
Artifact filenames above describe the default trusted-pickle format. Attacks may override
artifact persistence, and their files stay inside `artifacts/`. Segment numbers are global unit
indexes. Each run directory contains
`manifest.json`, published Parquet step, event and capture parts, and payload shards in its
[RunLog](runtime.md#runlog). Runs with ROS topic processing also publish `messages/` parts.
Offline report content is described in
[Optimization](../guides/optimization.md), and evaluation content
in [Evaluation](../guides/evaluation.md).

Preparation manifests list attacks with saved artifacts in `attacks_with_artifacts`, in
configuration order. `preparation-progress.json` includes only completed attacks in this list.
Loading uses the list and ignores unlisted artifact directories. Manifests without this field
load artifacts for every completed attack.

## Exit codes

| Code | Meaning |
|---|---|
| 0 | Command completed without failed runs, selection errors or metric errors. This does not establish attack effectiveness |
| 1 | Unexpected configuration loading errors, preparation load or validation errors, selection, execution, result reading, metric evaluation or service startup/stop failed |
| 2 | Invalid command usage, configuration or metadata contents, output conflict or incompatible saved preparation |
| 130 | Keyboard interruption. Available partial output is retained |

Service scene mismatch, a missing service at a nonlocal endpoint, absent `--trafficsim-models`
and invalid automatic GPU selection are configuration errors (exit 2). The explicit service
commands pass `--gpu` to Docker, whose startup failures exit 1. Container readiness and
explicit stop failures also exit 1. `evaluate` exits 1 if `prepared.json` is missing or if
expected runs without manifests are marked incomplete.

## Supplied configurations

See the complete [configuration list](../examples/README.md) and
[configuration reference](configuration.md).
