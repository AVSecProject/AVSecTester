# CLI reference

> **Audience:** Experiment operators.
>
> **Outcome:** Look up commands, outputs and exit codes.
>
> **Prerequisites:** [Installation](../getting-started/installation.md).

## Commands

```bash
avsectester version
avsectester run EXPERIMENT.yaml
avsectester run EXPERIMENT.yaml --prepared DIRECTORY
avsectester run EXPERIMENT.yaml --resume
avsectester evaluate EXPERIMENT.yaml
```

`python -m avsectester.cli` provides the same commands from a source checkout.

## Run

`run EXPERIMENT.yaml [--prepared DIRECTORY | --resume]` loads configuration, prepares artifacts,
executes treatments and evaluates their logs. With --prepared, it validates and loads an existing
complete preparation instead of selecting and optimizing again. Prepared artifacts must be fully
loaded before execution. All attack inputs and returned artifacts remain explicit.

The configuration must provide runs.output and at least one attack. Treatments without attack_name
run once per unit and seed. Attacked treatments run on units selected by their attack, and different
attacks selecting identical complete metadata share a comparison group. With split, execution is
restricted to split.execute, which defaults to test. See [Data split](configuration.md#data-split).

Without --resume an existing nonempty output is rejected rather than overwritten. With --prepared,
choose a fresh runs.output for execution. Reusing preparation requires compatible dataset,
simulator, attack inputs, first seed, split counts/seed and offline environments. Execution parts,
other run settings and metrics can change; see
[Saved preparation compatibility](configuration.md#saved-preparation-compatibility).

--resume continues the same runs.output. If prepared.json exists it is loaded; otherwise preparation
resumes from preparation-progress.json after compatibility checks. Saved metadata and parts are
reused, completed artifacts are loaded and only unfinished attacks run offline. The execution phase
keeps matching completed manifests and reads their logs, reruns only missing/failed/incomplete or
mismatched runs, and skips initialization for groups already complete. It then evaluates the result.
--resume and --prepared together are a configuration error (exit code 2).

The terminal prints the output directory, selection/execution errors including exception types, and
failed runs in non-strict mode. Reports retain every available run status and summarize each part
separately. A partial report does not imply successful completion.

## Evaluate

`evaluate EXPERIMENT.yaml` reads runs.output and recomputes all treatment-pair metrics. It
replaces evaluation outputs and leaves saved runs and preparation intact. No dataset scenarios or
frames, model inference, simulator connection or attack artifacts are needed. Missing saved
preparation/results produces a clear error. Corrupt run logs are recorded as failed; other logs
continue. Obsolete treatment directories are reported and skipped.

Metric types/options can change before reevaluation. Known configuration errors are rejected before
evaluation. A metric-code failure is an evaluation failure: report.json and runs.csv still record
run states and the metric error. Terminal output includes one treatment table per part and metric class, plus notes and errors.
Headers include units; numeric cells use four decimal places, mean ± population standard deviation
and n = k (standard deviation is omitted for n < 2). Boolean summaries show a true fraction and N of K groups. JSON/CSV retain full precision.
All metric fields are shown; pairwise differences are retained in evaluation/report.json and evaluation_report.html.
Both run and evaluate print the offline_report.html and evaluation_report.html paths.
Evaluate rebuilds both HTML reports from saved output.

## NRE

`nre serve CONFIG --gpu N [--name NAME] [--image IMAGE] [--cache-size 2] [--timeout 600] [--dry-run]`
loads configuration without validating the experiment environment, starts Docker and waits for
"Serving on". It requires NuRecDataset and NuRecRenderer with a local endpoint: 127.0.0.1,
localhost, 0.0.0.0 or ::1. Dataset archive paths are resolved and mounted separately read-only;
duplicate archive filenames are a configuration error.

```bash
avsectester nre serve configs/nurec_stop_sign_patch_2601.yaml --gpu 0
avsectester nre serve configs/nurec_stop_sign_patch_2601.yaml --gpu 0 --dry-run
avsectester nre stop configs/nurec_stop_sign_patch_2601.yaml
```

`--gpu` is the host GPU selected by Docker's `--gpus device=N`, independently of top-level
YAML device. The default image is nvcr.io/nvidia/nre/nre-ga:26.04 and the default container name
is avsectester-nre-<port>. The probe directory is created before launch; its server mount uses
server_probe_directory, or the same absolute client path when no server path is configured.

`--dry-run` only prints the shell-quoted command: it creates no directories, starts no process and
does not require Docker on PATH. A normal launch without Docker reports an error and suggests
--dry-run. A pre-existing container with the chosen name is left untouched and reported as an error.
On successful startup the command prints endpoint, GPU, scene count, container name and a stop
command. On early exit or timeout it prints the last 50 lines of logs and stops/removes the
container it started. Containers do not use --rm, allowing startup diagnostics to be read first.

`nre stop CONFIG [--name NAME]` only loads configuration and derives the same default name from
the renderer endpoint. It does not enumerate datasets or build a serve command. It runs docker stop followed by
docker rm, and exits 1 if the container does not exist. Invalid configuration exits 2; Docker
startup, stop and timeout errors exit 1. Check nvidia-smi before selecting an available card and
stop the service as soon as it is no longer needed.

## Interruptions

Ctrl-C prints an interruption message, cleans up owned resources and exits with code 130. Selection
and splitting are checkpointed before offline optimization; a preparation interruption after that
point retains preparation-progress.json and artifacts of completed attacks. A complete prepared.json
is written only after all offline phases succeed. An interruption during selection may precede any
checkpoint. Resume with the same configuration and --resume.

Interrupted execution logs have cancelled status; missing expected logs are incomplete. A later
--resume preserves matching completed runs and reruns other expected runs individually. An
interruption during evaluation does not restart evaluation; some report outputs may already be
replaced or incomplete. Re-run evaluate to regenerate them without simulation.

## Output layout

```text
<output>/
  preparation-progress.json
  prepared.json
  attacks/<attack-name>/metadata.json
  attacks/<attack-name>/artifacts.pkl
  attacks/<attack-name>/manifest.json
  offline/<attack-name>/report.json
  offline/<attack-name>/<table>.csv
  offline/<attack-name>/<image-or-series>.png
  offline/<attack-name>/files/
  offline_report.html
  offline-incomplete/<attack-name>-<UTC timestamp>/
  maps/<scenario-id>.json
  runs/segment-<NNNNN>/seed-<S>/<treatment>/
  evaluation/report.json
  evaluation_report.html
  evaluation/runs.csv
  evaluation/plots/
```

Artifact filenames above describe the default trusted-pickle format; attacks may override it. Each
run directory is a [RunLog](runtime.md#runlog). Segment numbers are global unit indexes. The old
attack-prefixed layout, including v5/v6 outputs, cannot be loaded or evaluated with this version. Metrics and all-pair summaries are described in
[Evaluation](../guides/evaluation.md).

## Exit codes

| Code | Meaning |
|---|---|
| 0 | All runs completed and evaluation succeeded without selection errors |
| 1 | Selection, execution, result reading or metric evaluation failed |
| 2 | Known invalid configuration, output conflict or incompatible saved preparation |
| 130 | Keyboard interruption; available partial output is retained |

Attack impact is assessed through metrics; successful execution is not an attack-impact
claim.

## Supplied configurations

See [Quickstart](../getting-started/quickstart.md#supplied-configurations) and
[Configuration](configuration.md).
