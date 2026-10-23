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
avsectester evaluate EXPERIMENT.yaml
```

`python -m avsectester.cli` provides the same commands from a source checkout.

## Run

`run EXPERIMENT.yaml [--prepared DIRECTORY]` loads configuration, selects metadata, prepares
artifacts, executes treatments and evaluates their logs. With --prepared, it validates and loads an
existing preparation instead of selecting and optimizing again. Prepared artifacts must be fully
loaded before execution. All attack inputs and returned artifacts remain explicit.

The configuration must provide runs.output and at least one attack. A zero-attack configuration
is an error; use a selection-only Attack for environment comparisons or a composite Attack for
combined actions. Treatments vary attack_name and environment; each runs on the segments of its
attack, and treatments without attack_name run on every attack's segments.

Configuration controls models, devices, decision limits, treatments, recording and evaluation. The
CLI does not override them. Run settings are nested under runs. Existing runs.output/runs is
rejected rather than overwritten. Choose a fresh runs.output when reusing preparation. Reuse
requires the same dataset, simulator and attack inputs, the same first seed and the same offline
environments; other run settings and metrics can change. See
[Saved preparation compatibility](configuration.md#saved-preparation-compatibility).

The terminal prints the output directory, selection/execution errors including exception types, and
failed runs in non-strict mode. Reports contain every available run status. A partial report uses
only preparation saved in this invocation; stale preparation from another run is never used.

## Evaluate

`evaluate EXPERIMENT.yaml` reads runs.output and recomputes all treatment-pair metrics. It
replaces evaluation outputs and leaves saved runs and preparation intact. No dataset scenarios or
frames, model inference, simulator connection or attack artifacts are needed. Missing saved
preparation/results produces a clear error. Corrupt run logs are recorded as failed; other logs
continue. Obsolete treatment directories are reported and skipped.

Metric types/options can change before reevaluation. Known configuration errors are rejected before
evaluation. A metric-code failure is an evaluation failure: report.json and runs.csv still record
run states and the metric error. Terminal output includes summaries, difference matrices and notes.

## Interruptions

Ctrl-C prints an interruption message, cleans up owned resources and exits with code 130. During
execution, the CLI attempts a partial report only when this invocation saved preparation. An
interruption during preparation does not imply that an output directory exists. An interruption
during evaluation does not restart evaluation; some evaluation outputs may already be replaced or
incomplete. Interrupted logs have cancelled status; missing expected logs are incomplete. A partial
report never implies successful completion.

## Output layout

```text
<output>/
  prepared.json
  attacks/<attack-name>/metadata.json
  attacks/<attack-name>/artifacts.pkl
  attacks/<attack-name>/manifest.json
  maps/<scenario-id>.json
  runs/<attack-name>/segment-<NNNNN>/seed-<S>/<treatment>/
  evaluation/report.json
  evaluation/runs.csv
  evaluation/plots/
```

Artifact filenames above describe the default trusted-pickle format; attacks may override it. Each
run directory is a [RunLog](runtime.md#runlog). Metrics and all-pair summaries are described in
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
