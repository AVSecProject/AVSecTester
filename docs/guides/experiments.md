# Experiments

> **Audience:** Experiment authors.
>
> **Outcome:** Prepare artifacts, execute treatments and resume saved work.
>
> **Prerequisites:** [Architecture](../concepts/architecture.md).

## Prepare and execute attacks

`Experiment` accepts a typed configuration, mapping or YAML path. `prepare` selects or loads
segments and runs each configured attack's offline phase. `execute` runs the selected treatment
groups; `evaluate` measures their recorded evidence. `PreparedExperiment` contains metadata,
units, artifacts and offline reports, without live simulator or `AVStack` history.

This complete example uses the repository's memory components and a temporary output directory.
It needs no simulator service, model weights or GPU.

<!-- runnable -->
```python
from pathlib import Path
from tempfile import TemporaryDirectory

from avsectester.experiment import Experiment, evaluate, load_configuration

with TemporaryDirectory() as directory:
    configuration = load_configuration({
        "dataset": {"type": "tests.fakes:MemoryDataset"},
        "simulator": {"type": "tests.fakes:MemorySimulator"},
        "avstack": {"type": "tests.fakes:ConstantAVStack"},
        "attacks": {"probe": {"type": "tests.fakes:MinimalAttack"}},
        "runs": {"output": str(Path(directory) / "run")},
    })
    Experiment.validate(configuration)
    prepared = Experiment.prepare(configuration)
    result = Experiment.execute(prepared)
    report = evaluate(result, configuration)
    assert not result.failures
    assert not report.metric_errors
    print(report.format_summary())
```

See [automatic local services](../reference/cli.md#automatic-local-services) for CLI and Python use.

## Preparation

Preparation selects every attack before running any offline phase. It checks selected
boundaries, applies the common deployment tick budget and builds shared segment units. A
supplied metadata file replaces selection. `Attack.validate_segment` checks selected and
loaded segments.

After `build_metadata` returns, preparation drops segments that fail `can_start_at` or whose
`validate_start` raises `UnsupportedStartError`. The experiment-wide budget recheck drops
segments that then fail `can_start_at`. Each drop records its reason; see
[deployment tick budgets](attacks.md#online-phase).

Preparation uses the first configured run seed and reuses its metadata for all execution seeds.
Its environment rules are defined in [Environments](environments.md): an offline phase receives
the merged experiment and attack environment, while selection requests an
environment explicitly.

`PreparedExperiment` retains all units, including those excluded from execution. Unit identity,
the split schema and deterministic allocation are defined in
[Data split](../reference/configuration.md#data-split).

**Not yet implemented.** Sharing unattacked runs across units with different `user_data`; see
[shared runs across different user data](../development/TODO.md#shared-runs-across-different-user-data).

## Input validation

`validate` and `prepare` reject existing preparation output without `resume=True`. Environment
checks run for fresh and resumed preparations. A fresh loaded metadata file is validated once
per call before offline work.

`Experiment.validate` checks inputs that can be checked without a `SimulatorRuntime` or service
connection. Loaded metadata follows
[Save and reuse selected metadata](#save-and-reuse-selected-metadata).

For resume with saved preparation or progress, validation checks frames and compatibility,
skipping eligibility and `validate_start`. The CLI skips this preflight when it
has already loaded preparation for `--prepared` or a completed preparation resume.
[CLI exit codes](../reference/cli.md#exit-codes) distinguish input failures from
execution failures.

## Treatments and common origins

The [architecture](../concepts/architecture.md#terms) defines units and comparison groups.
Each enabled treatment starts from the group's common origin; membership follows
[Evaluation](evaluation.md#comparison-groups). The same configured hooks apply to every treatment.

Execution follows the [deployment tick budget](attacks.md#online-phase).
Unattacked runs do not create an `AttackRuntime`. Each run creates fresh simulator and `AVStack`
state, with cleanup on success, failure or interruption. The
[treatment schema](../reference/configuration.md#run-settings) defines how to enable attacks and
environment overrides.

## Results and failures

`ExperimentResult.runs` holds every treatment result; `failures` lists failed or incomplete
runs; `selection_errors` lists selection errors; the `runs.strict` setting raises the first error.
See [Run settings](../reference/configuration.md#run-settings).

## Save and reuse selected metadata

Loading validates dataset type, scenario and frame membership, decision windows, each attack's
`validate_segment`, the shared deployment tick budget and simulator start requirements. An
invalid segment fails with its index and reason.

Metadata files contain segments, independently of offline artifacts or run results. Use
`metadata.save` to export all units after selection and deployment checks, before the offline
phases. Existing identical content is accepted; different content is an error.

Use `metadata.load` to reuse those segments with other attack settings or driving systems. Every
attack receives the same supplied segments, selection is skipped and split still applies. The
offline phases run for the configured attacks; loading segments does not reuse their artifacts.

Python callers can generate segments with their own criteria using `save_metadata` and
`load_metadata`. The file schema, duplicate checks, path rules and content-hash compatibility
are defined in [Metadata](../reference/configuration.md#metadata). Resuming preparation and
reusing a prepared experiment do not perform selection or write `metadata.save` again.

## Save and reload preparation

Preparation checkpoints selected metadata and units before the offline phases. Each completed
phase saves its artifacts and report before its name is marked complete. The complete prepared
manifest is written after every configured attack has finished.

`PreparedExperiment.save` and load persist and restore artifacts together with metadata and
reports. Loading requires an explicit compatible configuration, does not repeat selection and
does not run the offline phases. Only load trusted preparation files; artifact encoding is
controlled by the attack. See
[compatibility rules](../reference/configuration.md#saved-preparation-compatibility) and
[the preparation API](../reference/python-api.md#avsectesterexperimentmodels).

## Resume interrupted work

Prepare with `resume=True` to reuse saved selection and completed offline phases. Execute with
`resume=True` to retain completed runs whose expected metadata matches; missing, failed or
mismatched runs are retried individually. An already complete group needs no new initialization.

Execution resume compares `attack_name`, `segment_index`, `part`, the complete `segment`,
`treatment`, `seed`, `environment`, `decision_steps`, `selected_decision_steps`
and `deployment_ticks`.

An interrupted offline phase leaves progress and a partial failed report. Resuming archives that
attack's report directory before restarting its offline phase. The
[CLI resume workflow](../reference/cli.md#preparation-reuse) defines command options and output
handling; [Offline reports](optimization.md) explains phase output.

## Saved maps

Execution reads each selected scene's semantic map once and retains it in
`ExperimentResult.scenario_maps`, including `Unavailable` with its reason. Persistent results
save those maps with the run evidence. `ExperimentResult.load` restores them for
`ComparisonGroup.scenario_map`; evaluation does not call a dataset to fill in missing maps.

A saved result can therefore be evaluated without its dataset or metadata file. Configuration
compatibility still applies; see
[saved preparation compatibility](../reference/configuration.md#saved-preparation-compatibility).
The [CLI output layout](../reference/cli.md#output-layout) defines saved file locations.

## Reproducibility

Keep the configuration, selected metadata, artifacts and `RunLog` manifests together. Fresh
runtime and hook state isolate treatments; a common seed does not guarantee bitwise-identical
native simulator or model kernels. Sensor noise uses the
[environment seed rules](environments.md#randomness).

## Environment-only robustness experiments

Use a selection-only `Attack` and configure treatments with environment overrides. Its offline
and online phases may both be empty operations:

<!-- runnable -->
```python
from avsectester.attack import Attack
from avsectester.attack.segments import split_segments

class SelectionOnlyAttack(Attack):
    def __init__(self, decision_steps):
        self.decision_steps = decision_steps

    def build_metadata(self, scenario):
        return split_segments(scenario.frames, lambda frame: True,
                              decision_steps=self.decision_steps)
```

See [Environments](environments.md) for merge rules and supported conditions.

## Next steps

Read [Attacks](attacks.md), [Environments](environments.md), [Evaluation](evaluation.md)
and [CLI](../reference/cli.md).
