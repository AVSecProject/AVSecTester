# Experiments

> **Audience:** Experiment authors.
>
> **Outcome:** Prepare attacks once and compare isolated treatments.
>
> **Prerequisites:** [Architecture](../concepts/architecture.md).

## Prepare and execute attacks

```python
from avsectester.experiment import Experiment, evaluate

# Requires the configured dataset, simulator, models and device.
experiment = Experiment()
prepared = experiment.prepare("configs/carla_phantom.yaml")
result = experiment.execute(prepared)
report = evaluate(result, prepared.configuration)
print(report.format_summary())
```

`prepare` first selects every attack's metadata, validates segments against the experiment-wide
deployment budget and builds shared segment units. With split, it assigns these units to train,
validation and test before any offline phase. Each attack receives only its train segments as
OfflineContext.metadata and its validation segments as validation_metadata. Without split the
offline metadata contains all its selected segments. PreparedExperiment retains all metadata, units
explicit artifacts and offline_reports without live simulator or model history. To combine actions in one run,
implement one composite Attack.

Preparation uses runs.seeds[0]. OfflineContext.environment merges runs.environment with the attack
environment. The metabuilder requests its environment explicitly; it does not automatically inherit
the base environment. Its selected metadata is reused
for every execution seed. The framework does not interpret actor IDs in custom metadata; each
attack checks actors when it uses them and reports a missing actor as a run failure.

`execute` iterates unit and seed groups, restricted to split.execute when configured. Each run creates a simulator runtime,
AVStack and AttackRuntime. Cleanup closes owned resources on success and failure. Generated CARLA
frame data is cached across execution groups with seed, environment and deployment settings included
in its key.

## Treatments and common origins

```yaml
runs:
  treatments:
    - {name: clean}
    - {name: attacked, attack_name: phantom}
    - name: rain
      environment:
        world: {weather: SoftRainNoon}
```

A unit is one complete segment: scenario, start, stop and user_data. Identical segments selected by
different attacks share a unit and its global segment_index. A comparison group is that unit and
one seed. Treatments share its origin, decision length and initialization tick budget. Each
treatment without attack_name runs once; each attacked treatment runs when its attack selected
this unit, using that attack's artifacts. All treatment pairs, including different attacks, are
compared in configuration order. Environment-only treatments are valid; the configured hooks run
for every treatment. There is no baseline setting.

Preparation and execution use the same deployment budget: the largest hook/attack setup requirement
across all attacks, selected segments, seeds and treatments. Clean runs execute the same tick
budget without the attack. Initialization records actual state; there is no replay error threshold
or automatic retry/reselection.

**Not yet implemented.** Defense treatments and attack-plus-defense comparisons. See
[TODO](../development/TODO.md#defenses).

## Train, validation and test

Configure the optional top-level split with train_segments, validation_segments, seed and execute.
The framework orders units by dataset scenario, start/stop positions and configured attack order,
then assigns parts using a seeded NumPy permutation. The training and validation counts must leave
at least one test unit. execute defaults to [test]; omitting split executes all units. See
[Data split](../reference/configuration.md#data-split).

Units include user_data. Different attacker settings for the same window may therefore enter train
and test separately, and test images may have been seen during training. No conditional transfer or
cross-user_data sharing of un-attacked runs is implemented.

## Save and reload preparation

```python
from avsectester.experiment import Experiment, PreparedExperiment

# Requires an existing experiment configuration and saved preparation.
prepared = PreparedExperiment.load("configs/carla_phantom.yaml", "results/previous")
result = Experiment().execute(prepared)
```

Use `prepared.save(directory)` to persist metadata and explicit artifacts. The Attack controls
artifact encoding through `save_artifacts(artifacts, directory)` and `load_artifacts(directory)`;
the default format is trusted pickle with a content hash. Only load trusted preparations.

Loading requires an explicit compatible configuration and does not select or optimize again. See
[compatibility checks](../reference/configuration.md#saved-preparation-compatibility).

## Offline and evaluation reports

Each offline stage receives `OfflineContext.report`, an `OfflineReport` for its structured results.
It stores complete frozen ReportValue, ReportSeries, ReportTable and ReportImage entries and
can publish scalar values, numeric series, flat tables and RGB/RGBA images through
`set_value`, `set_series`, `set_table` and `set_image`. `report.directory` is
`runs.output/offline/<attack>/files` when an output is configured, otherwise None. Attacks can use
that directory for their own additional files; they do not need an output-directory parameter.

A saved preparation writes `offline/<attack>/report.json`, CSV tables and PNG images, with the
attack's additional files under `files/`. `offline_report.html` combines all attack reports,
including attacks without optimization. The HTML is rebuilt after each completed stage and when
saving a preparation. `PreparedExperiment.offline_reports` restores these reports on load;
saving into another directory copies their additional files along with artifacts. Older
preparations without offline reports load with empty reports.

Offline exceptions, including interruption, trigger a partial report save with status failed and an
HTML refresh before propagating. A reporting failure is attached as a note to the original exception
so the original offline error is preserved. Resuming an unfinished attack preserves its previous
`offline/<attack>/` directory under `offline-incomplete/<attack>-<UTC timestamp>/` before starting a
new report. The framework archives the whole report rather than requiring individual attacks to
manage old rollout directories.

Evaluation writes `evaluation/report.json`, `evaluation/runs.csv`, `evaluation/plots/` and the
root-level `evaluation_report.html`.
The offline and evaluation HTML files use saved data and relative image links, so the CLI
`evaluate` command can rebuild both without loading driving models or executing attacks.
The pages link to each other and to their structured data. All visible report text is English.
The console reports every metric field by part and metric label with units; pairwise differences
are available in the evaluation JSON and HTML. Displayed floats always use four decimal places, including integral-valued floats,
while JSON and CSV keep full precision. Offline train/validation counts come from saved units.

## Resume interrupted work

```python
from avsectester.experiment import Experiment

# Requires the original configuration and its output directory.
prepared = Experiment.prepare("configs/carla_phantom.yaml", resume=True)
result = Experiment.execute(prepared, resume=True)
```

Preparation writes each attack's metadata and preparation-progress.json immediately after selection
and splitting. After every completed offline phase it saves that attack's artifacts and report, then checkpoints
the completed attack names. prepared.json is written only after all attacks finish. With resume=True,
a complete preparation is loaded directly; otherwise the progress configuration and preparation
record are checked before restoring saved selection and units. Completed artifacts are loaded and
only remaining attacks run offline. Selection is not repeated on a partial resume. Incompatible
configuration raises ConfigurationMismatchError with its differences.

Execution preserves a completed run only when all expected metadata matches: segment_index, seed,
treatment, attack_name, part, segment, environment, decision_steps, selected_decision_steps and
deployment_ticks. Segment data is compared in its canonical serialized representation. A change in
the decision limit, treatment environment or deployment budget therefore reruns the affected run.
Otherwise it reads that log into the result without rewriting the manifest. A missing run is filled
in; an existing failed, incomplete, corrupt or mismatched run directory is removed individually and
rerun. A fully completed group requires no world/AVStack initialization.
The CLI equivalent is `avsectester run CONFIG --resume`. Without --resume, existing output is an
error; --resume cannot be combined with --prepared.

## Results and failure handling

`ExperimentResult.runs` contains TreatmentResult records, including failed runs. Each identifies
attack_name (None for an un-attacked treatment), segment_index, segment, part, treatment, seed, log
and error. `groups` is keyed by (segment_index, seed), `failures`
exposes non-completed runs, and `selection_errors` distinguishes metabuilder errors from ordinary
no-match decisions.

By default, selection errors and run failures are recorded while remaining work continues.
Setting runs.strict to true stops at the first error. Metrics consume completed logs and report
failed-run counts separately. A failure must not become a zero-valued successful observation.

`ExperimentResult.load(output, configuration)` reads only saved preparation summaries and run logs
for independent analysis. It never enumerates or loads the dataset, and does not load artifacts.
Unreadable logs are failed runs; obsolete treatment directories are reported and skipped.
`avsectester evaluate` uses this path after simulation closes. Runs use
runs/segment-xxxxx/seed-s/treatment/; historical attack-prefixed layouts such as v5/v6 results cannot
be loaded or evaluated with this version.

Execution also reads each scene with selected segments once through Dataset.scenario_map. The result
is retained in ExperimentResult.scenario_maps, including Unavailable and its reason. Persistent
experiments write maps/<scenario_id>.json using the prepared-metadata encoding. Loading results
restores these maps for ComparisonGroup.scenario_map; evaluation does not call the dataset for
missing maps.

## Reproducibility

Keep the experiment configuration, metadata, artifact hashes and RunLog manifests together.
Configured AVStack constructors receive their options and create fresh histories. Stage-hook
factories must return new HookSet and stateful handler instances per run. Sensor randomness uses the
run seed and absolute decision timeline position, so offline and online requests for the same frame
receive matching configured sensor noise.

Native simulators and model kernels may retain their own numerical nondeterminism. A common seed and
origin define the intended comparison; they do not prove bitwise identical native execution.

## Environment-only robustness experiments

Use a selection-only Attack whose default runtime applies no actions. Pair its selected segments
with clean and sensor/world-environment treatments:

<!-- runnable -->
```python
from avsectester.attack import Attack, AttackSegment
from avsectester.attack.segments import split_segments

class SelectionOnlyAttack(Attack):
    def __init__(self, decision_steps):
        self.decision_steps = decision_steps

    def build_metadata(self, scenario):
        return split_segments(scenario.frames, lambda frame: True,
                              decision_steps=self.decision_steps)
```

Configure it under `attacks`, then use runs.treatments with only environment overrides. The same
reporting machinery compares robustness across environments.

## Next steps

Read [Attacks](attacks.md), [Environments](environments.md), [Evaluation](evaluation.md) and
[CLI](../reference/cli.md).
