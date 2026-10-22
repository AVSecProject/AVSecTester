# Experiments

> **Audience:** Experiment authors.
>
> **Outcome:** Prepare attacks once and compare isolated treatments.
>
> **Prerequisites:** [Architecture](../concepts/architecture.md).

## Prepare and execute attacks

```python
from avsectester.experiment import Experiment
from avsectester.evaluation.report import evaluate

# Requires the configured dataset, simulator, models and device.
experiment = Experiment()
prepared = experiment.prepare("configs/carla_phantom.yaml")
result = experiment.execute(prepared)
report = evaluate(result, prepared.configuration)
print(report.format_summary())
```

`prepare` opens the closed-loop services, calls each attack's metabuilder once per scene, validates
segment structure and runs each attack's offline phase once over all its metadata. Preparation
returns metadata and artifacts without live simulator or model history. An experiment accepts
exactly one attack; multiple attacks sharing metadata are future work. To combine actions, implement
one composite Attack.

Preparation uses runs.seeds[0]. OfflineContext.environment merges runs.environment with the attack
environment. The metabuilder requests its environment explicitly; it does not automatically inherit
the base environment. Its selected metadata is reused
for every execution seed. The framework does not interpret actor IDs in custom metadata; each
attack checks actors when it uses them and reports a missing actor as a run failure.

`execute` iterates attack, segment, seed and treatment groups. Each run creates a simulator runtime,
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

Each comparison group has one attack's selected segment and one seed. Treatments share its origin,
decision length and initialization tick budget. All treatments use the experiment's single attack
metadata. They differ only in whether attack_name enables the attack and in world/sensor environment
settings. Environment-only treatments are valid. The same configured hooks run for every treatment.
Every treatment pair is compared in configuration order; there is no baseline setting.

Preparation and execution use the same deployment budget: the largest hook/attack setup requirement
across this attack's selected segments, seeds and treatments. Clean runs execute the same tick
budget without the attack. Initialization records actual state; there is no replay error threshold
or automatic retry/reselection.

**Not yet implemented.** Defense treatments and attack-plus-defense comparisons. See
[TODO](../development/TODO.md#defenses).

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

## Results and failure handling

`ExperimentResult.runs` contains TreatmentResult records, including failed runs. Each identifies
attack_name, segment, treatment, seed, log and error. `groups` exposes grouped logs, `failures`
exposes non-completed runs, and `selection_errors` distinguishes metabuilder errors from ordinary
no-match decisions.

By default, selection errors and run failures are recorded while remaining work continues.
Setting runs.strict to true stops at the first error. Metrics consume completed logs and report
failed-run counts separately. A failure must not become a zero-valued successful observation.

`ExperimentResult.load(output, configuration)` reads only saved preparation summaries and run logs
for independent analysis. It never enumerates or loads the dataset, and does not load artifacts.
Unreadable logs are failed runs; obsolete treatment directories are reported and skipped.
`avsectester evaluate` uses this path after simulation closes.

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
