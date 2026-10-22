# Evaluate experiments

> **Audience:** Researchers analyzing experiments.
>
> **Outcome:** Calculate metrics and treatment comparisons from recorded runs.
>
> **Prerequisites:** [Experiments](experiments.md).

## Saved logs are the input

Evaluation reads saved preparation and RunLog records after simulators close. It does not enumerate
scenarios, read dataset frames, load attack artifacts or run model inference. Missing observations
cannot be reconstructed. Unavailable metric values carry a reason; failed runs remain failures.

ComparisonGroup.scenario_map contains the map saved for that scene, or Unavailable with its reason.
Custom metrics can use it without reopening the dataset. ExperimentResult.load restores maps from
the output's maps/ directory alongside run records; see [Semantic maps](scenarios.md#semantic-maps).

```python
from avsectester.experiment import ExperimentResult, load_configuration
from avsectester.evaluation.report import evaluate

# Requires saved preparation and run logs under configuration.runs.output.
configuration = load_configuration("configs/carla_phantom.yaml")
result = ExperimentResult.load(configuration.runs.output, configuration)
report = evaluate(result, configuration)
print(report.format_summary())
```

`avsectester evaluate EXPERIMENT.yaml` performs the same evaluation and replaces evaluation outputs.
Unreadable runs are recorded as failed with their read error; other runs can still be evaluated.
Directories belonging to treatments removed from the configuration are reported and skipped.

## Comparison groups

One experiment contains exactly one attack. Its selected metadata is shared by every treatment. A
comparison group is one selected segment and one seed. Treatments can enable that attack and change
world or sensor environments. The framework does not compare independently selected attacks. To
apply several attack actions together, write one composite Attack.

**Not yet implemented.** Comparing multiple attacks in one experiment on shared metadata. See
[TODO](../development/TODO.md#multiple-attacks-on-shared-metadata).

There is no baseline setting. Within a group, treatments are paired in configuration order. For A
before B, every single-run metric difference is B minus A. Boolean values also have numeric
differences, such as 1 minus 0. A pair with an unfinished treatment has unavailable results and an
explanation; it does not become a zero difference.

TrajectoryDeviation computes one symmetric distance result per pair. StopImpact evaluates both
A-to-B and B-to-A directions. These are already comparison metrics and have no additional
difference.

## Built-in metrics

| Metric | Parameters | Evidence and result |
|---|---|---|
| DrivingMetrics | stop_speed=0.5 | Speed, braking, stopping, actor distances and geometric collision |
| TrajectoryDeviation | — | Mean, maximum and final matched-decision ego position distance |
| DetectionMetrics | match_distance=2.0, max_range=50.0, category_map=— | TP, FP, FN, precision, recall and F1 |
| StopImpact | stop_speed=0.5, reference_speed=1.0, min_overdrive_steps=3 | Stopping and braking comparisons in both directions |

The default metric set is chosen once for the entire experiment: DrivingMetrics and
TrajectoryDeviation, plus DetectionMetrics when all completed logs provide detection evidence.
StopImpact is available when explicitly configured. An experiment with no completed logs reports
missing run data rather than claiming its AVStack cannot provide detections. Report notes are also
printed in the terminal.

DrivingMetrics counts executed braking when brake is positive and throttle is zero. Collision means
bird's-eye oriented-box overlap, not a native collision-sensor event. Zero extents are valid known
geometry; absent geometry is unavailable evidence. Trajectory commands alone do not prove braking.
Units are m/s for speed and metres for distance. There is no universal attack-success threshold.

TrajectoryDeviation compares true ego positions at equal decision indices. It does not align
trajectories in time or measure road-relative error. StopImpact compares stopping in both
directions; duration-aware stop events remain
[planned](../development/TODO.md#driving-impact-metrics).

DetectionMetrics transforms true actors with `StepRecord.physical_pre.ego` into the capture-time
true ego frame. Detections use sensor mounting extrinsics, independently of model localization. Both
detections and truth are filtered by the same bird's-eye range, then matched once within
match_distance. Set max_range to null to disable range filtering. With category_map configured, a
detection without a category cannot match a class. Zero denominators and missing detection evidence
produce unavailable values with reasons. TN and AP/mAP are not defined by these thresholded outputs.

## Configure metrics

```yaml
evaluation:
  metrics:
    - {type: DrivingMetrics, options: {stop_speed: 0.5}}
    - {type: TrajectoryDeviation}
    - {type: DetectionMetrics, options: {max_range: null}}
    - {type: StopImpact}
```

Types and options are validated at configuration loading. Unknown keys and duplicate metric types
are rejected. DetectionMetrics is rejected early when the configured AVStack type explicitly
declares it cannot provide detections. A metric implementation failure is an evaluation error, not a
configuration error: the report and runs.csv still preserve run status and the metric error.

## Reports

`evaluate(result, configuration)` writes these files under `output/evaluation/`:

| Output | Content |
|---|---|
| report.json | Metric types/options, groups, treatment values, pair values, reasons and errors |
| runs.csv | One row per run, with identity/status and metric-prefixed value columns |
| plots/ | Per-group speed and braking by decision index |

Treatment summaries show completed group counts, total groups, failures, numeric mean/population
standard deviation (ddof=0), and boolean true proportions. Missing evidence is excluded from numeric
denominators. Boolean fields retain proportion formatting even when every value is unavailable. Pair
summaries include single-run differences and pair metrics, grouped by ordered treatment pair. The
terminal table shows available primary fields: final_speed, brake_steps, stopped, collision,
precision and recall. It also prints major metric difference matrices, notes and output location.
The JSON and CSV retain all metric fields.

## Add a metric

Implement Metric.evaluate(group), returning treatment names mapped to field/value mappings. Values
can be numbers, booleans or UnavailableMetricValue(reason). PairMetric evaluates a TreatmentPair
instead. Keep missing-value reasons in the returned value rather than mutable metric instance state.

<!-- runnable -->
```python
from avsectester.evaluation.metric import Metric

class DecisionCount(Metric):
    def evaluate(self, group):
        return {name: {"decision_count": len(log.steps)}
                for name, log in group.runs.items()}
```

Select the class with `type: my_project.metrics:DecisionCount` and constructor options. No metric
should require simulator startup, dataset access or additional model inference.

## Visualize saved observations

Retain sensor payloads before execution. camera_view and lidar_bev read saved inputs; labels_view
projects true actors from StepRecord.physical_pre; detections_view projects recorded detections.
StepRecord.sensor_geometry retains typed camera geometry without copying full observations to obtain
it. Missing images or calibration are errors rather than guessed geometry.

```python
from avsectester.runlog import RunLog
from avsectester.evaluation.visualization import record_run, save_gif, camera_view

# Requires a saved run with retained camera payloads and geometry.
log = RunLog.read("results/experiment/runs/attack/segment-00000/seed-0/clean")
record_run(log, "export/frames", visualize=camera_view)
save_gif(log, "export/drive.gif", fps=5)
```

record_run exports saved frames, filmstrip creates a grid and save_sequence exports both images and
animation. Caller-supplied view functions receive StepRecord. None of these functions drives a
world.

## Offline evaluation example

<!-- runnable -->
```python
import numpy as np
from avsectester.attack import AttackSegment
from avsectester.datasets import ActorState, FrameReference
from avsectester.closed_loop.datatypes import Control, WorldSnapshot
from avsectester.evaluation.metric import ComparisonGroup, DrivingMetrics
from avsectester.runlog import RunLog, StepRecord

pose = np.eye(4)
ego = ActorState(pose, (1, 0, 0), "vehicle", (4, 2, 1.5))
state = WorldSnapshot(0.1, 1, ego_speed=1.0, ego=ego)
log = RunLog(steps=[StepRecord(0, physical_pre=state, physical_post=state,
                             executed_command=Control())], status="completed")
start = FrameReference("road", 0, 0, 0.0)
stop = FrameReference("road", 1, 1, 0.1)
group = ComparisonGroup("sample", AttackSegment("road", start, stop), 0, {"clean": log})
values = DrivingMetrics().evaluate(group)
assert values["clean"]["final_speed"] == 1.0
```

## Next steps

Read [Runtime](../reference/runtime.md#runlog), [Python
API](../reference/python-api.md#evaluation-api) and
[Configuration](../reference/configuration.md#evaluation).
