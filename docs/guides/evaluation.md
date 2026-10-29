# Evaluate experiments

> **Audience:** Researchers analyzing recorded experiments.
>
> **Outcome:** Measure treatments and comparison groups without running a simulator.
>
> **Prerequisites:** [Experiments](experiments.md).

## Saved logs are the input

Evaluation reads saved preparation, maps and `RunLog` records after execution. It does not read
dataset frames, load deployment artifacts or run model inference. Missing evidence cannot be
reconstructed. Failed or unreadable runs keep their failure status. Unavailable measurements
carry reasons rather than becoming zero.

`ExperimentResult.load` restores the saved evidence. Report rebuilding and skipped-run notices
follow the [CLI evaluate command](../reference/cli.md#evaluate).

## Comparison groups

Every unattacked treatment runs once per unit and seed. An attacked treatment runs only in units
selected by its attack. All applicable treatments in a group are compared pairwise.

The [architecture terms](../concepts/architecture.md#terms) define units and comparison groups.
`RunGroup` retains all supplied logs in their order, including failed, cancelled and incomplete
logs, plus optional errors, part, treatment attacks and a scenario map.

`evaluate_groups` passes only completed logs to the `ComparisonGroup` supplied to each metric. All
treatment pairs are considered in supplied order. For treatments A then B, single-run metric
differences are B − A. Boolean values are treated as 0 or 1. An unfinished pair or unavailable
value produces a reason, not a zero difference.

`PairMetric` receives a `TreatmentPair` directly. `TrajectoryDeviation` measures symmetric
trajectory distance. `StopImpact` evaluates both A-to-B and B-to-A stopping effects. These pair
measurements do not receive another difference operation.

## Built-in metrics

| `Metric` | Measurements |
|---|---|
| `DrivingMetrics` | Physical speed, executed actuator braking, stopping, actor distance and geometric collision |
| `TrajectoryDeviation` | Ego-position deviation at matched decision indices |
| `DetectionMetrics` | Thresholded detection matching, precision, recall and F1 |
| `StopImpact` | Stopping and braking comparisons in both directions |
| `StopLineMetrics` | Physical stopping, crossing and signed distance relative to the selected map stop line |
| `ReasoningPhraseRate` | Case-insensitive phrase mentions in recorded reasoning text |

The [metric API](../reference/python-api.md#avsectesterevaluationmetric) defines constructor
parameters and unavailable conditions. The default set and YAML validation are described in
[evaluation configuration](../reference/configuration.md#evaluation).

`DrivingMetrics` uses executed actuator commands for `brake_steps`. Trajectory commands do not
provide that evidence. Collision is bird's-eye oriented-box overlap, not a native
collision-sensor event. `TrajectoryDeviation` compares true ego positions at equal decision
indices, without time or route alignment.

**Not yet implemented.** Matched-window driving-impact measures with prior-motion and duration
requirements. See [Driving-impact metrics](../development/TODO.md#driving-impact-metrics).

`DetectionMetrics` matches detections to true actors in the capture-time ego frame, using sensor
mounting geometry independently of model localization. Its range filtering is horizontal.
Category matching follows `category_map` when configured. The thresholded detections do not
define TN or AP/mAP.

`StopLineMetrics` uses `segment.user_data["stop_line_id"]` to identify a stop line in the
saved map and assumes the run starts before it. It uses the initial physical state and each
decision's physical result. Distances are horizontal and positive before the first crossing,
then negative. Crossing speed is interpolated between adjacent samples. Three distances
distinguish stopping and physical braking:

-  `distance_to_line_at_stop` uses the first sample below `stop_speed`, also used by
  `time_to_stop`. It is unavailable if the ego never stops.
-  `distance_to_line_at_min_speed` uses the first minimum-speed sample before crossing, or the
  whole run when no crossing occurs.
-  `distance_to_line_at_brake_onset` uses the start of the first adjacent-sample interval whose
  speed decrease divided by elapsed time reaches `brake_deceleration_mps2`. It is unavailable
  with fewer than two samples or no qualifying interval.

Brake onset uses physical speed rather than an actuator command. A stop after crossing has
negative `distance_to_line_at_stop`. Missing line, map or physical evidence produces unavailable
measurements. `Metric.describe` supplies field definitions to the reports. The
[metric API](../reference/python-api.md#avsectesterevaluationmetric) defines
parameter constraints.

`ReasoningPhraseRate` reads native component attributes or mapping fields. Its rate uses all
decisions as the denominator and is unavailable if any decision lacks text.

## Reports and statistics

Boolean values display as `yes` or `no`, floats with four decimal places, and unavailable or
nonfinite values as an em dash (—). Numeric summaries use `mean ± std` and `n = k`, omitting the
standard deviation for one observation. Boolean summaries show the true fraction and
`x of K groups`. HTML pages contain inline styling without scripts or external resources. Saved
images and CSV files are linked relative to the page. JSON and CSV retain full precision.

Treatment summaries group measurements by part and treatment. Pair summaries group by part and
treatment pair. Numeric summaries use the mean and population standard deviation (ddof=0), with
a separate available-group count for each field. Boolean summaries use the true fraction.
`Unavailable` values are excluded from that field's denominator.

Paired differences are computed inside each group before aggregation. `PairMetric` measurements
are summarized separately. Statistics do not infer an attack-success label. See the
[example interpretation rule](../examples/README.md#attack-impact).

The HTML report presents experiment setup, metric definitions, run status, per-part results,
paired differences and group details. Group details retain unavailable reasons and saved plots.
The command line shows per-part metric tables, notes and errors. Paired tables are in the HTML
and structured report. `format_summary` returns a string, so Python callers print it explicitly.

Output locations follow the [CLI reference](../reference/cli.md#output-layout).
Offline phase results are separate. [Optimization](optimization.md) describes their report
interface.

Saved metric specifications include `type`, `options`, `label`, `kind`, `summary` and `fields`.

## Custom metrics

`Metric.evaluate` returns a mapping from treatment names to scalar field mappings. Scalars can
be numbers, booleans or `UnavailableMetricValue` with a reason and a number/boolean type.
`PairMetric` instead returns one field mapping for its `TreatmentPair`. `MetricScalar` is a
scalar or unavailable value. `FieldMeasurements` maps field names to those values, and
`TreatmentMeasurements` maps treatment names to field mappings. Return missing evidence
explicitly rather than keeping mutable per-run state. See the
[metric types](../reference/python-api.md#avsectesterevaluationmetric).

The default `Metric.describe` uses the first paragraph of the class docstring as its summary and
supplies no field descriptions.

Describe each field's meaning, unit and unavailable conditions through `Metric.describe`.
Reports retain the complete metric label, so dotted field names remain attached to the correct
metric. Description failures are recorded as metric errors. Absent descriptions display "No
description provided".

<!-- runnable -->
```python
from avsectester.evaluation import Metric, MetricDescription, MetricField

class DecisionCount(Metric):
    def describe(self):
        return MetricDescription(
            "Number of recorded decisions in a run.",
            {"decision_count": MetricField("Decisions saved in the run log.", "decision steps")},
        )

    def evaluate(self, group):
        return {name: {"decision_count": len(log.steps)}
                for name, log in group.runs.items()}
```

Configure the class with a path such as `my_project.metrics:DecisionCount`. A metric reads
recorded evidence and does not require simulator startup, dataset access or model inference.

## Visualize saved observations

Retain the needed sensor payloads before execution using
[recording configuration](../reference/configuration.md#recording). `camera_view` and
`lidar_bev` read saved inputs. `labels_view` projects recorded true actors, while
`detections_view` projects recorded detections. `sensor_view` reconstructs only logged geometry.

The following example requires retained camera payloads in a saved run:

```python
from avsectester.runlog import RunLog
from avsectester.evaluation.visualization import camera_view, record_run, save_gif

log = RunLog.read("results/experiment/runs/segment-00000/seed-0/clean")
record_run(log, "export/frames", visualize=camera_view)
save_gif(log, "export/drive.gif", fps=5)
```

`record_run` exports frames, filmstrip creates a grid and `save_sequence` exports images and
animation. These functions consume saved records and do not drive a world. Missing input
produces no view. Overlays use only the recorded camera geometry. See the
[visualization API](../reference/python-api.md#avsectesterevaluationvisualization)
for constraints.

## Evaluate existing `RunLog` objects

Use `RunGroup` and `evaluate_groups` directly when logs are already in memory. They require
neither `Experiment` nor a dataset. Save the returned report when persistent files are needed.

<!-- runnable -->
```python
import numpy as np
from avsectester.attack import AttackSegment
from avsectester.datasets import FrameReference
from avsectester.data_types import ActorState, Control, WorldSnapshot
from avsectester.evaluation import DrivingMetrics, RunGroup, evaluate_groups
from avsectester.runlog import RunLog, StepRecord

ego = ActorState(np.eye(4), (1, 0, 0), "vehicle", (4, 2, 1.5))
state = WorldSnapshot(0.1, 1, ego_speed=1.0, ego=ego)
log = RunLog(steps=[StepRecord(0, physical_pre=state, physical_post=state,
                             executed_command=Control())], status="completed")
start = FrameReference("road", 0, 0, 0.0)
stop = FrameReference("road", 1, 1, 0.1)
group = RunGroup(0, AttackSegment("road", start, stop), 0, {"clean": log},
                 treatment_attacks={"clean": None})
report = evaluate_groups((group,), (DrivingMetrics(),), notes=("Existing in-memory log",))
assert len(report.groups) == 1
assert report.notes == ["Existing in-memory log"]
print(report.format_summary())
```

## Next steps

Read [Runtime](../reference/runtime.md#runlog),
[the evaluation API](../reference/python-api.md#avsectesterevaluation)
and [Configuration](../reference/configuration.md#evaluation).
