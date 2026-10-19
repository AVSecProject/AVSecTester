# Evaluate results

> **Audience:** Users who interpret experiment outputs.  
> **Outcome:** Read RunLog records and compatibility traces, compute the driving-impact metric,
> compare component outputs and run robustness evaluations.
> **Prerequisites:** [Run experiments](experiments.md).

## Run logs and saved results

`ClosedLoopRunner.run(...)` returns a `RunLog` containing one `StepRecord` per completed decision,
initialization/termination events, metadata and a final status. The records distinguish raw and
model-visible observations, physical pre/post-state, produced and executed commands, component
outputs and attack effects. The existing `run`, `run_logged` and visualization helpers use this
same loop; their `Trace` and `ComponentTrace` results are views of its captured records.

Configure a direct run with `RunRequest` and `RecordingPolicy` from
`avsectester.closed_loop.runner`. By default, the run retains its records in memory and records no
sensor payloads. Set `output_path` to an empty or new directory to save Parquet records and separate
array/image payload shards. Set `retain_steps=False` as well to stream records without keeping the
full run in memory. In that mode, returned `log.steps` and `log.events` are empty; analyze
`log.path`. The default `retain_steps=True` preserves in-memory access and compatibility traces.

This small example needs no simulator server, model or dataset:

<!-- runnable -->
```python
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from avsectester.closed_loop.avstack import AVStack
from avsectester.closed_loop.simulator import WorldBackend
from avsectester.closed_loop.runner import ClosedLoopRunner, RecordingPolicy, RunRequest
from avsectester.closed_loop.datatypes import Control, Observation
from avsectester.runlog import RunLog, read_table


class ExampleWorld(WorldBackend):
    def reset(self):
        self.frame = 0
        return self.capture()

    def capture(self):
        return Observation(self.frame * 0.1, self.frame,
                           sensor_data={"front": np.zeros((2, 3, 3), dtype=np.uint8)})

    def step(self, control):
        self.frame += 1
        return self.capture()


class Coast(AVStack):
    def __call__(self, observation):
        return Control()


with TemporaryDirectory() as directory:
    world = ExampleWorld()
    try:
        log = ClosedLoopRunner().run(world, Coast(), RunRequest(
            decision_steps=3,
            output_path=Path(directory) / "run",
            retain_steps=False,
            recording=RecordingPolicy(sensor_ids=("front",), every_n_steps=2),
        ))
    finally:
        world.close()  # The direct runner leaves caller-owned resources open.

    assert log.status == "completed" and log.steps == []
    table = read_table(log.path, columns=["step", "physical_post", "executed_command"])
    assert table.column("step").to_pylist() == [0, 1, 2]

    saved = RunLog.read(log.path, load_payloads=False)
    reference = saved.steps[0].model_observation.sensor_data["front"]
    assert reference["shape"] == [2, 3, 3] and reference["chunks"]
    assert saved.steps[1].model_observation.sensor_data == {}  # Payload sampling only.
```

`RecordingPolicy.sensor_ids` selects payloads. `representations` accepts `"raw"`, `"model_input"` or
both; its default is `("model_input",)`. `every_n_steps` samples those payloads from decision 0.
Physical state, commands, attack effects and available component outputs are still recorded at every
completed decision. Missing requested sensors fail explicitly.

Use `read_table(path, columns=...)` for selected top-level Parquet columns; it returns an Arrow
table with typed nested data and payload references, without loading the payload bytes. Use
`RunLog.read(path)` for decoded records and arrays. `load_payloads=False` retains references
containing dtype/shape and shard locations instead of loading arrays. This option still loads
all step/event records; it is not a streaming reader.

The writer publishes closed `part-*.parquet` chunks after their payloads are available and writes
completion metadata last. Readers ignore unfinished `.open` files and reject unknown schema
versions. Interrupted runs remain `incomplete`; completed chunks can be inspected without
claiming that the whole experiment finished.

The runner supplies lazy `default_codecs()` from `avsectester.runlog.codecs` for supported Modular
and Alpamayo native outputs. Creating the registry imports no simulator or model package. Saved
native objects decode to ordinary data, so reading a log does not reconstruct live model or
simulator objects. Unknown native types need an explicit codec before disk recording. To customize
chunk limits or codecs, pass a `Recorder` to the runner; the supplied recorder controls its own
path, metadata and retention. See [RunLog and recording API](../reference/python-api.md#runlog-and-recording).

**Not yet implemented.** Dataset-driven candidate services and the complete selection/preparation
experiment API belong to later migration phases. The direct runner currently uses already
configured backend/stack instances; `RunRequest.source` and `initialization` must remain `None`.

## Driving traces

The compatibility runners return [run records](../concepts/architecture.md#terminology) as a `Trace`
(`avsectester.closed_loop.datatypes`), derived from the unified RunLog. `run_configured` returns a
`dict[str, Trace]` keyed by treatment name. `run_prepared` returns the result of its `run_fn` for
each named treatment, which is a `Trace` by default. Their CLI summary and existing export options
are unchanged. Traces are held in memory; use the direct runner's `output_path` for persisted RunLog
records. See [Run from Python](experiments.md#run-from-python).

| `Trace` member | Content |
|---|---|
| `records` | One `FrameRecord` per driving step |
| `run_log` | The source `RunLog` on traces returned by the shared runner; `None` on manually constructed traces |
| `outputs` | `RecordedOutput` records from handlers. See [Intervention outputs and provenance](#intervention-outputs-and-provenance) |
| `provenance` | How the run was produced, as a dictionary |
| `replay_scenario` | For CARLA, the resolved scene configuration with the actual spawn settings of the run. `None` for other simulators. See [Random choices and seeds](simulators.md#random-choices-and-seeds) |
| `peak_speed`, `final_speed` | Maximum and last recorded speed, in m/s |
| `braking_frames` | Records with `throttle == 0` and `brake > 0` |
| `mean_detections` | Mean of `n_detections` over the records |

A `FrameRecord` describes the physical outcome of one driving step:

| Field | Meaning |
|---|---|
| `frame` | Zero-based driving-step index. It is not the simulator frame ID |
| `t`, `speed` | Physical time (s) and ego speed (m/s) after the command was executed |
| `throttle`, `brake`, `steer` | The executed command, after all `command` handlers |
| `n_detections` | Count from the available `perception` component output. It is 0 when that output is unavailable |

Records contain the physical state, not the model-visible observation. A spoofed speed or an
attacked image does not appear in them. Only its effect on the executed command and the resulting
motion does.

The examples in this guide run offline. They use NuRec with the stub renderer, which returns
black camera frames, a toy driving system and a toy observation attack:

<!-- runnable -->
```python
from avsectester.closed_loop.avstack import AVStack
from avsectester.closed_loop.datatypes import Control
from avsectester.runner import run_prepared
from avsectester.closed_loop.hooks import Hook, Runtime
from avsectester.scenarios import scenario_from_config


class BrightSpotStack(AVStack):
    """A toy driving system: bright pixels are an obstacle, and an obstacle means stop."""

    def __call__(self, observation):
        image = observation.sensor_data["camera_front"]
        detections = ["obstacle"] if image.max() > 200 else []
        plan = "stop" if detections else "cruise"
        self.latest = {"perception": detections, "planning": plan}
        return Control(brake=1.0) if plan == "stop" else Control(throttle=0.5)

    def component_log(self):  # used in "Component outputs"
        return getattr(self, "latest", None)


def paint_obstacle(observation, context):
    """An observation attack: a white square in the camera image from driving step 10."""
    if context.step >= 10:
        image = observation.sensor_data["camera_front"].copy()
        image[200:280, 280:360] = 255
        observation.sensor_data["camera_front"] = image
        context.record_output("painted_box", (280, 200, 360, 280))
    return observation


# NuRec with the offline stub renderer, which returns black 640x480 camera frames.
scenario = scenario_from_config({"backend": {"type": "NuRecSimulator"}})
runtimes = {"clean": None, "attack": Runtime([Hook("observation", paint_obstacle)])}
traces = run_prepared(scenario, BrightSpotStack, frames=40, runtimes=runtimes)
clean, attacked = traces["clean"], traces["attack"]

for record in attacked.records[8:12]:
    print(record.frame, round(record.t, 2), round(record.speed, 2), record.throttle, record.brake)
print(round(clean.peak_speed, 2), attacked.final_speed, attacked.braking_frames)
```

```text
8 0.45 0.68 0.5 0.0
9 0.5 0.75 0.5 0.0
10 0.55 0.35 0.0 1.0
11 0.6 0.0 0.0 1.0
3.0 0.0 30
```

The decision at step 10 sees the painted square and brakes. Record 10 already shows the reduced
speed, because each record is taken after its command was executed.

Trajectory-command driving systems such as Alpamayo return a `Control` with a `trajectory` and
leave `throttle`, `brake` and `steer` at 0. Their records carry speed and time, but no braking
information.

## Driving-impact metric

`impact(clean, attacked)` in `avsectester.evaluation.metric` compares a clean and a treated trace
and returns an `Impact`. It measures stops: an attack either forces a stop that the clean run did
not make, or prevents a stop that the clean run made.

```python
from avsectester.evaluation.metric import impact

result = impact(clean, attacked)
print(result)
```

```text
clean:    peak_speed= 3.00  final_speed= 3.00  brake_frames=0
attacked: final_speed= 0.00  brake_frames=30  overdrive_frames=0 (kept moving while clean was stopped)
=> ATTACK SUCCEEDED (forced an unsafe stop)
```

The keyword arguments are `stop_speed=0.5` (m/s at or below which the ego counts as stopped),
`baseline_speed=1.0` (m/s the clean run must reach) and `min_overdrive_frames=3`. The `Impact`
fields are computed as follows:

| Field | Definition |
|---|---|
| `clean_drove` | `clean.peak_speed >= baseline_speed`. The clean run established a driving baseline |
| `induced_braking` | The treated run has more braking frames than the clean run |
| `induced_stop` | The treated final speed is at most `stop_speed` and the clean final speed is above it |
| `overdrive_frames` | Driving steps, compared by index up to the shorter trace, where the clean speed is at most `stop_speed` and the treated speed exceeds `baseline_speed` |
| `suppressed_stop` | The clean final speed is at most `stop_speed` and `overdrive_frames >= min_overdrive_frames` |
| `attack_succeeded` | `clean_drove and (induced_stop or suppressed_stop)` |

The fields `clean_peak_speed`, `clean_final_speed`, `attacked_final_speed`, `clean_brake_frames`
and `attacked_brake_frames` hold the underlying measurements.

`result.verdict` interprets the pair. It checks the conditions in this order:

1. **Inconclusive** if the clean run never reached `baseline_speed`. A vehicle that never moved
   cannot show a driving impact. Run more driving steps or check the spawn.
2. **Attack succeeded** with a suppressed stop, then with an induced stop.
3. **Attack induced braking** without a full stop. This does not count as success.
4. **No impact**.

`str(result)` prints the measurements with the label `attacked` and the verdict.
`result.comparison(treated_label)` prints the measurements only. Use it for treatments that are
not attacks, such as a defense compared with the clean baseline. The CLI prints `comparison` for
defense and clean treatments. For attack treatments, it prints the verdict and maps it to an exit
code. See [Exit codes](../reference/cli.md#exit-codes).

This criterion covers stop-based effects only. It is not a universal definition of attack
success. Experiments that need other criteria compute them from the traces.

### Plot a comparison

`plot_impact` draws the ego speed and the brake command of both traces over the driving steps. It
returns the absolute path of the saved figure and requires matplotlib (the `viz` extra).

```python
from avsectester.evaluation.metric import plot_impact

print(plot_impact(clean, attacked, "tmp/evaluation/impact.png", result=result))
```

The upper panel has a dotted reference line at 0.5 m/s, the default `stop_speed`. Pass `result`
when you computed `impact` with other thresholds, because the function otherwise recomputes it
with the defaults for the caption. The keyword options `treated_label="attacked"` and
`show_verdict=True` control the legend and the verdict caption. For a defense or baseline
comparison, pass `show_verdict=False`, which also selects a neutral default title. An explicit
`title` replaces the default. These options do not change any measurement. From the CLI, `--plot`
produces the same figure. See [Save a plot](../getting-started/quickstart.md#save-a-plot).

## Intervention outputs and provenance

Handlers record diagnostic values with `context.record_output(name, value)`. See
[Recording outputs](../reference/runtime.md#recording-outputs). Each call appends an
`RecordedOutput` to `Trace.outputs`, in recording order:

| Field | Meaning |
|---|---|
| `name` | The name passed to `record_output` |
| `value` | A deep copy of the recorded value |
| `step` | The driving step of the callback |
| `stage` | The stage of the callback |
| `frame`, `t` | The simulator frame ID and physical time at the callback. A `step.end` output belongs to the resulting frame |
| `sensor` | The sensor of a render callback, otherwise `None` |

```python
painted = [output for output in attacked.outputs if output.name == "painted_box"]
first = painted[0]
print(len(painted), first.step, first.stage, first.frame, round(first.t, 2), first.value)
```

```text
30 10 observation 10 0.5 (280, 200, 360, 280)
```

The output's `t` is the time when the observation was produced. Record 10 in the driving trace is
one time step later, after the command. Outputs never change the command or the driving trace.
The framework defines no metric over them. Group or score them for your own experiment, for
example to count defense alarms. See [Record alarms and scores](defenses.md#record-alarms-and-scores).

`Trace.provenance` describes how a run was produced. The runners write these keys:

| Key | Written by | Content |
|---|---|---|
| `runtime_seed` | `run_prepared`, `run_pair` | The `Runtime` seed for handlers |
| `runtime_stages` | `run_prepared`, `run_pair` | The stage of every hook in execution order, including sensor-condition handlers |
| `treatment` | `run_prepared` | The treatment name, such as `clean`, `attack` or `baseline` |
| `conditions` | `run_prepared` | The scenario's world conditions, condition seed and sensor conditions |
| Scenario metadata | `run_prepared` | The prepared scenario's provenance, merged at the top level |
| `interventions` | `run_configured`, `evaluate_robustness` | One `{"name", "kind", "provenance"}` entry per prepared intervention of the treatment |

Scenario metadata depends on its source. The CLI adds `config_path`. A configured scenario adds
the `provenance` mapping supplied to `scenario_from_config`. A selected scenario adds the
selected scene's source metadata, such as `scene_id` for NuRec datasets or `sample_token` for
nuScenes. The CARLA selection source also adds `backend` and `validation`.

The `interventions` list is empty for the `clean` treatment. It is also empty when an explicit
`treated_runtime` replaces the configured interventions, because no prepared interventions are
known then. Raw `runtime.hooks` are not listed. The `provenance` of each entry is the
`PreparedIntervention.provenance` record. `prepare_intervention` stores the scenario provenance
at preparation time under `scenario`. The direct loop `run(...)` returns a trace with empty
provenance.

```python
print(attacked.provenance["treatment"], attacked.provenance["runtime_stages"])
```

```text
attack ['observation']
```

For what each entry point records and how to reproduce a run, see
[Outputs and provenance](experiments.md#outputs-and-provenance) and
[Reproducibility](experiments.md#reproducibility).

## Component outputs

Component logging captures the intermediate outputs of the driving system during a run, so you can
see where an intervention changes the pipeline. It collects outputs the driving system already
produced and runs no additional inference. The driving system must expose `component_log()`. See
[Component logging interface](driving-systems.md#component-logging-interface) for the
protocol and what each supplied driving system provides.

Pass `run_logged` as the `run_fn` of `run_prepared`. Each result is then a
`(Trace, ComponentTrace)` tuple. The prepared runner applies conditions to each treatment, starts
every run from the same origin and closes the backend.

```python
from avsectester.evaluation.component_log import run_logged

results = run_prepared(scenario, BrightSpotStack, frames=40, runtimes=runtimes,
                       run_fn=run_logged)
clean_trace, clean_components = results["clean"]
attacked_trace, attacked_components = results["attack"]
print(attacked_components.stage_names)
print(attacked_components.steps[10].stages)
print(attacked_components.counts("perception")[8:12])
print(clean_components.degradation(attacked_components, "perception")[8:12])
```

```text
['perception', 'planning']
{'perception': ['obstacle'], 'planning': 'stop'}
[0, 0, 1, 1]
[0, 0, -1, -1]
```

The attack adds one perception output from step 10, so the clean count minus the attacked count
is negative. For the modular driving system, instrument the stack in its factory:

<!-- runnable -->
```python
from avsectester.closed_loop.avstacks.modular import ModularAVStack


def make_stack():  # config is a CARLA configuration with a modular pipeline.
    stack = ModularAVStack(config["ego"]["pipeline"])
    stack.instrument()  # perception, tracking, planning and control
    return stack
```

Use a compatible `AlpamayoAVStack` factory for the end-to-end path. When you own simulator
assembly, call `run_logged(simulator, avstack, frames, perturb=None, *, hooks=None)` directly. It
uses the same driving loop and leaves the simulator open. Apply the scenario's sensor conditions to
the hooks yourself, as described in [Lower-level execution](conditions.md#lower-level-execution).

| API | Meaning |
|---|---|
| `components.steps[i]` | The `StepLog` of driving step `i`. `StepLog.frame` is the driving-step index |
| `components.steps[i].stages[name]` | Snapshot of the named component output at step `i` |
| `components.stage_names` | Component names of the first logged step |
| `components.counts(name)` | Per-step `len(output)`. 0 for a missing or `None` output, 1 for a value without `len` |
| `clean_components.degradation(attacked_components, name)` | Clean count minus attacked count per step, up to the shorter trace |
| `components.performance(name, truths, assign_radius=4.0)` | Per-step detection or tracking metrics from `avstack.metrics.get_instantaneous_metrics`, with nearest-neighbour assignment |

The shared recorder snapshots each step's output graph, so later updates do not change earlier
entries; `ComponentTrace` reads those snapshots rather than running a second collector.
Native in-memory outputs must support copying. A stack without `component_log()` yields
no steps. An uninstrumented `ModularAVStack` yields steps with empty `stages`. Count differences do
not measure position errors and do not establish attack success. For `performance`, supply one
ground-truth collection per driving step, with compatible object types, reference frames and
input times. When a `perception` output is present, the compatibility driving record also carries
its count. Disk-decoded native component outputs are plain data; native-object performance
helpers require an explicit compatible decoder or a metric that consumes that data representation.

**Timing.** `components.steps[i]` describes inference on the input of driving step `i`.
`trace.records[i]` describes the physical state after that step's command was executed. These
are different instants. Match ground truth to the component input, not to the record.

**Executed commands.** A component's `control` or `action` output is captured before the final
`command` handlers, which can replace the command. Read the executed throttle, brake and steer
from the driving trace.

## Robustness across corruption settings

`evaluate_robustness` in `avsectester.evaluation.robustness` measures how an attack's success
changes under image corruptions. For each prepared scenario from a source, it runs a clean and
an attacked treatment under each corruption setting, scores each pair with `impact` and returns a
`RobustnessReport`. The grid always starts with the baseline (labelled `clean`), which adds no
corruption, followed by one entry per corruption setting.

```python
from avsectester.evaluation.robustness import evaluate_robustness
from avsectester.preparation import Intervention
from avsectester.scenarios import source_from_config


class PaintObstacle(Intervention):
    """The observation attack packaged as an intervention, without preparation."""

    def hooks(self, artifacts):
        return (Hook("observation", paint_obstacle),)


source = source_from_config({"backend": {"type": "NuRecSimulator"}})
report = evaluate_robustness(source, None, PaintObstacle(), BrightSpotStack, frames=40)
print(report.summary())
```

```text
Attack robustness — 1 valid baseline pair(s). baseline ASR = 100%
condition                     valid  skipped inconclusive    ASR  resilience
clean                             1        0            0   100%  (baseline)
Brightness@0.50                   1        0            0   100%        100%
ChromaticAberration@0.50          1        0            0   100%        100%
ColorTemp@0.50                    1        0            0   100%        100%
Contrast@0.50                     1        0            0     0%          0%
DefocusBlur@0.50                  1        0            0   100%        100%
Fog@0.50                          1        0            0   100%        100%
Gamma@0.50                        1        0            0   100%        100%
GaussianNoise@0.50                1        0            0   100%        100%
JPEGCompression@0.50              1        0            0   100%        100%
LowLight@0.50                     1        0            0   100%        100%
MotionBlur@0.50                   1        0            0   100%        100%
Rain@0.50                         0        0            1    N/A         N/A
ShotNoise@0.50                    1        0            0   100%        100%
Snow@0.50                         1        0            0   100%        100%
mean resilience (corruptions): N/A
```

Reduced contrast dims the painted square below the toy detector's threshold, so the attack fails.
Rain streaks make the toy driving system stop in the clean run, so that pair is inconclusive.

Each corruption is added to the scenario's existing conditions as a sensor condition at the
`observation` stage, after all observation interventions. The scenario's own world and sensor
conditions apply in every setting, including the baseline. The corruption acts on the match's
camera, or on the first readable RGB camera when the scenario has no match. See
[Order relative to attacks and defenses](conditions.md#order-relative-to-attacks-and-defenses).

The attack can be given in three forms:

- An `Intervention`. It is prepared once per scenario, and its artifacts are reused across the
  corruption settings. Handlers are constructed afresh for each attacked run.
- A `PreparedIntervention` of kind `attack`. Its artifacts are reused as supplied, including
  across scenarios. Preparation is never inferred from a file name or an earlier run.
- A callable `attack_for(match, backend)`. It is called once per pair, after world conditions are
  configured and before the backend is reset. It returns a `Runtime`, a one-argument observation
  function or `None` to skip the pair. The clean run reuses the returned runtime's localizer and
  seed without its hooks. Stateful localizers shared this way must implement `reset`.

For the first two forms, the traces record the intervention under `provenance["interventions"]`.

### Parameters

`evaluate_robustness(source, req, attack_for, stack, frames, *, conditions=None, severity=0.5,
limit=None, run_fn=run)`:

| Parameter | Meaning |
|---|---|
| `source` | A `ScenarioSource`. See [Select a scenario](scenarios.md#select-a-scenario) |
| `req` | The `ScenarioRequirement` passed to `source.scenarios`. Use `None` for a source from `source_from_config`, which uses its configured requirement or yields its unselected scenario |
| `attack_for` | The attack, in one of the three forms above |
| `stack` | A factory that returns a fresh `AVStack` for each run |
| `frames` | Driving steps per run |
| `conditions` | An iterable of `AugmentationPipeline`, one per corruption setting. These are image-corruption pipelines, not `ScenarioConditions`. `None` uses `common_corruptions(severity)`: one pipeline per built-in operator. Pass `albumentations_corruptions(severity)` for the Albumentations operators, or `[]` for the baseline only |
| `severity` | Strength of the default corruption settings, default `0.5` |
| `limit` | Maximum number of scenarios taken from the source, or `None` for all |
| `run_fn` | Per-run recorder. It accepts `runtime=` and must return a `Trace`. The default is the shared driving loop |

Corruption settings are labelled by the pipeline's `name`, such as `Fog@0.50`. Pipelines with the
same name share one entry in the report. For the operators and their parameters, see
[Built-in image corruptions](conditions.md#built-in-image-corruptions).

Selection filters run once, before the grid, on the source's own conditions. They are not
re-evaluated on corrupted observations. To select by conditioned measurements, configure
conditions on the source. See
[Conditions during selection](conditions.md#conditions-during-selection).

### Statuses, ASR and resilience

`report.rows` holds one `ScenarioResult(scenario_id, condition, status, reason)` per attempted
pair. `scenario_id` comes from the scenario's `scene_id`, `sample_token` or `image_path`
provenance, or is the scenario's index. `reason` is the impact verdict or the skip reason.

| Status | Meaning | Counted in ASR |
|---|---|---|
| `success` | The clean run drove and `impact(...).attack_succeeded` is true | Yes |
| `failure` | The clean run drove and the attack did not succeed | Yes |
| `inconclusive` | The clean run did not reach the driving baseline | No |
| `skipped` | The `attack_for` callback returned `None` | No |

`report.per_condition` maps each label to a `ConditionResult` with the counts `successes`,
`failures`, `skipped` and `inconclusive`, the number of valid pairs `n` and the success rate
`asr`.

- **ASR** of a corruption setting is `successes / (successes + failures)`. It is `NaN` without
  valid pairs, and `summary()` shows `N/A`.
- **Resilience**, `report.resilience(label)`, is `ASR(setting) / ASR(baseline)`, capped at 1.
  `ASR(baseline)` is the ASR of the grid entry labelled `clean`, not of a clean treatment. It is
  `NaN` when either ASR is `NaN`, and 0 when the baseline ASR is 0.
- `report.baseline_asr` is the baseline ASR. `report.mean_resilience` averages the resilience over
  the corruption settings. It is `NaN` when any of them is `NaN`, and 0 without corruption
  settings.

Direct callers of `report.record(scenario_id, condition, status, reason="")` must pass a status
string. Success follows the stop-based criterion of [the metric](#driving-impact-metric).

## Scene recordings

`avsectester.closed_loop.simulators.viz` saves what the driving system saw at each driving step.
`record_run` runs the same loop as `run` and saves each image returned by `visualize(observation)`
as `frame_0000.png`, `frame_0001.png`, ... in `out_dir` (default `<repo>/tmp/frames`). The
`prefix` argument replaces `frame`. The observation is the model
input after all observation handlers, so recordings show attacks and sensor conditions. With
`collect=True`, the images are also kept in `trace.frames`.

To record several treatments, wrap `record_run` as the `run_fn` of `run_prepared`. The runner
does not pass the treatment name, but it runs treatments in the order of `runtimes`:

```python
from pathlib import Path

from avsectester.closed_loop.simulators.viz import record_run, save_sequence

out = Path("tmp/evaluation/scenes")
labels = iter(runtimes)  # run_prepared runs treatments in this order


def record(backend, stack, frames, *, runtime):
    return record_run(backend, stack, frames, out_dir=out / next(labels), hooks=runtime,
                      collect=True)


recorded = run_prepared(scenario, BrightSpotStack, frames=20, runtimes=runtimes, run_fn=record)
for name, trace in recorded.items():
    print(save_sequence(trace.frames, out, name=name, cols=5))
```

```text
('tmp/evaluation/scenes/clean_filmstrip.png', 'tmp/evaluation/scenes/clean.gif')
('tmp/evaluation/scenes/attack_filmstrip.png', 'tmp/evaluation/scenes/attack.gif')
```

A view is a function from an `Observation` to an RGB image, or `None` to skip the step:

| View | Use |
|---|---|
| `viz.camera_view` | The default. Camera payloads that are RGB arrays, such as NuRec |
| `simulators.carla.camera_view`, `simulators.carla.lidar_bev` | CARLA camera payloads and a lidar bird's-eye view |
| `viz.detections_view(detect, base, threshold=0.3)` | Draws boxes from your `detect(rgb)` function over a base view |
| `viz.labels_view(get_scene, base, ...)` | Draws ground-truth labels from your `get_scene(observation)` function |

`filmstrip`, `save_gif` and `save_sequence` assemble collected images into a contact sheet and an
animation. For signatures, see [Visualization](../reference/python-api.md#visualization). The
CARLA and NuRec demo scripts that save images are listed in
[Demo scripts and outputs](../examples/image-insertion.md#demo-scripts-and-outputs).

## Limitations

- The driving-impact metric scores induced and suppressed stops only. `induced_stop` compares
  final speeds, so a stop followed by resumed driving is not detected.
  **Not yet implemented.** Experiments cannot detect stopping events across a run or select another success criterion.
- **Not yet implemented.** No metric summarizes recorded intervention outputs, such as defense alarms.
- **Not yet implemented.** The CLI cannot compare clean, attack and combined treatments from one origin.
  In Python, pass three named runtimes to `run_prepared`.
- Trajectory-command driving systems record no throttle or brake, so braking counts and the brake
  panel of `plot_impact` carry no information for them.
- Traces and component traces are returned in memory. No API serializes all native outputs or
  generates an experiment report.
- Robustness corruptions are image-space approximations applied to one camera. See
  [Fidelity limits](conditions.md#fidelity-limits).

## Next steps

- [Implement a defense](defenses.md#evaluate-a-defense) and evaluate its side effects.
- [Apply operating conditions](conditions.md) to build custom corruption settings.
- [Component logging interface](driving-systems.md#component-logging-interface) for custom driving systems.
- [Python API reference](../reference/python-api.md#metrics-and-evaluation) for signatures.
