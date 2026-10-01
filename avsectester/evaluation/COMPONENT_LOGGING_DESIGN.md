# Component-level logging — in-system analysis of an attack

## Why

The black-box metrics so far — `metric.impact` (did the drive change?) and `evaluation.robustness`
(does the attack survive corruption?) — see only the *outcome* at the ego's wheels. They cannot say
**where inside the stack** an attack takes effect. For a hide-a-vehicle patch the interesting story is the
*propagation*: the patch makes **perception** miss the target → **tracking** drops the track →
**prediction** no longer forecasts a lead → **planning** does not slow → **action** holds throttle. A
phantom injection is the dual: a detection appears at perception and ripples forward to an unnecessary
brake.

Component-level logging records, **per frame, per layer**, the signals needed to tell that story. It is
the data behind the report's **in-system analysis** section, and it complements (does not replace) the
outcome metrics.

## Where it lives (and the dependency direction)

Three pieces, matching the request:

1. **Schema** — `evaluation/component_log.py`: pure, dependency-free dataclasses (the canonical per-layer
   record). Both producers and consumers import this; it pulls in nothing heavy.
2. **Producers (stack-specific)** — `stacks/`: each stack knows its own internals, so *extraction* is
   stack-specific. The modular stack emits layer logs via avstack **post-hooks** (the existing
   `_DetectionCounter` is the prototype). A stack advertises the capability with an
   `InstrumentedStack` protocol method.
3. **Gatherer/processor** — `evaluation/`: a `ComponentLogger` collects per-frame `StepLog`s across a run
   into a `ComponentTrace`, and the processing turns that into the analysis (target propagation,
   clean-vs-attacked per-layer degradation).

`stacks` imports the schema from `evaluation.component_log` (a leaf module, no inversion risk); `evaluation`
pulls logs from a stack *through the protocol method* and never imports `stacks`. The sim↔stack contract
(`plane.Observation`/`Control`) stays untouched — component logs are a **side channel**, not part of the
contract.

## Canonical schema (`evaluation/component_log.py`)

Per-layer records are **target-centric**: an attack acts on one object (the `ScenarioMatch.target`), so the
most diagnostic signal is that target's fate at each layer. Layers are **optional** (a stack emits only the
ones it has — key for the E2E case below).

```
StepLog:                       # one frame
  frame: int;  t: float
  perception: PerceptionLog | None
  tracking:   TrackingLog   | None
  prediction: PredictionLog | None
  planning:   PlanningLog   | None
  action:     ActionLog     | None

PerceptionLog:  n_detections, scores:[float],
                target_detected:bool, target_score:float|None, target_iou:float|None   # vs GT target box
TrackingLog:    n_tracks, target_tracked:bool, target_track_id:str|None,
                target_track_confidence:float|None, id_switched:bool
PredictionLog:  target_predicted:bool, pred_min_gap_m:float|None, pred_ttc_s:float|None # forecast on target
PlanningLog:    target_speed_mps:float, maneuver:str|None, plan_min_gap_m:float|None     # does the plan heed it
ActionLog:      throttle, brake, steer                                                   # (mirrors FrameRecord)

ComponentTrace:  steps:[StepLog]  + target metadata
  - series(layer, field) -> [values]                          # for plots / tables
  - target_survival() -> {layer: fraction of frames the target survived that layer}
  - break_layer() -> the first layer where target survival collapses (the attack's entry point)
```

Target matching (detection/track ↔ GT target) is by IoU / centre distance against the scenario target's
`box2d` / 3-D box; the logger is given the target once per run.

## Producers: modular stack (`stacks/`)

Generalise the `_DetectionCounter` post-hook into a small set of **probes**, one per stage, each mapping a
stage's avstack output to its canonical layer log:

```
stacks/probes.py:
  PerceptionProbe(target) -> fills PerceptionLog from the detections list
  TrackingProbe(target)   -> fills TrackingLog   from the tracks (+ id-switch bookkeeping)
  PredictionProbe(target) -> fills PredictionLog  (only if the pipeline has a prediction stage/output)
  PlanningProbe(target)   -> fills PlanningLog    from the plan/trajectory
  (action comes from the Control the stack already returns)
```

`ModularAVStack` gains:
- `instrument(target)` — attaches the probes as **post-hooks, last** on each stage (so they observe the
  *attacked* output, exactly as `attach_counter` does today), and remembers the target for matching;
- `component_log() -> StepLog | None` (the `InstrumentedStack` protocol) — assembles the probes' last
  outputs into one `StepLog`, called once per `__call__`.

The current avstack pipeline is perception → tracking → planning → control; **prediction** is mapped when a
prediction stage is present (the schema/field stays `None` otherwise), so adding it later needs no schema
change. This reuses the stack's existing hook machinery — no new extraction path.

## Gatherer/processor (`evaluation/`)

```
evaluation/component_log.py (schema, above) + evaluation/logging.py:
  ComponentLogger                       # accumulates StepLogs -> ComponentTrace
  run_logged(backend, stack, frames, perturb=None, target=None) -> (Trace, ComponentTrace)
```

`run_logged` is the instrumented twin of `backend.run`: it drives the same loop, and when the stack is an
`InstrumentedStack` it calls `stack.component_log()` each frame (after `stack(seen)`), building the
`ComponentTrace` alongside the ordinary driving `Trace`. A non-instrumented stack (or an E2E stack with no
layers) yields an empty/partial `ComponentTrace`, so the call site is uniform.

Processing (for the report):
- **target propagation** — per layer, the fraction of frames the target survived; `break_layer()` names
  where it first collapses.
- **clean-vs-attacked diff** — run `run_logged` twice (reusing the robustness harness's clean/attacked
  pairing) and diff the per-layer target-survival series: the layer with the largest clean→attacked drop is
  the attack's point of entry, and the forward layers show how it ripples to the action.

## How it feeds the end goal (paper-style report)

The report has five sections; this feature is section 5, and the modules that feed each already exist or
are planned:

| Report section            | Source |
|---------------------------|--------|
| Attack effectiveness      | `metric.impact` / robustness baseline ASR |
| Robustness                | `evaluation.robustness` (ASR + resilience vs corruptions) |
| Impacting-factor analysis | scenario factors (`SceneGT`: distance/viewpoint/visibility, corruption, severity) × outcome — *future* |
| Ablation                  | attack-config variants (patch size, harmonisation on/off, …) × outcome — *future* |
| **In-system analysis**    | **this component logging** (clean-vs-attacked per-layer propagation) |

A later `evaluation/report.py` assembles all five (tables + plots, reusing `simulators.viz` /
`metric.plot_impact`) into a single document.

## VLA / end-to-end stacks (future step)

An E2E/VLA stack (`AlpamayoAVStack`) has no perception/tracking/planning layers — it maps camera frames
straight to a trajectory. The optional-layer schema already accommodates this: such a stack implements
`component_log()` returning only an `action` log plus a new `policy` layer (e.g. the candidate-trajectory
distribution, chosen-mode index, and — where available — an attention/saliency map over the input). Target
propagation degrades gracefully to the layers present. Designing the `policy` layer's fields and extracting
them from Alpamayo is deferred to that step; nothing here needs to change to add it.

## Phasing

1. **Schema + modular probes + logger** — `StepLog`/`ComponentTrace`, `stacks/probes.py`,
   `ModularAVStack.instrument`/`component_log`, `evaluation/logging.run_logged`, target matching. Unit-test
   with stub detection/track/plan outputs (no simulator), mirroring `test_robustness.py`.
2. **Prediction layer + clean-vs-attacked in-system diff + plots** — wire a prediction probe when present;
   add the per-layer degradation analysis and a propagation figure.
3. **E2E `component_log`** — the `policy` layer for `AlpamayoAVStack`.
4. **`evaluation/report.py`** — assemble all five sections into the paper-style report.
