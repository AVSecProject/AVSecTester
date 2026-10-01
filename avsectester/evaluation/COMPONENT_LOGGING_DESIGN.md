# Component-level logging — in-system analysis of an attack

## Why

The black-box metrics so far — `metric.impact` (did the drive change?) and `evaluation.robustness`
(does the attack survive corruption?) — see only the *outcome* at the ego's wheels. They cannot say
**where inside the stack** an attack takes effect. For a hide-a-vehicle patch the story is the
*propagation*: perception misses the target → tracking drops it → planning doesn't slow → action holds
throttle. A phantom injection is the dual. Component-level logging records each stage's per-frame output
so that chain is visible.

## Design: reuse what avstack already provides

avstack **natively supports component logging** through its per-stage hook system (`BaseModule.register_post_hook` / the `@apply_hooks` decorator; it even ships `DetectionsLogger`/`TracksLogger`/`MetricsHook`). So we do **not** invent a parallel logging path or a heavy per-layer schema — we reuse the hooks and avstack's metrics, and keep the gatherer thin.

Three small pieces:

1. **Capture (stack side, `stacks/modular.py`).** `_StageCapture` is a generic avstack post-hook — the
   component-logging twin of the existing `_DetectionCounter` — that remembers a stage's most recent output
   and returns it unchanged. `ModularAVStack.instrument(stages=…)` attaches one per stage **last** (so it
   observes the *attacked* output, exactly as `attach_counter` does), and `component_log() -> {stage:
   output}` returns the latest per-stage raw avstack outputs. That is the whole stack-side addition.

2. **Gather (`evaluation/component_log.py`).** `run_logged(backend, stack, frames, perturb)` is the
   instrumented twin of `backend.run`: it reuses the *same* loop via `run`'s new `on_step` callback, and
   each frame appends the stack's `component_log()` to a `ComponentTrace` (a list of `StepLog{frame,
   stages}`). A non-instrumented stack yields an empty `ComponentTrace`, so the call site is uniform.

3. **Process (`evaluation/component_log.py`).** Deliberately thin, computed from the raw outputs:
   - `ComponentTrace.counts(stage)` — per-frame output size (`len`), e.g. n_detections / n_tracks.
   - `ComponentTrace.degradation(other, stage)` — per-frame count delta between two runs. The headline
     in-system signal is `clean.degradation(attacked, stage)`: which layer the attack first changes and how
     it ripples forward. **Needs no ground truth.**
   - `ComponentTrace.performance(stage, truths)` — per-frame TP/FP/FN vs ground truth, reusing
     `avstack.metrics.get_instantaneous_metrics` (nearest-neighbour assignment). Works on detections or
     tracks alike.

The sim↔stack contract (`plane.Observation`/`Control`) is untouched — component logs are a side channel
pulled through `component_log()`; `evaluation` never imports `stacks`.

## How it feeds the end goal (paper-style report)

Five sections; this is section 5, and each has its source module:

| Report section            | Source |
|---------------------------|--------|
| Attack effectiveness      | `metric.impact` / robustness baseline ASR |
| Robustness                | `evaluation.robustness` (ASR + resilience vs corruptions) |
| Impacting-factor analysis | scenario factors (`SceneGT`: distance/viewpoint/visibility) × outcome — *future* |
| Ablation                  | attack-config variants × outcome — *future* |
| **In-system analysis**    | **this component logging** (clean-vs-attacked per-layer propagation) |

A later `evaluation/report.py` assembles all five.

## Status / phasing

1. **DONE — capture + gather + counts/degradation.** `_StageCapture`, `ModularAVStack.instrument` /
   `component_log`, `run`'s `on_step` hook, `run_logged`, `ComponentTrace` (counts / degradation /
   performance). Unit-tested with a fake pipeline + stub loop (`tests/test_component_log.py`), no simulator.
2. **Prediction layer + GT plumbing + plots.** Map a `prediction` stage when the pipeline has one (schema
   needs no change — layers are just stage keys); feed per-frame `truths` (from `SceneGT` / live CARLA GT)
   into `performance`; add a propagation figure and the clean-vs-attacked in-system diff to the report.
3. **E2E / VLA (`AlpamayoAVStack`).** No perception/tracking/planning layers — it maps frames to a
   trajectory. It will implement `component_log()` returning a `policy` layer (candidate-trajectory
   distribution, chosen mode, optional saliency) plus `action`. `component_log()` is an optional protocol,
   so `run_logged` already handles a stack that returns different/fewer stage keys; only the Alpamayo-side
   extraction and the report's policy view are new.
4. **`evaluation/report.py`** — assemble all five sections (tables + plots, reusing `simulators.viz` /
   `metric.plot_impact`).
