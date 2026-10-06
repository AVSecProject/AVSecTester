# Component-level logging — in-system analysis of an attack

## Why

The black-box metrics so far — `metric.impact` (did the drive change?) and `evaluation.robustness`
(does the attack survive corruption?) — see only the *outcome* at the ego's wheels. They cannot say
**where inside the stack** an attack takes effect. For a hide-a-vehicle patch the story is the
*propagation*: perception misses the target → tracking drops it → planning doesn't slow → action holds
throttle. A phantom injection is the dual. Component-level logging records each stage's per-frame output
so that chain is visible.

## Design: reuse what avstack already provides

avstack supports component logging through `BaseModule.register_post_hook` and the `@apply_hooks`
decorator. This implementation uses those hooks to capture already-produced outputs and reuses
avstack's matching metrics for analysis.

### The common interface (both stack shapes)

One contract, defined in `evaluation/component_log.py`:

```python
class InstrumentedStack(Protocol):
    def component_log(self) -> dict[str, Any] | None: ...   # {component_name: free_output}, or None
```

A plain dict of the stack's *already-produced* outputs — no bespoke schema — so it fits either shape:
- **modular** (`ModularAVStack`): `{"perception": detections, "tracking": tracks, "planning": plan,
  "control": control}` (the stage outputs, captured via avstack post-hooks).
- **end-to-end** (`AlpamayoAVStack`): `{"policy": prediction, "action": control}` — just what `predict()`
  already returns (candidate trajectories `(K,T,3)` + `selected_index`) and the emitted command.

The collector accepts additional component keys if the E2E adapter later exposes more outputs.

Three small pieces:

1. **Capture (stack side, `stacks/`).** Modular: `_StageCapture` — a single generic avstack post-hook that
   remembers a stage's output and returns it unchanged. `ModularAVStack.instrument(stages)` attaches one per
   stage **last** (so it observes the *attacked* output). It is the *one* per-stage capture mechanism — used
   for plain detection telemetry (`instrument(("perception",))`, driving `FrameRecord.n_detections` via
   `run_logged`) and full in-system analysis (all stages) alike. E2E: `AlpamayoAVStack.component_log()`
   returns the prediction + control it already computed in `__call__`. Both satisfy `InstrumentedStack`.

2. **Gather (`evaluation/component_log.py`).** `run_logged(backend, stack, frames, perturb)` is the
   instrumented twin of `backend.run`: it reuses the *same* loop via `run`'s new `on_step` callback, and
   each frame deep-copies the stack's `component_log()` into a `ComponentTrace` (a list of
   `StepLog{frame, stages}`). The snapshot includes mutable tracks, plans and reference frames so
   subsequent steps cannot rewrite earlier outputs. A stack without the component logging interface
   yields an empty `ComponentTrace`.

3. **Process (`evaluation/component_log.py`).** Deliberately thin, computed from the raw outputs:
   - `ComponentTrace.counts(stage)` — per-frame output size (`len`), e.g. n_detections / n_tracks.
   - `ComponentTrace.degradation(other, stage)` — per-frame count delta between two runs, without
     requiring ground truth. Counts alone cannot measure position errors or trajectory changes.
   - `ComponentTrace.performance(stage, truths)` — per-frame TP/FP/FN vs ground truth, reusing
     `avstack.metrics.get_instantaneous_metrics` (nearest-neighbour assignment). Works on detections or
     tracks alike.

The sim↔stack contract (`plane.Observation`/`Control`) is untouched — component logs are a side channel
pulled through `component_log()`. `evaluation` never imports `stacks`.

## How it feeds the end goal (paper-style report)

The planned report has five sections, each with a source module:

| Report section            | Source |
|---------------------------|--------|
| Attack effectiveness      | `metric.impact` / robustness baseline ASR |
| Robustness                | `evaluation.robustness` (ASR + resilience vs corruptions) |
| Impacting-factor analysis | scenario factors (`SceneGT`: distance/viewpoint/visibility) × outcome — *future* |
| Ablation                  | attack-config variants × outcome — *future* |
| **In-system analysis**    | **this component logging** (clean-vs-attacked per-layer propagation) |

A later `evaluation/report.py` assembles all five.

## Status / phasing

1. **Implemented: component interfaces, collection and basic processing.** CPU tests cover collection
   with synthetic backends, the real modular pipeline, and the Alpamayo adapter with controlled model
   predictions. See `tests/core/test_component_log.py`, `tests/core/test_alpamayo.py` and
   `tests/avstack/test_component_log_real.py`. These tests do not require a simulator or model weights.
2. **Prediction stage + GT plumbing + plots.** Map a `prediction` stage when the pipeline has one (just
   another stage key). Feed per-frame `truths` (from `SceneGT` / live CARLA GT) into `performance`. Add a
   propagation figure + the clean-vs-attacked in-system diff to the report.
3. **E2E processing.** The E2E `policy`/`action` logs are collected now. Later add processing of them
   (e.g. clean-vs-attacked trajectory divergence / planned-speed delta) — no new model internals, no
   speculative probes.
4. **`evaluation/report.py`** — assemble all five sections (tables + plots, reusing `simulators.viz` /
   `metric.plot_impact`).
