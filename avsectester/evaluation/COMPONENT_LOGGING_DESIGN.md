# Component-level logging — in-system analysis of an attack

## Why

The black-box metrics so far — `metric.impact` (did the drive change?) and `evaluation.robustness`
(does the attack survive corruption?) — see only the *outcome* at the ego's wheels. They cannot say
**where inside the stack** an attack takes effect. For a hide-a-vehicle patch the story is the
*propagation*: perception misses the target → tracking drops it → planning doesn't slow → action holds
throttle. A phantom injection is the dual. Component-level logging records each stage's per-frame output
so that chain is visible.

## Design: reuse what avstack already provides

avstack **natively supports component logging** through its per-stage hook system (`BaseModule.register_post_hook` / the `@apply_hooks` decorator; it even ships `DetectionsLogger`/`TracksLogger`/`MetricsHook`). So we do **not** invent a parallel logging path or a heavy per-layer schema — we reuse the hooks and avstack's metrics, collect only **free, already-produced outputs**, and keep the gatherer thin.

### The common interface (both stack shapes)

One contract, defined in `evaluation/component_log.py`:

```python
class InstrumentedStack(Protocol):
    def component_log(self) -> dict[str, Any] | None: ...   # {component_name: free_output}, or None
```

A plain dict of the stack's *already-produced* outputs — no bespoke schema — so it fits either shape:
- **modular** (`ModularAVStack`): `{"perception": detections, "tracking": tracks, "planning": plan,
  "control": control}` (the stage outputs, captured via avstack post-hooks);
- **end-to-end** (`AlpamayoAVStack`): `{"policy": prediction, "action": control}` — just what `predict()`
  already returns (candidate trajectories `(K,T,3)` + `selected_index`) and the emitted command.

The E2E log is **sparse** (no semantic layers exist), but it is the *same* contract, so the logger and the
report need no change when the E2E stack later exposes more. Both stacks implement it today.

Three small pieces:

1. **Capture (stack side, `stacks/`).** Modular: `_StageCapture` — a single generic avstack post-hook that
   remembers a stage's output and returns it unchanged; `ModularAVStack.instrument(stages)` attaches one per
   stage **last** (so it observes the *attacked* output). It is the *one* per-stage capture mechanism — used
   for plain detection telemetry (`instrument(("perception",))`, driving `FrameRecord.n_detections` via
   `run_logged`) and full in-system analysis (all stages) alike. E2E: `AlpamayoAVStack.component_log()`
   returns the prediction + control it already computed in `__call__`. Both satisfy `InstrumentedStack`.

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

1. **DONE — the common interface + free logs, both stacks.** `InstrumentedStack` protocol; `_StageCapture`
   + `ModularAVStack.instrument`/`component_log` (modular stage outputs); `AlpamayoAVStack.component_log`
   (E2E `policy`+`action`, free from `predict()`); `run`'s `on_step` hook; `run_logged`; `ComponentTrace`
   (counts / degradation / performance). Unit-tested with a fake pipeline, a stub loop, and both stack
   shapes (`tests/test_component_log.py`), no simulator.
2. **Prediction stage + GT plumbing + plots.** Map a `prediction` stage when the pipeline has one (just
   another stage key); feed per-frame `truths` (from `SceneGT` / live CARLA GT) into `performance`; add a
   propagation figure + the clean-vs-attacked in-system diff to the report.
3. **E2E processing.** The E2E `policy`/`action` logs are collected now; later add *free* processing of them
   (e.g. clean-vs-attacked trajectory divergence / planned-speed delta) — no new model internals, no
   speculative probes.
4. **`evaluation/report.py`** — assemble all five sections (tables + plots, reusing `simulators.viz` /
   `metric.plot_impact`).
