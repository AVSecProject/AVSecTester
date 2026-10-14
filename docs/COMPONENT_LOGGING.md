# Component logging

Component logging captures intermediate outputs to help researchers examine how an intervention
propagates through a driving system. It collects already-produced outputs without running
additional model inference.

## Stack interface

A stack can expose its latest outputs through this protocol, defined in
`avsectester.evaluation.component_log`:

```python
class InstrumentedStack(Protocol):
    def component_log(self) -> dict[str, Any] | None: ...
```

| Stack | Outputs |
|---|---|
| `ModularAVStack` | Outputs of configured stages. `stack.instrument()` requests `perception`, `tracking`, `planning` and `control`. Pass `stages=(...)` for a smaller set. Requested stages must support effective native hooks. |
| `AlpamayoAVStack` | `policy`: the driver prediction, and `action`: the resulting `Control`. These are available without calling `instrument()`. |
| Custom stack | Any named outputs returned by `component_log()`. The collector does not require a fixed component schema. |

Register attacks through Runtime. Capture hooks observe returned outputs, including runtime
replacements. An uninstrumented modular stack yields an empty `ComponentTrace`.

Instrumentation raises `ValueError` for unavailable stages. For a custom pipeline, select
only the components whose post-stages appear in `stack.supported_stages`.

## Collect outputs

Given a CARLA configuration with a modular pipeline compatible with its sensors:

```python
from avsectester.evaluation.component_log import run_logged
from avsectester.runner import run_prepared
from avsectester.scenarios import scenario_from_config
from avsectester.stacks.modular import ModularAVStack

scenario = scenario_from_config(config)

def make_stack():
    stack = ModularAVStack(config["ego"]["pipeline"])
    stack.instrument()
    return stack

results = run_prepared(scenario, make_stack, frames=30, run_fn=run_logged)
trace, components = results["clean"]
perception_counts = components.counts("perception")
first_outputs = components.steps[0].stages
```

To compare both treatments with configured attacks, reuse that scenario and stack factory:

```python
from avsectester.interventions import configured_runtime

attack_runtime = configured_runtime(config, instance=scenario, kinds={"attack"})
results = run_prepared(scenario, make_stack, frames=30, runtimes={
    "clean": None,
    "attack": attack_runtime,
}, run_fn=run_logged)
clean_trace, clean_components = results["clean"]
attacked_trace, attacked_components = results["attack"]
count_difference = clean_components.degradation(attacked_components, "perception")
```

Each result is a `(Trace, ComponentTrace)` tuple. The prepared runner applies world and sensor
conditions to each treatment, starts both runs from the same scene and closes the backend.
For Python-selected cases, use the `ScenarioInstance` returned by the selection source directly.
Use a compatible `AlpamayoAVStack` factory for the end-to-end path.

`run_logged(backend, stack, frames, perturb=None, *, runtime=None)` is also available when the
caller owns backend assembly and cleanup. Pass a Runtime with
[scenario sensor conditions applied](SCENARIOS.md#lower-level-execution). It reuses the same
driving loop. See [Interventions](INTERVENTIONS.md) for custom attack and defense handlers.

Each `StepLog` contains the zero-based decision-step index in `frame` and a `stages` dictionary.
The collector deep-copies the output graph, including mutable state and reference frames, so
later updates do not change historical entries. Native outputs must support this copying.

## Read and compare outputs

| API | Meaning |
|---|---|
| `components.stage_names` | Component names from the first logged step |
| `components.steps[i].stages[name]` | Raw output snapshot for the named component at decision step `i` |
| `components.counts(name)` | Per-step `len(output)`, 0 for missing/`None`, or 1 for a scalar without `len` |
| `clean_components.degradation(attacked_components, name)` | Clean count minus attacked count at corresponding list indices, up to the shorter trace |
| `components.performance(name, truths, assign_radius=4.0)` | Per-step detection/tracking metrics from `avstack.metrics.get_instantaneous_metrics`, using nearest-neighbour assignment |

Count differences do not measure position error or establish attack success. For `performance`,
supply one ground-truth collection per decision step, with compatible native object types,
reference coordinates and input time. The caller aligns both traces before comparing them.
`run_logged` does not fill `FrameRecord.n_detections` from component captures.

## Timing and executed commands

`components.steps[i]` describes inference from the input to decision step `i`.
`trace.records[i]` describes the physical state after executing that decision. These are different
instants. Match ground truth to the component input rather than the post-control state.

A component's `control` or `action` output is captured before final `command` handlers.
Those handlers can change the command that is executed and recorded in `Trace`. Use the driving
trace when inspecting executed throttle, brake and steer.

Both traces are returned in memory. This API does not automatically serialize every native
output or generate a complete experiment report. Use `simulators.viz.record_run` and
`simulators.viz.save_gif` for scene recording. The [script index](../scripts/README.md) lists
examples that save images and animations.
