# Phantom injection

> **Audience:** Researchers who attack a modular driving system's perception output.  
> **Outcome:** Understand how phantom injection adds a fabricated detection inside the driving system, and how to configure and run it.  
> **Prerequisites:** [Architecture](../concepts/architecture.md) and [Implement an attack](../guides/attacks.md).

**What it does.** `PhantomInjection` appends one fabricated `BoxDetection` to the detector output
at every driving step. The phantom obstacle propagates through tracking and planning. In
`configs/carla_scenario.yaml`, the `ForwardCollisionPlanner` can respond to it with a stop that the
clean run does not make. Besides `target_xyz`, its options are `obj_type` (default `Car`), `score`
(`0.9`), `extent` (height, width and length, `(1.6, 1.8, 4.0)`) and the detection ID `oid`
(`90002`).

**Preparation.** No artifact is computed. The configured intervention returns a copy of the hook
configuration as its artifact.

**Execution.** The attack acts at `perception.post`, inside the modular driving system.
`PhantomInjection` is a native avstack post-hook. Explicitly import
`avsectester.attacks.pipeline.phantom` to register it in avstack's `HOOKS` registry; constructing
`ModularAVStack` does not register attacks. The `NativeHook` adapter connects it to the runtime:

- It builds a fresh native hook from the configuration at the start of every run.
- A native post-hook returns `(output,)`. `NativeHook` unwraps the tuple, because runtime
  handlers return the output directly. Use `NativeHook(config, phase="pre")` at a `.pre` stage.
- Component logging records the modified perception output.

`target_xyz` is the bottom centre of the phantom box in the source sensor's frame (forward, left,
up), in metres. The hook places the box in `detections.source_reference`, and it requires that
reference even when there are no detections. The avstack perception base class copies the input's
reference before inference and attaches it to the output before post-hooks run. Later ego motion
therefore does not move an earlier phantom. A passthrough detector input must supply
`DataContainer(..., source_reference=reference)` itself. A missing reference raises `ValueError`.
Container copy, filtering, mapping, addition and serialization keep the source reference.

**Configure and run.** `configs/carla_scenario.yaml` declares the attack in the
[`attacks`](../reference/configuration.md#attacks) list. The CLI runs a clean baseline and the attacked
run, then prints their [driving-impact](../guides/evaluation.md#driving-impact-metric) comparison:

```yaml
attacks:
  - stage: perception
    imports: [avsectester.attacks.pipeline.phantom]
    hook:
      type: PhantomInjection
      target_xyz: [6.0, 0.0, -1.5]
      obj_type: Car
      score: 0.9
```

```bash
avsectester run configs/carla_scenario.yaml --frames 40
```

The same hook can be written as a [`runtime.hooks`](../reference/configuration.md#runtime) entry with
`stage: perception.post` and an `avstack:` mapping. Both forms are enabled in the `attack` and
combined (`attack-defense`) treatments. Python experiment hook entries instead use
`native_hook` with `imports: [avsectester.attacks.pipeline.phantom]`. From Python, import the
registration module and register `NativeHook` directly:

```python
import yaml

from avsectester.legacy_experiment import configured_stack
import avsectester.attacks.pipeline.phantom  # Register PhantomInjection explicitly.
from avsectester.closed_loop.avstacks.native_hook import NativeHook
from avsectester.runner import run_prepared
from avsectester.closed_loop.hooks import Hook, Runtime
from avsectester.scenarios.setup import scenario_from_config

config = yaml.safe_load(open("configs/carla_scenario.yaml"))
scenario = scenario_from_config(config)  # The scene view ignores the attacks list.
phantom = Runtime([Hook("perception.post", NativeHook({
    "type": "PhantomInjection", "target_xyz": (6.0, 0.0, -1.5),
    "obj_type": "Car", "score": 0.9,
}))])
results = run_prepared(
    scenario, lambda: configured_stack(config), 40,
    runtimes={"clean": Runtime(), "attack": phantom},
)
```

**Code.** `avsectester/attacks/pipeline/phantom.py` (`PhantomInjection`),
`avsectester/closed_loop/avstacks/native_hook.py` (`NativeHook`) and
`avsectester/interventions.py` (`NativeHookIntervention`, legacy configuration). Adapter signatures are in
[Runtime](../reference/python-api.md#runtime).
