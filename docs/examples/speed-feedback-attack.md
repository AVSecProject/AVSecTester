# Speed-feedback adaptive attack

> **Audience:** Researchers who write adaptive attacks.  
> **Outcome:** Understand how a stateful plugin reads feedback after each driving step and changes its behavior during a run.  
> **Prerequisites:** [Architecture](../concepts/architecture.md) and [Implement an attack](../guides/attacks.md).

**What it does.** The attack adds a bias to the reported `ego_speed` in the observation. After each
driving step it reads the physical ego speed and updates the bias for the next step:

```text
bias = clip(bias + gain * (target_speed - true_speed), 0, max_bias)
```

The physical state is only read. This example shows reading feedback during a run, stateful
decisions and lifecycle management. It is not a sensor model and makes no claim of effectiveness.
It affects driving systems that read `Observation.ego_speed`, such as Alpamayo. The modular driving
system reads `vehicle_state` instead.

**Preparation.** `SpeedFeedbackAttack(target_speed, gain=0.1, max_bias=5.0, initial_bias=0.0)`
validates its parameters at construction. `prepare` returns them as a dictionary artifact.

| Parameter | Meaning |
|---|---|
| `target_speed` | Reference speed in m/s, nonnegative |
| `gain` | Nonnegative feedback gain per completed driving step |
| `max_bias` | Maximum reported-speed bias in m/s |
| `initial_bias` | Starting bias in m/s, between zero and `max_bias` |

All parameters must be finite.

**Execution.** `hooks` creates a fresh `SpeedFeedbackPlugin` at `observation` for each run. The
plugin's `reset` clears the bias and feedback history, `transform` adds the bias to `ego_speed`,
and `after_step` reads `context.physical_state.ego_speed` and updates the bias. Handler lifecycle and
driving-system lifecycle are separate. See [Plugin lifecycle](../reference/runtime.md#plugin-lifecycle)
and [Keep state and adapt](../guides/attacks.md#keep-state-and-adapt).

**Not yet implemented.** An adaptive reference attack that estimates other actors during the run.

**Configure and run.** Add an [`interventions`](../reference/configuration.md#interventions) entry.
Its `kind` defaults to `attack`:

```yaml
interventions:
  - factory: avsectester.attacks.adaptive:SpeedFeedbackAttack
    options: {target_speed: 4.0, gain: 0.2, max_bias: 3.0}
```

This example runs offline with the stub NuRec renderer and a driving system that reads the
reported speed:

<!-- runnable -->
```python
from avsectester.closed_loop.avstack import AVStack
from avsectester.experiment import run_configured
from avsectester.closed_loop.datatypes import Control


class SpeedKeeper(AVStack):
    """Accelerate below a reported 5 m/s, otherwise brake lightly."""

    def __call__(self, observation):
        if observation.ego_speed < 5.0:
            return Control(throttle=0.5)
        return Control(brake=0.3)


config = {
    "backend": {"type": "NuRecSimulator", "renderer": {"type": "StubRenderer", "height": 2, "width": 2}},
    "interventions": [{
        "factory": "avsectester.attacks.adaptive:SpeedFeedbackAttack",
        "options": {"target_speed": 4.0, "gain": 0.2, "max_bias": 3.0},
    }],
}
traces = run_configured(config, 60, stack_factory=SpeedKeeper)
for name, trace in traces.items():
    print(name, round(trace.records[-1].speed, 2))
```

The attacked run ends slower than the clean run:

```text
clean 4.5
attack 1.96
```

**Code.** `avsectester/attacks/adaptive.py` (`SpeedFeedbackAttack`, `SpeedFeedbackPlugin`).
