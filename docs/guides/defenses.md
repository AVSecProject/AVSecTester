# Implement a defense

> **Audience:** Researchers implementing a new defense.  
> **Outcome:** Write a defense with the shared intervention interface, order it relative to attacks and conditions, record alarms and run it in defense treatments.  
> **Prerequisites:** [Architecture](../concepts/architecture.md) and [Implement an attack](attacks.md).

This guide continues the running example of [Implement an attack](attacks.md): `CruiseControl`
holds 5 m/s using the reported speed, and the attack reports 5 m/s more than the measured speed.
The defense written here detects and limits that spoofed speed.

## Defenses and attacks

A defense is an [intervention](../concepts/architecture.md#terminology) of kind `defense`. It uses
the same interfaces as an attack: handlers registered with `Hook`, stateful `Plugin` handlers, and
an optional `Intervention` with `prepare` and `hooks`. Everything in
[Implement an attack](attacks.md) about stages, contexts, lifecycle and artifacts applies
unchanged.

The kind is the only difference. It decides which configured treatments enable the intervention:
`attack` enables attacks, `defense` enables defenses and the combined treatment (`attack-defense`)
enables both. See [Treatments and modes](experiments.md#treatments-and-modes). The runtime itself
never classifies handlers. A `Runtime` built in Python can combine attack and defense handlers
freely.

A defense acts on Vehicle I/O or inside the driving system. It should base its decisions on data
the vehicle could have: the handler's value, the observation and the driving-system components.
`context.ground_truth` is the simulator's physical state. Use it in a defense only to model a
trusted reference, and state that assumption in your evaluation.

## Write a defense handler

Successive reported positions imply a speed. This defense clamps the reported speed to within a
tolerance of that odometry speed. The NuRec state estimate carries `x` and `y` in metres:

```python
import math
from dataclasses import replace

from avsectester.backend import run
from avsectester.closed_loop.hooks import Hook, Plugin, Runtime
from avsectester.closed_loop.simulators.nurec import NuRecSimulator, StubRenderer

# CruiseControl and overreport_speed are defined in Write your first attack.


class OdometryClamp(Plugin):
    """Clamp the reported speed to the speed implied by successive reported positions."""

    def __init__(self, tolerance=1.0):
        self.tolerance = tolerance  # m/s

    def reset(self, context):
        self.previous = None  # (t, x, y) of the previous observation in this run

    def transform(self, observation, context):
        state = observation.vehicle_state
        previous, self.previous = self.previous, (observation.t, state.x, state.y)
        if previous is None or observation.t <= previous[0]:
            return observation
        distance = math.hypot(state.x - previous[1], state.y - previous[2])
        odometry = distance / (observation.t - previous[0])
        low, high = odometry - self.tolerance, odometry + self.tolerance
        return replace(observation, ego_speed=min(max(observation.ego_speed, low), high))


chains = {
    "attack": [Hook("observation", overreport_speed)],
    "attack, then defense": [Hook("observation", overreport_speed),
                             Hook("observation", OdometryClamp())],
    "defense, then attack": [Hook("observation", OdometryClamp()),
                             Hook("observation", overreport_speed)],
}
backend = NuRecSimulator({"dt": 0.1}, renderer=StubRenderer(height=8, width=8))
try:
    for name, hooks in chains.items():
        trace = run(backend, CruiseControl(), frames=80, runtime=Runtime(hooks))
        print(f"{name:21} final_speed={trace.final_speed:.2f}")
finally:
    backend.close()
```

```text
attack                final_speed=0.00
attack, then defense  final_speed=4.05
defense, then attack  final_speed=0.00
```

The clean run ends at 5.10 m/s. With the defense after the attack, the driving system drives
again. The reported speed can still exceed the odometry speed by the tolerance, so the vehicle
settles near 4 m/s. With the defense before the attack, the defense checks an unattacked value
and the attack takes full effect. Both handlers are at `observation`, so their registration order
decides which one sees the other's result.

The defense trusts the reported position. An attack that spoofs position and speed consistently
would pass this check. A defense is only as strong as the data it treats as trusted.

## Order defenses relative to attacks and conditions

The framework never moves defenses after attacks. Place each defense where it sees the data it
must check:

- **Different stages.** Stage order decides. A defense at `observation` always runs after an
  attack at `sensors.post`. A defense at `sensors.post` never sees an `observation` attack.
- **Same stage, direct `Runtime`.** Handlers run in registration order, as in the example above.
- **Same stage, configured treatments.** The convenience `attacks` and `patches` entries run
  first, then raw `runtime.hooks`, then `interventions` in definition order. List a defense after
  the attack it should see. Configured attacks from `attacks`, `patches` and `runtime.hooks`
  therefore always precede an `interventions` defense at the same stage.
- **Prepared interventions.** `prepare_intervention(..., position="before")` places hooks before
  the base Runtime's hooks in `Treatment.runtime(base)`, and `"after"` (the default) after them.
- **Sensor conditions.** At their stage, conditions with `order: after` (the default) run after
  every attack and defense. A defense at that stage then inspects the input before the condition
  is applied, while the driving system receives it after. `order: before` places the condition
  ahead of both, so the defense sees the conditioned, attacked input.

The complete rule is in
[Ordering at a shared stage](../reference/runtime.md#ordering-at-a-shared-stage). The effect of
condition placement on attacks, defenses and the driving system is tabulated in
[Order relative to attacks and defenses](conditions.md#order-relative-to-attacks-and-defenses).

## Record alarms and scores

A defense that changes a value can also report what it detected. `context.record_output(name,
value)` saves a named value in `trace.outputs` without changing execution. Add these lines to
`OdometryClamp.transform`, before its final `return`:

```python
alarm = not low <= observation.ego_speed <= high
context.record_output("defense.speed_alarm", {
    "alarm": alarm, "reported": observation.ego_speed, "odometry": odometry,
})
```

Then count the alarms of the attack-then-defense run:

```python
# CruiseControl, overreport_speed and OdometryClamp are defined above.
backend = NuRecSimulator({"dt": 0.1}, renderer=StubRenderer(height=8, width=8))
try:
    trace = run(backend, CruiseControl(), frames=80, runtime=Runtime([
        Hook("observation", overreport_speed), Hook("observation", OdometryClamp()),
    ]))
finally:
    backend.close()

alarms = [output for output in trace.outputs if output.name == "defense.speed_alarm"]
raised = [output for output in alarms if output.value["alarm"]]
print(f"{len(raised)} alarms in {len(alarms)} checks, first at step {raised[0].step}:")
print(raised[0].value)
```

```text
79 alarms in 79 checks, first at step 1:
{'alarm': True, 'reported': 5.0, 'odometry': 0.0}
```

The first observation has no previous position, so the check starts at step 1. Each record also
holds the stage, the physical frame and time, and the driving step. Use a distinct name prefix,
such as `defense.`, when several methods record outputs. The record fields are in
[Recording outputs](../reference/runtime.md#recording-outputs), and reading them in
[Intervention outputs and provenance](evaluation.md#intervention-outputs-and-provenance).

## Package and configure a defense

Package the handler as an `Intervention` so configured treatments can enable it. `hooks` builds a
fresh `OdometryClamp` for every run:

<!-- runnable -->
```python
from avsectester.preparation import Intervention
from avsectester.closed_loop.hooks import Hook

# OdometryClamp is defined above.
class SpeedClamp(Intervention):
    """Package OdometryClamp as a configurable defense."""

    def __init__(self, tolerance=1.0):
        self.tolerance = tolerance

    def prepare(self, scenario):
        return {"tolerance": self.tolerance}

    def hooks(self, artifacts):
        return [Hook("observation", OdometryClamp(artifacts["tolerance"]))]
```

In Python, prepare it with `prepare_intervention(SpeedClamp(), scenario, kind="defense")`. In a
configuration, add an [`interventions`](../reference/configuration.md#interventions) entry with
`kind: defense`. With `OdometryClamp` and `SpeedClamp` in `my_research/defenses.py`, the
`interventions` list of the attack configuration in
[Configure and run the attack](attacks.md#configure-and-run-the-attack) becomes:

```yaml
interventions:
  - factory: my_research.attacks:SpeedSpoof
    kind: attack
    name: speed-spoof
    options: {bias: 5.0}
  - factory: my_research.defenses:SpeedClamp
    kind: defense
    name: speed-clamp
    options: {tolerance: 1.0}
```

The defense entry follows the attack entry, so it processes the attacked observation. Keep these
rules in mind:

- Raw `runtime.hooks` are attack handlers in configured treatments. A `kind` key there is
  rejected.
- There is no top-level `defenses` key. Its presence raises an error.
- A handler factory is not an intervention factory. Wrap an existing handler in an
  `Intervention`, as `SpeedClamp` wraps `OdometryClamp`.

## Evaluate a defense

A defense needs three comparisons with the clean run from the same origin:

| Comparison | Question |
|---|---|
| clean vs `attack` | How strong is the attack without the defense? |
| clean vs `attack-defense` | How much attack impact remains with the defense? |
| clean vs `defense` | What does the defense change without an attack, such as false alarms? |

`run_prepared` runs all of them from one origin. `Treatment.runtime(kinds=...)` builds each
treatment's runtime from the same prepared interventions:

```python
from avsectester.evaluation.metric import impact
from avsectester.preparation import Treatment, prepare_intervention
from avsectester.runner import run_prepared
from avsectester.closed_loop.hooks import Runtime
from avsectester.scenarios.source import ScenarioInstance
from avsectester.closed_loop.simulators.nurec import NuRecSimulator, StubRenderer

# CruiseControl and SpeedSpoof are defined in Implement an attack. SpeedClamp is defined above.
scenario = ScenarioInstance(
    factory=lambda: NuRecSimulator({"dt": 0.1}, renderer=StubRenderer(height=8, width=8)),
)
treatment = Treatment((
    prepare_intervention(SpeedSpoof(bias=5.0), scenario, kind="attack"),
    prepare_intervention(SpeedClamp(tolerance=1.0), scenario, kind="defense"),
))
results = run_prepared(scenario, CruiseControl, frames=80, runtimes={
    "clean": Runtime(),
    "attack": treatment.runtime(kinds={"attack"}),
    "defense": treatment.runtime(kinds={"defense"}),
    "attack-defense": treatment.runtime(kinds={"attack", "defense"}),
})
clean = results["clean"]
for name in ("attack", "attack-defense"):
    print(f"{name:14} {impact(clean, results[name]).verdict}")
defended = results["defense"]
alarms = sum(o.value["alarm"] for o in defended.outputs if o.name == "defense.speed_alarm")
print(f"defense        final_speed={defended.final_speed:.2f} "
      f"(clean {clean.final_speed:.2f}), false alarms={alarms}")
```

```text
attack         ATTACK SUCCEEDED (forced an unsafe stop)
attack-defense no impact (attack did not change the drive)
defense        final_speed=5.10 (clean 5.10), false alarms=0
```

The configured version of this experiment is in [Custom treatments](experiments.md#custom-treatments).

From the command line, a configured experiment runs the clean baseline and one selected treatment:

- `--treatment defense` compares the defense with the clean run. It prints
  `[pair] Defense side effects relative to clean.` and the measurements, without a verdict.
- `--treatment attack-defense` compares the combined treatment with the clean run. It prints the
  verdict of the remaining attack impact and labels the run `attack+defense`. With the
  configuration above, the verdict is `no impact (attack did not change the drive)`.

The verdict decides the exit code of an attack pair. A defense-only pair exits with `0` after
successful execution. See [Exit codes](../reference/cli.md#exit-codes).

## Current limitations

**Not yet implemented.** The repository supplies no reference defense, and no metric summarizes
recorded defense outputs such as alarm rates.

**Not yet implemented.** Configured and CLI runs cannot compare clean, attack and attack-defense
treatments from one origin.

Until then, run three-treatment comparisons with `run_prepared` as above, and compute alarm
statistics from `trace.outputs` yourself. The supplied stop-based verdict scores only the
remaining attack impact. It does not score detection.

## Next steps

- [Evaluate results](evaluation.md): the driving-impact metric, intervention outputs and
  robustness evaluation across corruption settings.
- [Apply operating conditions](conditions.md): test a defense under weather and sensor
  degradation.
- [Runtime reference](../reference/runtime.md): stages, ordering rules and the handler contract.
- [Run experiments](experiments.md): treatments, modes and custom treatments.
