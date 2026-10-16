# Implement an attack

> **Audience:** Researchers implementing a new attack.  
> **Outcome:** Write an attack, test it offline, package it with reusable artifacts, run it against a clean baseline and read the results.  
> **Prerequisites:** [Architecture](../concepts/architecture.md) and the [core installation](../getting-started/installation.md#local-core-installation).

An attack is an [intervention](../concepts/architecture.md#terminology). It runs as handlers that
the runtime calls at named stages of the driving loop. Each handler receives the current value at
its stage and returns the value passed downstream. An attack can also prepare reusable artifacts
before any run. This guide follows one attack from a single handler to a configured experiment.
The complete stage, value and context tables are in the [Runtime reference](../reference/runtime.md).

## Choose an attack surface

The [architecture](../concepts/architecture.md#attacks-and-defenses) defines three surfaces an
attack can act on. Each surface corresponds to a set of stages:

| Surface | Stages | What a handler can change |
|---|---|---|
| World | `world.setup`, `world.step.pre`, `world.step.post` | Actors, environment and the physical ego state, through native simulator operations. These handlers return `None` |
| World, NuRec rendering | `render.pre`, `render.post` | The pose and camera of one rendering, and the rendered image. The physical ego does not move |
| Vehicle I/O | `sensors.post` | Sensor payloads and their metadata |
| | `localization.pre`, `localization.post` | The localizer's input, and the state estimate |
| | `observation` | The final model input, including the reported speed and clock |
| | `command` | The command that the simulator executes |
| Driving system | `perception`, `tracking`, `planning`, `control` with `.pre` or `.post` (modular) | Component inputs and outputs |
| | `policy.pre`, `policy.post` (Alpamayo) | The model input and its prediction |

Choose the stage closest to where your attacker acts. A physical attacker who places a patch or
changes the scene uses a world stage, so every downstream component sees the effect. An attacker
who spoofs a measurement uses `sensors.post`. An attacker who tampers with the state estimate or
the command uses `localization.post` or `command`. A driving-system stage models a compromised
component, or replaces a perception result as a proxy for an attack on a sensor.

Stage availability depends on the simulator and the driving system. A hook at an unavailable stage
raises an error before the world is reset. See [Stages](../reference/runtime.md#stages). The
supplied attacks and their stages are listed in [Examples](../examples/README.md#how-to-read-these-examples).

## Write your first attack

This example runs offline with `NuRecBackend` and `StubRenderer`, without a GPU, simulator server
or model weights. `CruiseControl` is a minimal `AVStack` that accelerates until the reported speed
reaches 5 m/s. It is an example policy, not a driving model. The attack reports 5 m/s more than
the measured speed. Save the code as `first_attack.py`:

<!-- runnable -->
```python
from dataclasses import replace

from avsectester.backend import AVStack, run
from avsectester.plane import Control
from avsectester.runtime import Hook, Runtime
from avsectester.simulators.nurec import NuRecBackend, StubRenderer


class CruiseControl(AVStack):
    """Accelerate until the reported speed reaches 5 m/s, then hold it."""

    @property
    def command_modes(self):
        return frozenset({"actuator"})

    def __call__(self, observation):
        if observation.ego_speed < 5.0:
            return Control(throttle=0.5)
        return Control()


def overreport_speed(observation, context):
    """Report 5 m/s more than the measured speed."""
    return replace(observation, ego_speed=observation.ego_speed + 5.0)


backend = NuRecBackend({"dt": 0.1}, renderer=StubRenderer(height=8, width=8))
try:
    clean = run(backend, CruiseControl(), frames=80)
    attacked = run(backend, CruiseControl(), frames=80,
                   runtime=Runtime([Hook("observation", overreport_speed)]))
finally:
    backend.close()  # run() leaves the backend open. The caller closes it.

print(f"clean    final_speed={clean.final_speed:.2f}")
print(f"attacked final_speed={attacked.final_speed:.2f}")
```

Run it with `python first_attack.py`, or with the Docker command in
[Run offline without a GPU](../getting-started/quickstart.md#run-offline-without-a-gpu):

```text
clean    final_speed=5.10
attacked final_speed=0.00
```

The driving system believes it already drives at 5 m/s and never accelerates. The handler changes
only the model-visible speed. The physical speed changes only through the commands the driving
system issues. Each `run` resets the world, so both runs start from the same pose.

`Hook(stage, handler)` registers a handler, and `Runtime` holds hooks in registration order. A
handler is a function `handler(value, context)` or an object with `transform(value, context)`. It
returns a value of the stage's type. `run` without `runtime` executes a clean run. At one stage,
handlers run in registration order, and each receives the previous handler's result. Handlers at
different stages run in the loop's stage order, whatever their registration order. The framework
never reorders handlers by attack or defense. See
[Handler contract](../reference/runtime.md#handler-contract) and
[Ordering at a shared stage](../reference/runtime.md#ordering-at-a-shared-stage).

## Read inputs and state

A handler can read four kinds of data:

- **The handler's value** is the current value in its stage's chain, including changes made by
  earlier handlers. Base measurement and state attacks on it.
- **`context.observation`** is the current observation when one exists. It is `None` during
  `world.setup` and the first sensor capture.
- **`context.ground_truth`** is a fresh snapshot of the physical ego state, independent of what
  the driving system sees. Editing it changes nothing.
- **Native resources** in `context.backend.native` and `context.stack.native` give access to
  simulator actors, the renderer and driving-system components. They are live and unsandboxed.

The physical state and the
[model-visible data](../concepts/architecture.md#physical-state-and-model-visible-data) stay
separate. This camera attack reads the physical speed to decide when to act, then changes the
image that the driving system receives. `StubRenderer` names its camera `camera_front`:

<!-- runnable -->
```python
from dataclasses import replace

from avsectester.runtime import Hook, Runtime

def camera_attack(observation, context):
    """Paint a white square into the front camera while the vehicle moves at 2 m/s or more."""
    if context.ground_truth.ego_speed < 2.0:  # physical speed, not the reported one
        return observation
    image = observation.sensor_data["camera_front"].copy()
    image[:4, :4] = 255
    context.record_output("attack.painted", True)
    return replace(observation, sensor_data={**observation.sensor_data, "camera_front": image})

runtime = Runtime([Hook("sensors.post", camera_attack)])
```

Pass this runtime to `run` with `CruiseControl` for 80 driving steps. `trace.outputs` then holds
66 records, starting at step 14, when the physical speed first reaches 2 m/s. `CruiseControl`
does not read images, so this run checks only that the handler acts when intended.
`context.record_output` saves a diagnostic value without changing execution. See
[Recording outputs](../reference/runtime.md#recording-outputs). Every readable and writable value
is listed in [Read and write interfaces](../reference/runtime.md#read-and-write-interfaces), and
the context fields in [Contexts](../reference/runtime.md#contexts).

## Keep state and adapt

A handler that remembers earlier steps or reacts to feedback during the run is adaptive. Subclass
`Plugin` and implement the lifecycle methods you need. This attack reports the physical speed of
several driving steps ago, with a delay drawn once per run:

<!-- runnable -->
```python
from dataclasses import replace

from avsectester.runtime import Hook, Plugin, Runtime

class StaleSpeed(Plugin):
    """Report the physical speed of several driving steps ago."""

    def reset(self, context):
        self.history = []                               # per-run state, cleared before every run
        self.delay = int(context.rng.integers(5, 11))   # repeatable random choice per run

    def transform(self, observation, context):
        if len(self.history) < self.delay:
            return observation
        return replace(observation, ego_speed=self.history[-self.delay])

    def after_step(self, context):
        self.history.append(context.ground_truth.ego_speed)  # feedback after each step

runtime = Runtime([Hook("observation", StaleSpeed())], seed=0)
```

Run twice with the same backend and `CruiseControl` for 80 driving steps, it ends both runs at
`final_speed=6.45`. The driving system sees an old, lower speed and overshoots 5 m/s. `reset` runs
before every run and clears the plugin's own state. The framework cannot reset variables it does
not know about, so a closure with captured state needs a fresh closure and `Runtime` for every
run. `after_step` runs after each completed driving step and sees the physical state after the
command. `context.rng` is a NumPy generator that `Runtime(seed=...)` creates afresh for every run.
It does not seed the simulator or the driving model.

The lifecycle order is in [Plugin lifecycle](../reference/runtime.md#plugin-lifecycle) and the
generator rules in [Randomness](../reference/runtime.md#randomness). The supplied
[speed-feedback adaptive attack](../examples/speed-feedback-attack.md) uses the same
pattern with a feedback controller.

## Keep component decisions in one runtime

Use `AttackRuntime` when one online attack reads both a step's state and a component result.
Declare the results in `reads`; `step` and `on_component` operate on the same instance. This
offline example uses a toy perception output and keeps its TTC history in that instance:

<!-- runnable -->
```python
from avsectester.attack_runtime import AttackRuntime, ResultSubscription, StageAction
from avsectester.backend import AVStack
from avsectester.closed_loop import ClosedLoopRunner, RunRequest
from avsectester.plane import Control
from avsectester.simulators.nurec import NuRecBackend, StubRenderer

class TTCPhantom(AttackRuntime):
    reads = (ResultSubscription("physical_state"),
             ResultSubscription("component", "perception.post"))

    def start(self, context):
        self.history = []
        return ()

    def step(self, context):
        self.speed = max(context.physical_state.ego_speed, 0.1)
        return ()

    def on_component(self, context):
        ttc = context.component_output[0]["distance"] / self.speed
        self.history.append(ttc)
        if ttc < 2.0:
            return (StageAction("perception.post", [{"distance": 0.0}]),)
        return ()

class ToyDriver(AVStack):
    supported_stages = frozenset({"perception.post"})

    def __call__(self, observation):
        detections = self._emit("perception.post", [{"distance": 0.1}])
        return Control(brake=1.0 if detections[0]["distance"] == 0.0 else 0.0)

attack = TTCPhantom()
backend = NuRecBackend({"dt": 0.1}, renderer=StubRenderer(height=8, width=8))
try:
    log = ClosedLoopRunner().run(backend, ToyDriver(), RunRequest(2, attacks=[attack]))
finally:
    backend.close()
assert attack.history == [1.0, 1.0]
assert all(step.executed_command.brake == 1.0 for step in log.steps)
```

The real driving adapter determines its native component-output schema. Create a fresh runtime
for each independent run. Callbacks receive subscribed data and a read-only physical clock;
they do not receive a simulator handle. `StageAction` can replace a current or later pipeline
value. Physical insertions still use existing world hooks. See
[Online attack runtime](../reference/runtime.md#online-attack-runtime) for lifecycle and limits.

## Attack localization

Without a localizer, the state estimate is taken from the observation after `sensors.post`, so a
change to `vehicle_state` or `ego_speed` there is kept. To test a measurement attack against a
localization algorithm, configure [Vehicle I/O](driving-systems.md#vehicle-io) on the driving
adapter. The compatibility form `Runtime(localizer=...)` below remains supported. It is called as
`localizer(observation, context)` and returns a `StateEstimate`:

```python
from avsectester.plane import StateEstimate
from avsectester.runtime import Hook, Runtime

# estimator and measurement_attack are supplied by the caller.
def localize(observation, context):
    state = estimator.update(observation.sensor_data, observation.calibration)
    return StateEstimate(vehicle_state=state, ego_speed=state.speed)

runtime = Runtime([Hook("sensors.post", measurement_attack)], localizer=localize)
```

A handler at `sensors.post` changes a measurement, such as a GNSS or IMU payload. The change
reaches the driving system only if the localizer reads that measurement. A handler at
`localization.post` changes the estimate after the localizer, as an attacker who controls the
estimate directly. `localization.pre` exists only with a localizer and changes its input. The
localizer's result is copied before `localization.post`, so handlers there cannot change the
localizer's cached state. A stateful localizer can define the same lifecycle methods as a plugin.
A configured localizer uses the `localizer` key of [`runtime`](../reference/configuration.md#runtime).

## Change the world

World handlers at `world.setup`, `world.step.pre` and `world.step.post` receive `None`, return
`None` and change the world through native operations in `context.backend.native`. Editing an
observation or `context.ground_truth` never moves anything. Simulators capture sensor data at
different times: in CARLA, a change at `world.step.post` appears only from the observation after
next, so use `world.setup` or `world.step.pre` when the next observation must show it. Resolve
actors in `world.setup` or later, because the world reset creates them after the handlers' `reset`
methods run. Undo changes that outlive a run, such as dynamics parameters, and release created
actors in `close`. A CARLA world handler that needs ticks before driving declares `setup_ticks`
(see [World setup ticks](../reference/runtime.md#world-setup-ticks)).

What each simulator lets you read and change, with examples, is in
[Change the world from handlers](simulators.md#change-the-world-from-handlers). A world attack is
packaged like any other attack, as the next section shows. The supplied
[physical patch](../examples/physical-patch.md) is a packaged world attack.

## Package the attack as an intervention

An `Intervention` separates preparation from execution. `prepare(scenario)` runs once before any
run and returns the artifacts, such as parameters, a texture or a trained policy. The default
returns `None`. `hooks(artifacts)` returns the ordered hooks for one run. Return fresh handlers,
or handlers whose `reset` clears their state. The first attack becomes:

<!-- runnable -->
```python
from dataclasses import replace

from avsectester.preparation import Intervention
from avsectester.runtime import Hook

class SpeedSpoof(Intervention):
    """Add a bias to the speed reported to the driving system."""

    def __init__(self, bias=5.0):
        self.bias = bias

    def prepare(self, scenario):
        return {"bias": self.bias}  # the artifact: a JSON-compatible parameter set

    def hooks(self, artifacts):
        def spoof(observation, context):
            return replace(observation, ego_speed=observation.ego_speed + artifacts["bias"])
        return [Hook("observation", spoof)]
```

`prepare_intervention(method, scenario, kind=..., name=...)` calls `prepare` once and returns a
`PreparedIntervention`. A `Treatment` groups prepared interventions, and `treatment.runtime()`
builds a `Runtime` with fresh hooks. `run_prepared` runs named treatments of one
[prepared scenario](../concepts/architecture.md#terminology) from the same origin:

```python
from avsectester.preparation import Treatment, prepare_intervention
from avsectester.runner import run_prepared
from avsectester.runtime import Runtime
from avsectester.scenarios.source import ScenarioInstance
from avsectester.simulators.nurec import NuRecBackend, StubRenderer

# CruiseControl is defined in the first example. SpeedSpoof is defined above.
scenario = ScenarioInstance(
    factory=lambda: NuRecBackend({"dt": 0.1}, renderer=StubRenderer(height=8, width=8)),
    provenance={"scene": "offline-example"},
)
prepared = prepare_intervention(SpeedSpoof(bias=5.0), scenario, kind="attack", name="speed-spoof")
treatment = Treatment((prepared,))

results = run_prepared(scenario, CruiseControl, frames=80, runtimes={
    "clean": Runtime(),
    "attack": treatment.runtime(case=scenario.match),
})
for name, trace in results.items():
    print(f"{name:6} final_speed={trace.final_speed:.2f} treatment={trace.provenance['treatment']}")
```

```text
clean  final_speed=5.10 treatment=clean
attack final_speed=0.00 treatment=attack
```

`run_prepared` creates the backend with `scenario.make_backend()`, applies the scenario's
conditions to every treatment, builds a fresh driving system for each run and closes the backend.
`prepare_intervention` also takes `position="before"` or `"after"` (the default), which places the
hooks before or after a base Runtime's hooks in `treatment.runtime(base)`. Signatures are in
[Preparation](../reference/python-api.md#preparation). More treatments, such as an attack-defense
comparison, are described in [Custom treatments](experiments.md#custom-treatments), and
optimization inside `prepare` in
[Use optimization in attack preparation](optimization.md#use-optimization-in-attack-preparation).

### Use the selected scenario

`prepare` receives the complete `ScenarioInstance`. When the scenario comes from
[selection](scenarios.md#select-a-scenario), `scenario.match` holds the selected actor roles and
insertions. Without selection it is `None`. During execution, handlers receive only the match, as
`context.case`, which is also `None` without selection. A method that needs a selected role
checks for it during preparation:

<!-- runnable -->
```python
from avsectester.preparation import Intervention
from avsectester.runtime import Hook

class TargetedAttack(Intervention):
    role = "attacker"

    def prepare(self, scenario):
        if scenario.match is None or self.role not in scenario.match.binding_ids:
            raise ValueError(f"TargetedAttack needs a selected scenario with the role {self.role!r}")
        return {"role": self.role}

    def hooks(self, artifacts):
        def locate(value, context):
            actor_ids = context.case.binding_ids[artifacts["role"]]
            context.record_output("attack.target", actor_ids)
            return None  # Look up and change the actor through context.backend.native here.
        return [Hook("world.setup", locate)]
```

`binding_ids` maps each role to a tuple of actor IDs, and `context.case.insertions` holds the
insertion specifications. See [Roles and bindings](scenarios.md#roles-and-bindings) and, for how
`context.case` is set, [Contexts](../reference/runtime.md#contexts). Explicit world positions and
native actor names work without selection.

Preparation runs before the driving backend is reset. Resolve actors and live calibration for the
driving runs in handlers, not in `prepare`. Preparation that needs trial runs creates its own
backend with `scenario.make_backend()`, which applies the world conditions, and closes it.
`scenario.factory()` alone skips them. A direct `run` skips the sensor conditions, so pass
`scenario.conditions.apply_runtime(runtime)` as its Runtime. See
[Lower-level execution](conditions.md#lower-level-execution).

### Save and reuse artifacts

A `PreparedIntervention` can be reused in several treatments and scenarios. The caller decides
where artifacts remain valid. The framework does not cache them. To save and restore them:

```python
from avsectester.preparation import PreparedIntervention, Treatment, prepare_intervention

# prepared comes from prepare_intervention. other_scenario is supplied by the caller.
prepared.save("tmp/artifacts/speed-spoof.json")
restored = PreparedIntervention.load(SpeedSpoof(), "tmp/artifacts/speed-spoof.json")

# Reuse artifacts in another scenario without preparing again.
reused = prepare_intervention(SpeedSpoof(), other_scenario, name="speed-spoof",
                              artifacts=restored.artifacts)
runtime = Treatment((reused,)).runtime(case=other_scenario.match)
```

`save` writes a JSON record with the name, kind, position, preparation provenance and artifacts.
`load` takes the method explicitly and imports no code from the file. Passing `artifacts=`,
including `None`, skips `prepare`. The default codec stores JSON values only. For arrays, tensors
or other objects, override `export_artifacts(artifacts)` to return JSON data, for example
`{"texture": texture.tolist()}`, and `import_artifacts(data)` to rebuild the objects, for example
with `np.asarray(data["texture"], dtype=np.uint8)`. The codec may instead write a file and return
its path. Saving the record does not copy that file. The record format is in
[Artifact codec](../reference/python-api.md#artifact-codec).

## Configure and run the attack

To run the attack from a configuration, put `SpeedSpoof` in an importable module, such as
`my_research/attacks.py`, and add an [`interventions`](../reference/configuration.md#interventions)
entry. This configuration runs offline. `my_research/stacks.py` holds `CruiseControl`, configured
as a [custom driving system](../reference/configuration.md#driving-system-keys):

```yaml
backend:
  type: nurec
  renderer: {type: stub, height: 8, width: 8}
  dt: 0.1
stack:
  factory: my_research.stacks:CruiseControl
frames: 80
interventions:
  - factory: my_research.attacks:SpeedSpoof
    kind: attack
    name: speed-spoof
    options: {bias: 5.0}
```

Save it as `attack.yaml` next to the `my_research` package. Then run a clean baseline and the
attack from the same origin:

```bash
PYTHONPATH=. avsectester run attack.yaml --treatment attack
```

```text
[pair] 80 frames per run, treatment=attack
[clean] mean_detections=0.0 peak_speed=5.10 final_speed=5.10 brake_frames=0 outputs=0
[attacked] mean_detections=0.0 peak_speed=0.00 final_speed=0.00 brake_frames=0 outputs=0
clean:    peak_speed= 5.10  final_speed= 5.10  brake_frames=0
attacked: final_speed= 0.00  brake_frames=0  overdrive_frames=0 (kept moving while clean was stopped)
=> ATTACK SUCCEEDED (forced an unsafe stop)
```

The default mode is `pair` and the default treatment is `attack`. From Python,
`run_configured(config, 80, treatment="attack")` returns the traces by treatment name. Options,
output lines and exit codes are in the [CLI reference](../reference/cli.md#avsectester-run), and
what each treatment enables in [Treatments and modes](experiments.md#treatments-and-modes).

Configurations also accept three other attack forms:
[`attacks`](../reference/configuration.md#attacks) adds native avstack hooks on modular-pipeline
component outputs, such as the [phantom injection](../examples/phantom-injection.md).
[`patches`](../reference/configuration.md#patches) adds CARLA physical panels, such as the
[physical patch](../examples/physical-patch.md). [`runtime.hooks`](../reference/configuration.md#runtime)
adds raw handler factories without preparation, which configured treatments enable as attack
handlers. At a shared stage, `attacks` and `patches` run first, then `runtime.hooks`, then
`interventions` in definition order. Sensor conditions surround this chain. See
[Ordering at a shared stage](../reference/runtime.md#ordering-at-a-shared-stage).

## Read the results

Each treatment returns a `Trace` with the physical state and executed command of every driving
step, the recorded intervention outputs and the provenance. `impact(clean, attacked)` compares
two traces with the supplied stop-based criterion and gives the verdict shown above. An attack
whose goal is not a stop, such as speeding, needs its own criterion computed from the traces.
See [Driving-impact metric](evaluation.md#driving-impact-metric) and
[Intervention outputs and provenance](evaluation.md#intervention-outputs-and-provenance).

## Next steps

- [Implement a defense](defenses.md): write a defense against your attack and evaluate both.
- [Examples](../examples/README.md): how the supplied attacks use these interfaces.
- [Runtime reference](../reference/runtime.md): every stage, value, ordering rule and context field.
- [Prepare scenarios](scenarios.md): select scenarios with roles for targeted attacks.
- [Evaluate results](evaluation.md): metrics, component outputs and robustness evaluation.
