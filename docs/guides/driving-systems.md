# Configure driving systems

> **Audience:** Users who run the supplied driving systems or integrate their own.  
> **Outcome:** Configure the modular and Alpamayo driving systems, check command compatibility, expose component outputs and add a custom driving system.  
> **Prerequisites:** [Architecture](../concepts/architecture.md) and [Installation](../getting-started/installation.md).

The [driving system](../concepts/architecture.md#terminology) is the system under test. It turns
each observation into a [command](../concepts/architecture.md#terminology). It is connected to the
driving loop through a stack, an `AVStack` adapter. The supplied stacks are `ModularAVStack`
(`avsectester/stacks/modular.py`) and `AlpamayoAVStack` (`avsectester/stacks/alpamayo.py`).
Configured runs select one with the `stack` key. Python runs pass a `stack_factory`.

## Vehicle I/O

Each `AVStack` owns `vehicle_io`, a `VehicleIO` that applies sensor hooks, localization and
observation hooks in that order. Set its localizer when constructing the driving adapter:

<!-- runnable -->
```python
from avsectester.backend import AVStack
from avsectester.plane import Control, StateEstimate
from avsectester.vehicle_io import VehicleIO

class Driver(AVStack):
    def __init__(self):
        self.vehicle_io = VehicleIO(self.localize)

    def localize(self, observation, context):
        return StateEstimate(observation.vehicle_state, observation.ego_speed)

    def __call__(self, observation):
        return Control()

driver = Driver()
assert driver.vehicle_io.localizer == driver.localize
```

The example keeps the provided state estimate; replace `localize` with your estimator. A stateful
localizer may implement the [plugin lifecycle](../reference/runtime.md#plugin-lifecycle).
The dispatcher resets it once before world initialization. `stack.reset` receives the first
processed input once and must not reset localization or process that input again.

Without a localizer, Vehicle I/O uses the observation's reported state after sensor hooks.
Existing `Runtime(localizer=...)` remains supported through a compatibility adapter. Configure
one localizer per run; different localizers on the stack and legacy runtime are rejected before
reset. Stage values and ordering are in the [runtime reference](../reference/runtime.md#stages).

## Modular driving system

`ModularAVStack` runs an avstack `ModularDrivingPipeline` (`avstack.modules.pipeline`). The
pipeline is called with the observation's `sensor_data` and `vehicle_state` and runs four modules
in series: perception, tracking, planning and control. The stack converts the control output to
an actuator `Control` (`throttle`, `steer`, `brake`).

The default configuration in `configs/carla_scenario.yaml` is:

```text
PointPillars (MMDetObjectDetector3D) → BasicBoxTracker3D → ForwardCollisionPlanner → VehiclePIDController
```

PointPillars is a neural LiDAR detector. The tracker, planner and PID controller are algorithmic
components. `ForwardCollisionPlanner` (`avstack.modules.planning.vehicle`) drives straight ahead:

- At every driving step it replaces the plan with one waypoint `d_forward` metres (default 3)
  ahead along the current heading.
- The waypoint's target speed is `target_speed`. It is zero when any tracked object lies between
  `min_forward` (default 1.0) and `brake_distance` (default 12.0) metres ahead and within
  `brake_corridor` (default 2.5) metres laterally, measured in the ego body frame. The
  controller then brakes.
- It does not follow a route. `ego.destination` is not used.

A perception attack that adds an object in this corridor therefore makes the ego stop. See
[Phantom injection](../examples/phantom-injection.md).

The pipeline runs outside the CARLA actor. `CarlaBackend` gives the avcarla actor an empty internal
pipeline, and the driving loop sends the stack's command back to the backend.

### Pipeline configuration

Configured CARLA runs build the modular stack from `ego.pipeline` when `stack` is omitted. The
mapping is an avstack configuration passed to `ModularAVStack(pipeline_cfg)`:

```yaml
ego:
  pipeline:
    type: ModularDrivingPipeline
    perception: {type: MMDetObjectDetector3D, model: pointpillars, dataset: carla-vehicle, gpu: 0,
                 threshold: 0.3}
    tracking: {type: BasicBoxTracker3D}
    planning: {type: ForwardCollisionPlanner, target_speed: 6.0, brake_distance: 12.0,
               brake_corridor: 2.5}
    control:
      type: VehiclePIDController
      args_lateral: {K_P: 1.2, K_D: 0.1, K_I: 0.05}
      args_longitudinal: {K_P: 0.4, K_D: 0.0, K_I: 0.05}
```

- `perception`, `tracking`, `planning` and `control` are avstack module configurations. Each
  `type` names a module registered in avstack. Other keys are the module's constructor
  arguments.
- With several sensors in `sensor_data`, set the pipeline's `perception_input` to the sensor name
  that feeds perception. A single sensor is used automatically. Several sensors without
  `perception_input` raise `ValueError`. When the selected sensor has no data in a step,
  perception reuses the previous data.
- The CLI's `--gpu` option sets `pipeline.perception.gpu`. Other keys and validation are in
  [`ego`](../reference/configuration.md#ego).

CARLA-trained PointPillars checkpoints need MMDetection3D's `CarlaDataset` metadata, even for
inference. `ModularAVStack` calls `register_carla_dataset()` (`avsectester/stacks/mmdet3d.py`)
before building the pipeline when perception is `MMDetObjectDetector3D` with a `dataset` starting
with `carla-` and `deploy` unset. Call it yourself before building such a detector outside the
stack. It keeps an existing registration. Otherwise it imports the declaration from the installed
MMDetection3D package or, when absent there, from the nested `third_party/mmdetection3d` submodule
of avstack-core. It raises `FileNotFoundError` when neither is available. The checkpoints are
downloaded as described in
[CARLA perception checkpoints](../getting-started/installation.md#carla-perception-checkpoints).

Constructing a `ModularAVStack` imports the avstack modules and registers the supplied avstack
hooks, such as `PhantomInjection`. Importing `avsectester.stacks` does not require avstack.

`stack.reset` clears only the component-logging captures. It does not rebuild the state of the
avstack tracker, planner or controller, which have no common reset contract. Use a fresh
`ModularAVStack` for each independent run. The runners call `stack_factory` for every treatment.
See [Plugin lifecycle](../reference/runtime.md#plugin-lifecycle).

### Component stages

`ModularAVStack.supported_stages` contains `<component>.pre` and `<component>.post` for each of
`perception`, `tracking`, `planning` and `control` whose call dispatches avstack hooks. The four
modules of the default pipeline all do, so it exposes eight stages. A module that bypasses avstack
hook dispatch exposes no stage.

| Stage | Handler receives and returns |
|---|---|
| `<component>.pre` | `CallInputs` with the module's `args` and `kwargs`. Return a `CallInputs` to replace them |
| `<component>.post` | The module's native output, after the module's own avstack post-hooks |

The arguments of each component are listed in
[Component inputs](../reference/runtime.md#component-inputs). For a custom module, inspect
`stack.native[component]`. `stack.native` maps `pipeline`, `perception`, `tracking`, `planning`
and `control` to the built objects, including modules without hook dispatch. `stack.models` maps
each component to its loaded `model` attribute, or to the module itself when it has none.

`NativeHook` (`avsectester.interventions`) adapts a registered avstack hook to these stages. The
`attacks` shorthand binds avstack post-hooks at `<component>.post`. A native pre-hook needs a
`runtime.hooks` entry with an `avstack` mapping at a `.pre` stage. See
[`attacks`](../reference/configuration.md#attacks) and
[`runtime`](../reference/configuration.md#runtime).

`stack.instrument(stages=...)` captures the output of each named component after all `.post`
handlers have run. The default is all four components. A component without a stage raises
`ValueError`. Configured runs without `stack` instrument `perception`. The CARLA runner uses it to
fill `FrameRecord.n_detections`. See
[Component logging interface](#component-logging-interface).

## Alpamayo

`AlpamayoAVStack` wraps NVIDIA's Alpamayo-1.5-10B end-to-end camera policy
(`alpasim_driver.models.alpamayo1_5_model.Alpamayo15Model`). It outputs a planned trajectory, not
actuator values, so it needs a backend that accepts `trajectory` commands. The supplied path is
NuRec with `TrajectoryFollower` dynamics. A complete configured run is in
[Backend configuration](simulators.md#backend-configuration). It runs in the AlpaSim
[driver environment](../getting-started/installation.md#driver-environment).

Constructor options are `checkpoint_path`, `camera_ids` (default `["camera_front_wide_120fov"]`),
`device` (`"cuda:0"`), `context_length` (`4`), `output_frequency_hz` (`10`) and `model`. In YAML,
use `stack: {type: AlpamayoAVStack, options: {...}}`. See
[Driving system keys](../reference/configuration.md#driving-system-keys).

### Inputs and outputs

At each driving step, the stack builds an AlpaSim `PredictionInput` from the observation:

- **Camera frames.** The image of each camera in `camera_ids` is read from
  `observation.sensor_data` and appended to a buffer of the last `context_length` frames. At the
  start of a run the buffer is padded by repeating its earliest frame. A step without an image for
  a camera adds no frame.
- **Speed.** `observation.ego_speed`. Acceleration is fixed at `0.0`, the drive command is
  `STRAIGHT` and no route is given.
- **Ego history.** Poses at intervals of `1 / output_frequency_hz` seconds over at least 1.6
  seconds before the current time, which covers the model's required 1.5 seconds. They are
  synthesized from the current `x`, `y`, `yaw` and `ego_speed` at constant velocity, not taken
  from earlier steps.
- **Previous plan and seed.** The `selected_plan` of the previous prediction and an inference seed
  that increases by one at each call.

**Not yet implemented.** An ego history built from the states the driving system actually
received.

Model-facing timestamps are `observation.t` plus a 10-second offset, so that the backward history
keeps positive timestamps. The output waypoints use the backend clock instead.

`model.predict` returns a `ModelPrediction`. The stack takes the candidate at `selected_index`
from `candidate_positions` (K×T×3, rig frame) and returns it as `Control.trajectory`. Waypoint `i`
is timestamped `observation.t + i / output_frequency_hz`, starting at `i = 1`, in microseconds.
The format is in [Trajectory commands](../reference/runtime.md#trajectory-commands). The
prediction's `reasoning_text`, when present, is kept in `stack.last_reasoning`.

Two stages surround the model call:

| Stage | Value |
|---|---|
| `policy.pre` | The `PredictionInput`, before `model.predict`. Return a replacement to change the model input |
| `policy.post` | The `ModelPrediction`, before the trajectory is built |

Their fields are listed in [Values at each stage](../reference/runtime.md#values-at-each-stage).
Alpamayo has no separate perception, tracking, planning or control stages. `stack.native` exposes
`model` and `camera_history`, the per-camera frame buffers.

### Loading and sharing the model

Importing the module does not import Torch or `alpasim_driver`. The model loads at `reset`, or at
the first call when no reset preceded it. The checkpoint is resolved in this order:

1. `model=`: an already loaded model is used and no checkpoint is read.
2. `checkpoint_path=`: an explicit directory.
3. The `AVSECTESTER_ALPAMAYO_CHECKPOINT` environment variable, read when the stack is constructed.

Without any of them, loading raises `ValueError` before the driver package is imported. Weights are
described in [Alpamayo checkpoint](../getting-started/installation.md#alpamayo-checkpoint).

`stack.models` is `{"policy": model}` once loaded, and empty before. Pass it as `model=` to create
further adapters around the same weights. Each adapter keeps its own camera history, previous plan
and inference seed. `reset` clears these and `last_reasoning`. The supplied model must implement
the AlpaSim driver's prediction contract. Share it only when its own mutable state is safe to
reuse. This factory loads the model once and reuses it for every later treatment:

<!-- runnable -->
```python
from avsectester.stacks.alpamayo import AlpamayoAVStack


class SharedAlpamayo:
    """Create Alpamayo adapters that reuse the model loaded by the first one."""

    def __init__(self, **options):
        self.options = options
        self.previous = None

    def __call__(self):
        model = self.previous.models.get("policy") if self.previous is not None else None
        self.previous = AlpamayoAVStack(model=model, **self.options)
        return self.previous


stack_factory = SharedAlpamayo(checkpoint_path="/path/to/Alpamayo-1.5-10B", device="cuda:0")
```

The runners create the stacks one after another, so the first run loads the model before the
second stack is created.

## Models used in this repository

The repository uses models for driving, auxiliary detection and image preparation. They are not
stages of one pipeline:

| Model | Role and entry point | Takes part in driving decisions |
|---|---|---|
| Alpamayo-1.5-10B | End-to-end camera policy through `AlpamayoAVStack` | Yes, outputs a trajectory |
| PointPillars, CARLA vehicle weights | LiDAR perception of the modular pipeline in `configs/carla_scenario.yaml` | Yes, detections feed tracking, planning and control |
| Faster R-CNN, CARLA vehicle weights, MMDetection | `build_detector` in `scripts/common/demo_common.py`, and the patch optimization scripts | Only in `patch_driving_demo.py`, where detections feed a braking rule. Other uses inspect or optimize detector response |
| Faster R-CNN ResNet-50 FPN, COCO weights, torchvision | `build_coco_detector` in `scripts/common/demo_common.py`, used by the NuRec and nuScenes object demos with `--eval` | No, auxiliary detection scores only |
| PCTNet | `PCTNetHarmonizer` adjusts the appearance of inserted images | No, optional harmonization |
| SAM, default `facebook/sam-vit-huge` | `scripts/preparation/extract_person_cutouts.py` prepares pedestrian RGBA assets | No, offline segmentation |
| NuRec through `nre-ga` | `NuRecRenderer` renders camera images at the requested pose | No, scene rendering |

COCO names the detector's pretraining dataset. It does not add a COCO scene source, and the
auxiliary detector's scores are neither Alpamayo predictions nor closed-loop outcomes.

Two demo driving systems are not models. `CruiseStack` (`scripts/common/demo_common.py`) holds a
fixed throttle. `CameraForwardCollisionStack` (`scripts/demos/carla/patch_driving_demo.py`) brakes
when a detected car box exceeds an area fraction of the image. Dependencies and loading are in
[Model weights and assets](../getting-started/installation.md#model-weights-and-assets). Each
script's components are listed in the [script index](../../scripts/README.md).

## Commands and compatibility

A stack declares the command modes it produces in `command_modes`, and a backend declares the
modes it accepts. `ModularAVStack` produces `actuator` commands. `AlpamayoAVStack` produces
`trajectory` commands. `CarlaBackend` accepts `actuator`. `NuRecBackend` accepts the mode of its
dynamics: `actuator` by default, `trajectory` with `TrajectoryFollower`. The full table and rules
are in [Command compatibility](../reference/runtime.md#command-compatibility). In summary:

- Before the world resets, every declared mode of the stack must be accepted by the backend.
  Otherwise `CommandCompatibilityError` (a `ValueError`) is raised.
- A `command` handler may convert between modes. Its presence skips the check before reset. The
  final command is still validated after all `command` handlers.
- An undeclared side (`None`) skips the check before reset. An undeclared backend also accepts any
  final command.

Matching command modes is necessary but not sufficient. The interfaces do not convert sensor,
state or coordinate formats. `ModularAVStack` expects avstack sensor wrappers and an avstack ego
state, which CARLA provides and NuRec does not. `AlpamayoAVStack` expects `HWC` `uint8` camera
arrays under its `camera_ids` and an ego state with `x`, `y` and `yaw`, which NuRec provides. The
supplied combinations are listed in
[Supported combinations](../concepts/architecture.md#supported-combinations).

**Not yet implemented.** A trajectory-to-actuator controller with vehicle dynamics, which would let
Alpamayo drive in CARLA.

A `command` handler can convert a trajectory to actuator values so that a trajectory-producing
stack runs with actuator dynamics. The converted command must set `trajectory` to `None`. Only the
command is converted.

## Component logging interface

A stack exposes its latest intermediate outputs through the `InstrumentedStack` protocol in
`avsectester.evaluation.component_log`:

```python
class InstrumentedStack(Protocol):
    def component_log(self) -> dict[str, Any] | None: ...
```

`component_log()` returns the outputs of the most recent call to the stack, keyed by component
name, or `None` when nothing is captured. It returns outputs the driving system already produced
and runs no extra inference.

| Stack | Outputs |
|---|---|
| `ModularAVStack` | The components selected with `stack.instrument(stages=...)`: by default `perception`, `tracking`, `planning` and `control`. `None` until `instrument()` is called |
| `AlpamayoAVStack` | `policy`, the `ModelPrediction`, and `action`, the returned `Control`. No `instrument()` call is needed |
| Custom stack | Any named outputs. The collector does not require a fixed schema |

The supplied stacks capture outputs after their stage handlers, so the outputs include runtime
replacements made at those stages. They do not include replacements made later at `command`.
Collect the outputs with `run_logged` and analyze them with `ComponentTrace`, as described in
[Component outputs](evaluation.md#component-outputs).

## Use a custom driving system

Subclass `avsectester.backend.AVStack`. Only `__call__` is abstract:

| Member | Default | Contract |
|---|---|---|
| `__call__(observation) -> Control` | Abstract | Decide the command for the processed observation |
| `reset(observation)` | No-op | Called once per run with the first processed observation, before the first call. Clear per-run state here |
| `command_modes` | `None` (undeclared) | `frozenset` of `"actuator"` and/or `"trajectory"`. Declare it so that incompatible backends fail before reset |
| `supported_stages` | Empty | The component stages the stack emits |
| `models` | Empty mapping | Loaded models that handlers can read through `context.stack.models` |
| `native` | Empty mapping | Native objects for handlers, available as `context.native["stack"]` |
| `component_log()` | Absent | Optional. See [Component logging interface](#component-logging-interface) |

Rules for component stages, emitted with `self._emit(stage, value)`:

- Declare every emitted stage in `supported_stages`. A handler at an undeclared stage fails before
  reset, and emitting an undeclared stage during a run raises `ValueError`.
- Use the returned value in the next computation: `value = self._emit(stage, value)`. Outside a
  run, `_emit` returns the value unchanged.
- Stage names are free. Handlers at `perception.pre`, `tracking.pre`, `planning.pre` and
  `control.pre` must return `CallInputs`, so emit those names only with `CallInputs` values.
- A stack is bound to one run at a time.

Keep heavy imports, such as Torch or a model package, inside the methods that load the model, so
that the module imports without them. Load large models in `reset` or on first use.

This example runs offline with NuRec and the stub renderer, which returns black frames. The stack
has a perception and a planning component, each followed by a stage, and logs both. An attack at
`perception.post` adds a phantom obstacle:

<!-- runnable -->
```python
# my_driver.py
from avsectester.backend import AVStack
from avsectester.plane import Control


class TwoStageStack(AVStack):
    """Bright pixels are an obstacle, and an obstacle means stop. Each component has a stage."""

    def __init__(self, camera="camera_front", threshold=200, throttle=0.5):
        self.camera, self.threshold, self.throttle = camera, threshold, throttle
        self.latest = None

    @property
    def command_modes(self):
        return frozenset({"actuator"})

    @property
    def supported_stages(self):
        return frozenset({"perception.post", "planning.post"})

    def reset(self, observation):
        self.latest = None

    def __call__(self, observation):
        image = observation.sensor_data[self.camera]
        detections = ["obstacle"] if image.max() > self.threshold else []
        detections = self._emit("perception.post", detections)  # use the returned value
        plan = self._emit("planning.post", "stop" if detections else "cruise")
        self.latest = {"perception": detections, "planning": plan}
        return Control(brake=1.0) if plan == "stop" else Control(throttle=self.throttle)

    def component_log(self):
        return self.latest
```

In Python, pass a `stack_factory`, a callable without arguments that returns a new stack. The
runners call it for every treatment:

```python
from avsectester.evaluation.component_log import run_logged
from avsectester.runner import run_prepared
from avsectester.runtime import Hook, Runtime
from avsectester.scenarios import scenario_from_config

from my_driver import TwoStageStack


def phantom(detections, context):
    """Add a phantom obstacle from driving step 10."""
    return [*detections, "phantom"] if context.step >= 10 else detections


scenario = scenario_from_config({"backend": {"type": "nurec"}})  # stub renderer, black frames
runtimes = {"clean": None, "attack": Runtime([Hook("perception.post", phantom)])}
results = run_prepared(scenario, TwoStageStack, frames=20, runtimes=runtimes, run_fn=run_logged)
trace, components = results["attack"]
print(components.counts("perception")[8:12])
print([step.stages["planning"] for step in components.steps[8:12]])
print([record.brake for record in trace.records[8:12]])
# [0, 0, 1, 1]
# ['cruise', 'cruise', 'stop', 'stop']
# [0.0, 0.0, 1.0, 1.0]
```

The logged planning output changes at the same step as the perception output, and the executed
command brakes from that step.

In a configuration, name an importable factory under `stack.factory` as `package.module:callable`.
It is called as `factory(**options)` and must return an `AVStack`. A stack class is such a
factory. The module must be importable, for example from a directory on `PYTHONPATH`:

```yaml
backend: {type: nurec}
stack:
  factory: my_driver:TwoStageStack
  options: {throttle: 0.3}
```

`run_configured(config, 20, mode="single", treatment="clean")` with this mapping drives the clean
treatment to a final speed of 0.9 m/s. An import failure, a non-callable factory or a returned
object that is not an `AVStack` raises `ConfigurationError`. The stack must consume the backend's
sensor and state formats. For a custom simulator, see
[Write a custom simulator adapter](simulators.md#write-a-custom-simulator-adapter).

## Next steps

- [Implement an attack](attacks.md)
- [Evaluate results](evaluation.md)
- [Configure simulators](simulators.md)
- [Runtime reference](../reference/runtime.md)
- [Configuration reference](../reference/configuration.md#driving-system-keys)
