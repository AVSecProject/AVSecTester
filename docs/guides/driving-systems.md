# Driving systems

> **Audience:** Experiment authors and driving-adapter implementers.
>
> **Outcome:** Configure an AVStack and expose its component outputs when available.
>
> **Prerequisites:** [Architecture](../concepts/architecture.md).

## AVStack interface

An AVStack accepts Observation and returns Control. It owns model-visible ObservationPipeline and per-run
model history. `reset(first_observation)` receives the already processed first input;
ObservationPipeline does not transform that input or reset the localizer twice. Its processing order
is sensors.post hooks, the localizer, localization hooks and observation hooks. It does not process
output commands.

Declare command_modes and supported_stages. Unsupported commands fail before a world starts unless
an explicit command hook converts them. Actual commands are checked again before execution. The
interface never silently translates incompatible sensor or command formats.

## Modular driving system

ModularAVStack(pipeline_cfg) builds avstack's ModularDrivingPipeline, including perception,
tracking, planning and control. It advertises only component stages that actually dispatch native
hooks. Native models remain avstack objects. CARLA-trained MMDetection3D metadata is registered
before the detector is built.

Each formal run constructs a fresh pipeline, preserving independent tracker/planner/controller
histories. `instrument()` captures optional component outputs, while detections() returns canonical
perception boxes lazily without another inference call. Unsupported perception formats return None
rather than interrupt driving. Coordinates use the true capture-time ego frame derived from sensor
extrinsics, independently of model-visible localization.

Concrete attacks are not registered merely by constructing ModularAVStack. For a native hook, import
its registration module explicitly or list it in `hooks[].imports`, and configure it with
native_hook. Supplied PhantomInjectionAttack creates its detection action directly, without hook
registration.

## Camera forward-collision policy

CameraForwardCollisionAVStack is a lightweight 2D detector/box-area braking policy used by camera
patch and recorded-playback configurations. Its options include gpu, camera, area_brake, throttle,
detector_dataset and threshold. The perception model loads lazily and is exposed for bound scorers.
It is an experimental policy, not the modular 3D planner or Alpamayo.

Its detections() returns None: 2D image boxes are not canonical 3D detections. DetectionMetrics
therefore cannot be applied without a true 3D detection output. Component/image diagnostics remain
available through recorded outputs.

## Alpamayo

AlpamayoAVStack wraps AlpaSim's trajectory-output model. It uses camera histories and the
model-visible ego state, returns Control.trajectory and exposes policy.pre/policy.post. It has no
separate perception or tracking stage. `detections()` returns None, so DetectionMetrics is not a
default metric for this driving system. Its component log contains `policy` for the model
prediction and `command` for the produced ego command.

The driver environment is Python 3.12 and distinct from the avstack Python 3.11 environment. Run
from a source checkout using the driver interpreter. See [NuRec
installation](../getting-started/installation.md#nurec-and-alpamayo-environment).

### Loading and sharing the model

Pass checkpoint_path explicitly or set AVSECTESTER_ALPAMAYO_CHECKPOINT. An explicit path takes
precedence. Model loading is lazy. Separate AVStack instances can share an injected model only if
their camera histories, previous plans and inference state remain independent.

AlpamayoAVStack.close() is terminal and idempotent: it clears that wrapper's history and model
references, and later reset or inference calls fail. A model supplied through model= is borrowed;
closing the wrapper does not close the shared model or release its device allocator. A model loaded
by the wrapper is owned and receives its optional close callback during cleanup.

<!-- runnable -->
```python
from avsectester.closed_loop.avstacks.alpamayo import AlpamayoAVStack

avstack = AlpamayoAVStack(checkpoint_path="/path/to/Alpamayo-1.5-10B", device="cuda:1")
assert avstack.command_modes == frozenset({"trajectory"})
assert avstack.detections() is None
```

**Not yet implemented.** A complete received-state history and a physical trajectory controller. See
[TODO](../development/TODO.md#vehicle-dynamics).

## Component logging interface

AVStack.component_log() optionally returns a mapping of outputs captured during the latest decision.
The runner snapshots it into StepRecord.components. Return an empty mapping when nothing is
available. It must not trigger another model inference.

`detections() -> Sequence[Detection] | None` supplies canonical 3D detections. None means the system
offers no detection output; an empty sequence means it observed zero detections. The AVStack base
class supplies default component_log(), detections() and close() methods. Position, extent and yaw
use true capture-time ego x-forward, y-left, z-up axes, metres and radians.

## Components and scorers for offline work

DrivingAdapter binds a fresh AVStack factory to ComponentDescriptor and ScorerDescriptor records.
Descriptors declare inputs, outputs, codecs and differentiable inputs. A scorer also declares
whether gradients are available. Closed-loop services reject unsupported gradient requests instead
of detaching values and continuing.

An optional `avstack.bindings` constructor receives the configured AVStack factory. It returns a
DrivingAdapter from closed_loop/avstacks/driving_adapter.py. The modular_driving_adapter binding lives in
closed_loop/avstacks/modular.py; alpamayo_driving_adapter lives in closed_loop/avstacks/alpamayo.py.
Detector scorers live in closed_loop/avstacks/scorers.py.

## Use a custom driving system

Implement AVStack, with __call__ and any required reset, detections, component_log and close
behavior. Keep heavyweight imports in the implementation rather than shared data contracts.

<!-- runnable -->
```python
from avsectester.closed_loop.avstacks.base import AVStack
from avsectester.data_types import Control, Observation

class CruiseControl(AVStack):
    def __init__(self, target_speed=5.0):
        self.target_speed = target_speed

    @property
    def command_modes(self):
        return frozenset({"actuator"})

    def __call__(self, observation):
        error = self.target_speed - observation.ego_speed
        return Control(throttle=min(1.0, max(error, 0.0)),
                       brake=min(1.0, max(-error, 0.0)))

assert CruiseControl()(Observation(0, 0, ego_speed=0)).throttle == 1
```

Select an importable class as `avstack.type: my_project.driving:CruiseControl`; place constructor
parameters under options. Keep the class in your own package; no file under
avsectester/closed_loop/avstacks needs editing. The loader calls it with those options for every
run, without passing a seed. See [Driving
specification](../reference/configuration.md#driving-specification).

Annotate filesystem constructor parameters as Path or str | Path. The loader uses these annotations
for [relative paths](../reference/configuration.md#constructor-filesystem-parameters).

## Models used in this repository

| Model | Role |
|---|---|
| CARLA PointPillars | Modular 3D perception |
| CARLA/nuScenes image detector | Camera forward-collision policy and bound scorers |
| Alpamayo-1.5-10B | End-to-end trajectory generation |
| User-bound detector scorer | Optional offline objective |
| PCTNet | Optional attack image harmonization, not a driving system |
| SAM | Offline extraction of insertion assets |

Download weights separately as described in
[Installation](../getting-started/installation.md#model-weights-and-assets).

## Next steps

Read [Configuration](../reference/configuration.md), [Optimization](optimization.md) and [Python
API](../reference/python-api.md#driving-system-api).
