# Driving systems

> **Audience:** Experiment authors and driving-adapter implementers.
>
> **Outcome:** Configure an AVStack and expose its component outputs when available.
>
> **Prerequisites:** [Architecture](../concepts/architecture.md).

## AVStack interface

An AVStack accepts Observation and returns Control. It owns model-visible ObservationPipeline and per-run
model history. `reset(first_observation, past=())` receives the processed first input and
recorded past observations ordered oldest to newest. ObservationPipeline does not transform that
input or reset the localizer twice. Its processing order
is sensors.post hooks, the localizer, localization hooks and observation hooks. It does not process
output commands.

Declare command_modes and supported_stages. Unsupported commands fail before a world starts unless
an explicit command hook converts them. Actual commands are checked again before execution. The
interface never silently translates incompatible sensor or command formats.

### Recorded input history

Declare the required number of preceding decision observations with the AVStack class attribute
`past_steps`, which defaults to 0. The simulator must advertise `can_observe_past` and the selected
start must have that much recorded history. NuRec provides these inputs through
`observe_past(count)`; unavailable or insufficient recorded history is an error, never a reason to
fabricate or repeat observations.

The runner and candidate rollouts apply the treatment's sensor corruptions to each past observation
using its own absolute frame position. Past inputs do not pass through stage hooks,
ObservationPipeline or attack actions; world objects present at the start, such as objects inserted
during setup, appear in them because they existed before it. Their t and frame values are negative
relative to the start.
Reset receives them separately from the first processed current input. The RunLog initialization
event records `past_steps` and each past observation's t and vehicle_state in `past_observations`,
without storing those images.

During selection, `session.can_start_at(position)` checks both the deployment window and the
required history. Built-in attacks use this check; experiment preparation also discards segments
that cannot start.

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
prediction and `command` for the produced ego command. It also records route_waypoints,
navigation_command, navigation_note, history_pose_count and selected_image_times.

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

Alpamayo declares past_steps=15: its model consumes 16 poses at 0.1 s intervals, covering 1.5 s
(`NUM_HISTORY_STEPS` and `HISTORY_TIME_STEP` in AlpaSim). reset requires that real observation
history and checks rounded observation timestamps are exactly 100 ms apart before loading its
model. Its `(context_length - 1) * frame_subsample` image span must not exceed 15 decisions.
AlpamayoAVStack declares `seed`. Like AlpaSim's driver, a seeded stack loads the model with
force_determinism and seeds each decision from the run seed and the decision count since reset.
Unlike AlpaSim's seed plus count, distinct run seeds yield non-overlapping inference seeds.
force_determinism enables deterministic PyTorch algorithms for the whole process, and cuBLAS then
requires CUBLAS_WORKSPACE_CONFIG to have been set before the process's first GPU matrix product.
Importing the adapter, which configuration loading does before any model runs, sets it to
:4096:8 unless it is already set, so detector optimization and Alpamayo can share one process.
Each decision appends the received pose. As in AlpaSim's driver, the model input's speed is the
horizontal speed of Observation.vehicle_state (ego_speed and lateral_speed) and its acceleration is the
state's longitudinal acceleration; Alpamayo 1.5 models do not read either value.
Images are taken from the received camera history at frame_subsample intervals. Missing history or
images raises an error; neither poses nor images are fabricated or repeated.

frame_subsample defaults to 1. The policy selects context_length images from the latest
`(context_length - 1) * frame_subsample + 1` observations, retaining the current frame. The supplied
NuRec configurations use four images with stride 1, matching Alpamayo's 10 Hz image history: at
decision zero, their times are −0.3, −0.2, −0.1 and 0.0 s relative to the selected start.

Observation.route becomes an AlpaSim Route at the same rounded microsecond epoch as the pose
history. AlpaSim's determine_command_from_route supplies the navigation command. Without a route,
the model receives route=None and STRAIGHT, with the fallback recorded in navigation_note.
Control.trajectory begins at the current rig origin with identity rotation and t0, followed by the
selected model waypoints and their rotations. Model-provided waypoint timestamps are preserved.

For one-shot offline policy evaluation, alpamayo_driving_adapter's policy component takes
`{"observation": current, "past": past_observations}`. The caller must supply the actual history;
a single image is insufficient. This component input contains the same data used by reset.

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

DrivingAdapter exposes `past_steps` without constructing an AVStack or loading its model. If the
factory has an `avstack_type` attribute naming an AVStack subclass, its class-level declaration wins.
Otherwise the adapter's explicit `past_steps` option is used, defaulting to 0. Configuration loading
attaches the configured AVStack type to its factory. alpamayo_driving_adapter explicitly supplies
15 as its fallback, so a plain Python factory without avstack_type still selects valid start frames.

An optional `avstack.bindings` constructor receives the configured AVStack factory. It returns a
DrivingAdapter from closed_loop/avstacks/driving_adapter.py. The modular_driving_adapter binding lives in
closed_loop/avstacks/modular.py; alpamayo_driving_adapter lives in closed_loop/avstacks/alpamayo.py.
Detector scorers live in closed_loop/avstacks/scorers.py.

Your own binding adds scorers without editing framework files. This one keeps Alpamayo's policy
component and adds a black-box scorer over the planned trajectory: the forward progress of the
waypoint closest to 3 s. Reference it as `avstack.bindings.type: my_project.bindings:alpamayo_progress`.
A white-box scorer instead runs a differentiable forward pass through `avstack.models["policy"]`,
returns a tensor and declares its differentiable inputs.

<!-- runnable -->
```python
from avsectester.closed_loop.avstacks.alpamayo import alpamayo_driving_adapter
from avsectester.closed_loop.avstacks.driving_adapter import DrivingAdapter, ScorerDescriptor

def planned_progress(avstack, inputs, target):
    avstack.reset(inputs["observation"], past=inputs["past"])
    trajectory = avstack(inputs["observation"]).trajectory
    position, _, _ = min(trajectory, key=lambda waypoint: abs(waypoint[2] - 3_000_000))
    return float(position[0])

def alpamayo_progress(factory):
    policy = alpamayo_driving_adapter(factory)
    descriptor = ScorerDescriptor(
        "planned_progress", {"observation": "current Observation", "past": "15 past Observations"},
        {"score": "forward metres at 3 s"}, {"score": "float"}, direction="minimize")
    return DrivingAdapter(factory, components=tuple(policy.component_bindings.values()),
                          scorers=((descriptor, planned_progress),), past_steps=policy.past_steps)

adapter = alpamayo_progress(lambda seed: None)
assert set(adapter.component_bindings) == {"policy"} and set(adapter.scorer_bindings) == {"planned_progress"}
```

`ClosedLoopSession.score("planned_progress", inputs, target)` then evaluates it on a fresh policy
history for each request.

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
run. Declare a `seed` constructor parameter to receive the run seed, so repeated runs with one seed
are reproducible and treatments sharing a seed are paired. See [Driving
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
