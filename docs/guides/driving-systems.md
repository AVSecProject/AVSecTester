# Driving systems

> **Audience:** Experiment authors and driving-adapter implementers.
>
> **Outcome:** Configure an AVStack and expose its component outputs when available.
>
> **Prerequisites:** [Architecture](../concepts/architecture.md).

The interfaces live in `avsectester.closed_loop.avstacks` (`base.py` and
`driving_adapter.py`). Framework-provided driving systems live in
`avsectester.closed_loop.avstacks.implemented`; import each implementation from its own module.
The interfaces package does not import concrete driving systems.

User-defined driving systems belong in their own package. Configure them with
`type: package.module:Class`; adding one does not require editing the framework.

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

## NuRec rule-based driving system

RuleBasedAVStack is a trajectory-only stop-sign policy. Its implementation lives in
closed_loop/avstacks/implemented/rule_based/: detector.py loads and runs the COCO stop-sign detector,
ranging.py estimates a sign position from known size and camera rays, tracker.py tracks stationary
signs with a bearing EKF, planner.py follows the route with a free-road cruise term and
constant-deceleration stopping, and stack.py connects these stages to AVStack. It provides no
canonical 3D detections.

The stack composes a required CocoStopSignDetector with an optional StopSignTracker and
StopSignPlanner. Omitting either creates its default instance. These classes and the frozen
SignTrack record are exported from the rule_based package. Only the top-level RuleBasedAVStack
has a built-in short name. Nested component specifications use their full module:Class paths.
A top-level device value is injected into the detector, rather than into the stack:

```yaml
device: cuda:0
avstack:
  type: RuleBasedAVStack
  options:
    detector:
      type: avsectester.closed_loop.avstacks.implemented.rule_based:CocoStopSignDetector
      options:
        weights_path: ../models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth
```

The stack reads camera RGB and requires CameraCalibration for the chosen camera in
Observation.calibration. NuRec supplies this optical-camera-to-rig transform and camera model;
missing calibration is an explicit error explaining the current NuRec requirement. Ranging
unprojects the top and bottom box centres and uses their angular separation with sign_height_m
(default 0.762 m). For ray angle θ, range is sign_height_m / (2 tan(θ / 2)); the box-centre ray is then
transformed from the optical camera frame into the rig frame. Torchvision and Camera use the same
continuous pixel coordinates; no half-pixel conversion is applied. Range error scales in direct
proportion to the assumed sign height.

The stack validates calibration.rig_from_camera before taking its translation as the camera
optical centre in rig coordinates; ranging also validates this rigid transform. CameraCalibration
does not validate array fields at construction, so RunLog.read(load_payloads=False) can retain a
payload reference without loading it.

StopSignTracker converts each measured sign position and the camera optical centre into horizontal
world coordinates using the model-visible PlanarEgoState. A new SignTrack contains a track_id,
a two-dimensional world position and a 2×2 position covariance. Existing tracks persist through
missing detections; reset clears them and starts track IDs at zero again. For measured camera-to-sign
range r, the measurement covariance has radial standard deviation
hypot(range_sigma_fraction * r, min_position_sigma_m) and tangential standard deviation
hypot(bearing_sigma_rad * r, min_position_sigma_m). Association uses the smallest Mahalanobis
distance under the sum of track and measurement covariances, accepting distances at most
max_association_mahalanobis_distance. Unmatched measurements initialize new stationary tracks.

After association, only the measured bearing updates an existing track through an EKF; the
known-size range is used for initialization and association, avoiding repeated fusion of its
systematic size bias. Bearing residuals wrap to (−π, π], and there is no process noise for these
stationary signs. The default parameters are range_sigma_fraction=0.3, bearing_sigma_rad=0.007,
min_position_sigma_m=0.3 and max_association_mahalanobis_distance=3.5. The measured bearing standard
deviation was 0.0067 rad; 0.3 fractional range uncertainty covers the observed systematic bias and
jitter (worst relative error −35%). The 0.3 m position floor limits overconfidence in measurement
and association noise and helps avoid duplicate tracks.

At every decision StopSignPlanner takes the tracker output, transforms each world position into the
current rig frame and projects it onto Observation.route, a navigation intent with X forward and
Y left. The planner does not maintain its own sign-position memory or radius-based association.
Handled signs and the current target are identified by track_id. When the route is absent it uses
a forward straight line; endpoint tangents continue short routes.

Signs behind the route origin or beyond sign_lookahead_m are ignored. A sign is also ignored when
its distance to the nearest route point exceeds max_sign_lateral_offset_m. The default lateral
limit is 10.0 m: in 32 selected NuRec segments the mapped sign centres are 2.8108–7.2283 m from the
recorded route; adding 2 m to the maximum and rounding up gives 10 m. A stop sign governing another
intersection direction can still trigger a stop if it falls within these ranges.

The stop target is the projected sign arc length minus stop_before_sign_m. The default 5.5 m uses
map stop-line geometry and ego front overhang in the same 32 segments: the geometric target places
the vehicle front at least 0.5 m before the line in 29/32 segments (90.6%, meeting the 90% condition).
This calculation excludes monocular ranging error.

The planner parameters make units and roles explicit:

| Parameter | Default | Meaning |
|---|---:|---|
| stop_before_sign_m | 5.5 | Route distance before the sign at which to stop (m) |
| stop_duration_s | 2.0 | Time held in STOPPED (s) |
| stop_speed_mps | 0.1 | Speed below which the ego is considered stopped (m/s) |
| stop_position_tolerance_m | 2.0 | Maximum remaining target distance for entering STOPPED (m) |
| sign_lookahead_m | 60.0 | Furthest forward route projection considered for a sign (m) |
| max_sign_lateral_offset_m | 10.0 | Maximum sign-to-route lateral distance (m) |
| cruise_speed_mps | None | Desired cruise speed; None uses reset speed (m/s) |
| min_cruise_speed_mps | 5.0 | Lower bound on reset-derived cruise speed (m/s) |
| max_acceleration_mps2 | 1.0 | Maximum free-road acceleration (m/s²) |
| comfortable_deceleration_mps2 | 3.0 | Required-deceleration threshold for starting a stop (m/s²) |
| max_deceleration_mps2 | 6.0 | Maximum commanded planning deceleration (m/s²) |
| acceleration_exponent | 4.0 | IDM free-road exponent |
| trajectory_horizon_s | 5.0 | Planned duration (s) |
| trajectory_step_s | 0.1 | Integration and waypoint interval (s) |

Cruising and approach motion before braking use the IDM free-road acceleration term:

```text
a_free = max_acceleration_mps2 * [1 - (v / v0)^acceleration_exponent]
```

The standard IDM free-road symbols are a = max_acceleration_mps2 and δ = acceleration_exponent.
There is no virtual leader, standstill-gap or time-headway term. comfortable_deceleration_mps2
instead determines when distance-based stopping begins.

For a STOPPING target, let d be the remaining route distance to that target. At each integration
substep the required constant deceleration is required = v² / (2 d) for d > 0. Braking starts when
required reaches comfortable_deceleration_mps2, or immediately when d ≤ 0. Once braking starts it
stays latched for that target, including across decisions, until the target changes, the planner
enters STOPPED, or no target remains. Before this threshold the planner uses a_free.

After braking begins, acceleration is -required, capped at -max_deceleration_mps2. At d ≤ 0 it
uses -max_deceleration_mps2. The planner does not accelerate again for that target while braking
is latched. If detection is so late that required exceeds max_deceleration_mps2, braking is capped
and the ego may overshoot the target. CRUISE always uses the free-road term.

The state machine is CRUISE → STOPPING → STOPPED → CRUISE. It enters STOPPED when speed is below
stop_speed_mps and the remaining target gap is at most stop_position_tolerance_m. It holds for
stop_duration_s using observation time, marks that track_id handled and resumes. Handled tracks
are ignored. Each trajectory begins at the rig origin and identity quaternion at the observation
timestamp, with strictly increasing integer microsecond timestamps. STOPPED repeats the origin.
trajectory_horizon_s must cover at least one trajectory_step_s, and the step must round to a
positive integer number of microseconds.

The component log records boxes and confidence, estimated rig positions and ranges, tracks,
planner state, the target track_id (None without a stop target), the latched braking flag and
target gap. Each track entry contains its track_id,
world position and maximum position standard deviation (the square root of the covariance's largest
eigenvalue). acceleration_mps2 and planned_speed_mps describe the first integration step of the
planned trajectory. The controller follows the whole trajectory and may begin slowing earlier,
so the logged first-step acceleration can be positive while the physical vehicle is decelerating.

The detector reads a local Faster R-CNN COCO_V1 checkpoint; it downloads no weights.
CocoStopSignDetector.close() releases its loaded or injected model, after which detect and target_score raise
RuntimeError; RuleBasedAVStack.close() calls it. The supplied nurec_stop_sign_patch_modular_2601.yaml
and nurec_stop_sign_patch_modular_2604.yaml configurations use the same checkpoint for driving and
white-box scoring.

This policy only handles stop signs. It does not perceive vehicles or obstacles and follows the
navigation route centreline. With recorded, non-reactive traffic, following vehicles can rear-end
the stopped ego, and vehicles intruding into the lane can be sideswiped. DrivingMetrics.collision
counts the recorded top-down bounding-box overlaps, including these cases.

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
from avsectester.closed_loop.avstacks.implemented.alpamayo import AlpamayoAVStack

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
selected model waypoints and their rotations. Model-provided waypoint timestamps are preserved. output_frequency_hz (default 10) is only the
fallback when the model returns no waypoint timestamps; it does not set the experiment step.

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
closed_loop/avstacks/implemented/modular.py; alpamayo_driving_adapter lives in closed_loop/avstacks/implemented/alpamayo.py.
Detector scorers live in closed_loop/avstacks/implemented/scorers.py.

Your own binding adds scorers without editing framework files. This one keeps Alpamayo's policy
component and adds a black-box scorer over the planned trajectory: the forward progress of the
waypoint closest to 3 s. Reference it as `avstack.bindings.type: my_project.bindings:alpamayo_progress`.
A white-box scorer instead runs a differentiable forward pass through `avstack.models["policy"]`,
returns a tensor and declares its differentiable inputs.

<!-- runnable -->
```python
from avsectester.closed_loop.avstacks.implemented.alpamayo import alpamayo_driving_adapter
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
