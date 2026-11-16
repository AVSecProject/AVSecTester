# Driving systems

> **Audience:** Experiment authors and driving-system implementers.
>
> **Outcome:** Select a supplied driving system or bind a custom system.
>
> **Prerequisites:** [Architecture](../concepts/architecture.md).

## `AVStack` interface

`AVStack` receives an `Observation` and returns a `Control`. It declares supported command
modes, stages and optional recorded-input requirements. The simulator and driving system must
agree on commands and decision timing.
See [Configuration](../reference/configuration.md#driving-specification).

Interfaces live in `avsectester.closed_loop.avstacks`. Supplied implementations live below
`avstacks.implemented`. User implementations belong in the user's package.

`ObservationPipeline` processes current input in this order:

1. `sensors.post` hooks.
2. `localization.pre` hooks and the configured localizer, when present.
3. `localization.post` hooks.
4. `observation` hooks.

The pipeline owns the localizer lifecycle and does not process output commands.
When localization changes the planar ego pose, the pipeline re-expresses the route in the
estimated ego frame and keeps its world positions. Navigation then follows the model-visible
pose, as a real vehicle locates its route with its own localization. A route requires finite
planar `x`, `y` and `yaw` before and after localization.
[Runtime](../reference/runtime.md) defines the complete per-decision order and record formats.

### Recorded input history

`AVStack.history` is a `HistoryRequirement(duration_us, samples)` for the first decision.
`duration_us` requests recorded poses on the same decision and `pose_reporting_interval_us`
clock used during the run. Its interval ends at the origin, where every sensor's first delivered
frame ends. Include the closest preceding pose-clock timestamp when the lower boundary lies
between timestamps. With zero reporting interval, only decision-grid poses are requested.
`samples` requests each sensor's recent
frames, including its origin frame. Pose context also covers preceding frame ends for pose
interpolation. Exposure starts extend selection's source reservation without extending delivered
pose history. The preceding `samples - 1` payloads come from the recording,
using that sensor's `frame_interval_us` and `shutter_duration_us`. They bypass online hooks,
localization and sensor attack actions. Sensor environments still apply.

`AVStack.required_history(options)` resolves option-dependent needs before model loading.
When the factory's `avstack_type` is an AVStack subclass, `DrivingAdapter.history` uses
`avstack_type.required_history(factory.avstack_options)` and `decision_time_step` uses the
type's value. Otherwise the adapter's own declarations apply. That classmethod defaults to
the type's `history`. The constructed instance must declare the same history as its adapter.
A mismatch raises an error rather than changing the instance. `required_sensors` names required
inputs, or is `None` to accept the configured
sensor set. The shared history plan controls selection, recorded reads and simulator caches.

The framework delivers each ended frame once. Read new frames, oldest first, with
`observation.delivered(sensor_id)` and new poses with `observation.estimated_states()`.
Frame estimates interpolate the model-visible pose trajectory at frame end. Use the decision
state for planning. See [Runtime](../reference/runtime.md#delivery-and-coordinates).

`reset(observation)` receives the first input with its recorded context. There is no separate
history argument. Drivers retain their own references. `AlpamayoAVStack` uses bounded frame
caches matching AlpaSim's FrameCache and a timestamped pose cache. Other supplied drivers retain
their latest input references without adding old frames to the framework observation. A sensor
whose due frames were all dropped is missing for that decision. Supplied drivers do not fill it
from an earlier frame.

The normal path calls `__call__` once per decision. When `skip_driver_during_force_gt` skips
inference, the runner calls `ingest(observation)` instead. Stateful drivers can retain input
there without predicting. Do not ingest and call the same observation twice.


## avstack modular pipeline (`ModularAVStack`)

The `*_modular_*` stop-sign configurations use `RuleBasedAVStack`, described below.

`ModularAVStack` wraps avstack's perception, tracking, planning and control pipeline. Each run
gets fresh component histories. Native models and component outputs remain avstack objects.
Before inference, each native sensor body reference uses that sensor's capture-time
model-visible pose. Native payload timestamps remain acquisition timestamps. The current
decision state separately reaches planning and control. Localization therefore affects perception
in the frame the driver believes without moving physical truth. Recorded inputs retain their
capture poses. When no new perception frame arrives, the native pipeline uses its latest
retained frame with its original timestamp, while planning and control use the decision state.

`instrument` captures selected component outputs. `detections` converts supported perception
boxes to the capture-time ego frame without another inference call. These ego-frame boxes do
not depend on localization. Unsupported perception output returns `None`, while an empty
sequence means supported perception with no detections.

`NativeHook` wraps explicitly registered native hook implementations. Registration modules can
be listed in the hook configuration. The framework's phantom action can instead modify the
declared perception stage directly. See [Hooks](../reference/configuration.md#hooks).

## Camera forward-collision policy

`CameraForwardCollisionAVStack` uses a 2D image detector and image-box area to choose braking or
throttle. It uses the latest delivered camera image and retains its acquisition metadata.
A decision without a new image reuses its most recent detection result.
Its learned model loads lazily and can be shared with bound scorers.

`CameraForwardCollisionAVStack` exposes its scorer through `modular_driving_adapter`.

This policy exposes 2D component diagnostics, not canonical 3D detections. `DetectionMetrics`
requires a driving system that provides 3D detections. See [Evaluation](evaluation.md).

## NuRec rule-based driving system

`RuleBasedAVStack` composes a `YOLOXDetector`, `StopSignTracker` and `RuleBasedPlanner`.
It requires the selected camera's `CameraCalibration` and a planar ego state. Detection runs
once per decision on the latest delivered camera. Stop-sign detections use known-size ranging
and bearing EKF tracking. Tracking uses the camera sample's pose and acquisition time, while
planning uses the decision pose and clock. A decision without a new camera frame retains tracks
without updating perception or tracking, then replans with the current decision state. LiDAR
frames are retained independently, so camera updates can use the latest earlier LiDAR scan.
Supplying a `VehicleTracker` also enables vehicle following, which additionally requires the
selected LiDAR's `PointCloud` and `LidarCalibration`. With `vehicle_tracker: null`, vehicle
detections are ignored and the stop-sign driving behavior is preserved.

The cruise reference `v0` is `cruise_speed_mps`, or the speed at reset bounded below by
`min_cruise_speed_mps` when cruise speed is omitted. The stop-sign tracker never deletes tracks,
adds no process noise, and wraps bearing residuals across ±π. Stop signs facing another direction
can still trigger a stop if they pass the route filters. `provides_detections=False` makes this
stack incompatible with `DetectionMetrics`.

The stack selects stop-sign boxes and confidence. Ranging uses the angular height of a
box and the assumed sign height, then transforms the point from optical-camera coordinates into
rig coordinates. Distance error scales with error in the assumed physical sign size.

The tracker stores stationary signs in world coordinates. Range and bearing uncertainty
determine association. Bearing-only EKF updates refine associated tracks. The planner identifies
a target and served signs by track ID.

The planner projects tracks onto the navigation route, filters signs outside its forward/lateral
range, and places the stop target before a relevant sign. With no route it uses a straight
route. Cruise acceleration is the IDM free-road term, `a_max * (1 - (v / v0) ** delta)`. Here
`a_max` is `max_acceleration_mps2` and `delta` is `acceleration_exponent`.

For a stop target at remaining route distance `d`, the required deceleration is `v² / (2d)`.
Reaching `comfortable_deceleration_mps2` latches braking. The planner then follows that required
deceleration and does not return to cruise for the same target. Acceleration is limited by the
configured acceleration/deceleration bounds. Late detection can require more braking than
available and can produce a stop beyond the target.

After the speed and position criteria are met, the planner holds its stop for the configured
duration, marks that track served, and resumes cruise. Reset clears tracking and planning state.
The [Python API](../reference/python-api.md) lists defaults, parameter meanings and errors.

### Optional vehicle following

Enable the tracker in the driving-system options:

```yaml
avstack:
  type: RuleBasedAVStack
  options:
    vehicle_tracker:
      type: avsectester.closed_loop.avstacks.implemented.modules:VehicleTracker
```

Keep the detector options from the supplied configuration. `vehicle_labels` defaults to
`[car, truck, bus]` and can select other detector class names. `lidar_id` defaults to
`lidar_gt_top_p128`. The calibrated transform converts LiDAR points to its capture rig. When
camera and LiDAR times differ, their capture-time model-visible poses move the points into the
camera capture rig before projection. Same-time inputs retain their original coordinate path.

With following enabled, detection uses
`min(detector.score_threshold, vehicle_tracker.low_score_threshold)` for that call.
Stop signs still require `detector.score_threshold`. This remains the detector's configured
threshold and the bound scorer's `decision_threshold`. With following disabled, detection uses
the configured threshold and does not read LiDAR.

`estimate_vehicle_face_points(box_xyxy, calibration, rig_points)` projects the rig-coordinate
LiDAR points into the camera. It keeps points more than 0.5 m forward in camera coordinates,
then selects the box's middle 80% horizontally and 10% to 60% vertically from its top.
Horizontal distance to the camera defines a near cluster: points no farther than the 20th
percentile plus 1 m. Both the selected region and this cluster require at least five points.
Insufficient points raise `ValueError`, and the stack skips and counts that vehicle detection.

The result is a `(3, 3)` rig-coordinate array in meters, ordered left, center and right, with
`z = 0` as a horizontal representation. All three x coordinates use the near cluster's 10th
percentile. Left and right y use its 95th and 5th percentiles, respectively. Center y is the
midpoint of those two endpoints.

`VehicleTracker.update(vehicle_state, t, face_points_rig, scores)` transforms each three-point
measurement into horizontal world coordinates using ego's planar pose. Scores correspond
one-to-one with measurements and must be finite in `[0, 1]`. Each track has a constant-velocity
Kalman state `[x, y, vx, vy]`. Prediction uses white acceleration noise.
The face midpoint supplies the position measurement, with isotropic standard deviation
`max(min_measurement_noise_m, measurement_noise_fraction * horizontal_range)`.
Here horizontal range is measured from the rig origin to the face midpoint.
Initial position covariance uses that measurement variance. Initial velocity is zero, with
standard deviation `initial_velocity_sigma_mps`.

ByteTrack-style association has two stages. Measurements at or above `high_score_threshold`
first associate with all retained tracks, including lost tracks. Measurements from
`low_score_threshold` up to but excluding the high threshold then associate only with tracks
matched in the previous update and still unmatched in the current update. Each stage greedily
selects the smallest squared Mahalanobis innovation within `association_gate_chi2`, using each
track and measurement once. Only unmatched high-score measurements create tracks.
The score thresholds must satisfy `0 <= low_score_threshold <= high_score_threshold <= 1`.

Associated tracks receive a Kalman update. `half_width_m` is half the measured visible face's
lateral span. Unassociated tracks retain their predicted motion for re-association and are
deleted after `max_missed_updates` consecutive misses. `tracks` returns independent snapshots
only for tracks matched in the latest update and with at least `min_hits` measurements,
counting initialization as the first hit. Lost tracks remain internal. Equal timestamps allow
measurement updates without prediction. Backward timestamps are rejected. `reset()` clears
the tracks and timestamp history.

| Tracker option | Default |
|---|---:|
| `acceleration_noise_mps2` | 2 m/s² |
| `min_measurement_noise_m` | 0.3 m |
| `measurement_noise_fraction` | 0.05 |
| `initial_velocity_sigma_mps` | 5 m/s |
| `association_gate_chi2` | 9.21 |
| `high_score_threshold` | 0.5 |
| `low_score_threshold` | 0.1 |
| `min_hits` | 2 |
| `max_missed_updates` | 10 |

The planner projects each vehicle onto the navigation route, with signed lateral offset
positive to the left. The vehicle's lateral interval is its center offset plus or minus
`half_width_m`. It is relevant when that interval overlaps the ego corridor
`[-(ego_half_width_m + lateral_margin_m), +(ego_half_width_m + lateral_margin_m)]` and its
forward arc distance is positive and within `vehicle_lookahead_m`. Touching interval edges
counts as overlap. The nearest eligible vehicle ahead supplies the lead.
The planner takes its velocity along the route tangent, bounded below by zero.
The distance from ego's origin to that visible face is reduced by `ego_front_offset_m` to
obtain the following gap `s`. The points represent the near side of the detected vehicle,
so the planner does not subtract another vehicle half-length.

Following uses the IDM interaction term:

```text
s_star = s0 + max(0, v*T + v*(v - v_lead)/(2*sqrt(a_max*b)))
a_follow = a_max * (1 - (v/v0)**delta - (s_star/s)**2)
```

Here `v` is ego speed, `v0` the cruise reference, `v_lead` the lead speed and `delta` the existing
acceleration exponent. `a_max` is `max_acceleration_mps2`, and `b` is
`comfortable_deceleration_mps2`. A nonpositive gap requests maximum deceleration. Every forecast
step uses the smaller of the sign/free-road acceleration and following acceleration, then
applies the existing acceleration limits. The lead advances along the route at constant speed
within the forecast. Timed stop-sign holds still keep ego stationary.

| Planner option | Meaning | Default |
|---|---|---:|
| `vehicle_lookahead_m` | Maximum forward route distance of an eligible lead | 60 m |
| `ego_half_width_m` | Half of ego's horizontal width | 1 m |
| `lateral_margin_m` | Additional corridor width on each side | 0.3 m |
| `min_gap_m` (`s0`) | Desired stationary following gap | 2 m |
| `time_headway_s` (`T`) | Desired time headway | 1.5 s |
| `ego_front_offset_m` | Distance from the rig origin to ego's front | 3.7 m |

`min_gap_m` is a desired gap, not a guaranteed minimum clearance. A very close lead can cease
to be detected. Without a matched lead, the planner resumes cruise unless a stop sign constrains
it. Occlusion and truncated boxes can distort the estimated face position and width.
NuRec LiDAR returns from scene-inserted assets have larger surface error near the ground. See
[Scene insertion](simulators.md#scene-insertion). The system follows the centerline without lane
changes. Limited braking and late detection can cause an overlap. Recorded non-reactive traffic
can also hit ego from behind. `DrivingMetrics.collision` records physical box overlaps.

### Component outputs

The component log retains sign boxes, position estimates, world tracks, target distance, state
and braking. `state` describes stop-sign handling. Vehicle following adds:

| Field | Contents |
|---|---|
| `vehicle_detections` | Vehicle boxes, scores and class labels at the per-call detection threshold |
| `vehicle_estimates` | `rig_face_points_m` left/center/right points and `detection_index` |
| `vehicle_tracks` | Currently matched, confirmed track IDs, world positions/velocities, `half_width_m` and `hits` |
| `skipped_vehicle_detections` | Number of vehicle boxes without enough LiDAR points |
| `lead_track_id` | Selected vehicle track ID, or `None` |
| `lead_gap_m` | Along-route distance from ego's front to the visible face, or `None` |
| `lead_speed_mps` | Lead speed along the route, or `None` |
| `lead_lateral_interval_m` | Lead's signed lateral interval in meters, or `None` |

`lead_gap_m` is an estimated face distance. It differs from the ground-truth bounding-box clearance
reported by `DrivingMetrics`. `acceleration_mps2` and `planned_speed_mps` describe the first
planned trajectory step. A controller tracks the whole forecast and can slow the vehicle while
that planned first-step acceleration is positive. Reset clears both trackers and planning state.

## Alpamayo

`AlpamayoAVStack` uses AlpaSim's policy interface to turn camera input, ego pose history, speed,
acceleration and navigation into a rig-frame trajectory. Navigation comes from
`Observation.route`, which the simulator generates as route intent. It is independent of the
executed future path. See [Navigation](simulators.md#navigation). By default it uses the four
cameras of AlpaSim's Alpamayo 1.5 driver: front wide, front tele and both cross cameras. The
renderer must provide every listed camera.

Component logs contain `policy`, `command`, `route_waypoints`, `navigation_command`,
`navigation_note`, `history_pose_count`, `selected_image_times` and `selected_capture_ids`.
Without a route, the navigation command falls back to `STRAIGHT`.

Images and poses use acquisition timestamps. The stack requires 1.5 seconds of pose history
and `(context_length - 1) * frame_subsample + 1` camera acquisitions. The defaults are
`context_length=4` and `frame_subsample=1`. It selects every Nth frame
backward from the newest delivered frame, then supplies them oldest first, matching AlpaSim's
FrameCache. Each camera cache retains exactly the required sample capacity. The planning origin
is the newest image timestamp across the configured cameras. A separate pose cache spans from
1.5 seconds before that image through the current decision, including a preceding pose when the
boundary lies between estimates. Both caches retain delivered references without copying
payloads. The model resamples timestamped poses at its own required spacing. Acquisitions need
not be 100 ms apart. Missing context is rejected rather than fabricated.

The returned trajectory preserves the model's generation-time rig frame and native waypoint
timestamps. The controller handles reference-frame conversion, as in AlpaSim. The adapter does
not move the trajectory to the decision rig. Native policy output, including optional reasoning,
remains available through component logging.

### Loading and sharing the model

Local checkpoint loading happens lazily. Injecting a model allows the caller to supply an
already-constructed policy. Closing releases a model loaded by the stack. An injected model is
borrowed and is not closed. Closing the stack is terminal and idempotent. Checkpoint locations
and installation are specified in
[Installation](../getting-started/installation.md#alpamayo-weights).

## Component logging interface

`component_log` returns optional component outputs for the most recent decision. The runner
snapshots them independently. `detections` returns canonical 3D detections or `None` when
unsupported. The [Runtime reference](../reference/runtime.md) describes field layout
and serialization.

## Components and scorers for offline work

A `DrivingAdapter` binds reusable component/scorer declarations to a factory that creates fresh
driving systems. Discovery reads static declarations and configured options without constructing
a stack or loading model weights. Constructor injection belongs
to [Configuration](../reference/configuration.md#driving-specification).

`rule_based_driving_adapter` exposes `rule_based.detection_confidence`. Its `sensor_id` comes
from the configured `camera_id` or the stack default, and `decision_threshold` comes from
`detector.score_threshold`. Missing detector configuration or a detector other than
`YOLOXDetector` raises `ValueError`. The scorer accepts any COCO class name as `target.label`.

| Binding | Purpose |
|---|---|
| `modular_driving_adapter` | Modular perception components and bound image scorers |
| `rule_based_driving_adapter` | The rule-based driver's detection-confidence scorer |
| `alpamayo_driving_adapter` | Policy prediction from one observation containing delivered context |

The Alpamayo binding uses the stack's option-dependent acquisition-history declaration,
including when the factory has no `AVStack` type metadata.

Scorer callbacks have signature `(avstack, images, targets)` and return one score per input.
Built-in descriptors name the input `images` and output `scores`. `output_shape=()` describes
one input's scalar score. Differentiable scorers declare `"BCHW float tensor [0,1]"` and
`differentiable_inputs=("images",)`. Inference scorers declare `"list of HWC RGB uint8"`.
Image and target batch lengths must agree.

Scorers also declare objective direction, camera and decision threshold where applicable. A
white-box attack can use the same detector as the driving system through its scorer binding.
See [Optimization](optimization.md).

## Use a custom driving system

Implement `AVStack` in a user package and reference it as
`type: my_project.driving:CustomAVStack`. Declare required history, compatible command modes and
optional decision timing. A separate binding factory exposes components or scorers when the
experiment needs them.

A minimal actuator policy needs no simulator connection:

<!-- runnable -->
```python
from avsectester.closed_loop.avstacks import AVStack
from avsectester.data_types import Control, Observation


class CruiseStack(AVStack):
    command_modes = frozenset({"actuator"})

    def __call__(self, observation):
        return Control(throttle=0.2)


stack = CruiseStack()
observation = Observation(0.0, 0)
stack.reset(observation)
assert stack(observation).throttle == 0.2
stack.close()
```

## Models used in this repository

| Model | Role |
|---|---|
| CARLA PointPillars | Modular 3D perception |
| CARLA/nuScenes image detector | Camera forward-collision policy and bound scorers |
| COCO YOLOX-m | COCO detection and its bound scorer. The rule-based stack selects stop signs and optional vehicles |
| Alpamayo-1.5-10B | Camera/history/navigation policy |
| PCTNet | Optional image harmonization |
| SAM | Extraction of insertion assets |

[Installation](../getting-started/installation.md) describes sources and weight setup.

## Next steps

-  [Configure an experiment](../reference/configuration.md).
-  [Use component/scorer services](attacks.md#offline-phase).
-  [Evaluate driving output](evaluation.md).
