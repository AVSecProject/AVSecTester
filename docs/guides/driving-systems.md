# Driving systems

> **Audience:** Experiment authors and driving-system implementers.
>
> **Outcome:** Select a supplied driving system or bind a custom system.
>
> **Prerequisites:** [Architecture](../concepts/architecture.md).

## `AVStack` interface

`AVStack` receives an `Observation` and returns a `Control`. An externally controlled ego uses
`None` instead. It declares supported command modes, stages and optional recorded-input
requirements. The simulator and driving system must agree on commands and decision timing.
See [Configuration](../reference/configuration.md#driving-specification).

Interfaces live in `avsectester.closed_loop.avstacks`. Supplied implementations live below
`avstacks.implemented`. User implementations belong in the user's package.

`ObservationPipeline` processes current input in this order:

1. `sensors.post` hooks.
2. If a localizer is configured, `localization.pre` hooks followed by the localizer.
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
`duration_us` requests recorded poses from the run origin backward on the decision and
`pose_reporting_interval_us` clock. When the duration's lower boundary falls between pose
timestamps, the nearest preceding one is included. The pose context also covers preceding frame
ends. `samples` requests each sensor's recent frames, including the frame ending at the origin.
The preceding `samples - 1` frames use that sensor's `frame_interval_us` and
`shutter_duration_us`. Recorded context bypasses hooks, localization and sensor actions, but
sensor environments apply.

`AVStack.required_history(options)` resolves needs before loading a model and defaults to the
class's `history`. When the factory declares an `AVStack` subclass, `DrivingAdapter` uses that
type's `required_history(factory.avstack_options)` and `decision_time_step`. Otherwise the
adapter's own declarations apply. An instance whose history differs from its adapter raises
an error. History requirements control selection and recorded reads.

Read newly delivered frames, oldest first, through `observation.delivered(sensor_id)` and new
pose estimates through `observation.estimated_states()`. Convert measurements with their
frame's pose and use the decision state for planning. See
[delivery and coordinates](../reference/runtime.md#delivery-and-coordinates).

`reset(observation)` receives the first input with its recorded context. Driving systems cache the
history they need. A sensor whose due frames were all dropped is missing for that decision.
Supplied driving systems leave dropped input absent.

The runner calls `__call__` once per decision, or `ingest(observation)` when
`skip_driver_during_force_gt` skips inference. Stateful driving systems can retain input in `ingest`
without predicting. Do not ingest and call the same observation twice.

## Modular avstack pipeline

`ModularAVStack` wraps avstack's perception, tracking, planning and control pipeline. Each run
constructs its own stack and component histories. Native models and component outputs remain avstack
objects. Before inference, each native sensor body reference uses that sensor's capture-time
model-visible pose. Native payload timestamps remain acquisition timestamps. The current decision
state separately reaches planning and control. Localization therefore affects perception in the
frame the driving system uses without moving physical truth. Recorded inputs retain their capture
poses. When no new perception frame arrives, the native pipeline uses its latest retained frame with
its original timestamp, while planning and control use the decision state.

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

This policy exposes 2D component diagnostics. `DetectionMetrics`
requires a driving system that provides 3D detections. See [Evaluation](evaluation.md).

## NuRec rule-based driving system

`RuleBasedAVStack` composes a `YOLOXDetector`, `StopSignTracker` and `RuleBasedPlanner`.
It requires the selected camera's `CameraCalibration` and a planar ego state. Detection runs
when a newly delivered camera frame is available. Stop-sign detections use known-size ranging
and bearing EKF tracking. Tracking uses the camera sample's pose and acquisition time, while
planning uses the decision pose and clock. A decision without a new camera frame retains tracks
without updating perception or tracking, then replans with the current decision state. LiDAR
frames are retained independently, so camera updates can use the latest earlier LiDAR scan.
Supplying a `VehicleTracker` also enables vehicle following, which additionally requires the
selected LiDAR's `PointCloud` and `LidarCalibration`. With `vehicle_tracker: null`, vehicle
detections are ignored and the stop-sign driving behavior is preserved.

The planner stops before relevant route-side signs, waits, then resumes cruise. A sign
facing another approach can trigger a stop if it passes the route filters. With late
detections, the vehicle can stop farther along the route than intended.

The supplied modular stop-sign and vehicle insertion configurations use this driving system.

### Optional vehicle following

To enable following, start from the
[vehicle insertion configuration](../../configs/nurec_vehicle_insertion_modular_2604.yaml).
Its `vehicle_tracker` option enables following. Keep the detector and configure the NuRec
renderer's LiDAR and the stack's matching `lidar_id`. The stack projects LiDAR points into
the camera capture frame and tracks visible vehicle faces. It follows the nearest relevant
vehicle and continues to stop for signs. Late detections, occlusion and limited braking can
cause a collision. Recorded traffic follows its trajectories without responding to the ego
vehicle and can hit it from behind. `DrivingMetrics.collision` records top-down contact or
overlap of physical bounding boxes.

The API reference gives the formulas and association rules for
[ranging](../reference/python-api.md#avsectesterclosed_loopavstacksimplementedmodulesranging),
[tracking](../reference/python-api.md#avsectesterclosed_loopavstacksimplementedmodulestracker),
[planning](../reference/python-api.md#avsectesterclosed_loopavstacksimplementedmodulesplanner)
and the [stack](../reference/python-api.md#avsectesterclosed_loopavstacksimplementedmodulesstack).

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
cameras of AlpaSim's Alpamayo 1.5 policy: front wide, front tele and both cross cameras. The
renderer must provide every listed camera.

Component logs contain `policy`, `command`, `route_waypoints`, `navigation_command`,
`navigation_note`, `history_pose_count`, `selected_image_times` and `selected_capture_ids`.
Without a route, the navigation command falls back to `STRAIGHT`.

Images and poses use acquisition timestamps. The stack requires 1.5 seconds of pose history
and a frame cache of `(context_length - 1) * frame_subsample + 1` camera acquisitions. Defaults
are `context_length=4` and `frame_subsample=1`. Starting from the newest frame, it selects
`context_length` frames at a stride of `frame_subsample`, then supplies them oldest first.
The planning origin is the newest image timestamp across the configured cameras.
Retained poses span from 1.5 seconds before that image through the current decision, including a
preceding pose when the boundary lies between estimates. The model resamples timestamped poses
at its own required spacing. Acquisition intervals may vary. Missing context is rejected rather
than fabricated.

The returned trajectory preserves the model's generation-time rig frame and native waypoint
timestamps. The controller handles reference-frame conversion. Native policy output, including
optional reasoning, remains available through component logging.

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
driving systems. Discovery reads static declarations and configured options without constructing a
stack or loading model weights. Select a binding with `avstack.bindings`. Constructor injection
belongs to [Configuration](../reference/configuration.md#driving-specification).

`rule_based_driving_adapter` exposes `rule_based.detection_confidence`. Its `sensor_id` comes
from the configured `camera_id` or the stack default, and `decision_threshold` comes from
`detector.score_threshold`. Missing detector configuration or a detector other than
`YOLOXDetector` raises `ValueError`. The scorer accepts any COCO class name as `target.label`.

| Binding | Purpose |
|---|---|
| `modular_driving_adapter` | Modular perception components and the `mmdet.inference` and `mmdet.rpn_objectness` scorers |
| `rule_based_driving_adapter` | Detection-confidence scoring for the rule-based system |
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

## Hosted Autoware

`ExternalAVStack` lets CARLA accept commands from a process outside the framework.
`AutowareDrivingSystem` hosts that process for each closed-loop run. Configure it under
`simulator.options.external_driving_system` and select `ExternalAVStack` as the driving
system. Follow [Hosted Autoware](../examples/hosted-autoware.md) for the supplied comparison
and [Simulators](simulators.md#hosted-external-driving-systems) for the lifecycle.

## Use a custom driving system

Implement `AVStack` in a user package and reference it as
`type: my_project.driving:CustomAVStack`. Declare required history, compatible command modes and
optional decision timing. A separate binding factory exposes components or scorers when the
experiment needs them.

Use `ExternalAVStack` to represent a driving system that runs outside the framework and controls
ego through the simulator. It declares empty command modes and returns `None`. For CARLA setup
and runtime restrictions, see [native ROS 2 control](simulators.md#native-ros-2-control).

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
