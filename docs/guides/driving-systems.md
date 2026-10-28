# Driving systems

> **Audience:** Experiment authors and driving-system implementers.
>
> **Outcome:** Select a supplied driving system or bind a custom system.
>
> **Prerequisites:** [Architecture](../concepts/architecture.md).

## `AVStack` interface

`AVStack` receives an `Observation` and returns a `Control`. It declares supported command
modes, stages and optional recorded-input requirements. The simulator and driving system must
agree on commands and decision timing;
see [Configuration](../reference/configuration.md#driving-specification).

Interfaces live in `avsectester.closed_loop.avstacks`. Supplied implementations live below
`avstacks.implemented`; user implementations belong in the user's package.

`ObservationPipeline` processes current input in this order:

1. `sensors.post` hooks.
2. `localization.pre` hooks and the configured localizer, when present.
3. `localization.post` hooks.
4. `observation` hooks.

The pipeline owns the localizer lifecycle and does not process output commands.
[Runtime](../reference/runtime.md) defines the complete per-decision order and record formats.

### Recorded input history

History acquisition and processing follow
[Past observations](simulators.md#recorded-past-observations). `AVStack.past_steps` declares the
required history and is read without constructing the model.

## avstack modular pipeline (`ModularAVStack`)

The `*_modular_*` stop-sign configurations use `RuleBasedAVStack`, described below.

`ModularAVStack` wraps avstack's perception, tracking, planning and control pipeline. Each run
gets fresh component histories. Native models and component outputs remain avstack objects.

`instrument` captures selected component outputs. `detections` converts supported perception
boxes to the true capture-time ego frame without another inference call; model-visible
localization does not change that coordinate conversion. Unsupported perception output returns
`None`, while an empty sequence means supported perception with no detections.

`NativeHook` wraps explicitly registered native hook implementations. Registration modules can
be listed in the hook configuration. The framework's phantom action can instead modify the
declared perception stage directly. See [Hooks](../reference/configuration.md#hooks).

## Camera forward-collision policy

`CameraForwardCollisionAVStack` uses a 2D image detector and image-box area to choose braking or
throttle. Its learned model loads lazily and can be shared with bound scorers.

`CameraForwardCollisionAVStack` exposes its scorer through `modular_driving_adapter`.

This policy exposes 2D component diagnostics, not canonical 3D detections. `DetectionMetrics`
requires a driving system that provides 3D detections; see [Evaluation](evaluation.md).

## NuRec rule-based driving system

`RuleBasedAVStack` composes a `YOLOXDetector`, `StopSignTracker` and `StopSignPlanner`.
Its processing chain is detection, known-size ranging, bearing EKF tracking and trajectory
planning. `YOLOXDetector` is a general COCO detector; the stack uses its `stop sign` detections.
It requires the selected camera's `CameraCalibration` and a planar ego state.

The cruise reference `v0` is `cruise_speed_mps`, or the speed at reset bounded below by
`min_cruise_speed_mps` when cruise speed is omitted. The tracker never deletes tracks, adds no
process noise, and wraps bearing residuals across ±π. Stop signs facing another direction can
still trigger a stop if they pass the route filters. `provides_detections=False` makes this
stack incompatible with `DetectionMetrics`.

The stack selects stop-sign boxes and confidence. Ranging uses the angular height of a
box and the assumed sign height, then transforms the point from optical-camera coordinates into
rig coordinates. Distance error scales with error in the assumed physical sign size.

The tracker stores stationary signs in world coordinates. Range and bearing uncertainty
determine association; bearing-only EKF updates refine associated tracks. The planner identifies
a target and served signs by track ID.

The planner projects tracks onto the navigation route, filters signs outside its forward/lateral
range, and places the stop target before a relevant sign. With no route it uses a straight
route. Cruise acceleration is the IDM free-road term, `a_max * (1 - (v / v0) ** delta)`. Here
`a_max` is `max_acceleration_mps2` and `delta` is `acceleration_exponent`.

For a stop target at remaining route distance `d`, the required deceleration is `v² / (2d)`.
Reaching `comfortable_deceleration_mps2` latches braking; the planner then follows that required
deceleration and does not return to cruise for the same target. Acceleration is limited by the
configured acceleration/deceleration bounds. Late detection can require more braking than
available and can produce a stop beyond the target.

After the speed and position criteria are met, the planner holds its stop for the configured
duration, marks that track served, and resumes cruise. Reset clears tracking and planning state.
The [Python API](../reference/python-api.md) lists defaults, parameter meanings and errors.

The component log records boxes, estimated positions, tracks with covariance summaries, target
distance, state and braking. `acceleration_mps2` and `planned_speed_mps` describe the first
planned trajectory step. A controller tracks the whole forecast and can slow the vehicle while
that planned first-step acceleration is positive.

This system handles stop signs only. It follows the navigation centerline without perceiving
other vehicles or obstacles. Recorded non-reactive traffic can hit ego from behind; intruding
vehicles can also overlap its path. `DrivingMetrics.collision` records those physical overlaps.

## Alpamayo

`AlpamayoAVStack` uses AlpaSim's policy interface to turn camera input, ego pose history, speed,
acceleration and navigation into a rig-frame trajectory. Navigation comes from
`Observation.route`, which the simulator generates as route intent. It is independent of the
executed future path; see [Navigation](simulators.md#navigation).

Component logs contain `policy`, `command`, `route_waypoints`, `navigation_command`,
`navigation_note`, `history_pose_count` and `selected_image_times`. Without a route, the
navigation command falls back to `STRAIGHT`.

Camera images and pose history must match the declared decision timeline. History validation
rejects missing or inconsistent observations. The stack transforms policy outputs into the
common trajectory command format. Native policy output, including optional reasoning, remains
available through component logging.

### Loading and sharing the model

Local checkpoint loading happens lazily. Injecting a model allows the caller to supply an
already-constructed policy. Closing releases a model loaded by the stack; an injected model is
borrowed and is not closed. Closing the stack is terminal and idempotent. Checkpoint locations
and installation are specified in
[Installation](../getting-started/installation.md#alpamayo-weights).

## Component logging interface

`component_log` returns optional component outputs for the most recent decision; the runner
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
| `alpamayo_driving_adapter` | Policy prediction from current and preceding observations |

The Alpamayo binding declares 15 preceding decisions when a factory has no `AVStack`
type metadata.

Scorers declare supported input formats, differentiable inputs, objective direction, camera and
decision threshold where applicable. A white-box attack can use the same detector as the driving
system through its scorer binding; see [Optimization](optimization.md).

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
| COCO YOLOX-m | General COCO detection and its bound scorer; the rule-based stack selects stop signs |
| Alpamayo-1.5-10B | Camera/history/navigation policy |
| PCTNet | Optional image harmonization |
| SAM | Extraction of insertion assets |

[Installation](../getting-started/installation.md) describes sources and weight setup.

## Next steps

-  [Configure an experiment](../reference/configuration.md).
-  [Use component/scorer services](attacks.md#offline-phase).
-  [Evaluate driving output](evaluation.md).
