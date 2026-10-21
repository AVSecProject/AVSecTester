# Runtime reference

> **Audience:** Adapter and attack implementers.
>
> **Outcome:** Look up decision order, stages, contexts and recorded evidence.
>
> **Prerequisites:** [Architecture](../concepts/architecture.md).

## Initialization

The caller creates SimulatorRuntime and AVStack. The runner validates stages, world settings and
command compatibility before resetting the world. For selected CARLA origins, reset follows
prescribed motion without attack hooks to the shared deployment window. simulator.setup applies
world actions, then deployment and capture ticks produce the selected observation. At position zero
these ticks hold base actors. A positive origin earlier than the window is rejected.

AttackRuntime.setup receives subscribed setup state and returns only WorldAttackAction. Its
setup_ticks declares its deployment budget. Preparation and execution share the maximum budget over
the attack's segments, seeds and treatments. Actions register cleanup on the provided ExitStack. The
simulator runtime records actual initial state; it does not compare against replay-error thresholds
or reselect the origin.

## One decision

```mermaid
sequenceDiagram
    participant R as ClosedLoopRunner
    participant S as SimulatorRuntime
    participant A as AttackRuntime
    participant V as VehicleIO
    participant D as AVStack
    participant L as Recorder
    R->>S: observe and state (no advancement)
    R->>A: step(subscribed snapshots)
    A-->>R: world, sensor and AVStack actions
    R->>R: apply sensor actions, then sensor environment
    R->>V: sensor hooks, localization, observation hooks
    R->>D: processed observation
    D-->>R: produced command and component outputs
    R->>R: command actions and hooks, validate
    R->>S: apply queued world actions, step(executed command)
    S-->>R: post-state and next capture
    R->>L: step record, actions and receipts
```

Decision zero resets AVStack history with the same already processed observation used for its first
call. Input transforms execute once; the localizer is not reset again. World actions returned during
a decision cannot change its already captured input. They affect subsequent capture. Sensor and
AVStack actions operate on current-decision values.

Attack actions have three boundaries: WorldAttackAction changes the simulated world;
SensorAttackAction changes any sensor capture, including LiDAR, GNSS and IMU; AVStackAttackAction
changes driving-system values from localization.pre through command. Sensor actions chain in issue
order on private data, preserving the original capture. Native data wrappers retain calibration
and acquisition metadata; camera buffers remain RGB uint8. AVStackAttackAction rejects sensors.post.
That stage remains a normal HookSet stage.

SensorView.camera and cam_from_world are optional and only describe optical camera geometry.
sensor_from_world is the world-to-sensor-body transform (x forward, y left, z up), including for
cameras. It is distinct from cam_from_world (x right, y down, z forward) and is None when pose data
is unavailable. SimulatorRuntime.sensors lists all sensor views, and camera_sensors is its subset.

### CARLA LiDAR data

A CARLA LiDAR view's data is an owned SensorCapture. Its raw_data buffer contains packed float32
records in the order x, y, z, intensity: four values, or 16 bytes, per point. Coordinates use
CARLA's local left-handed sensor frame: x forward, y right, z up. These raw bytes have not been
flipped; they differ from sensor_from_world's canonical body axes, whose y points left.

Writing a replacement capture retains the native LidarData wrapper and its flipy setting. With
flipy=True, the native export conversion negates y. Return points in the original CARLA axes;
do not pre-flip y and then apply that conversion again. Preserve the capture's frame, timestamp
and metadata when constructing replacement bytes.

This action decodes the points, shifts them one metre forward and returns a new SensorCapture:

<!-- runnable -->
```python
import numpy as np
from avsectester.attack.runtime import SensorAttackAction
from avsectester.runlog.codecs import SensorCapture

class ShiftLidarPointsAction(SensorAttackAction):
    sensor = "lidar"

    def apply(self, data, view):
        points = np.frombuffer(data.raw_data, dtype=np.float32).reshape(-1, 4).copy()
        points[:, 0] += 1.0
        return SensorCapture(
            native_type=data.native_type,
            raw_bytes=points.tobytes(),
            frame=data.frame,
            timestamp=data.timestamp,
            metadata=dict(data.metadata),
        )
```

Return the action from AttackRuntime.step; the runner supplies its current capture and SensorView.

## Hook stages

A hook is `Hook(stage, handler)`. The handler is callable `(value, StageContext) -> value`, or
provides transform with that signature. Handlers run in registration order, consuming previous
replacements. Adapter-specific stages must be declared; unsupported stages fail before reset.

| Stage | Value |
|---|---|
| simulator.setup | None; native world setup |
| simulator.step.pre/post | None; before/after physical advancement |
| sensors.post | Observation |
| localization.pre | Observation |
| localization.post | StateEstimate |
| observation | Observation after localization |
| perception.pre/post | CallInputs / component output, when declared |
| tracking.pre/post | CallInputs / component output, when declared |
| planning.pre/post | CallInputs / component output, when declared |
| control.pre/post | CallInputs / component output, when declared |
| policy.pre/post | End-to-end policy inputs/output, when declared |
| command | Control before execution |
| render.pre/post | CameraRenderRequest / rendered sensor, when declared |

Simulator setup and step handlers use explicit native operations and return None. A pre-hook uses
CallInputs(args, kwargs) to retain native calling conventions. A post-hook returns the replacement
output directly. NativeHook adapts an explicitly registered avstack-library hook's tuple convention.
It lives in closed_loop/avstacks/native_hook.py.

## Contexts and lifecycle

StageContext exposes step, stage, sensor, physical_state, observation, control, run resources, rng
and record_output. Native access is explicit under simulator_runtime and avstack. A handler that
uses native SDK objects gives up adapter portability.

OnlineContext exposes only subscribed observation/physical_state snapshots and physical time; it
cannot advance simulation. AVStackAttackAction bound methods also receive the actual stage value.
All online state should remain in one AttackRuntime.

Plugin optionally provides reset(RunContext), transform(value, StageContext), after_step(context)
and close(). Plugin.after_step receives the next raw capture and real post-step clock. HookSet
manages each handler owner's lifecycle once even if registered at several stages. Stateful objects
must reset per run or be newly constructed. VehicleIO owns its localizer, and the dispatcher manages
its lifecycle with other handlers.

## Randomness

HookSet.seed creates a fresh per-run hook RNG. Environment transforms use a separate generator keyed
by run seed, absolute dataset position and transform index. A segment beginning at position p uses
p+i at decision i. Offline requests and formal runs at the same position align their sensor noise.
Native frames can include initialization ticks and are not the dataset position.

## Physical and model-visible records

| Record | Fields and coordinates |
|---|---|
| Observation | t, frame, sensor_data, calibration, vehicle_state, ego_speed; model-visible |
| Control | throttle, steer, brake, optional trajectory |
| WorldSnapshot | t, frame, vehicle_state, ego_speed, ego, other actors; physical |
| ActorState | world-from-actor pose, velocity, category, extent, local box center |
| Detection | ego-frame position, full extent, yaw, category and score |
| RecordedOutput | name, copied value, decision step, stage, physical frame/time and sensor |

ActorState.extent is full length, width and height in metres; None means unavailable. box_center is
the actor-local box center; ActorState.box_pose() returns its oriented world box with optional
extent. Unknown extent is None, distinct from known zero dimensions. Pose-only use remains valid;
operations requiring dimensions report missing evidence rather than substituting zero. Detection
position is its box center in the true ego frame at capture time, using sensor mounting extrinsics,
independently of driving localization, with ego x forward, y left, z up; yaw is radians. None
detections means unsupported, while an empty sequence means supported with no detections.

## RunLog

StepRecord stores raw_observation, model_observation, physical_pre, physical_post, produced_command,
executed_command, components, detections, actions, outputs, typed sensor_geometry and telemetry.
physical_pre and physical_post are StepRecord fields. Action requests and receipts distinguish what
was requested from what actually applied. EventRecord captures initialization, termination and
failure/cleanup information outside a decision. RunLog.status distinguishes completed, failed,
cancelled and incomplete logs.

Parquet stores typed scalars and nested struct/list values. Large arrays and images occupy bounded
payload shards referenced by records. Each steps/part-NNNNN.parquet and event part is closed before
publication. Payloads publish before referring records; completion metadata is written last. A crash
leaves completed parts readable and can lose the open part of each stream.

Records include typed camera geometry under sensor_geometry, with bounding-box offsets for
log-only evaluation. Readers consume the current record layout without legacy-format conversion.
Persisted native objects decode to plain
data unless a codec explicitly reconstructs a supported type. In-memory capture preserves supported
Python container types and snapshot isolation.

RecordingPolicy(sensor_ids=(), representations=("model_input",), every_n_steps=1) controls sensor
payload retention. State, commands and available component outputs/detections remain per decision.
Recorder supports chunk_steps, chunk_bytes, payload_shard_bytes and retain_steps. Disabling
in-memory retention requires a persistent output directory.

## Failures and cleanup

Unsupported stages, commands and environments are configuration errors. A failing decision stops
attacks, records unapplied actions and cleans resources. Simulator reservations are released only
after native cleanup. Never interpret a requested action as applied without its receipt, or a
partial log as a completed run.

See [Python API](python-api.md#runlog-api) for readers and recorder signatures.
