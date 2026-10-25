# Implement an attack

> **Audience:** Attack authors.
>
> **Outcome:** Define selection, optional preparation and per-run attack actions.
>
> **Prerequisites:** [Architecture](../concepts/architecture.md).

## What a user implements

| Entry point | Input | Return |
|---|---|---|
| `build_metadata(scenario)` | ScenarioView, user assets and parameters | Sequence of AttackSegment |
| `run_offline(context)` | ClosedLoopServices, training metadata and optional validation metadata | Artifacts; defaults to None |
| `create_runtime(segment, artifacts)` | One selected segment and prepared artifacts | Fresh AttackRuntime |

Only `build_metadata` is abstract. The base Attack owns no assets. An attack that needs assets
defines its own constructor parameter and run_offline implementation, which may return those
assets unchanged. A selection-only attack uses the default None artifacts and empty runtime.
Attack configuration holds reusable parameters; mutable online history belongs to its runtime.
Artifacts are explicit return values, never a second mutable Attack property.

One experiment supports one or more attacks, each selecting its own metadata. Identical complete
segments form one unit, including scenario, start, stop and user_data. A comparison group is one
unit and seed: it contains each treatment without attack_name once and treatments for every attack
that selected that unit. All treatment pairs are evaluated, including pairs from different attacks.
Use one composite Attack to apply several actions together in one run. Different user_data creates
a different unit even for the same window; sharing un-attacked runs across these units remains
[planned](../development/TODO.md#shared-runs-across-different-user-data).

## Write your first attack

This attack changes the model-visible ego speed at the observation stage. It uses the same selection
helper as environment-only experiments and keeps no live simulator handles.

<!-- runnable -->
```python
from dataclasses import replace
from avsectester.attack import Attack, AttackRuntime, AVStackAttackAction
from avsectester.attack.segments import split_segments

class SpeedSpoofAttack(Attack):
    def __init__(self, decision_steps, reported_speed=0.0):
        self.decision_steps = decision_steps
        self.reported_speed = reported_speed

    def build_metadata(self, scenario):
        return split_segments(scenario.frames, lambda frame: True,
                              decision_steps=self.decision_steps)

    def create_runtime(self, segment, artifacts):
        return SpeedSpoofAttackRuntime(self.reported_speed)

class SpeedSpoofAttackRuntime(AttackRuntime):
    def __init__(self, reported_speed):
        self.reported_speed = reported_speed

    def step(self, context):
        return [AVStackAttackAction("observation", self.spoof)]

    def spoof(self, observation, context):
        context.record_output("reported_speed", self.reported_speed)
        return replace(observation, ego_speed=self.reported_speed)
```

Place user attacks under implemented_attacks/speed_spoof_attack/ with __init__.py exporting
only SpeedSpoofAttack. Use type: SpeedSpoofAttack and constructor options. The same layout applies
to supplied and user attacks: attack.py connects entry points, metabuilder.py selects, optional
offline.py prepares, online.py implements AttackRuntime/actions, and optional assets.py owns
attack-specific assets. See the [folder
guide](../../avsectester/implemented_attacks/README.md). An importable external class can
instead use type: my_project.attacks:SpeedSpoofAttack. The modified speed reaches the driving system
while StepRecord.physical_pre and physical_post retain actual world state.

Mark filesystem constructor parameters as Path or str | Path so YAML resolves relative paths; see
[filesystem parameters](../reference/configuration.md#constructor-filesystem-parameters).

## Use the selected scenario

`ScenarioView.frames` is a supported decision timeline. `frame_data(frame)` supplies ego and actor
states through recorded data or a preparation-owned cache. `record_reason(text)` aggregates
selection explanations by reason with occurrence counts in the experiment result, without recording
every rejected frame or host. Each scenario keeps at most 100 distinct reasons; later new reasons
are counted under "Other selection reasons". Within `scenario.services.session(seed)`, request
sensor samples, actions-based rendering, component evaluations, scores or rollouts.

Use SensorRequest(..., views=True) when offline preparation needs each sample's camera geometry,
actors or ego pose. SensorSample.views contains one detached SensorView per requested sensor;
view.native is empty. view.data is the capture before sensor corruptions, while
sample.observation includes those transforms. views defaults to False and then the mapping is
empty, avoiding depth computation and extra data copies. Requesting views requires can_render.

read_sensors reads a world without attack actions. When can_replay is True, it can reuse one
simulator runtime for forward scans; NuRec resets to the requested recorded frame each time. render
applies world or sensor attack actions and creates a fresh world for each request.
Both can return calibrated views; render remains the entry point for evaluating deployed actions.

This offline example uses the repository's shared in-memory components to collect several samples.
A real attack substitutes its ScenarioView.services and selected frames, then passes the returned
images and geometry into its own optimizer:

<!-- runnable -->
```python
import numpy as np
from avsectester.execution.services import ClosedLoopServices, SensorRequest
from tests.fakes import MemoryDataset, MemorySimulator

dataset = MemoryDataset()
environment = {"sensors": [{"type": "GaussianNoise", "camera": "front",
                             "options": {"severity": 0.2}}]}
with ClosedLoopServices(dataset, MemorySimulator()) as services:
    scenario = services.scenario("scene", seed=7)
    with services.session(scenario.seed) as session:
        samples = [session.read_sensors(SensorRequest(
            scenario.scenario_id, frame, ("front",), environment, views=True))
            for frame in scenario.frames[1:4]]
        plain = session.read_sensors(SensorRequest("scene", scenario.frames[3], ("front",)))

optimization_samples = [
    {"image": sample.views["front"].data, "camera": sample.views["front"].camera,
     "cam_from_world": sample.views["front"].cam_from_world,
     "actors": sample.views["front"].actors, "ego": sample.views["front"].ego}
    for sample in samples]
assert len(optimization_samples) == 3 and not plain.views
assert all(sample.views["front"].native == {} for sample in samples)
assert all(sample["cam_from_world"].shape == (4, 4) for sample in optimization_samples)
assert np.any(samples[0].observation.sensor_data["front"] != optimization_samples[0]["image"])
```

`AttackSegment` has exactly four fixed fields: scenario_id, start, stop and user_data. Put all
custom data, including actor IDs, placement parameters and any scores or ranks, in user_data.
Data that must be persisted can contain only JSON-like values or numpy arrays; convert custom
placement objects to data before storing them. The framework validates segment boundaries and
saves records without interpreting custom data, checking selected actors, ranking records or
supplying filters. An attack must report missing actors when it uses them. Each attack owns the
metadata of the treatments that run on its segments.

`split_segments(frames, eligible, *, decision_steps, keep_tail=False)` returns AttackSegment
records. `eligible(frame)` runs once in order for every nonterminal frame. It never runs on the
terminal state. A short eligible tail is dropped unless keep_tail is true.

<!-- runnable -->
```python
from avsectester.datasets import FrameReference
from avsectester.attack.segments import split_segments

frames = tuple(FrameReference("road", i, i, i * 0.1) for i in range(80))
segments = split_segments(frames, lambda frame: frame.position >= 5,
                          decision_steps=20, keep_tail=True)
assert [(s.start.position, s.stop.position) for s in segments] == [
    (5, 25), (25, 45), (45, 65), (65, 79)]
assert [s.decision_steps for s in segments] == [20, 20, 20, 14]
```

No default segment length is imposed. N decisions require N+1 timeline states. Requests beyond
recorded support fail rather than padding or silently shortening the run.

## Offline preparation

`run_offline(OfflineContext)` receives training segments in metadata and validation segments in
validation_metadata when split is configured. Both retain the attack's selection order. Without
split, metadata contains all selected segments and validation_metadata is empty. The context uses
the same closed-loop service boundary as selection and the experiment-wide deployment budget.
The attack chooses whether to use validation_metadata. The user defines optimization and objectives. A candidate rollout
declares every request field, for example:

```python
from avsectester.execution.services import CandidateRequest
from avsectester.execution.runner import RecordingPolicy

# Requires an Attack self, selected segment, artifacts and OfflineContext context.
request = CandidateRequest(attack=self, segment=segment, artifacts=artifacts,
                           seed=context.seed, decision_steps=segment.decision_steps,
                           environment=context.environment, recording=RecordingPolicy())
```

The request calls the same runtime factory and actions as formal execution.

Optimization does not cause reselection or a geometry-preservation check. Keep visibility and
placement assumptions valid in your optimizer if the attack requires them. See
[Optimization](optimization.md).

## Reuse framework tools

The public avsectester.attack.insertion package provides insertion assets, placement resolution,
compositing and visibility. avsectester.rendering provides simulator-independent ObjectAsset,
PlaneSurface, surface rendering, composition and harmonizer extension points. The public
avsectester.attack.optimization package provides DataSource, objective, perturbation and scorer
contracts plus PGD, NES and SessionScorer. An attack composes these tools in its own offline and
online implementations. See [Image
insertion](../examples/image-insertion.md#use-insertion-tools-in-your-attack) and
[Optimization](optimization.md#use-the-optimization-tools).

## Choose world or sensor insertion

The two insertion paths are independent; an attack uses whichever matches its threat model.

| | World insertion: InsertObject | Sensor insertion: InsertionAttackAction |
|---|---|---|
| Action kind | WorldAttackAction | SensorAttackAction |
| Affects | Every camera, from the next capture | One named camera's data, in the current decision |
| World state and RunLog | WorldSnapshot.objects; poses, extents and per-camera visibility | Receipts only; not part of world state |
| Placement | Fixed world_from_object | World or host-attached placement, resolved per capture |
| Occlusion | Scene depth, otherwise actor cuboids; 0.1 m tolerance | Scene depth or actor cuboids and other world objects; millimetre tolerances |
| Appearance | PlacedObject.harmonizer | PatchCompositor and its harmonizer |
| Differentiable path | SensorView.composition (or composite_plan with object_occlusion="geometry"), then CompositePlan.composite | PatchPerturbation over plane_sampling |
| Simulator requirement | can_insert_objects | A calibrated camera SensorView |

Use world insertion for a physical object that every sensor and the recorded world should contain,
such as a sign. Use sensor insertion for a perturbation of one camera's data, such as an injected
image or a patch that follows a host vehicle.

## Insert a world object

Import InsertObject from avsectester.attack.insertion. Return it from AttackRuntime.setup to make
an arbitrary textured object visible from the first capture. Returning it from step makes it
visible from the next capture. The action registers
cleanup, so the object is removed when the run ends. The framework composites it into every camera;
the native simulator does not need a matching blueprint or reconstructed actor.

ObjectAsset contains rectangular PlaneSurface faces with RGBA textures. Its right-handed local
frame has its origin at the bottom center, +X pointing out of the front, +Y to the left and +Z up.
extent is the full (length, width, height), with the horizontal box centered on the origin and its
bottom at z=0. Every corner must lie inside that box. ObjectAsset.from_surfaces infers those extents
without moving the origin. ObjectAsset.from_panel creates a front-facing textured panel with
optional support posts. Pass an RGBA array, a Texture implementation or your own image path.
Existing built-in attacks retain their own deployment implementations.

world_from_object is a rigid 4×4 transform from that local frame into canonical world coordinates.
Selection can derive it from ScenarioView.frame_data(frame).ego.pose, an actor pose or a semantic
map feature, then save it in AttackSegment.user_data. Account for the source feature's reference
point: a TrafficControl pose and an asset's bottom-center origin need not coincide. Compute any
mounting offset before returning InsertObject.

This minimal example inserts a sign eight metres ahead, facing ego:

<!-- runnable -->
```python
import numpy as np
from avsectester.attack import AttackRuntime
from avsectester.attack.insertion import InsertObject
from avsectester.execution.runner import ClosedLoopRunner, RunRequest
from avsectester.rendering import ObjectAsset
from tests.fakes import ConstantAVStack, MemorySimulator

texture = np.full((96, 96, 4), (180, 20, 20, 255), dtype=np.uint8)
texture[12:-12, 12:-12, :3] = (240, 240, 240)
asset = ObjectAsset.from_panel(texture, category="stop_sign", width_m=0.9,
                               mount_height_m=1.5, post_count=1)
world_from_object = np.eye(4)
world_from_object[:3, :3] = np.diag([-1.0, -1.0, 1.0])
world_from_object[:3, 3] = (8.0, 0.0, 0.0)

class PlaceSignAttackRuntime(AttackRuntime):
    def setup(self, context):
        return [InsertObject("demo_stop_sign", asset, world_from_object)]

with MemorySimulator().create_runtime() as simulator_runtime:
    log = ClosedLoopRunner().run(simulator_runtime, ConstantAVStack(),
                                RunRequest(1, attacks=(PlaceSignAttackRuntime(),)))
    assert "demo_stop_sign" in log.steps[0].physical_pre.objects
    assert not simulator_runtime.state().objects
assert log.status == "completed"
```

The following more complete example uses a semantic map supplied by the in-memory Dataset. The sign's
pose_reference is face_center. Its ground height comes from the closest centerline point over all
governed lanes, in world coordinates. If c is the vertical distance from that point to the face
center, the object pose is control.pose @ translate(0, 0, -c), and the sign's mount height is
c - face_height/2, where face_height is the inserted sign's own face height. This keeps the face at
its mapped location and the post on the estimated ground.
The same calculation supports nonzero world-ground heights. It needs no native simulator or model
weights and imports no concrete attack:

<!-- runnable -->
```python
import numpy as np
from avsectester.attack import AttackRuntime
from avsectester.attack.insertion import InsertObject
from avsectester.execution.runner import ClosedLoopRunner, RecordingPolicy, RunRequest
from avsectester.geometry.poses import transform
from avsectester.maps import Lane, ScenarioMap, StopLine, TrafficControl
from avsectester.rendering import ObjectAsset
from tests.fakes import ConstantAVStack, MemoryDataset, MemorySimulator

centerline = np.array([[0.0, 0.0, 0.0], [12.0, 0.0, 0.0]])
lane = Lane("approach", centerline, centerline + (0.0, 2.0, 0.0),
            centerline - (0.0, 2.0, 0.0))
face_pose = np.eye(4)
face_pose[:3, :3] = np.diag([-1.0, -1.0, 1.0])
face_pose[:3, 3] = (8.0, 0.0, 2.0)
control = TrafficControl("stop", "stop_sign", "demo_stop_sign", face_pose,
                         (0.02, 0.9, 0.9), pose_reference="face_center",
                         lane_ids=(lane.id,), stop_line_id="stop-line")
stop_line = StopLine("stop-line", np.array([[7.0, -2.0, 0.0], [7.0, 2.0, 0.0]]),
                     native_category="STOP", kind="stop_sign",
                     lane_ids=(lane.id,), control_id=control.id)
dataset = MemoryDataset(maps={"scene": ScenarioMap(
    lanes={lane.id: lane}, traffic_controls={control.id: control},
    stop_lines={stop_line.id: stop_line})})
scenario_map = dataset.scenario_map("scene")
control = scenario_map.traffic_controls["stop"]
assert control.pose_reference == "face_center"
center = control.pose[:3, 3]
candidates = []
for lane_id in control.lane_ids:  # the nearest centerline point over every governed lane
    centerline = scenario_map.lanes[lane_id].centerline
    starts, vectors = centerline[:-1], np.diff(centerline, axis=0)
    fraction = np.clip(np.sum((center - starts) * vectors, axis=1)
                       / np.sum(vectors * vectors, axis=1), 0.0, 1.0)
    candidates.append(starts + fraction[:, None] * vectors)
points = np.concatenate(candidates)
ground_z = points[np.argmin(np.linalg.norm(points - center, axis=1))][2]
c = control.pose[2, 3] - ground_z
width = control.extent[1]
texture = np.full((96, 96, 4), (180, 20, 20, 255), dtype=np.uint8)
texture[12:-12, 12:-12, :3] = (240, 240, 240)
# The face height follows the texture aspect ratio, which may differ from the mapped box.
face_height = width * texture.shape[0] / texture.shape[1]
asset = ObjectAsset.from_panel(texture, category="stop_sign", width_m=width,
                               mount_height_m=c - face_height / 2, post_count=1)
offset = np.eye(4)
offset[2, 3] = -c
world_from_object = control.pose @ offset
face_corners = transform(world_from_object, asset.surfaces[-1].corners)
post_corners = transform(world_from_object, asset.surfaces[0].corners)
np.testing.assert_allclose(face_corners.mean(axis=0), control.pose[:3, 3])
np.testing.assert_allclose(post_corners[:, 2].min(), ground_z)

class PlaceSignAttackRuntime(AttackRuntime):
    def setup(self, context):
        return [InsertObject("demo_stop_sign", asset, world_from_object)]

with MemorySimulator().create_runtime() as simulator_runtime:
    start = dataset.scenario_start("scene", dataset.frames("scene")[0])
    simulator_runtime.reset(start)
    clean = simulator_runtime.observe().sensor_data["front"].copy()
    log = ClosedLoopRunner().run(simulator_runtime, ConstantAVStack(),
        RunRequest(1, start=start, attacks=(PlaceSignAttackRuntime(),),
                   recording=RecordingPolicy(sensor_ids=("front",), representations=("raw",))))
    record = log.steps[0]
    np.testing.assert_allclose(record.physical_pre.objects["demo_stop_sign"].pose,
                               world_from_object)
    assert np.any(record.raw_observation.sensor_data["front"] != clean)
    assert record.object_visibility["demo_stop_sign"]["front"].visible_pixels > 0
    assert not simulator_runtime.state().objects
assert log.status == "completed"
```

WorldSnapshot.objects keeps inserted objects separate from actors, and RunLog preserves their
poses, categories, extents and zero velocities. Per-camera visibility is measured at every capture;
[Inserted world objects](simulators.md#inserted-world-objects) defines it and
[RunLog](../reference/runtime.md#runlog) describes the recorded receipts and visibility records.

Selection may check scenario.services.capabilities.insert_objects, which reflects the simulator's
can_insert_objects declaration. Returning InsertObject to an unsupported simulator raises
NotImplementedError during the run and records failure. Setup actions are created inside reset, so
the framework cannot inspect them during preflight.

Pass harmonizer=None for unchanged appearance, a built-in ClassicHarmonizer or PCTNetHarmonizer,
or an instance of your own avsectester.rendering.Harmonizer subclass. In an attack's YAML options,
reference your class as package.module:Class and pass the constructed instance to InsertObject.
See the [custom harmonizer example](../examples/image-insertion.md#harmonization). Harmonization
changes only visible object pixels. When using composite_objects directly, set each
PlacedObject.harmonizer; appearance belongs to the placed object, not a separate global argument.

Inserted objects affect camera images and recorded world state. They do not add native collision
geometry. **Not yet implemented.** Inserted objects in LiDAR or other non-camera sensor data. See
[TODO](../development/TODO.md#inserted-objects-in-non-camera-sensors).

## Online actions

| Action | Method | Application |
|---|---|---|
| WorldAttackAction | `apply(simulator_runtime, resources)` | Setup or before the world advances |
| SensorAttackAction | `apply(data, view)` | Any captured sensor data, before sensor environment |
| AVStackAttackAction | `apply(value, context)` | Driving-system stages from localization.pre to command |

SensorAttackAction is the attack entry for camera, LiDAR, GNSS and IMU data. Actions for one sensor
chain in issue order on a private copy; the view retains the original capture. The runner uses
write_camera_rgb when SensorView.camera is present, keeping camera data RGB uint8. Other
sensors use write_sensor_data, which preserves array shape and dtype, native wrappers and
calibration. SensorView.camera and cam_from_world are None for non-camera sensors.
sensor_from_world uses sensor-body axes, including for cameras, independently of the optical
camera transform. See [CARLA LiDAR data](../reference/runtime.md#carla-lidar-data) for raw point
decoding and a replacement action.

AVStackAttackAction cannot target sensors.post; use SensorAttackAction for sensor data. The
sensors.post stage remains available to ordinary HookSet handlers. Actions returned dynamically
by step are validated before their requests are queued or any action is applied.

`setup(context)` returns only world actions. Declare required setup_ticks for deployment.
`step(context)` returns actions once each decision. `stop(context)` runs on completion and failure.
Use its instance to hold histories, trigger counts and bound component-stage callbacks.

Declare `ResultSubscription("observation")` or `ResultSubscription("physical_state")` when the
runtime reads those snapshots in its callbacks. The OnlineContext exposes no simulator or service
that can advance time. An AVStack action receives its current stage value; a bound method can use it
without introducing another independently stateful handler.

Multiple actions for one stage are applied sequentially in returned and registration order. Every
action receives the preceding action's replacement; receipts reflect actual application. Action
requests and actual application receipts are both recorded. Any AttackAction may return
additional evidence from receipt_details(); every receipt has details, defaulting to {}. The
built-in actions' details are listed in [RunLog](../reference/runtime.md#runlog).

## Persist artifacts

The default `save_artifacts(artifacts, directory)` stores trusted pickle plus a content hash.
`load_artifacts(directory)` returns the decoded value. Override both methods together for a portable
custom format. Neither method should modify Attack configuration.

Direct Python objects may override `configuration_parameters()` when constructor arguments are
stored under different names. This is needed only for validated persistence, not for an in-memory
experiment. Return configured inputs, excluding counters and live resources.

## Next steps

Read [Examples](../examples/README.md), [Runtime](../reference/runtime.md) and [Python
API](../reference/python-api.md#attack-api).

A selected start must satisfy `session.can_start_at(position)`: the shared deployment window
must fit and the position must provide the driving adapter's declared past_steps. Built-in
metabuilders exclude earlier positions before splitting windows. Preparation also drops invalid
custom-builder segments and records why they cannot start. Past observations initialize the
driving system from the recording; attack actions do not modify them.
