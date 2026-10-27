# Implement an attack

> **Audience:** Attack authors.
>
> **Outcome:** Select segments, prepare artifacts and deploy actions through public interfaces.
>
> **Prerequisites:** [Architecture](../concepts/architecture.md).

## What a user implements

Every attack has an offline phase and an online phase; either phase can be a no-op. Keep
reusable parameters on `Attack` and mutable per-run state on a fresh `AttackRuntime`.

| Entry point | Responsibility |
|---|---|
| `build_metadata(scenario)` | The metabuilder selects `AttackSegment` records; optional with `metadata.load` |
| `validate_segment(segment)` | Reject metadata that lacks the data or geometry the attack requires |
| `deployment_ticks(segment)` | Declare the deployment tick budget needed before the first capture |
| `run_offline(context)` | Prepare and return artifacts; the default returns `None` |
| `create_runtime(segment, artifacts)` | Create a fresh runtime for the online phase |

Use [Experiment workflows](experiments.md) for preparation, execution and resume semantics.
Constructor signatures, constraints and errors are in the
[Python API](../reference/python-api.md).

## Write your first attack

Put user implementations in your own importable package. Reference them by a full type path:

```yaml
attacks:
  speed_spoof:
    type: my_project.attacks:SpeedSpoofAttack
    options:
      decision_steps: 20
      reported_speed: 0.0
```

A supplied attack uses the
[implemented-attacks layout](../../avsectester/implemented_attacks/README.md). Keep explicit
constructor parameters as attributes with the same names, or override
`configuration_parameters()` to describe the configured inputs.
Filesystem options follow
[Path resolution](../reference/configuration.md#constructor-filesystem-parameters).

This example defines a selection rule and an online phase that changes model-visible speed:

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

The modified observation reaches the driving system while physical states remain unchanged; the
[Runtime reference](../reference/runtime.md) defines the recorded inputs, commands and
action receipts.

## Select segments

`ScenarioView` provides the decision timeline, frame data, semantic map and preparation
services. Use `record_reason()` to explain rejected candidates, and
`session.can_start_at(position)` to check start requirements before returning a segment.
Placement and visibility criteria belong to the attack; the framework does not infer them
from `user_data`.

`AttackSegment` contains `scenario_id`, `start`, `stop` and `user_data`. The decision interval
is `[start, stop)`, with `stop` as its terminal state. Store custom actor IDs, positions, ranks
and other attack data in `user_data`, using JSON-like values and NumPy arrays for persisted
metadata. Check these values in `validate_segment()` so user-supplied metadata follows the same
attack rules. The [metadata configuration](../reference/configuration.md#metadata) controls saving and
loading segments.

`split_segments` groups adjacent eligible decisions. The following self-contained example
includes an eligible tail:

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

## Offline phase

A candidate rollout request is created from the chosen segment and deployment:

```python
from avsectester.execution.services import CandidateRequest
from avsectester.execution.runner import RecordingPolicy

request = CandidateRequest(
    segment=segment, seed=context.seed, decision_steps=segment.decision_steps,
    environment=context.environment, recording=RecordingPolicy(),
    attack=self, artifacts=artifacts)
```

The metabuilder is not run again after the offline phase. Use
[data split settings](../reference/configuration.md#data-split) to control training, validation
and execution membership.

Use [Optimization tools](optimization.md) to sample inputs, score candidates and collect
offline reports.

The services available through the context provide sensor reads, rendering, component
evaluations, scores and candidate rollouts. This example reads detached geometry and separates
captured pixels from sensor-environment output using the shared in-memory components:

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

Use `session.render` to evaluate deployed world or sensor actions. Use `session.rollout` when
the objective depends on a driving response; each candidate has its own run state. Service
request records and capability checks are defined in the
[Python API](../reference/python-api.md).

## Online phase

The deployment tick budget is the maximum of
`hooks.setup_ticks + attack.deployment_ticks(segment)` across all configured attacks, segments,
seeds and treatments. Every treatment uses that budget. Larger budgets can make early starts
fail `can_start_at`.

A runtime's `setup` returns world actions; `step` returns the actions for a decision; `stop`
releases its own resources. Declare snapshot reads with `ResultSubscription`. The context
supplies snapshots and a clock, rather than a service that advances the world.

Choose `WorldAttackAction` for physical-world changes, `SensorAttackAction` for sensor-data
replacement, and `AVStackAttackAction` for a driving-system stage value. The
[Runtime reference](../reference/runtime.md) owns their application order and stage semantics.
For insertion, `InsertObject` is a `WorldAttackAction`; `InsertionAttackAction` is a
`SensorAttackAction`. Object coordinates use +X forward, +Y left and +Z up.

Placement tools belong to `attack.insertion`, not the dataset. A `SensorView` supplies camera
geometry, actor poses and available depth for sensor composition. See
[Image insertion](../examples/image-insertion.md) and
[Physical patch](../examples/physical-patch.md) for the supplied deployment paths.

A runtime must satisfy `runtime.setup_ticks <= attack.deployment_ticks(segment)`.

## Insert a world object

Return `InsertObject` from setup to include an object in the first capture. The action registers
removal with the run's resources. This example needs no simulator server or model weights:

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

`ObjectAsset` uses a bottom-center local origin. Convert a map feature's reference point before
placing the asset; geometry records and asset constraints are in the
[Python API](../reference/python-api.md). This example derives a sign placement from a
semantic map:

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

World insertion, visibility, depth, occlusion and non-camera limitations are defined by the
[Simulator guide](simulators.md). Pass a harmonizer through the placed object or insertion
action; [Optimization tools](optimization.md) explains the differentiable composition path.

## Persist artifacts

Return artifacts explicitly from `run_offline`; pass them to each runtime. Override
`save_artifacts` and `load_artifacts` together when a custom storage format is needed.
Persistence must not change the attack's configured inputs. See the
[Attack API](../reference/python-api.md#avsectesterattackattack) for the storage contract.

## Next steps

Read [Optimization tools](optimization.md), [Examples](../examples/README.md)
and [Evaluation](evaluation.md).
