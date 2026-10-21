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
| `run_offline(context)` | ClosedLoopServices and all selected metadata | Artifacts; defaults to assets unchanged |
| `create_runtime(segment, artifacts)` | One selected segment and prepared artifacts | Fresh AttackRuntime |

Only `build_metadata` is abstract. An attack without assets leaves `assets=None`. An attack without
optimization uses the default offline phase. A selection-only attack uses the empty runtime. Attack
configuration holds reusable parameters; mutable online history belongs to its runtime. Artifacts
are explicit return values, never a second mutable Attack property.

Each experiment requires exactly one Attack. Every treatment shares its metadata and changes only
attack enablement and environment settings. Use one composite Attack to apply several actions
together.

**Not yet implemented.** Comparing multiple attacks in one experiment on shared metadata. See
[TODO](../development/TODO.md#multiple-attacks-on-shared-metadata).

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

Place user attacks under attack/implemented_attacks/speed_spoof_attack/ with __init__.py exporting
only SpeedSpoofAttack. Use type: SpeedSpoofAttack and constructor options. The same layout applies
to supplied and user attacks: attack.py connects entry points, metabuilder.py selects, optional
offline.py prepares, online.py implements AttackRuntime/actions, and optional assets.py owns
attack-specific assets. See the [folder
guide](../../avsectester/attack/implemented_attacks/README.md). An importable external class can
instead use type: my_project.attacks:SpeedSpoofAttack. The modified speed reaches the driving system
while StepRecord.physical_pre and physical_post retain actual world state.

## Use the selected scenario

`ScenarioView.frames` is a supported decision timeline. `frame_data(frame)` supplies ego and actor
states through recorded data or a preparation-owned cache. `record_reason(text)` aggregates
selection explanations by reason with occurrence counts in the experiment result, without recording
every rejected frame or host. Each scenario keeps at most 100 distinct reasons; later new reasons
are counted under "Other selection reasons". Within `scenario.services.session(seed)`, request
sensor samples, actions-based rendering, component evaluations, scores or rollouts.

`AttackSegment` includes scenario_id, start, stop, roles, placement and optional selection_score,
selection_rank and user_data. Roles map user-defined names to stable actor IDs. They can be empty.
The framework validates records and required actor existence; it does not interpret scores, rerank
records or supply filters. A single attack owns all metadata used by its experiment treatments.

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

`run_offline(OfflineContext)` receives all selected metadata and the same closed-loop service
boundary used by selection. The user defines optimization and objectives. A candidate rollout
declares every request field, for example:

```python
from avsectester.closed_loop.services import CandidateRequest
from avsectester.closed_loop.runner import RecordingPolicy

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

The public avsectester.attack.insertion package provides asset surfaces, compositing, visibility and
harmonizer extension points. The public avsectester.attack.optimization package provides DataSource,
objective, perturbation and scorer contracts plus PGD, NES and SessionScorer. These are independent
tools; an attack composes them in its own offline and online implementations. See [Image
insertion](../examples/image-insertion.md#use-insertion-tools-in-your-attack) and
[Optimization](optimization.md#use-the-optimization-tools).

## Online actions

| Action | Method | Application |
|---|---|---|
| WorldAttackAction | `apply(simulator_runtime, resources)` | Setup or before the world advances |
| SensorAttackAction | `apply(data, view)` | Any captured sensor data, before sensor environment |
| AVStackAttackAction | `apply(value, context)` | Driving-system stages from localization.pre to command |

SensorAttackAction is the attack entry for camera, LiDAR, GNSS and IMU data. Actions for one sensor
chain in issue order on a private copy; the view retains the original capture. chain_sensor_actions
uses write_camera_rgb when SensorView.camera is present, keeping camera data RGB uint8. Other
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
requests and actual application receipts are both recorded.

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
