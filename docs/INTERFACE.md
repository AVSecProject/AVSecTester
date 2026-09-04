# Interfaces

AVSecTester has a tiny surface because it borrows avstack's. There are exactly three things it
adds — a **scenario**, an **attack hook**, and a **metric** — plus the closed-loop driving stack
that was contributed *into* avstack. Everything else (world, ego, sensors, perception, tracking,
planning, control, the hook mechanism, the config registries) is avstack/avcarla.

## 1. Scenario (config → a driving `Trace`)

A scenario is a plain config dict built through avstack/avcarla's registries. `run_scenario`
constructs it, drives the loop, and returns a `Trace`.

```python
from avsectester.scenario import run_scenario   # (scenario: dict, attacks: list|None, frames: int) -> Trace
```

The config *is* avcarla config — no AVSecTester-specific types:

```yaml
client:  { type: CarlaClient, connect_ip: 127.0.0.1, connect_port: 2000, synchronous: true, rate: 20.0, ... }
ego:                                    # an avcarla CarlaMobileActor
  type: CarlaMobileActor
  spawn: 0
  vehicle: vehicle.tesla.model3
  sensors: [ { type: CarlaLidar, name: lidar-0, rotation_frequency: 20, sensor_tick: 0.05 } ]
  pipeline:                             # an avstack ModularDrivingPipeline (the AV brain)
    type: ModularDrivingPipeline
    perception: { type: MMDetObjectDetector3D, model: pointpillars, dataset: carla-vehicle }
    tracking:   { type: BasicBoxTracker3D }
    planning:   { type: ForwardCollisionPlanner, target_speed: 6.0, brake_distance: 12.0 }
    control:    { type: VehiclePIDController, args_lateral: {...}, args_longitudinal: {...} }
npcs:    { count: 6, npc_type: vehicle }
attacks: [ { stage: perception, hook: { type: PhantomInjection, target_xyz: [6.0, 0.0, -1.5] } } ]
```

`Trace` (in `avsectester/scenario.py`) is a list of per-frame records with three convenience
properties the metric reads: `final_speed`, `braking_frames`, `mean_detections`.

## 2. Attack (an avstack `HOOKS` hook)

An attack is a callable registered in avstack's `HOOKS` registry and attached to a pipeline stage's
pre/post hooks. That is the entire interface — no base class, no seams enum.

```python
from avstack.config import HOOKS

@HOOKS.register_module()
class PhantomInjection:
    def __init__(self, target_xyz=(6.0, 0.0, -1.5), obj_type="Car", score=0.9, ...): ...
    def __call__(self, detections):
        detections.append(<fabricated BoxDetection>)
        return (detections,)          # avstack post-hook contract: return the value as a 1-tuple
```

The scenario attaches each configured attack with avstack's own
`stage.register_post_hook(HOOKS.build(hook_cfg))`. To act on a different stage, name it in the
config (`stage: tracking`, `stage: planning`, …); to write a pre-hook attack, use
`register_pre_hook`. A defense is the same thing — a hook that sanitizes a stage's output.

## 3. Metric (clean vs attacked → verdict)

```python
from avsectester.metric import impact   # (clean: Trace, attacked: Trace) -> Impact
```

`Impact` answers the differential question — did the attack induce braking / an unsafe stop the
clean run never had — via `induced_braking`, `induced_stop`, `attack_succeeded`.

## What was contributed into avstack

avstack shipped the modules but not a turnkey closed-loop driving stack, so these live in the
`avstack-core` / `lib-avstack-carla` forks (not in AVSecTester):

- **`ModularDrivingPipeline`** (`avstack.modules.pipeline`) — maps `(sensor_data, ego_state)` →
  control by running perception → tracking → planning → control; the modular counterpart to
  end-to-end / foundation-model stacks. Attacks/defenses attach as hooks on any stage.
- **`ForwardCollisionPlanner`** (`avstack.modules.planning.vehicle`) — drive straight, brake to a
  stop when a track occupies the forward corridor (body-frame check); the driving consequence a
  perception attack triggers.
- **`CarlaMobileActor`** (`avcarla`) — closed the control loop: `apply_control` + feeding ego state
  into the pipeline each tick.
