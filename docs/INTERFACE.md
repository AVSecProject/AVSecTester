# Interfaces

AVSecTester is two roles joined by a **pure data plane**. A **world backend** senses and actuates; an
**AV stack** (the box under test) turns observations into control. An attack is a transform on the
stream between them — *not* part of the contract. Everything else (the world, ego, sensors, the
perception/tracking/planning/control modules, the Alpamayo policy, the hook mechanism, the config
registries) comes from avstack / avcarla / alpasim_driver.

## 0. The spine: data plane + interfaces

Two files, both pure — no `carla`/`avcarla`/`torch`/`avstack` imports, serializable so the loop can
run over gRPC if ever split across processes.

**`avsectester/plane.py`** — the sim↔stack contract, two messages plus the driving record:

- **`Observation`** (down): `sensor_data` dict + `calibration` + ego `vehicle_state` + `ego_speed`.
  World state stays hidden in the backend.
- **`Control`** (up): `throttle` / `steer` / `brake`, plus an optional `trajectory` (rig-frame
  waypoints) for stacks that plan rather than actuate directly.
- **`Trace`** / **`FrameRecord`** — the per-frame driving record the metric reads
  (`final_speed`, `peak_speed`, `braking_frames`, `mean_detections`).

**`avsectester/backend.py`** — the interfaces:

- **`WorldBackend`**: `reset() -> Observation` and `step(control) -> Observation`. Owns the world
  state and the shared vehicle dynamics.
- **`AVStack`**: `__call__(observation) -> Control`. The box; it knows nothing of modular vs
  end-to-end. `reset(obs)` lets it clear per-episode history.
- **`run(backend, stack, frames, perturb=None) -> Trace`** drives the loop:

  ```python
  obs = backend.reset(); stack.reset(obs)
  for i in range(frames):
      seen = perturb(obs) if perturb else obs      # the single attack seam
      control = stack(seen)
      obs = backend.step(control)
      trace.records.append(FrameRecord(... obs.ego_speed ...))   # TRUE ego state, not the perturbed view
  ```

  **Causal invariant:** control at *t* affects only *t+1*, so the loop serializes cleanly.
  `perturb: Observation -> Observation` is the **single universal attack seam** — it works against
  any stack, black-box included.

## 1. The 2×2: any backend × any stack

| | **ModularAVStack** | **AlpamayoAVStack** |
|---|---|---|
| **CarlaBackend** | phantom-injection demo (`avsectester run`) | Alpamayo in a CARLA world |
| **NuRecBackend** | modular stack on reconstructed sensors | ✔ Alpamayo on photoreal NuRec frames |

### 1a. CarlaBackend (`simulators/carla.py`) + ModularAVStack (`stacks/modular.py`)

`CarlaBackend` owns an avcarla `CarlaClient` + ego (`CarlaMobileActor` with a no-op internal pipeline
— driving is external, done by the `AVStack`) + `CarlaNpc` traffic. `reset()` spawns actors and
captures a strict-spawn `replay_scenario`; `step(control)` applies the `Control` through CARLA
physics; `_observe()` reads sensors + ego into an `Observation`. `ModularAVStack` wraps an avstack
`ModularDrivingPipeline` (perception → tracking → planning → control) and returns a `Control`.

`run_scenario` = `run(CarlaBackend, ModularAVStack, frames)` with the clean/attacked pairing below.
For a clean/attacked comparison, call `prepare_scenario(config)` once and run clean with its returned
configuration; then run attacked with `clean.replay_scenario`, which records the actual successful
spawn transforms. The CLI does this automatically. Preparation resolves random vehicle models, spawn
indices and destinations using `client.seed`, while preserving explicit selections. A missing seed is
generated once for the pair. The resolved configuration also retains the map, weather, traffic-light
settings, Traffic Manager seed and LiDAR noise seed; it is held in memory and the input configuration
is not mutated.

These experiments require a dedicated CARLA server: each run reloads the world after enabling
synchronous, fixed-step physics, then rebuilds its actors and driving pipeline. The clean run can
relocate vehicles when spawning fails; the attacked run must reproduce the successful clean spawns.

Optional client settings include `map_name`, `weather` (a dictionary of CARLA weather fields),
and `traffic_lights` (OpenDRIVE light ID to state, green/yellow/red duration and frozen flag).
When omitted, preparation uses the selected world's initial environment and the existing
`randomize_lights` setting. `reset_world` must be enabled for paired experiments.
Traffic continues to react to the ego after the common starting point; an attack can therefore
change NPC trajectories as part of its driving consequence.

#### Configuring a YAML scenario

Start with the complete [example configuration](../configs/carla_scenario.yaml). Keep its
`type` fields and nested pipeline settings when editing individual parameters, then run:

```bash
avsectester run configs/carla_scenario.yaml --frames 40 --gpu 0
```

Each command runs one experiment pair: clean first, then attacked. `--frames` overrides YAML
`frames` for each run; `--gpu` overrides `ego.pipeline.perception.gpu` and does not select the CARLA
server's GPU. The following YAML snippets are edits to the complete example, not standalone files.

| Field | Meaning |
| --- | --- |
| `frames` | Simulation steps per run; 40 steps at `client.rate: 20.0` represent 2 seconds per run. |
| `client.connect_ip`, `connect_port` | Address of the running CARLA server. |
| `client.traffic_manager_port` | Traffic Manager used by NPC autopilot. |
| `client.synchronous`, `rate` | Use `true` for paired experiments; `rate` sets simulation steps per second. |
| `ego.vehicle`, `ego.spawn` | Vehicle blueprint ID and map spawn-point index, or `random` for either field. Spawn indices depend on the selected map. |
| `ego.autopilot` | Keep `false` to let the configured AV pipeline control the ego. |
| `ego.sensors` | Sensor settings; the example uses a LiDAR with `sensor_tick: 0.05` and `rotation_frequency: 20`. |
| `ego.pipeline` | Perception, tracking, planning and control modules. The default planner drives straight and brakes for obstacles; `target_speed` is in m/s, and `brake_distance` and `brake_corridor` are in meters. |
| `npcs` | Background vehicles, using the compact form or explicit list below; omit it or use `[]` for none. |
| `attacks` | Hooks attached only during the attacked run. The supplied `PhantomInjection` is used at `stage: perception`; see the attack section for its coordinate convention. |

#### Fixed choices and random choices

Random choices are resolved **once per experiment pair**. Both runs use those choices, even when
each new command starts with a fresh seed. Explicit vehicle models and seeds are preserved; explicit
spawn points are the first positions attempted, with retries as described below. Omitting a field
does not generally mean random: fields can have defaults or be required.

| Setting | Behavior across separate commands |
| --- | --- |
| `client.seed: 0` (the example default), or another fixed integer | Repeats seeded scene choices with the same configuration and map/blueprint candidates. This is not a guarantee of identical model outputs or complete driving traces. |
| `client.seed: null`, or omit `seed` | Generates a fresh seed for each pair and selects random items again; individual choices can still repeat by chance. |
| A concrete value such as `ego.spawn: 0` | Always attempts this spawn point first, regardless of the seed. |

To select a new random ego model and spawn point for each command, edit these fields while keeping
the other client and ego settings:

```yaml
client:
  seed: null
  traffic_manager_seed: null
ego:
  vehicle: random
  spawn: random
```

An omitted or `null` `traffic_manager_seed` follows the resolved `client.seed`. The complete example
explicitly sets both seeds to `0`: changing only `seed` leaves the Traffic Manager seed fixed.
Similarly, an omitted or `null` LiDAR `noise_seed` is derived during preparation; an explicit
`noise_seed` is preserved. The CLI prints `[scenario] seed=...`; set `client.seed` to that integer
to repeat the seeded choices with the same remaining configuration and environment.

The seed does not randomly change every field. NPC count, map, weather, sensor specifications and
attack parameters are not automatically randomized. `map_name` and `weather` can be set explicitly
under `client`; when omitted, they come from the selected world's initial environment. Traffic
lights use the client's seeded `randomize_lights` behavior unless explicit light settings are given.

#### Configuring NPCs and spawn positions

The compact form attempts to create six NPCs at spawn indices 1 through 6. Here `vehicle` means a randomly
selected vehicle model, not a fixed model:

```yaml
npcs:
  count: 6
  npc_type: vehicle
  spawn_start: 1
```

To mix fixed and random NPC choices, replace the compact form with a list. Each entry creates one
NPC; `count` and `spawn_start` apply only to the compact form:

```yaml
npcs:
  - type: CarlaNpc
    npc_type: random
    spawn: random
  - type: CarlaNpc
    npc_type: vehicle.tesla.model3
    spawn: 3
```

Random spawn selection reserves explicit spawn indices first and selects unused indices for random
actors. Different indices do not guarantee that all vehicle geometries fit without collisions.
During clean setup, a failed spawn moves the requested position 3 meters along the vehicle heading
in the horizontal plane and tries again, up to 10 attempts in total. The runner records each
vehicle's actual position, orientation and model before advancing the simulation. After resetting
the world, the attacked run uses these recorded transforms in the same creation order, without
applying spawn offsets again. If a recorded spawn fails, the attacked run reports an error rather
than moving it. Separate pairs can therefore resolve different spawn positions.

`client.strict_spawn` defaults to `false` for the first run. Set it to `true` to disable clean
relocation as well. Recorded transforms are always replayed strictly, even if this setting is
`false`. If clean exhausts its retries, choose other indices or fewer vehicles.
`ego.destination` accepts `null`, a spawn index or `random`, but the default straight-driving
pipeline does not use it to navigate to a destination.

##### CARLA Python interface

```python
from avsectester.scenario import prepare_scenario, run_scenario

prepared = prepare_scenario(config)
clean = run_scenario(prepared, frames=40)
attacked = run_scenario(clean.replay_scenario, attacks=prepared.get("attacks", []), frames=40)
```

`Trace.replay_scenario` is an in-memory configuration, not a generated file. Each actor's
`spawn_transform` contains native CARLA world coordinates (`location`: x/y/z in meters;
`rotation`: pitch/yaw/roll in degrees). It already includes `reference_to_spawn` and any retry
displacement. The runner produces this field; users normally configure `spawn` instead.

### 1b. NuRecBackend (`avsectester/simulators/nurec.py`)

An in-process world driven by an NVIDIA **NuRec** neural reconstruction (a 3D `.usdz` scene rendered
per pose, not a pre-baked video). The backend owns an `EgoPose`, transparent dynamics
(`KinematicBicycle` for throttle/steer control, or `TrajectoryFollower` to track a rig-frame
`Control.trajectory`), and a pluggable `Renderer`:

- **`StubRenderer`** — deterministic black `HWC-uint8` frames; **no server or GPU**, so it runs the
  whole loop in CI and every `tests/test_nurec.py` case.
- **`NuRecRenderer`** — one stateless `SensorsimService.render_rgb(pose)` gRPC call per frame to an
  `nre-ga` renderer. The render pose is `world_rig @ rig_to_camera` (the camera extrinsic is composed
  in, matching AlpaSim's `construct_rgb_render_request`); the returned JPEG is decoded to `HWC-uint8`.

`step(control)` integrates dynamics → pose → `renderer.render(pose, camera)` → `Observation`.
`checkpoint()`/`restore()` snapshot the whole world state as a small picklable dict. Bringing up the
renderer + a scene is covered in [`SETUP.md`](SETUP.md); the runnable demo is
[`scripts/alpamayo_nurec_demo.py`](../scripts/alpamayo_nurec_demo.py).

### 1c. AlpamayoAVStack (`avsectester/stacks/alpamayo.py`)

Wraps NVIDIA's real **Alpamayo-1.5-10B** end-to-end policy
(`alpasim_driver.models.alpamayo1_5_model.Alpamayo15Model`) as an `AVStack`. `__call__(obs)` builds a
`PredictionInput` (camera frames + ego speed + ego-pose history) → `model.predict()` →
`ModelPrediction.candidate_positions` (K×T×3 rig-frame waypoints) → `Control.trajectory`, which a
`TrajectoryFollower` backend then tracks. The model input contract, learned by running it: it needs
`context_length` camera frames (padded at startup) and ≥1.5 s of backward ego history (synthesized
constant-velocity); model-facing timestamps carry an epoch offset while waypoints stay in the
backend's sim clock. Torch / `alpasim_driver` imports are **lazy**, so importing AVSecTester and
running the offline suite need only the base env. It runs in a dedicated Python-3.12 driver env — see
[`SETUP.md`](SETUP.md).

## 2. Attack — a stream transform, or an avstack hook

Two seams, chosen by how much of the stack the attack needs to see:

**Universal — `perturb(Observation) -> Observation`.** The `run(...)` seam. A sensor/camera/world
perturbation that works against *any* stack, black-box end-to-end policies included. The `Trace`
still records the backend's true ego state, so the metric measures the real driving consequence of a
perturbed *view*. This is where the AlpaSim adversarial-render camera attacks go.

**Modular white-box — an avstack `HOOKS` hook.** For `ModularAVStack`, an attack can attach to a
pipeline stage's pre/post hooks and see its internal tensors:

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

`PhantomInjection.target_xyz` is expressed in the source sensor's coordinates (forward, left, up).
It requires `detections.source_reference`, including when there are no detections. The perception
base class copies the input reference before inference and attaches it to the output before
post-hooks run, so later ego motion does not move an earlier phantom. Passthrough detector inputs
must supply `DataContainer(..., source_reference=reference)` explicitly. Missing source metadata
raises `ValueError`; the attack never guesses the frame from a box or falls back to world origin.
Container copy, filter, mapping, addition and serialization retain the source reference.

## 3. Metric (clean vs attacked → verdict)

```python
from avsectester.metric import impact   # (clean: Trace, attacked: Trace) -> Impact
```

`Impact` answers the differential question — did the attack induce braking / an unsafe stop the
clean run never had — via `induced_braking`, `induced_stop`, and `attack_succeeded`. The verdict is
guarded by a **driving baseline**: `attack_succeeded` requires both that the clean run actually drove
(`clean_drove`, i.e. `clean.peak_speed >= baseline_speed`) *and* `induced_stop`. If the clean ego
never got moving (too few frames, or stuck at the spawn) the result is **inconclusive**, not a
success — "an already-stopped car braking" proves nothing. So a real demo needs enough frames for the
clean run to reach cruising speed (the 40-frame demo peaks ~5 m/s).

## 4. What comes from avstack / AlpaSim

AVSecTester adds only the interface (§0), the two new backend/stack halves (§1b, §1c), the attack and
metric seams, and the viz. The heavy pieces are external:

- **`ModularDrivingPipeline`** (`avstack.modules.pipeline`) — maps `(sensor_data, ego_state)` →
  control by running perception → tracking → planning → control; the modular counterpart to
  end-to-end / foundation-model stacks. Attacks/defenses attach as hooks on any stage.
- **`ForwardCollisionPlanner`** (`avstack.modules.planning.vehicle`) — drive straight, brake to a
  stop when a track occupies the forward corridor (body-frame check); the driving consequence a
  perception attack triggers.
- **`CarlaMobileActor`** (`avcarla`) — the CARLA closed-loop actor: `apply_control` + feeding ego
  state into the pipeline each tick. (These closed-loop pieces were contributed *into* the
  avstack-core / lib-avstack-carla forks, not kept here.)
- **`nre-ga` NuRec renderer** and **`alpasim_driver` / Alpamayo-1.5** — NVIDIA
  [AlpaSim](https://github.com/NVlabs/alpasim) pieces that `NuRecRenderer` and `AlpamayoAVStack`
  wrap. See [`SETUP.md`](SETUP.md) for standing them up.
