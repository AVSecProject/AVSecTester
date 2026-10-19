# Configure simulators

> **Audience:** Users who configure CARLA or NuRec scenes, change the world from handlers, or integrate another simulator.  
> **Outcome:** Configure the supplied simulators, control random choices and actor spawning, and implement a custom simulator adapter.  
> **Prerequisites:** [Architecture](../concepts/architecture.md) and [Installation](../getting-started/installation.md).

A [simulator](../concepts/architecture.md#terminology) owns the world: physical state, vehicle
dynamics, other actors and sensor rendering. In code it has two parts: a `Simulator` holds the
configuration, and each `SimulatorRuntime` it creates is one live world. The supplied simulators
are `CarlaSimulator` and `NuRecSimulator`. A configuration selects one with `backend.type`.
Without `backend`, the scene is a CARLA scene.

## CARLA

`CarlaSimulator` (`avsectester/closed_loop/simulators/carla.py`) connects to a running CARLA 0.9.16
server through avcarla. It owns the client, the ego vehicle and its sensors, NPC traffic and an
optional stationary lead vehicle. The ego's own avcarla pipeline is replaced by an empty one. The
driving system decides each command, and the backend applies it to the ego and advances the server
by one fixed-length tick. CARLA accepts only actuator commands (`throttle`, `brake`, `steer`).

Start the server as described in
[Start a CARLA server](../getting-started/installation.md#start-a-carla-server).

### Scene configuration

Start from the complete example [`configs/carla_scenario.yaml`](../../configs/carla_scenario.yaml).
Keep its `type` fields and nested settings when you edit individual values. The `type` fields
select avcarla and avstack registry classes. The complete key table is in
[CARLA scene keys](../reference/configuration.md#carla-scene-keys).

A CARLA scene has these groups:

- `client` connects to the server (`connect_ip`, `connect_port`) and its Traffic Manager
  (`traffic_manager_port`). It also sets the simulation rate, seeds and optional `map_name`.
- `ego` defines the ego vehicle: model (`vehicle`), spawn point (`spawn`), `sensors` and the
  modular driving `pipeline`. Keep `ego.autopilot: false` so that the driving system controls
  the ego. The pipeline is described in
  [Pipeline configuration](driving-systems.md#pipeline-configuration).
- `npcs` defines background traffic. See [NPCs and spawn positions](#npcs-and-spawn-positions).
- `lead` optionally spawns a stationary vehicle `gap` metres (default 9.0) directly ahead of the
  ego, with an optional `lateral` offset. It is scene content and exists in every treatment.
  The [physical patch](../examples/physical-patch.md) example attaches its panel to it.
- `backend` is optional. For CARLA it accepts only `type: CarlaSimulator`.

Keep `client.synchronous: true`. Every reset reloads the world, and avcarla requires synchronous
mode for that. `client.rate` sets ticks per simulated second. One driving step is one tick, so
40 steps at `rate: 20.0` cover 2 seconds. In synchronous mode, set every ego sensor's
`sensor_tick` to 0 so that it captures on every tick. A `sensor_tick` equal to the step can skip
ticks, because CARLA accumulates sensor time in floating point. The example also sets LiDAR
`rotation_frequency: 20` to match the rate. When the ego carries several sensors, set
`ego.pipeline.perception_input` to the sensor that feeds perception, as in
[`configs/carla_patch_scenario.yaml`](../../configs/carla_patch_scenario.yaml).

World weather is a scenario condition. See [World conditions](conditions.md#world-conditions).
Selected CARLA scenarios use `CarlaSimulator(scenario, depth_camera="front")`, which adds a depth
camera aligned with that RGB camera. See [Select a scenario](scenarios.md#carla).

To check a scene, run one clean treatment:

```bash
avsectester run configs/carla_scenario.yaml --mode single --treatment clean --frames 40
```

The command prints `[scenario] seed=...`, then one `[clean]` line with the mean detections, peak
and final speed. A peak speed above zero shows that the ego was driven by the pipeline.

### Random choices and seeds

Random choices are resolved once per paired experiment, and both runs of the pair use them.
Explicit vehicle models and seeds are preserved. Explicit spawn points are the first positions
attempted, with the retries described under
[NPCs and spawn positions](#npcs-and-spawn-positions). Omitting a field does not generally mean
random: fields have defaults or are required.

| Setting | Behavior across separate commands |
|---|---|
| `client.seed: 0` (the example), or another integer | Repeats the seeded scene choices with the same configuration, map and blueprint library. It does not guarantee identical model outputs or driving traces |
| `client.seed: null`, or omitted | Generates a fresh seed for each command and selects random items again. Individual choices can still repeat by chance |
| A concrete value such as `ego.spawn: 0` | Always attempts this spawn point first, regardless of the seed |

To select a new random ego model and spawn point for each command, change these fields and keep
the other `client` and `ego` settings:

```yaml
client:
  seed: null
  traffic_manager_seed: null
ego:
  vehicle: random
  spawn: random
```

An omitted or `null` `traffic_manager_seed` follows the resolved `client.seed`. The example sets
both seeds to `0`, so changing only `seed` leaves the Traffic Manager seed fixed. Pair
preparation also derives an omitted or `null` LiDAR `noise_seed` from the seed. An explicit
`noise_seed` is preserved.

The CLI prints `[scenario] seed=...` before its first server operation, including when the run later
fails. Set `client.seed` to that integer to repeat the seeded choices. In Python,
`resolve_scenario_seed(config)` from `avsectester.closed_loop.simulators.carla` returns a copy with
a generated seed, as the CLI does. The seed of a paired run is also in
`trace.replay_scenario["client"]["seed"]`. Pair preparation and single runs resolve random choices
with different procedures, so one seed can select different vehicles and spawn points in `single`
and `pair` mode.

The scene seed is separate from `runtime.seed`, which seeds handlers' `context.rng` and not the
simulator. See [Randomness](../reference/runtime.md#randomness) and the
[reproducibility overview](experiments.md#reproducibility).

The seed does not randomize every field. NPC count, map, weather, sensor specifications and
attack parameters are never chosen at random. When `map_name` and weather are omitted, they come
from the world currently loaded on the server. Traffic lights are set from the seeded client
generator unless explicit light settings are given. Pair preparation records the resulting map,
weather and traffic-light settings and applies them to both runs.

### NPCs and spawn positions

The compact form attempts to create `count` NPCs at consecutive spawn indices from
`spawn_start` (default 1). `npc_type: vehicle` selects a random vehicle model for each NPC:

```yaml
npcs:
  count: 6
  npc_type: vehicle
  spawn_start: 1
```

To mix fixed and random choices, replace the compact form with a list. Each entry creates one
NPC. `count` and `spawn_start` apply only to the compact form:

```yaml
npcs:
  - type: CarlaNpc
    npc_type: random
    spawn: random
  - type: CarlaNpc
    npc_type: vehicle.tesla.model3
    spawn: 3
```

Omit `npcs` or set `npcs: []` for no traffic. NPCs drive on CARLA autopilot through the Traffic
Manager at `client.traffic_manager_port`.

Spawn indices refer to the spawn points of the loaded map. Random spawn selection reserves the
explicit indices first and picks unused indices for random actors. Different indices do not
guarantee that all vehicle geometries fit without collisions.

When a spawn fails during the first run, the backend moves the requested position 3 metres
forward along the vehicle heading in the horizontal plane and tries again, up to 10 attempts in
total. Set `client.strict_spawn: true` to fail instead of relocating. The backend records each
vehicle's actual position, orientation and model before the first driving step. Later runs of the
same experiment reuse these recorded transforms in the same creation order, without offsets or
retries. If a recorded spawn fails, the run reports an error rather than moving the vehicle.
Separate experiments can therefore resolve different spawn positions. If the first run exhausts
its retries, choose other indices or fewer vehicles.

`ego.destination` accepts `null`, a spawn index or `random`. The default straight-driving
pipeline does not navigate to it.

### Paired runs on one server

Use a dedicated CARLA server for each experiment. Treatments run sequentially on that server.
Each reset destroys the previous actors, reloads the world in synchronous fixed-step mode, and
rebuilds the ego, NPCs, lead vehicle and weather. The driving system is rebuilt by the stack
factory for every treatment.

Before the first of several runs, `run_pair` and `run_prepared` call
`backend.prepare_clean_attack_pair()`. For CARLA this:

1. Generates a missing `client.seed` and sets a missing `traffic_manager_seed` to it.
2. Connects once and resolves random vehicle models, spawn indices, destinations and LiDAR noise
   seeds. Sensors without `source_ID` receive `0`.
3. Records the map, weather and traffic-light states, and requires `client.reset_world` (an
   explicit `false` is rejected).

The first run starts from this resolved configuration and may relocate failed spawns. Its reset
captures `replay_scenario`: the same scene with each vehicle's actual model and spawn transform,
and `strict_spawn: true`. Every later run starts from `replay_scenario`. A new call to
`prepare_clean_attack_pair()` resolves random choices again from the original configuration.

The shared origin covers the initial scene only. NPC traffic keeps reacting to the ego after the
start, so an attack can change NPC trajectories as part of its driving consequence.

### Python entry points

`avsectester.carla_runner` assembles a CARLA backend and the modular stack:

```python
import yaml

from avsectester.carla_runner import run_scenario, run_scenario_pair

with open("configs/carla_scenario.yaml") as file:
    config = yaml.safe_load(file)

clean, attacked = run_scenario_pair(config, frames=40)
single = run_scenario(config, frames=40)  # One clean run. YAML attacks are not applied
print(attacked.replay_scenario["ego"]["spawn_transform"])
```

- `run_scenario_pair(scenario, frames=40, *, clean_runtime=None, attacked_runtime=None,
  settle_iters=100, stack_factory=None)` runs a clean and an attack treatment from one origin.
  The attack treatment uses the configured `runtime`, `attacks`, `patches` and `interventions`
  unless `attacked_runtime` is given.
- `run_scenario(scenario, attacks=None, frames=40, settle_iters=100, patches=None, *,
  runtime=None)` runs once. It ignores the configured interventions and applies only the
  `attacks`, `patches` and `runtime` arguments.

Both build the driving system from `stack` when it is configured. Otherwise they build a
`ModularAVStack` from `ego.pipeline`, instrumented at perception so that each `FrameRecord`
carries `n_detections`. `settle_iters` bounds the wait for asynchronous sensor data at each
observation, in 5 ms polls. Both close the backend when they finish. Signatures and the general
runners are in [Driving loop and runners](../reference/python-api.md#driving-loop-and-runners).

`Trace.replay_scenario` is an in-memory configuration, not a file. Each vehicle's
`spawn_transform` holds native CARLA world coordinates: `location` x, y, z in metres and
`rotation` pitch, yaw, roll in degrees. It already includes the configured spawn offset and any
retry displacement. The runner produces this field. Configure `spawn` instead.

## NuRec

`NuRecSimulator` (`avsectester/closed_loop/simulators/nurec.py`) renders an NVIDIA NuRec
reconstructed scene and runs the ego in process. It owns the ego pose (`EgoPose`: `x`, `y` in
metres, `yaw` in radians, `speed` in m/s and time `t`), a dynamics model and a renderer. It does not
run a physics engine or traffic. NuRec runs are configured from Python. The CLI does not run them,
as explained in [Why NuRec has no CLI
path](../getting-started/installation.md#why-nurec-has-no-cli-path).

`reset()` loads the scene in the renderer, places the ego at `ego0`, emits `simulator.setup` and
renders the first observation. `step(control)` validates the command, emits `simulator.step.pre`,
advances the pose with the dynamics model, emits `simulator.step.post` and renders every camera. Each
camera passes through `render.pre` and `render.post`. The observation contains the rendered
images, the renderer's calibration, a copy of the pose as `vehicle_state` and its speed as
`ego_speed`. See [Stages](../reference/runtime.md#stages).

### Renderers

- `StubRenderer(cameras=None, height=480, width=640)` returns black `HWC` `uint8` frames. It needs
  no server or GPU. Its default camera is `camera_front`. Use it for offline development and
  tests.
- `NuRecRenderer` sends one `SensorsimService.render_rgb` gRPC request per camera and
  observation to an `nre-ga` rendering service. Its options are `endpoint` (default
  `127.0.0.1:50051`), `scene_id`, `cameras` (default `["camera_front_wide_120fov"]`),
  `calibration`, `image_quality` (default 95), `start_timestamp_us` and `start_transform`.

`NuRecRenderer` connects during each reset. It selects the scene that matches the `scene` setting
or `scene_id` exactly or as a unique substring, and fails otherwise. Without either, it uses the
first scene the service offers. It reads each camera's intrinsics and mounting pose and the
recorded start time. The camera pose sent to the service is the ego's world pose composed with
the camera's mounting pose. The returned JPEG is decoded to an RGB `HWC` `uint8` array. After
loading, `renderer.service` exposes the native gRPC client. The ego moves in the plane. Camera
height comes from the first recorded ego pose, or from `start_transform`, which also keeps its
tilt. To start the service and obtain scenes, see
[NuRec renderer and scenes](../getting-started/installation.md#nurec-renderer-and-scenes).

A custom renderer subclasses `Renderer` and implements `render(pose, camera)`. It can override
`load_scene(scene)` and `close()`, and sets `cameras` and `calibration`.

### Dynamics and time step

The dynamics model turns the final command into ego motion and decides the backend's command
mode:

| `dynamics.type` (class) | Command mode | Behavior |
|---|---|---|
| `KinematicBicycle` (default) | `actuator` | Kinematic bicycle model. Options `wheelbase` (2.8 m), `max_accel` (3.0 m/s²), `max_brake` (8.0 m/s²) and `max_steer` (0.6 rad) scale the command |
| `TrajectoryFollower` | `trajectory` | Places the ego on the planned trajectory at the end of each step. Option `decel` (4.0 m/s²) applies without a trajectory |

`TrajectoryFollower` reads `Control.trajectory` as rig-frame waypoints with absolute microsecond
timestamps on the backend clock. See [Trajectory commands](../reference/runtime.md#trajectory-commands).
It interpolates from the current ego position to the first waypoint and between later waypoints,
then sets the heading to the direction of travel and the speed to distance over `dt`. It follows
the plan exactly. Without a trajectory, the ego keeps its position and its speed decreases by
`decel`.

**Not yet implemented.** A trajectory-to-actuator controller with a vehicle dynamics model for
this path.

The backend's `dt` (default `0.05`) is the physics and observation interval in seconds. It is
independent of the policy's waypoint interval. Alpamayo's 10 Hz waypoints are 0.1 seconds apart,
and the follower's interpolation makes a 0.05-second step valid. Set `dt: 0.1` to align steps
with the waypoints. Do not rescale predicted positions to change the simulation rate. The driving
system is called once per step, so `dt` also sets its call rate.

A custom dynamics object implements `step(pose, control, dt) -> EgoPose` and can declare
`command_modes`. Pass it as `NuRecSimulator(config, dynamics=...)`. Without `command_modes`, the
backend's command mode is undeclared.

`backend.checkpoint()` and `backend.restore(checkpoint)` save and restore the ego pose and frame
counter. They do not checkpoint the rendering service.

### Backend configuration

Configured entry points such as `scenario_from_config` and `run_configured` build a NuRec backend
from a `backend` mapping with `type: NuRecSimulator`.
The keys are listed in [NuRec backend keys](../reference/configuration.md#nurec-backend-keys).
Top-level scene keys, such as `dt`, are also passed to the backend. Keys inside `backend` take
precedence.

This example runs offline with the stub renderer, trajectory dynamics and a minimal driving
system:

<!-- runnable -->
```python
from avsectester.closed_loop.avstack import AVStack
from avsectester.closed_loop.datatypes import Control
from avsectester.runner import run_prepared
from avsectester.scenarios import scenario_from_config


class StraightPlanner(AVStack):
    """Plan rig-frame waypoints straight ahead at a constant speed."""

    def __init__(self, speed=5.0, horizon=10, interval_s=0.1):
        self.speed, self.horizon, self.interval_s = speed, horizon, interval_s

    @property
    def command_modes(self):
        return frozenset({"trajectory"})

    def __call__(self, observation):
        start_us = int(observation.t * 1e6)  # waypoint times use the backend clock
        step_us = int(self.interval_s * 1e6)
        return Control(trajectory=[
            ((self.speed * self.interval_s * i, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0), start_us + i * step_us)
            for i in range(1, self.horizon + 1)
        ])


scenario = scenario_from_config({
    "backend": {
        "type": "NuRecSimulator",
        "renderer": {"type": "StubRenderer", "cameras": ["camera_front_wide_120fov"]},
        "dynamics": {"type": "TrajectoryFollower"},
        "ego0": {"speed": 5.0},
        "dt": 0.05,
    },
})
trace = run_prepared(scenario, StraightPlanner, frames=4)["clean"]
print([(round(record.t, 2), round(record.speed, 2)) for record in trace.records])
# [(0.05, 5.0), (0.1, 5.0), (0.15, 5.0), (0.2, 5.0)]
```

For the rendering service and Alpamayo, replace the renderer and the driving system. Run this in
the [driver environment](../getting-started/installation.md#driver-environment):

```python
from avsectester.legacy_experiment import run_configured

traces = run_configured({
    "dt": 0.1,
    "backend": {
        "type": "NuRecSimulator",
        "ego0": {"speed": 5.0},
        "renderer": {"type": "NuRecRenderer", "endpoint": "127.0.0.1:50051",
                     "scene_id": "01d503d4", "cameras": ["camera_front_wide_120fov"]},
        "dynamics": {"type": "TrajectoryFollower"},
    },
    "stack": {"type": "AlpamayoAVStack", "options": {
        "checkpoint_path": "/path/to/Alpamayo-1.5-10B", "device": "cuda:1",
    }},
}, frames=8, mode="single", treatment="clean")
trace = traces["clean"]
```

Alpamayo produces trajectory commands, so it needs `dynamics.type: TrajectoryFollower`. With
`KinematicBicycle` dynamics, the run raises `CommandCompatibilityError` before reset unless a `command` handler
converts the command. See [Commands and compatibility](driving-systems.md#commands-and-compatibility).

`ego0` is the initial pose in the scene's world coordinates (`x`, `y`, `yaw`, `speed`, `t`). The
backend does not read the recorded start pose itself. `camera_aliases` renames native cameras in
`observation.sensor_data` and `calibration`, for example `{"camera_front": "front"}`. The render
stages report the native name in `context.sensor`. Unknown backend keys raise
`ScenarioConfigurationError` when the backend is built. NuRec accepts no world conditions, so a
`conditions.world` setting raises `ConditionConfigurationError`.

When you construct `NuRecSimulator(config, renderer=..., dynamics=...)` directly, pass renderer and
dynamics objects. The string and mapping forms are interpreted only by the `backend` mapping.
`NuRecSimulator` is the configuration; `create_runtime()` creates a live world. The renderer you
pass is a template that is never connected itself: each runtime renders through its own
`renderer.copy()`, with its own connection, loaded scene and origin, so access the live renderer as
`runtime.renderer`. A custom renderer that holds resources a shallow copy would share overrides
`copy()`.
The demo [`scripts/demos/nurec/alpamayo_nurec_demo.py`](../../scripts/demos/nurec/alpamayo_nurec_demo.py)
supplies such a backend through `backend_factory`. Scenarios selected from recorded NuRec scenes
come from `NuRecDataset`, whose `make_backend(scene)` sets `ego0` from the selected frame. See
[NuRec datasets](scenarios.md#nurec-datasets).

### Recorded actors

The renderer requests each image at the scene's start time plus the ego's simulated time `t`.
Other actors in the reconstruction appear where they were recorded at that time. They do not
react to the ego, and the backend does not detect collisions with them. `backend.native` exposes
no other actors. For their recorded poses, use `NuRecDataset.actor_poses(usdz_path, timestamp_us)`
with `renderer.timestamp_us(pose)`.

**Not yet implemented.** Other actors in NuRec that react to the ego.

## Change the world from handlers

Handlers at `simulator.setup`, `simulator.step.pre` and `simulator.step.post` receive `None`, must return
`None`, and change the world through the simulator's native resources in `context.simulator_runtime.native`.
Editing `context.physical_state` or an observation never moves anything. The general read and
write rules are in [Read and write interfaces](../reference/runtime.md#read-and-write-interfaces)
and [Native resources and copying](../reference/runtime.md#native-resources-and-copying).

**CARLA**

| Data | Read | Change |
|---|---|---|
| Configured vehicles | `native["actors"][name].get_transform()` or `.get_velocity()`, with `name` `"ego"`, `"lead"` or `"npc:<index>"` | Raw actor methods such as `.set_transform(...)` |
| Environment | `native["world"].get_weather()` and other `carla.World` getters | `native["world"].set_weather(...)` and other supported world methods |

`native` also contains `client` (the avcarla client), `world` (`carla.World`), `ego` (the avcarla
ego wrapper), `lead` (the raw lead actor or `None`) and `npcs` (the avcarla NPC wrappers).
Coordinates are native CARLA world coordinates. Getters reflect the latest completed tick, and
native commands can take effect asynchronously. CARLA renders sensors during the tick, before
`simulator.step.post`. A change at `simulator.step.post` therefore appears from the observation after next.
Use `simulator.setup` or `simulator.step.pre` when the next observation must show it. At `simulator.setup`,
no tick has completed since spawning, so `get_transform()` can still return the default origin.
Actors are rebuilt at every reset. Look them up in `native` at each callback instead of keeping
them across runs.

This handler shifts the ego by one metre along the world's x axis before the first command is
applied, after the initial sensor capture:

<!-- runnable -->
```python
from avsectester.closed_loop.hooks import Hook, Runtime


def move_carla_ego(value, context):
    if context.step != 0:
        return None
    actor = context.simulator_runtime.native["actors"]["ego"]
    transform = actor.get_transform()
    transform.location.x += 1.0
    actor.set_transform(transform)
    return None


runtime = Runtime([Hook("simulator.step.pre", move_carla_ego)])
```

A world plugin that needs extra ticks before driving, such as the physical panel, calls
`context.simulator_runtime.setup_tick()` during `simulator.setup`. See
[World setup ticks](../reference/runtime.md#world-setup-ticks).

**NuRec**

| Data | Read | Change |
|---|---|---|
| Ego | `context.physical_state.vehicle_state` (an independent copy) or `context.simulator_runtime.pose` (live) | Set `context.simulator_runtime.pose.x`, `.y`, `.yaw` or `.speed` |
| Ego dynamics | `native["dynamics"]` | Change its attributes, such as `max_brake`, for later steps. The dynamics belong to this runtime, so other runs keep the configured values |
| Renderer | `native["renderer"]`, including `service` for `NuRecRenderer` | Renderer-specific operations |
| Render requests | `context.simulator_runtime.render_request(camera)` and `native["render_requests"]` | Read-only copies. Use a `render.pre` handler to change the next request |

NuRec renders after `simulator.step.post`, so a change at that stage appears in the next
observation. Other actors follow their recordings and cannot be moved.
`render_request(camera)` accepts a native camera name or an alias. It returns an independent copy
of the request that produced the current image, and editing it has no effect on rendering.

This example runs offline. It moves the ego at `simulator.setup` and records the render pose:

<!-- runnable -->
```python
from avsectester.closed_loop.avstack import AVStack
from avsectester.closed_loop.datatypes import Control
from avsectester.runner import run_prepared
from avsectester.closed_loop.hooks import Hook, Runtime
from avsectester.scenarios import scenario_from_config


class Cruise(AVStack):
    @property
    def command_modes(self):
        return frozenset({"actuator"})

    def __call__(self, observation):
        return Control(throttle=0.3)


def move_initial_ego(value, context):
    context.simulator_runtime.pose.x += 1.0  # metres along the scene's x axis
    return None


def record_render_x(observation, context):
    request = context.simulator_runtime.render_request("camera_front")  # an independent copy
    context.record_output("render_x", request.pose.x)
    return observation


scenario = scenario_from_config({"backend": {"type": "NuRecSimulator", "renderer": "StubRenderer"}})
results = run_prepared(scenario, Cruise, frames=1, runtimes={
    "clean": Runtime([Hook("sensors.post", record_render_x)]),
    "moved": Runtime([Hook("simulator.setup", move_initial_ego), Hook("sensors.post", record_render_x)]),
})
for name, trace in results.items():
    print(name, [output.value for output in trace.outputs])
# clean [0.0]
# moved [1.0]
```

For a world-changing attack packaged as an intervention, see
[Change the world](attacks.md#change-the-world).

## Write a custom simulator adapter

A simulator has two classes, as CARLA, NuRec and recorded replay do:

- A subclass of `avsectester.closed_loop.simulator.Simulator` holds the configuration and the
  declarations: `dataset_kinds` (accepted dataset class names, such as `("CarlaDataset",)`),
  `decision_time_step`, `capture_ticks`, `can_render`, `command_modes`, `supported_stages`,
  `supported_world_settings`, `identity` and `validate_start`. Override
  `validate_world_settings(settings)` to validate supported values without opening a world. Its `create_runtime(seed=None)` returns a new runtime. Copy
  any mutable template it holds into each runtime, as NuRec does with its renderer and ego
  dynamics, so that one run cannot change another.
- A subclass of `avsectester.closed_loop.simulator.SimulatorRuntime` is one live world. The base
  owns `reset`, `step`, `observe`, `state()`, the deployment tick budget, the read-only `clock`,
  the process-local `exclusive()` reservation and closing as a context manager. It reads the
  declarations from its `simulator`. Overriding `reset` or `step` in a subclass raises
  `TypeError` at class definition.

The runtime's `reset(..., world=world_settings)` accepts only the native world mapping, such as
`{"weather": "ClearNoon"}`. The runner and closed-loop services handle the experiment's
`environment["sensors"]` after capture and pass only `environment["world"]` to reset. Passing
`sensors` or a nested `{world, sensors}` mapping to reset is an error.

A runtime implements:

| Member | Contract |
|---|---|
| `_initialize_world(start, initialization, environment)` | Abstract. Build the world at `start`, or at the configured origin when `start` is `None`. Recompute all run state, so a later plain reset keeps nothing of the previous start |
| `_advance_native(control)` | Abstract. Apply the command and advance the world by one decision |
| `_observe() -> Observation` | Abstract. Capture the sensor data. Fill every field: `t`, `frame`, `sensor_data`, `calibration`, `vehicle_state` and `ego_speed`, sharing no mutable objects with the physical state |
| `_state() -> WorldSnapshot` | Abstract. An independent snapshot of the physical state, read by `state()`. Every handler call reads it, so it must work at each stage, including `simulator.setup` during reset |
| `_replay_tick()`, `_hold_tick()` | Advance one tick along the start's prescribed motion, or one tick holding the origin. The defaults raise. Needed to start at a later frame or to support deployment ticks, as in CARLA |
| `_check_initialization(initialization)` | Handle simulator-specific initialization data and record state/provenance in `initialization_report`. The default rejects nonempty initialization; no replay-deviation tolerance is imposed |
| `_capture_frame_data(frame) -> FrameData` | Record the current state as `FrameData`, used by `generate_frame_data` |
| `sensor_view(sensor) -> SensorView` | Required when the simulator sets `can_render = True`. The view of the current capture: the RGB data, the camera model, `cam_from_world`, actor and ego cuboid poses, time and, when measured, depth |
| `native` | Live simulator resources for handlers. The default is empty |
| `close()` | Release the world and native connections. It can be called more than once |

The base emits `simulator.setup`, `simulator.step.pre` and `simulator.step.post`. Emit any further
stage with `value = self._emit(stage, value, sensor=None)` and declare it in the simulator's
`supported_stages`, as NuRec does for `render.pre` and `render.post`. `generate_frame_data(start,
frames)` is shared: it resets as for a run, then alternates `_replay_tick()` and
`_capture_frame_data(frame)`. A runtime that implements both can generate frame data for datasets
without it, as CARLA and recorded replay do. See [selected-origin
initialization](scenarios.md#selected-origin-deployment-and-reset-ownership).

### Legacy `WorldBackend` adapters

`WorldBackend` is the earlier single-class interface. `ClosedLoopRunner` and the legacy experiment
path still run it through an internal adapter, with plain resets only and without sensor views. It
is removed together with the legacy paths. Subclass `avsectester.closed_loop.simulator.WorldBackend`.
Only `reset` and `step` are abstract. Override the other members your simulator needs:

| Member | Default | Contract |
|---|---|---|
| `reset() -> Observation` | Abstract | Build the world, emit `simulator.setup` if declared, and return the first observation. The ego does not move until the first command |
| `step(control) -> Observation` | Abstract | Call `validate_command`, emit `simulator.step.pre`, apply the command, advance the world, emit `simulator.step.post` and return the next observation |
| `ground_truth() -> WorldSnapshot` | Copy of the state in the last observation returned by `reset` or `step` | An independent snapshot of the physical ego state. Override it when observations carry estimated state or when the backend emits stages |
| `close()` | No-op | Release simulator resources. It can be called more than once |
| `copy_observation(observation)` | `deepcopy` | Detach mutable model-visible data from backend state. Override it for native captures that cannot be copied |
| `supported_stages` | Empty | The stages the backend emits |
| `command_modes` | `None` (undeclared) | `frozenset` of `"actuator"` and/or `"trajectory"` |
| `validate_command(command)` | Rejects active fields outside `command_modes` | Raise `ValueError` for commands the simulator would ignore |
| `configure_conditions(world)` | Raises `ValueError` for any setting | Validate and store world conditions for later resets |
| `native` | Empty mapping | Live simulator resources for handlers |
| `setup_ticks`, `setup_tick()` | `0`, absent | Deployment ticks required by world plugins, advanced during `simulator.setup` |
| `prepare_clean_attack_pair()` | No-op | Resolve random scene choices once, so that later resets reproduce the first run's initial scene |

Rules for emitting stages with `self._emit(stage, value, sensor=None)`:

- Declare every emitted stage in `supported_stages`. Emitting an undeclared stage during a run
  raises `ValueError`, even without registered handlers. Registering a handler at an undeclared
  stage fails before reset.
- Use the returned value: `value = self._emit(stage, value)`. World stages pass and return
  `None`. `render.pre` passes a `CameraRenderRequest`. Pass `sensor=` for per-sensor stages.
- Emit `simulator.setup` after the world is built and before the first sensor capture,
  `simulator.step.pre` before actuation and `simulator.step.post` after the world has advanced. Document
  whether sensors are rendered during the advance, as in CARLA, or after `simulator.step.post`, as in
  NuRec.
- Every handler call reads `ground_truth()`, so it must work at each emitted stage, including
  `simulator.setup` during the first reset. The default implementation fails there.
- Outside a run, `_emit` returns the value unchanged. One run binds the adapter at a time.

Fill every `Observation` field: `t`, `frame`, `sensor_data`, `calibration`, `vehicle_state` and
`ego_speed`. Return observations that share no mutable objects with the physical state. If the
backend supports deployment ticks, implement `setup_tick()` to advance one tick only during
`simulator.setup`, and advance until `setup_ticks` have been used after emitting `simulator.setup`.
A deterministic backend needs no `prepare_clean_attack_pair`.

Custom backends enter scenario setup through `backend_factory(scene_config)`, accepted by
`scenario_from_config`, `source_from_config` and `run_configured`. The argument is a fresh copy
of the scene view of the configuration, without experiment keys such as `frames`, `stack` or
`conditions`. See [File layout](../reference/configuration.md#file-layout). The YAML
`backend.type` accepts only the supplied simulators. The factory returns a live world, for
example `lambda config: CarlaSimulator(config).create_runtime()`. `ScenarioInstance.make_backend()`
calls `configure_conditions` when the scenario has world conditions.

This example runs offline. It implements a straight road with a stopped obstacle, a range sensor
and a world condition:

<!-- runnable -->
```python
from avsectester.closed_loop.avstack import AVStack
from avsectester.closed_loop.simulator import WorldBackend
from avsectester.closed_loop.datatypes import Control, Observation, WorldSnapshot
from avsectester.runner import run_prepared
from avsectester.scenarios import scenario_from_config


class LineWorld(WorldBackend):
    """A straight road with a stopped obstacle and a forward range sensor."""

    def __init__(self, config):
        self.obstacle_x = float(config["obstacle_x"])
        self.initial_speed = float(config.get("speed", 5.0))
        self.dt = float(config.get("dt", 0.1))
        self.sensor_range = float("inf")
        self.x = self.speed = self.t = 0.0
        self.frame = 0

    @property
    def command_modes(self):
        return frozenset({"actuator"})

    @property
    def supported_stages(self):
        return frozenset({"simulator.setup", "simulator.step.pre", "simulator.step.post"})

    @property
    def native(self):
        return {"world": self}

    def configure_conditions(self, world):
        unknown = set(world) - {"sensor_range"}
        if unknown:
            raise ValueError(f"Unsupported LineWorld conditions: {sorted(unknown)}")
        self.sensor_range = float(world.get("sensor_range", self.sensor_range))

    def reset(self):
        self.x, self.speed, self.t, self.frame = 0.0, self.initial_speed, 0.0, 0
        self._emit("simulator.setup", None)  # after building the world, before the first capture
        return self._observe()

    def step(self, control):
        self.validate_command(control)
        self._emit("simulator.step.pre", None)  # before actuation
        accel = 3.0 * control.throttle - 8.0 * control.brake
        self.speed = max(0.0, self.speed + accel * self.dt)
        self.x += self.speed * self.dt
        self.t += self.dt
        self.frame += 1
        self._emit("simulator.step.post", None)  # after advancing, before the next capture
        return self._observe()

    def ground_truth(self):
        return WorldSnapshot(self.t, self.frame, {"x": self.x}, self.speed)

    def _observe(self):
        gap = self.obstacle_x - self.x
        return Observation(
            t=self.t, frame=self.frame,
            sensor_data={"range": gap if gap <= self.sensor_range else None},
            vehicle_state={"x": self.x}, ego_speed=self.speed,
        )


class RangeBrake(AVStack):
    @property
    def command_modes(self):
        return frozenset({"actuator"})

    def __call__(self, observation):
        gap = observation.sensor_data["range"]
        return Control(brake=1.0) if gap is not None and gap < 12.0 else Control(throttle=0.2)


scenario = scenario_from_config(
    {"obstacle_x": 30.0, "speed": 5.0, "conditions": {"world": {"sensor_range": 10.0}}},
    backend_factory=LineWorld,
)
trace = run_prepared(scenario, RangeBrake, frames=60)["clean"]
first = next(record for record in trace.records if record.brake > 0)
print(f"first brake recorded at t={first.t:.1f} s, final speed {trace.final_speed:.1f} m/s")
# first brake recorded at t=3.5 s, final speed 0.0 m/s
```

A matching driving system must consume the backend's sensor and state formats. See
[Use a custom driving system](driving-systems.md#use-a-custom-driving-system).

## Next steps

- [Configure driving systems](driving-systems.md)
- [Apply operating conditions](conditions.md)
- [Implement an attack](attacks.md)
- [Run experiments](experiments.md)
- [Configuration reference](../reference/configuration.md)
- [Runtime reference](../reference/runtime.md)
