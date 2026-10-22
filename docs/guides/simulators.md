# Simulators

> **Audience:** Experiment authors and simulator implementers.
>
> **Outcome:** Select a simulator or implement its configuration/simulator-runtime boundary.
>
> **Prerequisites:** [Architecture](../concepts/architecture.md).

## Simulator and simulator runtime

A Simulator stores reusable configuration and capability declarations. `create_runtime(seed)`
creates a fresh SimulatorRuntime owning one live world, clock, sensors and native connections. Its
creator closes it, including after errors. ClosedLoopRunner uses but does not own caller resources;
Experiment and ClosedLoopServices create and close their own resources.

SimulatorRuntime `reset(start, initialization=None, *, deployment=None, world=None)` initializes a
world and captures the first observation. `observe()` returns a copy of that capture without
advancing, requesting native renders or running hooks. `step(control)` advances one decision and
captures again.
`state()` returns independent physical state, available even at setup and step hooks.

Only world settings go into reset. The complete `{world, sensors}` envelope is resolved by the
experiment, runner and services; sensor corruptions run after capture.

### Read prescribed motion

start_replay(start, initialization=None, *, world=None, deployment=None) initializes at the selected
recorded or recipe boundary and captures it while keeping ego under prescribed motion. It is used
for raw sensor reads before driving-system control. advance_replay(count) then advances that motion
and captures the requested boundary without emitting stage hooks. Both methods reject bound hooks;
forward advancement also requires an initialized runtime and can_replay=True. The capability
defaults to False.

| Simulator | can_replay |
|---|---|
| CARLA | True |
| Recorded playback | True |
| NuRec | False |

Closed-loop services reuse a forward scan only when can_replay is True. NuRec reads another frame
by calling start_replay for that frame rather than advancing the previous read. This preserves the
recorded pose for each raw sensor request.

## Inserted world objects

Simulator.can_insert_objects declares support for framework-composited ObjectAsset geometry.

| Simulator | can_insert_objects | Camera occlusion source |
|---|---|---|
| In-memory example | True | Available depth, otherwise actor cuboids |
| CARLA | True | Aligned depth for depth_camera; actor cuboids for other cameras |
| NuRec | True | Static mesh and actor boxes with depth="mesh"; actor cuboids with depth="none" |
| Recorded playback | True | Actor cuboids |
| Base Simulator | False | Unsupported until declared by the implementation |

SimulatorRuntime.insert_object(object_id, asset, world_from_object, harmonizer=None) registers an
object; remove_object removes it. Images change from the next capture. state().objects reflects
the registered objects immediately, independently of state().actors. The shared simulator-runtime
implementation composites all cameras, so a new simulator declares camera_sensors and implements
native _sensor_view rather than a separate object renderer. A lightweight sensor_geometry override
is an optional optimization; the default derives geometry from the native view.

The framework first checks projection using camera geometry. CARLA, NuRec and recorded playback
provide lightweight geometry, so cameras with no projected object avoid view construction and
mesh-depth computation. For affected cameras, composite_objects uses aligned scene depth if
present, otherwise actor cuboids. The nearest object surfaces are composited in depth order.
sensor_view returns the composed data. If the native view supplies depth, the returned depth is
the pixelwise minimum of that depth and inserted-object depth. If native depth is None, it stays
None after insertion. Actor-cuboid fallback estimates are used only in candidate projection regions;
they are not promoted to full sensor-depth evidence. Repeated reads reuse the capture's composition.
The input order is object composition, sensor attack actions, sensor environment, then
ObservationPipeline.

WorldSnapshot.objects records each object's world pose, category, full extent and zero velocity,
with box_center=(0, 0, height/2). Inserted objects remain separate from native actors and do not
participate in native physics or collisions.

object_visibility(object_id) describes the latest capture. Every camera whose image contains the
object's projection has an ObjectVisibility entry with occlusion_source, ignored_occluders and
visible_pixels, including a fully hidden object (visible_pixels=0). It raises KeyError for an
object that is not part of the latest capture and re-raises a failed composition. SensorView.objects
holds the PlacedObject values of the same cameras; they are immutable and shared, not copied. A
sensor insertion in a view without depth uses their alpha-aware surfaces as occluders. The recorded
structure is described in [RunLog](../reference/runtime.md#runlog). See
[Insert a world object](attacks.md#insert-a-world-object) for coordinate conventions and a runnable
attack.

**Not yet implemented.** Inserted objects in LiDAR or other non-camera sensors; those captures are
unchanged. See [TODO](../development/TODO.md#inserted-objects-in-non-camera-sensors).

## CARLA

`CarlaSimulator(client, settle_iters=100, *, simulator_version="0.9.16", depth_camera=None)` accepts
simulator-runtime connection and timing settings. Scene content belongs in
`CarlaDataset(recipes=...)`, not the simulator. Each reset requires a ScenarioStart carrying its
recipe. Use matching CARLA 0.9.16 client/server packages and a dedicated server.

### Scene configuration

CarlaRecipe holds map name, ego and NPC spawns, sensors, explicit motion, time_step and horizon. The
simulator client chooses address, port, traffic-manager port, synchronous operation and rate.
CarlaSimulator.client_config holds these reusable settings; the live simulator runtime owns client.
An explicit create_runtime(seed) overrides client.seed for that run; absent an explicit run seed,
the configured seed is retained. traffic_manager_seed is fixed when explicitly supplied; otherwise
it is derived from the resolved run seed. Seeds must be nonnegative integers, and RPC/TM ports must
be integers from 1 through 65535. The client rate must agree with recipe time_step. PrescribedMotion
selects constant_control, per-step controls or an N+1 trajectory for each actor.

### Selected-origin initialization

The simulator runtime initializes base actors and sensors, replays prescribed motion without hooks
to the deployment window, applies simulator.setup actions and runs the shared deployment ticks.
Capture ticks then make world actions visible in the selected frame. The selected boundary is
preserved; a positive boundary earlier than the required window is rejected. At position zero,
actors are held while initialization ticks execute.

Clean treatments use the same tick budget. Setup does not count as a driving decision. Actual
initial states, actor identities and sensor times are recorded. No replay deviation threshold,
selection revalidation or automatic retry is performed. Before decision zero, ego is released from
recipe motion so driving commands control it.

Set depth_camera to a configured RGB sensor alias for aligned depth and a pinhole camera view. The
simulator runtime constructs SensorView directly from native camera geometry and captured data.

## NuRec

NuRecSimulator takes `config`, a required Renderer template, an optional ego-dynamics template and
a depth mode. Configure NuRecRenderer explicitly; there is no default renderer. Renderer is defined
in closed_loop/simulators/nurec/renderer.py, and the dynamics interfaces and implementations live
in closed_loop/simulators/ego_dynamics.py.
The default depth="mesh" requires the optional nurec dependencies at construction; depth="none"
disables depth and needs no mesh-intersection library. Each simulator
runtime copies them, so connections, loaded scenes and dynamics mutations are isolated. Set
`config.dt` to NuRecDataset.time_step; this mismatch is rejected before simulation starts.

With depth="mesh", a reset from a recorded Dataset ScenarioStart checks that its USDZ contains both
mesh.ply and mesh_ground.ply before its first NRE request. Missing entries raise ValueError listing
their filenames. depth="none" does not require either mesh entry.

NuRecRenderer connects to the reconstruction service and renders at requested times and ego poses.
Start NRE with --enable-editing-actors, as in the
[installation command](../getting-started/installation.md#nurec-renderer-and-scenes).
Actors follow their recorded tracks unless explicitly moved; they do not react to traffic.
SimulatorRuntime.restore immediately captures its restored state; later observe calls do not render
again.

The configuration can be constructed without connecting to NRE; a connection is opened when a
simulator runtime loads its scene:

<!-- runnable -->
```python
from avsectester.closed_loop.simulators.ego_dynamics import TrajectoryFollower
from avsectester.closed_loop.simulators.nurec import NuRecRenderer, NuRecSimulator

simulator = NuRecSimulator(
    {"dt": 0.1},
    renderer=NuRecRenderer(endpoint="127.0.0.1:50051"),
    dynamics=TrajectoryFollower(),
    depth="none",
)
assert simulator.command_modes == frozenset({"trajectory"})
```

### Move recorded actors

NuRecSimulatorRuntime.set_actor_state(actor_id, state) moves an actor already reconstructed in the
scene. The ID must be present in the recorded tracks at the current time. An unknown ID raises
KeyError listing available IDs, and ego is rejected with ValueError because driving commands own
ego motion. state.velocity must be an explicit world-coordinate vector; None raises ValueError.

The supplied pose is located at the call's physical time. Starting with the next capture, the
actor leaves its recorded trajectory and moves at constant state.velocity, keeping its supplied
orientation. Zero velocity holds it still. The override persists until reset or the end of the
run, or until another set_actor_state call replaces it. state() reports current physical actors;
sensor_view().actors retains the poses used for its capture and changes at the next capture.
control_actor remains unsupported and raises NotImplementedError; this is kinematic editing
rather than a traffic or actuator model.

Every render sends all tracked actors present at its timestamp: overridden poses for edited
tracks, interpolated recorded poses for the others. Once actor editing is enabled, NRE omits
dynamic actors absent from the request. Recorded track poses use [x, y, z, qx, qy, qz, qw], with
the same world transform sent back through named quaternion fields. No half-height offset is
added; recorded track ActorState uses a zero box_center. checkpoint/restore also preserves actor
overrides.

NRE can move only actors reconstructed in the source scene. Its editing interface cannot insert
new geometry or textures; framework object composition supplies those camera pixels without sending
the new objects to NRE.

KinematicBicycle consumes actuator controls. TrajectoryFollower consumes trajectory controls and
follows them ideally. Choose compatible AVStack commands; there is no automatic conversion.

**Not yet implemented.** Reactive other-actor traffic and a physical trajectory controller. See
[TODO](../development/TODO.md#vehicle-dynamics).

### Mesh depth

With depth="mesh", SensorView.depth combines the USDZ's mesh.ply and mesh_ground.ply static
surfaces with the current dynamic actors' oriented bounding boxes. Rays originate at calibrated
pixel centers and use the actual pinhole or f-theta lens. The value is optical-axis z depth in
metres, rather than distance along the ray; pixels without a hit contain inf. Each pixel keeps
the nearest mesh or actor-box depth.
Mesh depth requires known extents for every dynamic actor. Missing extents raise ValueError naming
the actor ID; normal NuRec track records supply these dimensions.

Depth is computed only when sensor_view(camera) is requested and cached for that capture and
camera. Successive requests for the same capture reuse it. The scene mesh's accelerated BVH is
shared through a two-scene in-process LRU cache. A plain observation scan does not compute depth.
InsertionAttackAction automatically uses DepthVisibilityEstimator when this depth is available.

Static surfaces absent from the archived meshes do not occlude inserted content. Dynamic actors
are represented by boxes, not detailed reconstructed geometry, so silhouettes can differ. The
static mesh can also contain parked or slowly moving vehicles. Moving a recorded actor with
set_actor_state does not erase that vehicle's archived mesh geometry, so its original location can
continue to occlude. These mesh contents are not classified or removed automatically.

geometry.mesh.camera_depth casts calibrated camera rays through supplied MeshRaycaster instances
and combines their nearest optical-axis depths. The simulator and validation tool use this same
function. Archived native inverse lens polynomials are used unchanged; missing inverses are not
fitted. The validation script checks vertical ground/map residuals offline and reports camera-range
depth statistics separately. NRE is optional and used only when requesting RGB overlays; see
[Validation tools](../../scripts/README.md#validation-and-visualization).

## Recorded playback

RecordedReplaySimulator reads a NuScenesDataset ScenarioStart and advances through recorded
FrameData.sensor_references. Dataset sensor names stay native, such as CAM_FRONT; configure
RecordedReplaySimulator(camera_aliases={"CAM_FRONT": "front"}) for the driving system
name front. It is open-loop: commands cannot change ego or traffic motion. N decisions require N+1
frames. Sensor environments and actions can change model inputs but do not turn playback into
closed-loop simulation.

**Not yet implemented.** Static-scene occlusion for recorded replay, which supplies no mesh depth.
Actor cuboids remain available for supported insertion occlusion estimates. See
[TODO](../development/TODO.md#recorded-static-occlusion).

## Write a custom simulator adapter

Annotate filesystem constructor parameters as Path or str | Path; see
[filesystem parameters](../reference/configuration.md#constructor-filesystem-parameters).

Declare dataset_kinds, supported_world_settings, command_modes, decision_time_step,
supported_stages, can_render, can_insert_objects, can_replay, capture_ticks and identity on
Simulator.
Validate start compatibility
and world-setting values before creating native resources. Values depending on live actors are
checked during reset.

World ownership is defined by SimulatorRuntime.world_key and claim_world(owner). The default key
("simulator", id(self)) gives each runtime its own world. Override world_key for an external world
so every runtime connected to that same world returns the same hashable key, as CARLA does.
Claims are process-local context managers: the same owner object can nest them, different owners
are rejected immediately, and exiting releases the claim. Use exclusive() to keep one operation
active on a runtime; it can run inside claim_world(owner).

Implement native operations on SimulatorRuntime. The base class owns reset, step, start_replay and
advance_replay, and rejects subclass overrides of those public methods. For forward replay, declare
can_replay=True and implement only _replay_tick to advance the prescribed native motion. The shared
methods own capture lifecycle, stage timing, reset isolation and deployment windows. This complete
example runs offline:

<!-- runnable -->
```python
import numpy as np
from avsectester.closed_loop.simulators.base import Simulator, SimulatorRuntime
from avsectester.closed_loop.avstacks.base import AVStack
from avsectester.data_types import ActorState, Control, Observation, WorldSnapshot
from avsectester.execution.runner import ClosedLoopRunner, RunRequest

class LineWorld(Simulator):
    @property
    def decision_time_step(self):
        return 0.1

    @property
    def command_modes(self):
        return frozenset({"actuator"})

    def create_runtime(self, seed=None):
        return LineRuntime(self)

class LineRuntime(SimulatorRuntime):
    def _initialize_world(self, start, initialization, environment):
        self.position = 0.0
        self.speed = 0.0
        self.index = 0

    def _advance_native(self, command):
        self.speed = max(0.0, self.speed + command.throttle - command.brake)
        self.position += self.speed * self.decision_time_step
        self.index += 1

    def _ego(self):
        pose = np.eye(4)
        pose[0, 3] = self.position
        return ActorState(pose, (self.speed, 0, 0), "vehicle", (4, 2, 1.5))

    def _observe(self):
        return Observation(self.index * 0.1, self.index,
                           vehicle_state=self._ego(), ego_speed=self.speed)

    def _state(self):
        return WorldSnapshot(self.index * 0.1, self.index,
                             ego_speed=self.speed, ego=self._ego())

class Cruise(AVStack):
    def __call__(self, observation):
        return Control(throttle=0.2)

with LineWorld().create_runtime() as simulator_runtime:
    log = ClosedLoopRunner().run(simulator_runtime, Cruise(), RunRequest(3))
assert log.status == "completed"
assert len(log.steps) == 3
assert log.steps[-1].physical_post.ego_speed > 0
```

Declare sensors for every available sensor_view, and camera_sensors for its camera subset.
Non-camera views have camera=None and cam_from_world=None; positioned sensors provide a
sensor_from_world body-frame transform, distinct from the optical camera transform.
Set can_render=True for calibrated views and can_insert_objects=True for shared object composition.
A rendering simulator declares camera_sensors and implements _sensor_view(sensor), supplying
canonical camera projection, data, camera transform, actor/ego poses with optional cuboid extents
and measured depth. Override sensor_geometry(sensor) when geometry can be read more cheaply without
images or depth. The base class exposes sensor_view and adds registered objects to the capture;
implementations do not override that public method.

## Next steps

Read [Runtime](../reference/runtime.md), [Scenarios](scenarios.md) and [Python
API](../reference/python-api.md#simulator-api).
