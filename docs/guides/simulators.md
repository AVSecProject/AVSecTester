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

### Scenario start inputs

ScenarioStart carries the selected frame and dataset payload, together with two read-only inputs:

- recorded_ego contains the complete recorded ego trajectory on the dataset's decision timeline.
  Its order and length match Dataset.frames, and each state matches FrameData.ego. It is empty
  when no recorded trajectory exists, such as a CARLA recipe.
- scenario_map is the dataset's cached ScenarioMap or an Unavailable reason. The start shares this
  read-only map without copying it. NuRec supplies both inputs, nuScenes supplies the trajectory,
  and CARLA supplies its exported map when available.

These inputs let the simulator reconstruct the recorded past and navigation intent without
receiving attack-selection metadata. Maps are cached per scene rather than reparsed for each run.

### Read prescribed motion

start_replay(start, initialization=None, *, world=None, deployment=None) initializes at the selected
recorded or recipe boundary and captures it while keeping ego under prescribed motion. It is used
for raw sensor reads before driving-system control. advance_replay(count) then advances that motion
and captures the requested boundary without emitting stage hooks. Both methods reject bound hooks;
forward advancement also requires an initialized runtime and can_replay=True. The capability
defaults to False.

| Simulator | can_replay | can_observe_past |
|---|---|---|
| CARLA | True | False |
| Recorded playback | True | False |
| NuRec | False | True |

Closed-loop services reuse a forward scan only when can_replay is True. NuRec reads another frame
by calling start_replay for that frame rather than advancing the previous read. This preserves the
recorded pose for each raw sensor request.

## Inserted world objects

Simulator.can_insert_objects declares support for framework-composited ObjectAsset geometry.

| Simulator | can_insert_objects | Camera occlusion source |
|---|---|---|
| In-memory example | True | Available depth, otherwise actor cuboids |
| CARLA | True | Aligned depth for depth_camera; actor cuboids for other cameras |
| NuRec | True | Probes rendered in the reconstruction (default object_occlusion="rendered_probes"); with object_occlusion="geometry", static mesh and actor boxes with depth="mesh", actor cuboids with depth="none" |
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

A composed view records how it was composed. SensorView.composition holds the native capture beneath
the objects (`background`) and the CompositePlan that composed them; `plan.apply(background)`
reproduces the view's data and `plan.composite` is the same blend, differentiable in surface colours.
SensorView.object_occlusion states how objects are hidden: "geometry" (aligned depth or actor
cuboids, as above) or "rendered_probes" (the simulator renders opaque probe copies of the objects
inside its own scene, as NuRec does by default). A simulator adapter supplies such measured
visibility by overriding `_composite_plan(view, objects)` and returning plan_composite with
scene_visibility; ObjectVisibility.occlusion_source is then "scene_visibility".

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

NuRecSimulator takes `config`, a required Renderer template, an optional ego-dynamics template, a
depth mode and an object_occlusion mode. Configure NuRecRenderer explicitly; there is no default renderer. Renderer is defined
in closed_loop/simulators/nurec/renderer.py, and the dynamics interfaces and implementations live
in closed_loop/simulators/ego_dynamics.py.
Mesh depth and ego ground contact require the optional nurec dependencies at construction.
To run without mesh queries, set both depth="none" and physics_update_mode="none". Each simulator
runtime copies its renderer and dynamics, so connections, loaded scenes and mutations are isolated. Set
`config.dt` to NuRecDataset.time_step; this mismatch is rejected before simulation starts.

With depth="mesh", a reset from a recorded Dataset ScenarioStart checks that its USDZ contains both
mesh.ply and mesh_ground.ply before its first NRE request. Missing entries raise ValueError listing
their filenames. depth="none" disables this depth requirement; ego ground contact still requires
its selected ground_mesh file unless physics_update_mode="none".

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
from avsectester.closed_loop.simulators.ego_dynamics import AlpaSimController
from avsectester.closed_loop.simulators.nurec import NuRecRenderer, NuRecSimulator

simulator = NuRecSimulator(
    {"dt": 0.1},
    renderer=NuRecRenderer(endpoint="127.0.0.1:50051"),
    dynamics=AlpaSimController(),
    depth="none",
    physics_update_mode="none",
)
assert simulator.command_modes == frozenset({"trajectory"})
```

### Recorded past observations

NuRec declares `can_observe_past=True`. After reset, `observe_past(count)` returns the preceding
recorded decision observations from oldest to newest, with negative start-relative time and frame.
The complete ScenarioStart.recorded_ego trajectory supplies the actual poses; each camera is
rendered with recorded traffic at that time. The current physical state and capture are unchanged.
World objects present at the start, i.e. inserted during reset or simulator setup, existed before
it: each past camera capture composes them exactly as live captures do, with the recorded camera
pose, recorded actors and rendered probes or mesh depth. Objects inserted later, sensor attack actions and stage hooks
are excluded. A request fails when history is insufficient, the runtime has not reset, or the
simulator does not declare this capability.

A custom simulator can declare can_observe_past and implement `_observe_recorded(position)`;
to compose start objects it also implements `_recorded_sensor_geometry(position, sensor)` and
`_recorded_sensor_view(position, sensor, observation)`. The public observe_past lifecycle belongs
to the base class and cannot be overridden. CARLA and
recorded replay do not currently provide past observations.

### Navigation

NuRecSimulator uses `route_generator="map"` by default. ScenarioStart supplies the complete
recorded ego path and cached semantic map. Driving lanes within 4 m and a heading difference
of π/8 are candidates; the route follows consistent lane sequences and extends along successors
with the smallest heading change. A projection farther than 10 m is rejected.

Each capture projects the physical ego origin onto this route and provides 20 forward points
spanning 80 m as Observation.route in ego coordinates (X forward, Y left, Z zero). Short routes
repeat their last point. This navigation intent does not constrain the driving command.

Map mode requires a map with driving lanes at reset. Select `route_generator="recorded"` to use
the recording without extension, or `route_generator="none"` to provide no route.

### Ground contact

The runtime owns the physical three-dimensional ego pose, starting from the recording.
After each dynamics step, `physics_update_mode="ego_only"` fits the ego box bottom to the
scene mesh using AlpaSim's vertical ray and plane-fit procedure. `ground_mesh="mesh"` is the
official default; `"mesh_ground"` selects the ground-only reconstruction. BVHs are cached by
archive and mesh name. Select `physics_update_mode="none"` to retain the predicted pose.

Updates require at least six hits within 100 m, at most 1.5 m displacement and at most 10°
rotation. Rejected updates retain the prediction; termination events count failures by reason.
Two checks are stricter than AlpaSim's PhysicsBackend: it fits a plane from three hits and leaves
its `min_intersections = 6` unused, and it compares only the largest signed Euler angle of the
correction with 10°, whereas the rotation here is the full angle between the fitted normals.
The renderer receives the resulting three-dimensional rig pose rather than retaining an initial
height or tilt. The plane-centroid fit rotates about the box centre, as in AlpaSim, so it is
an approximate contact correction rather than a penetration solver.

### Camera resolution

NuRecRenderer.resolution_hw maps native camera IDs to requested (height, width). Its default
is native resolution. NRE preserves aspect ratio, so actual width may differ from the request;
the renderer probes each camera on load and scales its lens model using the returned height.
All later captures must have the same dimensions. Sensor geometry, mesh depth and inserted
objects use that scaled model. The supplied NuRec configurations request (1080, 1900) from
camera_front_wide_120fov, matching AlpaSim's single-camera preset.

The lens model follows NRE's rendering cameras: the f-theta polynomials, the vehicle-windshield
distortion stored with the camera, and NRE's pixel convention, which places the principal point
half a pixel from the spec value in the framework's pixel-centre convention. Probe objects
inserted into NRE scenes at 3 m and 8 m land on their projected pixels with a mean error below
0.01 px and a standard deviation below 0.1 px, at native and half resolution. Without the
windshield model, projections drift 2–9 px from the rendered image across the frame.

### AlpaSim defaults

The supplied NuRec configurations follow AlpaSim affc2ea and its single-camera Alpamayo 1.5 preset:

| Setting | AlpaSim source/default | Supplied NuRec configurations |
|---|---|---|
| Controller | base_config.yaml: default linear MPC | AlpaSimController with linear MPC |
| Navigation | route_generator_type: MAP | route_generator: map |
| Ground contact | physics_update_mode: EGO_ONLY; use_ground_mesh=False | ego_only; ground_mesh: mesh |
| Camera | 1cam_1080.yaml: camera_front_wide_120fov, requested 1080×1900 | Same camera and request; actual NRE size is probed |
| Camera timing | frame_interval_us: 100000; subsample_factor: 3 | dt: 0.1; frame_subsample: 3 |
| Reactive traffic | AlpaSim trafficsim | **Not yet implemented.** Other actors follow recorded tracks or explicit overrides |
| Rolling shutter | shutter_duration_us: 30000 in the camera preset | **Not yet implemented.** Captures render at one instant |

The controller and camera preset are explicit configuration choices. NuRecSimulator still accepts
other dynamics implementations, navigation modes and camera resolutions.

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

### Ego dynamics

The supplied NuRec configurations explicitly select AlpaSimController, matching AlpaSim's default
linear MPC and dynamic bicycle vehicle model. Its parameters are mpc_implementation="linear",
n_horizon=20, dt_mpc=0.1 and gains=None. An explicit gains mapping overrides named fields of
AlpaSim's MPCGains; unspecified gains retain ControllerConfig's defaults. Choosing another
implementation requires its registered AlpaSim MPC entry point.

The controller uses the official VehicleModel.Parameters defaults, taken from the Ford Fusion:
2,014.4 kg mass, 3,414.2 kg·m² yaw inertia and 2.85 m wheelbase. These physical parameters are not
read from the USDZ. The USDZ ego extent and box centre still determine ground contact and geometry.
AlpaSim packages load only on the first reset; constructing this configuration does not load a
controller, model weights or a rendering service. See the
[driver installation](../getting-started/installation.md#driver-environment) for its dependencies.

PlanarEgoState contains x, y and yaw in world coordinates; longitudinal speed and lateral_speed in
ego axes; t in seconds; yaw_rate in radians per second; and longitudinal acceleration in m/s².
At each runtime reset, the recorded ego velocity is transformed into ego axes, and neighbouring
recorded frames supply yaw_rate and acceleration. The runtime then calls dynamics.reset(state).
Checkpoint/restore retains these planar state fields alongside the physical three-dimensional pose.

AlpaSimController accepts trajectory commands led by the rig origin at t0 with identity rotation,
followed by future rig-frame poses. It advances through AlpaSim's controller and vehicle model,
using a fixed internal timestamp offset to keep protobuf times nonnegative. State timestamps must
match the controller's propagated time. Repeated reset initializes a fresh controller system.

KinematicBicycle consumes actuator commands. TrajectoryFollower consumes trajectory commands and
realizes their positions ideally; it remains available through explicit selection. Both accept
reset(state) without warmup and return finite-difference yaw_rate and acceleration. Choose command
modes compatible with the driving system; no automatic command conversion is performed.

A custom dynamics component implements step(state, command, dt) and can provide reset(state) and
command_modes. Configure it without editing the simulator:

```yaml
dynamics:
  type: package.module:CustomDynamics
  options: {}
```

This dynamics specification belongs in simulator.options. Each runtime receives an independent
copy and calls its reset method when provided.

**Not yet implemented.** Reactive other-actor traffic. See
[TODO](../development/TODO.md#reactive-traffic-in-nurec).

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

### Rendered-probe occlusion

By default (object_occlusion="rendered_probes"), inserted world objects are hidden by the
reconstruction as NRE renders it, because the same reconstruction produces the images. With
object_occlusion="geometry" they are hidden by mesh depth and actor boxes instead, as described
under [Mesh depth](#mesh-depth). Rendered probes need no mesh.ply, so they also apply to
releases that ship only mesh_ground.ply (26.02 and 26.04), and it handles reconstructed vehicles,
poles and vegetation alike. For each camera whose image contains objects, the renderer inserts
each object's opaque Gaussian copy (probe) into the loaded scene, renders the capture once with
white probes and once with black ones, and treats pixels whose white render exceeds the black one
by half the colour range as visible. Object colours still come from the framework's composite, so
optimized textures remain differentiable; use SensorView.composition, because
SensorView.composite_plan cannot measure this evidence on a clean view and raises. The setting
applies to world objects; sensor insertion (InsertionAttackAction) keeps its own visibility
estimators. See [TODO](../development/TODO.md#nurec-rendered-probe-conventions) for the
measured NRE conventions this relies on.

NuRecRenderer writes probe PLY files to `probe_directory`; the NRE server must read that directory
at `server_probe_directory` (the same path by default), and must run with --enable-editing-actors.
A renderer without `probe_directory` raises at the first capture that composes a world object.
Track IDs derive from probe content, so runtimes and processes rendering the same object share the
server's probes. Probes are inserted when NRE reports them missing: on first use, after NRE reloads
an evicted scene, and after any failed edit of the scene, which drops all of its edits. Probes stay
in the server's memory until it reloads the scene; the framework never restores scene parameters,
because that would remove other clients' probes. Their trajectories lie 10 km below the scene,
because a render request that lists no dynamic objects renders every track along its trajectory.

Each composed camera adds two renders per capture, about 0.03 s each at 1080 x 1920. They use
maximum-quality JPEG: server-side PNG encoding takes about 0.5 s per image, and the measured
visibility differs from PNG in only a few edge pixels. NRE documents neither how a PLY's axes map
to the inserted track nor how `object_size` scales it; the framework uses the convention measured on
nre-ga 26.04, with probes landing within 0.15 px of the projected surfaces, and `track_padding_m`
is the server's track padding. Probe pixels outside the objects' projected boxes raise ValueError,
which flags a server that uses a different convention.

The evidence is the reconstruction's own geometry, with these consequences:

- Reconstructed surfaces have thickness. A panel placed within a few centimetres of a reconstructed
  sign is partly or wholly hidden by it; on three stop-sign scenes at 12–40 m, offsets of 10–20 cm
  kept 83–99% of a replica face visible, while 2 cm kept 6–33% in two of them.
- Mesh depth is not a substitute near reconstructed surfaces: in one of those scenes it hid a
  replica 10–20 cm in front of the sign completely at 12 m, where NRE renders 83–98% of it visible.
- The base of an object standing on the road can be hidden by the reconstructed road surface.
- Probe silhouettes differ from the composite's by about one pixel at edges.

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
