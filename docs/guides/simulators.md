# Simulators

> **Audience:** Experiment authors and simulator implementers.
>
> **Outcome:** Choose simulator behavior and implement a custom simulator.
>
> **Prerequisites:** [Architecture](../concepts/architecture.md).

## `Simulator` and simulator runtime

A `Simulator` stores configuration and capability declarations. Each `create_runtime(seed)`
creates a `SimulatorRuntime` whose creator closes its world and connections, including after
failures. Import interfaces from `avsectester.closed_loop.simulators` and concrete
implementations from `simulators.carla`, `simulators.nurec` or `simulators.recorded`.

`reset` initializes a selected `ScenarioStart` and captures its first observation. `step`
advances one decision and captures again. `observe` copies the current capture without
rendering, advancing time or invoking hooks. `state` copies physical ego, actors and
inserted objects.

Only world settings reach reset. Sensor corruptions apply after capture. See
[Environments](environments.md) for environment composition and
[Python API](../reference/python-api.md) for signatures and errors.

### Scenario start inputs

The dataset supplies [ScenarioStart](datasets.md#dataset-interface). Attack selection metadata
is not a simulator input.

### Read prescribed motion

`start_replay` initializes and captures a recorded or recipe boundary with prescribed ego
motion. `advance_replay` advances that motion and captures without stage hooks. Replay rejects
bound hooks. Advancement also requires an initialized runtime and replay capability.

| `Simulator` | Prescribed-motion advancement | Past observations |
|---|---|---|
| CARLA | Supported | Unsupported |
| Recorded playback | Supported | Unsupported |
| NuRec | Unsupported | Supported |

Services reuse forward scans where advancement is supported. NuRec instead resets prescribed
motion at each requested frame.

## Inserted world objects

`insert_object` registers an `ObjectAsset` under a unique ID and rigid world pose.
`remove_object` removes it. The physical `state().objects` changes immediately, while images
change at the next capture. Inserted objects stay separate from native actors, have zero
velocity and do not participate in native physics or collisions.

Composite insertion projects each asset into captured camera images, resolves nearest surfaces
and applies visibility. NuRec also supports scene insertion, where NRE renders assets with the
reconstruction into camera images and LiDAR. Set `object_insertion` to `composite` (the default)
or `scene` on `NuRecSimulator`. The insertion actions and `harmonizer` interface are the same.

| Simulator | Insertion | Composite occlusion |
|---|---|---|
| CARLA | Composite | Aligned scene depth for `depth_camera`, actor cuboids for other cameras |
| NuRec | Composite or scene | Reconstruction visibility from rendered probes, or mesh/actor geometry |
| Recorded playback | Composite | Actor cuboids only |

Both modes describe each object's visible pixels in the captured image.
`SensorView.object_masks` maps the same IDs as `SensorView.objects` to boolean H×W masks.
Composite insertion uses the pixels that the composition actually assigns to each object.
Scene insertion measures coverage with white and black asset copies and keeps coverage of at
least 0.5. Each `ObjectVisibility.visible_pixels` equals the corresponding mask's pixel count.
A projected but fully hidden object has an empty mask and a camera entry with zero visible
pixels. An object outside a camera projection has no entry. Captures retain their own object
membership until the next capture.

`SensorView.composition` exists only for composite insertion. It records the unmodified
background and its `CompositePlan`, whose `plan.apply(background).image` reproduces the capture.
Scene insertion has `composition=None`. An attack can build its own differentiable plan using
the captured camera geometry and `object_masks`. See [Optimization](optimization.md) and the
[compositing API](../reference/python-api.md#avsectesterrenderingcompositing).

Composite insertion combines native depth with object surface depth. Rendered visibility and
scene insertion replace depth only at visible object pixels and preserve the rest. A view
without native depth keeps `depth=None`.

Occlusion tolerance applies only to composite insertion. It keeps scene surfaces within the
configured tolerance in front of an object from hiding it, for both geometry and reconstruction
probes. See [Configuration](../reference/configuration.md#simulator) for defaults.

A single-sided asset can show its front texture from behind. Add a rear face when the object
must have a distinct back. Sensor insertion through `InsertionAttackAction` remains independent
of world insertion.

See [Writing attacks](attacks.md#insert-a-world-object) for insertion actions and
[Runtime records](../reference/runtime.md#runlog) for saved evidence.

**Not yet implemented.** CARLA scene insertion and non-camera insertion in composite mode.
See [TODO](../development/TODO.md#inserted-objects-in-non-camera-sensors).

## CARLA

A `CarlaDataset` supplies recipes describing map, spawns, sensors and prescribed motion.
`CarlaSimulator` supplies the connection and synchronous timing settings. Each reset requires a
recipe-bearing `ScenarioStart` and uses a dedicated server.

Traffic-light states are recorded under OpenDRIVE control IDs.

### Selected-origin initialization

The runtime creates actors and sensors, replays prescribed motion to the deployment window,
applies setup actions and runs the deployment tick budget before capturing the selected
boundary. At position zero, actors are held during setup ticks. Before decision zero, ego is
released from recipe motion for driving-system control. See the
[deployment tick budget](attacks.md#online-phase).

A positive selected boundary needs materialized frame data. Recorded initialization evidence
preserves the selected actor identities and sensor timing. See [Experiments](experiments.md) for
preparation and [Runtime](../reference/runtime.md) for decision order.

Setting `depth_camera` selects an RGB camera for aligned depth and an undistorted pinhole lens.
Other sensors keep their configured capture types. CARLA LiDAR retains native byte axes and its
wrapper's `flipy` setting. See [LiDAR records](../reference/runtime.md#carla-lidar-data).

## NuRec

NuRec requires a `ScenarioStart` from its dataset. The runtime initializes the recorded pose,
timeline and actors, then advances ego with the selected dynamics. Camera names remain native
unless `camera_aliases` maps them to observation names. Aliases also apply to views
and calibration.

NuRec provides no traffic-light states.

Constructing mesh-depth or ground-contact support requires the `nurec` extra.
Reset checks required archive meshes before connecting to NRE. Missing resources fail preflight.
A manually started NRE must enable `--enable-editing-actors`.

The decision step comes from `dataset.time_step` through configuration loading. A driving system
can require a specific step. See [Configuration](../reference/configuration.md#simulator) for
the injection and consistency rules.

NRE is required for camera rendering. See
[automatic local services](../reference/cli.md#automatic-local-services) for CLI and Python use.

Construction does not connect to NRE:

<!-- runnable -->
```python
from avsectester.closed_loop.simulators.ego_dynamics import AlpaSimController
from avsectester.closed_loop.simulators.nurec import NuRecRenderer, NuRecSimulator

simulator = NuRecSimulator(
    NuRecRenderer(endpoint="127.0.0.1:50051"),
    AlpaSimController(),
    time_step=0.1,
    depth="none",
    physics_update_mode="none",
)
assert simulator.command_modes == frozenset({"trajectory"})
```

### Scene insertion

Scene insertion requires `NuRecRenderer.asset_directory`. The renderer writes generated asset
files there for NRE to read. See
[Configuration](../reference/configuration.md#nurec-closed-loop-settings) for directory and mount
settings.

Assets use 5 mm grid spacing, capped at one Gaussian per texture pixel. Grid centers with alpha
below 128 are omitted. Colors are alpha-weighted area averages. Gaussian
lateral size equals grid spacing, thickness is 1 mm and opacity logit is 8, with zeroth-order
spherical harmonics only.

NRE registers one track for each object and texture. Moving an object reuses its track. Tracks
remain until the NRE process restarts.

For each projected object, visibility requires two additional camera renders with white and
black copies. A harmonizer also needs a background render. A differentiable harmonizer also
needs a nominal-texture render when its object has nominal textures. These extra
renders use the capture's request pose and bypass hooks. They add rendering work in proportion
to the projected objects and cameras. A harmonizer also adjusts partially covered pixels at
object edges, outside `object_masks`. With scene insertion, `render.post` receives images and
point clouds that already contain the inserted objects. With composite insertion, the hook
receives the native capture before camera composition.

Reconstructed objects remain in the scene and can hide an inserted object placed very close
to them.

**Not yet implemented.** Removing real objects from reconstructed scenes. See
[TODO](../development/TODO.md#remove-real-objects-from-reconstructed-scenes).

Scene-inserted assets appear in LiDAR for 26.04 scenes. In 26.01 scenes, LiDAR can pass through
inserted assets. This is a limitation of that scene release.

### LiDAR

Set `NuRecRenderer.lidar` to a scene's logical sensor name, such as `lidar_gt_top_p128`, to
collect LiDAR alongside the cameras. The default `None` disables LiDAR. The selected scene must
contain exactly one LiDAR with a `row-offset-spinning` model. The sensor-to-rig transform comes
from calibration. The scan period is the median duration of recorded LiDAR frames. See
[LiDAR records](../reference/runtime.md#lidar-data) for the payload format.

The decision interval must match the scan period within 1% of the decision interval. This is
checked after scene loading and before the first render. Each scan covers the previous decision
timestamp through the current decision timestamp. Its start pose is the previous decision pose.
For decision zero, that pose comes from the recorded trajectory at the scan's start time.
Recorded past observations also include LiDAR, using recorded ego and actor poses at both scan
endpoints.

With LiDAR enabled, one additional recorded step is needed before the earliest observation.
Segment selection therefore reserves at least one more step than the driving-system history
alone requires. A past-observation request whose scans begin before the recording raises
`ValueError`.

Cameras precede LiDAR in `Observation.sensor_data`. Current LiDAR captures pass through
`render.post` with the LiDAR sensor name. They do not pass through `render.pre`, whose
`CameraPoseRequest` is specific to cameras. A LiDAR `SensorView` has no camera geometry and uses
the scan-end sensor transform. Supplied driving systems continue to read their configured
cameras.

Reconstructed-scene defects may produce a small number of below-ground points in a scan.

Supplied scenes provide no intensity. Intensity and no-return-ray simulation are
**Not yet implemented.** See
[NuRec LiDAR simulation](../development/TODO.md#nurec-lidar-simulation).
Inserted objects appear in LiDAR with [scene insertion](#scene-insertion). Composite insertion
changes camera images only.

**Not yet implemented.** Offline renders of a dataset's first frame while LiDAR is enabled. See
[TODO](../development/TODO.md#nurec-offline-lidar-sampling).

### Recorded past observations

After reset, `observe_past(count)` returns actual recorded observations before the selected
start, oldest first, with negative run-relative timestamps and frames. It rejects insufficient
history and preserves the current world, clock and capture. Start eligibility uses the larger
of the simulator's own history and the `DrivingAdapter` history plus the simulator's observation
lead steps.

Past captures use recorded ego poses and actors, including the same short-track filtering as
current captures. Objects installed during reset or setup use those recorded poses with the
configured insertion mode. Later insertions and actor overrides are excluded. Past captures
bypass stage hooks and sensor attack actions. The runner applies sensor corruptions at each
absolute dataset position, then passes history directly to `AVStack.reset`. It bypasses
localization and the observation pipeline. See
[initialization records](../reference/runtime.md#initialization) and
[environment randomness](environments.md#randomness).

### Navigation

Map navigation matches the recorded ego path to driving lanes and extends the route through
successors. Every capture projects the current physical ego origin onto that route and provides
forward points in ego coordinates. Navigation expresses intent. It does not constrain the
produced or executed command.

Recorded-route mode uses the recording without extension. Disabled navigation supplies no route.
A map route needs driving lanes. Folded or inconsistent routes are rejected by `validate_start`
so selection can record the reason. With loaded metadata, an invalid segment fails preparation.
Algorithm parameters and constraints are in the [navigation API](../reference/python-api.md).

### Ground contact

The runtime retains a three-dimensional ego pose. Ground contact fits the predicted box bottom
to the selected scene mesh after each dynamics step. Rejected corrections retain the prediction.
Termination events count failures by reason. Disabling physical updates retains the
dynamics prediction.

The full scene mesh may include roadside objects. A ground-only mesh avoids using those surfaces
for ego contact where that asset is available. Ground contact is an approximate fit rather than
a penetration solver. It does not correct traffic actors.

### Camera resolution

Cameras render at their native resolution. A returned image whose dimensions differ from the
camera's native model raises `ValueError`.

Captures render a single instant.

The lens includes its f-theta model and recorded windshield distortion. Camera projection,
mesh depth and object placement share the resulting geometry. Camera coordinate conventions
and models are specified in the [Python API](../reference/python-api.md).

### Move recorded actors

Only controllable tracks can be moved by NRE. Parked tracks stay at recorded poses and
`set_actor_state` rejects them. Short controllable tracks are filtered by `min_track_duration_s`.
Zero disables filtering.

Filtered controllable tracks are absent from world snapshots, sensor actors, depth and metrics.
All captures and probes hide their recorded render instances below the scene. Parked tracks are
never duration-filtered.

An accepted override changes world state immediately and camera state at the next capture. Its
explicit world velocity advances the actor at fixed orientation until reset or another override.
`control_actor` is unsupported. Without traffic simulation, checkpoints preserve actor
overrides. Restoring a checkpoint creates a new capture of the restored state.

**Not yet implemented.** Checkpoint restoration with active render hooks. See
[TODO](../development/TODO.md#nurec-checkpoint-restoration-with-active-render-hooks).

### Reactive traffic

Enable `AlpaSimTraffic` through `simulator.options.traffic`. YAML, timing and history follow
[Reactive traffic configuration](../reference/configuration.md#reactive-traffic).

World actors begin with unfiltered recorded states, then use returned controllable-session
poses, then explicit overrides. Parked tracks stay at recorded poses within recorded support.
Sessions include duration-qualified tracks with nonempty clipped history. A first override
backfills constant-velocity history and subsequent requests resend its complete history.

Trajectories are not smoothed. Offline candidate rollouts also require the service.

**Not yet implemented.** Fitting traffic actors to the selected ground mesh, checkpointing
traffic sessions and controlling CATK sampling through the run seed. See
[Reactive traffic](../development/TODO.md#reactive-traffic).

### Ego dynamics

| Dynamics | Commands | Behavior |
|---|---|---|
| `KinematicBicycle` | Actuator | Integrates a kinematic bicycle |
| `TrajectoryFollower` | Trajectory | Follows planned positions ideally |
| `AlpaSimController` | Trajectory | Tracks a trajectory with AlpaSim's controller and vehicle model |

NuRec initializes longitudinal/lateral speed, yaw rate and acceleration from the recorded
trajectory before resetting dynamics. The three-dimensional ego geometry still comes from
the dataset.

`AlpaSimController` uses AlpaSim Ford Fusion vehicle parameters.

Trajectory commands use rig-frame poses and increasing timestamps, beginning at the current
origin. The controller tracks the full forecast. It need not execute its first planned
acceleration exactly. There is no automatic command conversion.

A custom dynamics component supplies `step` and may supply `reset` and `command_modes`. Each
runtime receives its own copy. Constructor defaults and trajectory constraints are listed in the
[dynamics API](../reference/python-api.md).

### Mesh depth

Mesh depth combines archived static surfaces and current actor cuboids. Calibrated camera rays
keep the nearest hit, expressed as optical-axis depth. Missing hits are infinite. Actor extents
must be known. Depth is computed lazily per camera and cached for the capture.

Mesh content is not classified or erased when an actor moves. Archived vehicle geometry can
therefore continue to occlude its recorded location. Cuboid silhouettes can differ from rendered
vehicles. Rendered probes instead measure the reconstruction's visibility directly.

### Rendered-probe occlusion

Opaque white and black probe copies measure whether scene content hides inserted surfaces. Only
sufficiently opaque asset pixels contribute probes. The renderer requires `asset_directory`
to be visible inside the NRE service. Path and mount rules are
in [Configuration](../reference/configuration.md#nurec-closed-loop-settings).

A captured `composition.plan` already contains this measured visibility. An arbitrary
replacement object list needs a fresh render request. `SensorView.composite_plan` cannot obtain
new reconstruction probes from a detached view. Geometry mode can build plans from its supplied
depth and actors.

## Recorded playback

`RecordedReplaySimulator` captures supported dataset frames while ego and other actors follow
their recorded motion. Commands do not change that trajectory. It supports camera aliases and
shared world-object composition. It provides no traffic-light states.
Environment support follows [Configuration](../reference/configuration.md#environment).

## Write a custom simulator

See the [simulator base module](../reference/python-api.md#avsectesterclosed_loopsimulatorsbase)
for its Extension contract.
[Architecture](../concepts/architecture.md#world-ownership) describes the world-claim and
operation-isolation rules.

Subclass `Simulator` for configuration and `SimulatorRuntime` for native world operations.
Declare dataset kinds, timing, command modes, stages, world settings, capture capabilities and
identity.

Declare the simulator's own recorded-history requirement in `past_steps`. If a sensor needs
recorded data before the earliest observation, declare it in `observation_lead_steps`, which
adds to the driving system's history.

Declare every view-capable sensor in `sensors` and cameras in `camera_sensors`. Native views
supply capture data and sensor geometry. The shared runtime owns object-capture integration. A
lightweight `sensor_geometry` implementation avoids image/depth work for projection-only
requests. Past-capable implementations supply recorded observations and recorded camera views
for start-object composition.

## Next steps

-  Configure timing and rendering in [Configuration](../reference/configuration.md#simulator).
-  Bind a [driving system](driving-systems.md).
-  Use [world insertion actions](attacks.md#insert-a-world-object).
