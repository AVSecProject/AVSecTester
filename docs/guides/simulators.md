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

`reset` initializes a selected `ScenarioStart` and captures its first observation.
`advance_interval` propagates one decision interval and reports intermediate physical states.
`capture_at` publishes the interpolated physical state and acquires selected sensors at an integer
frame-end timestamp. `step` is a convenience for one interval and its endpoint capture. The runner
always uses the separate operations. `observe` copies the latest acquisition batch without
advancing or invoking hooks. `sensor_view` reads a sensor's latest captured viewpoint.
`frame_range_us(sensor, end_time_us)` returns its half-open shutter or native scan interval.
`state` copies physical ego, actors and inserted objects. See
[Runtime](../reference/runtime.md#one-decision) for event order and clock meanings.

Only world settings reach reset. Sensor corruptions apply after capture. See
[Environments](environments.md) for environment composition and
[Python API](../reference/python-api.md) for signatures and errors.

### Scenario start inputs

The dataset supplies [ScenarioStart](datasets.md#dataset-interface). Attack selection metadata
is not a simulator input.

### Read prescribed motion

`start_replay` initializes and captures a recorded or recipe boundary with prescribed ego motion.
`advance_replay` advances that motion and captures without stage hooks. Replay rejects bound hooks.
Advancement also requires an initialized runtime and replay capability. Publish any pending decision
interval endpoint before advancing replay.

| `Simulator` | Prescribed-motion advancement | Past observations |
|---|---|---|
| CARLA | Supported | Supported |
| Recorded playback | Supported | Unsupported |
| NuRec | Unsupported | Supported |

Services reuse forward scans where advancement is supported. NuRec instead resets prescribed
motion at each requested frame.

## Inserted world objects

`insert_object` registers an `ObjectAsset` or `MeshAsset` under a unique ID and rigid world pose.
`remove_object` removes it. The physical `state().objects` changes immediately, while images
change at the next capture. Inserted objects stay separate from native actors, have zero
velocity when stationary and are excluded from native physics, collisions and traffic models.
Built-in metrics count overlap with them as a collision. See
[Built-in metrics](evaluation.md#built-in-metrics) and
[Moving inserted objects](../reference/runtime.md#moving-inserted-objects).

`ObjectAsset` contains textured planes. `MeshAsset` contains colored triangle meshes and requires
NuRec scene insertion. Both use forward-left-up axes and a bottom-center bounding box.
Only base colors are rendered, without material, reflection or lighting simulation.
Alpha values of at least 128 are opaque. Mesh triangles used to occlude sensor-inserted content
are treated as opaque regardless of their color alpha.

**Not yet implemented.** Mesh composite insertion, including white-box texture optimization.
See [TODO](../development/TODO.md#mesh-composite-insertion).

**Not yet implemented.** CARLA mesh insertion. See
[TODO](../development/TODO.md#carla-mesh-insertion).

Composite insertion projects each asset into captured camera images, resolves nearest surfaces
and applies visibility. NuRec also supports scene insertion, where NRE renders assets with the
reconstruction into camera images and LiDAR. Set `object_insertion` to `composite` (the default)
or `scene` on `NuRecSimulator`. The insertion actions and `harmonizer` interface are the same.
NuRec scene insertion uses the pose at each end of the shutter or scan window. Composite
insertion uses the pose at the capture timestamp.

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

A single-sided plane or mesh shows its front colors from behind. Add a rear face when the object
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
Framework actor-motion and physics changes are synchronized before each native tick.

CARLA's post-process profile determines RGB exposure, tone mapping and sensor noise. The default
profile adds noise to each frame, so detections near a threshold can produce different commands
in repeated runs. See [Simulator configuration](../reference/configuration.md#simulator).
Before holding actors, `ground_settle_s` lets vehicles without prescribed trajectories settle
with physics and gravity enabled and the handbrake applied. Their initial states retain the
complete settled pose with zero velocity. Holding preserves that pose with gravity enabled.
Before the scenario time origin, `camera_warmup_s` adds synchronous hold ticks with actors at
their initial poses. These ticks do not consume the deployment tick budget and are skipped
when the recipe has no RGB camera. Pre-origin settling and hold ticks advance traffic-light
timers by their configured durations for every run. See
[CARLA runtime configuration](../reference/configuration.md#simulator).

`Town10HD_Opt` has no map layers. A nonempty `map_layers` setting on it raises `ValueError`.
The native adapter's ParkedVehicles unload request has no effect on this map, so its parked
roadside vehicles remain in the scene. Layer settings are supported on `Town03_Opt`.

Native sensors use `sensor_tick=0` and produce packets on every physics tick. The scheduler retains
only planned acquisitions, discarding packets from other ticks. A recipe sensor with a nonzero
`sensor_tick` is rejected. Set acquisition rates with `runs.timing.sensors`. A spinning LiDAR packet
covers one native physics interval. Reducing its acquisition frequency does not concatenate
packets into a longer scan. Observations list sensors in the recipe's order, independently of
packet arrival.

Runtime methods accept a sensor's logical name or its native source ID.
Unknown names and off-tick requests raise `ValueError` before changing state. Timing support is
listed in [Scheduling](../reference/configuration.md#scheduling). See
[Simulator configuration](../reference/configuration.md#simulator) for `client.rate` constraints
and [Runtime](../reference/runtime.md#one-decision) for timestamps.

### Selected-origin initialization

The runtime creates actors and sensors, replays prescribed motion to the deployment window, applies
setup actions and runs the deployment tick budget before capturing the selected boundary. At
position zero, actors are held during setup ticks. Ego leaves recipe motion before decision zero
with default timing. With `force_gt_duration_us`, it leaves at the handoff decision.
See the [deployment tick budget](attacks.md#online-phase).

A positive selected boundary needs materialized frame data. Recorded initialization evidence
preserves the selected actor identities and sensor timing. See [Experiments](experiments.md) for
preparation and [Runtime](../reference/runtime.md) for decision order.

Setting `depth_camera` selects an RGB camera for aligned depth and an undistorted pinhole lens.
Other sensors keep their configured capture types. CARLA LiDAR retains native byte axes and its
wrapper's `flipy` setting. See [LiDAR records](../reference/runtime.md#carla-lidar-data).

### Native ROS 2 control

With `ego_control: ros2`, CARLA publishes sensor data directly to an external ROS 2 driving
system and accepts its control commands. The framework holds ego through initialization and
releases it for decision zero. External subscribers receive native tick-rate sensor packets
independently of the framework's recording clocks. See
[Runtime](../reference/runtime.md#one-decision) for command records and telemetry.

Native ROS 2 runs start at the scene origin. Selection skips later starts and records the
reason. When a run has a prepared initialization, its actual ego pose and velocity must
match the prepared values. Starting after the origin is **Not yet implemented.** See
[TODO](../development/TODO.md#native-ros-2-runs-after-the-origin).

Enable external control output only for execution. Preparation and rendering can reset the
world repeatedly. CARLA continues publishing ROS messages, so subscribers see clock jumps
and new ego actors. A control mismatch during replay raises an error if an external client
overrides the framework's command.

`realtime_factor` paces closed-loop ticks from the first hold tick onward. Preparation and
rendering run without pacing. For a paced run, the initialization and termination events
record the achieved speed ratio, overdue ticks and maximum delay. See
[Simulator configuration](../reference/configuration.md#simulator) for pacing and hold options.

During a run, only world actions and simulator world hooks can affect the external system.
The runner rejects sensor environments and hooks outside simulator world stages before the
first tick. Hook behavior cannot be classified as read-only, so this check is conservative.
Non-world attack actions are rejected before application because observation changes only
affect the framework's copy of the server-published data.
Framework-composited `InsertObject` assets are unavailable during a native run. Native physical
patch panels are supported. Offline selection and rendering can use sensor environments,
sensor actions and framework-composited objects on their returned observations.
Sensor attacks need a [ROS 2 relay](../development/TODO.md#native-ros-2-sensor-attacks).
Background traffic is managed on the external-system side. The framework is the only tick
initiator. Other clients must not tick the world. See
[ROS 2 server setup](../getting-started/installation.md#native-ros-2).

### Managed external driving systems

In closed-loop runs, CARLA reloads the world and spawns the recipe's ROS rig before starting a
new external driving session. Preparation and rendering spawn the same rig without starting
the session. CARLA holds ego and keeps ticking until the session reports readiness. It resets
traffic lights before the origin capture in preparation and execution. The origin check compares
each light's state and elapsed phase time with the prepared reference.

CARLA requests autonomous mode when it releases ego at decision zero. It checks the session
after each subsequent tick and stops it before destroying the sensor rig. See
[runtime records](../reference/runtime.md#one-decision) for telemetry and event data.

The [Town10HD example](../../configs/carla_autoware.yaml) hosts Autoware with native ROS 2.
It uses the official sensor rig, camera profile, launch and CARLA overrides. Each run gets a
fresh container and a new traffic-light source. Readiness waits for ADAPI, localization, pose
agreement, clean distortion diagnostics, route acceptance and autonomous availability.
Localization requests, pose checks and goal requests can retry. A goal is published again
when no route follows it. Autonomous engagement uses one request once the session is ready.
Missing override files fail startup. The framework sets the traffic-light phase at the origin
and keeps responsibility for pacing, actor control during setup and the run's decision limit.
See [installation](../getting-started/installation.md#managed-autoware) and
[configuration](../reference/configuration.md#managed-carla-driving-systems).

## NuRec

NuRec requires a `ScenarioStart` from its dataset. The runtime initializes the recorded pose,
timeline and actors, then advances ego with the selected dynamics. Camera names remain native
unless `camera_aliases` maps them to observation names. Aliases also apply to views
and calibration.

Camera frames end on their independent `frame_interval_us` clocks. A positive
`shutter_duration_us` renders `[frame_end_us - shutter_duration_us, frame_end_us)` using
interpolated ego and actor poses at both endpoints. Instantaneous frames use a one-microsecond
render request. `pose_reporting_interval_us` controls the physical trajectory used for
interpolation and is independent of camera frequency. Objects, coverage copies and
reconstruction-visibility probes use the same frame interval and endpoint poses.

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

Plane assets use 5 mm grid spacing, capped at one Gaussian per texture pixel. Grid centers
with alpha below 128 are omitted. Colors are alpha-weighted area averages. Gaussian
lateral sizes are twice the grid-cell dimensions. The textured asset and its white and black
coverage copies share their Gaussian geometry. Thickness is one millimeter and opacity logit
is eight, with zeroth-order spherical harmonics only.

Mesh triangles use the same grid spacing, and every nonzero-area triangle has at least one
sample. Samples with alpha below 128 are omitted. The others become Gaussians that lie flat on
their triangles and take their base colors. Their lateral sizes are twice the requested grid
spacing. Textured colors are alpha-weighted averages over each Gaussian's texture footprint.
Mesh assets have no nominal textures. See
[mesh loading settings](../reference/configuration.md#mesh-assets) for file formats and axes.

Composite visibility probes use a one-centimeter nearest-count grid, capped at one Gaussian
per texture pixel. Their lateral sizes are nine tenths of the grid-cell dimensions. White and
black probes share their Gaussian geometry. Because scene-asset Gaussians are wider than their
grid cells, image edges are slightly soft and coverage extends slightly beyond the exact asset
silhouette.

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

LiDAR returns from scene-inserted assets can lie behind the asset surface. The offset is larger
for surfaces near the ground and varies with the asset's position. NRE blends the Gaussians
along each LiDAR ray in an approximate order, so large road Gaussians can add range from the
road behind the asset. This is a limitation of NRE's LiDAR rendering and applies to plane and
mesh assets.

### LiDAR

Set `NuRecRenderer.lidar` to a scene's logical sensor name, such as `lidar_gt_top_p128`, to collect
LiDAR alongside the cameras. The default `None` disables LiDAR. The selected scene must contain
exactly one LiDAR with a `row-offset-spinning` model. The sensor-to-rig transform comes from
calibration. The calibrated spin period is the median duration of recorded LiDAR frames. See
[LiDAR records](../reference/runtime.md#lidar-data) for the payload format.

Each LiDAR scan covers `[frame_end_us - frame_interval_us, frame_end_us)`. Adjacent scans meet
without gaps or overlap. The calibrated spin period is used only to validate `frame_interval_us` and
must differ from it by at most 1%. It does not set the scan window. LiDAR does not accept
`shutter_duration_us`.

Scan endpoints use their physical ego poses and actor states. The runtime retains motion history for
interpolation. Pre-origin endpoints come from the recording. Insufficient source support raises
`ValueError` when the requested scan cannot be formed. Selection reserves enough preceding
dataset steps to cover the configured LiDAR `frame_interval_us`.

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

After reset, `observe_recorded_at(time_us, sensor_ids)` reads a negative integer offset from the
run origin. Pass `()` to read only the recorded pose without loading or rendering sensors.
NuRec interpolates recorded ego and actor poses between dataset grid points. CARLA reads the
requested captures and poses from its prescribed pre-roll. Reads return independent values
without changing the live world, clock or capture, and reject insufficient source data.
See the [recorded acquisition contract](../reference/python-api.md#avsectesterclosed_loopsimulatorsbase)
for returned fields.

The [recorded-input requirement](driving-systems.md#recorded-input-history) controls selection
and reads, including simulator observation lead time.

Past captures use recorded ego poses and actors, with the same short-track filtering as current
captures. Objects installed during reset or setup use recorded poses and the configured
insertion mode. Later insertions and actor overrides are excluded. Recorded context processing
follows [driving history](driving-systems.md#recorded-input-history).

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

### Recorded motion

With `force_gt_duration_us`, the controller follows a recorded reference trajectory. Dynamics
align to the recording during recorded motion and at handoff. With physics enabled, reference
poses blend from recorded poses toward ground-corrected poses over that duration. Zero-weight
poses and runs without physics use recorded poses directly.

Frame and scan endpoints interpolate the ego's control/report poses and the actors' control
poses. See [recorded-motion execution](../reference/runtime.md#planner-delay-and-recorded-motion)
for command and decision semantics.

### Camera resolution

Cameras render at their native resolution. A returned image whose dimensions differ from the
camera's native model raises `ValueError`.

Camera exposure uses `shutter_duration_us`. Zero selects an instantaneous frame.

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

Traffic hands over at the run origin plus `force_gt_duration_us` plus the decision period.
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

Trajectory commands use rig-frame poses and increasing absolute timestamps. `planner_delay_us`
selects an earlier command without changing its waypoints or timestamps. The controller anchors the
original rig-frame reference using its physical pose history. `AlpaSimController` performs this
conversion inside the native controller. `TrajectoryFollower` uses its own bounded pose history.
The controller tracks the full forecast and need not execute its first planned acceleration exactly.

A custom dynamics component supplies `step`, with optional initialization, reports and
recorded-motion alignment described in the
[dynamics API](../reference/python-api.md#avsectesterclosed_loopsimulatorsego_dynamics).
Each runtime receives its own dynamics copy. Intermediate reports follow
`pose_reporting_interval_us` independently of sensor clocks, and changing that interval can
change ego motion.

### Mesh depth

Mesh depth combines archived static surfaces and current actor cuboids. Calibrated camera rays keep
the nearest hit, expressed as optical-axis depth. Missing hits are infinite. Actor extents must be
known. Depth is computed on demand for the captured viewpoint.

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
their recorded motion. Commands do not change that trajectory. nuScenes uses a nominal scheduler
period of 500000 microseconds. Timestamp representation follows
[Runtime](../reference/runtime.md#one-decision), and timing support is listed in
[Scheduling](../reference/configuration.md#scheduling). It supports camera aliases and shared
world-object composition. It provides no traffic-light states. Environment support follows
[Configuration](../reference/configuration.md#environment).

## Write a custom simulator

Subclass `Simulator` for configuration and `SimulatorRuntime` for native world operations.
Declare supported datasets, commands and timing capabilities, then implement the native lifecycle
hooks in the [simulator API contract](../reference/python-api.md#avsectesterclosed_loopsimulatorsbase).
[Architecture](../concepts/architecture.md#world-ownership) defines world ownership and isolation.

## Next steps

-  Configure timing and rendering in [Configuration](../reference/configuration.md#simulator).
-  Bind a [driving system](driving-systems.md).
-  Use [world insertion actions](attacks.md#insert-a-world-object).
