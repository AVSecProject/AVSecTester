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

Only world settings reach reset; sensor corruptions apply after capture. See
[Environments](environments.md) for environment composition and
[Python API](../reference/python-api.md) for signatures and errors.

### Scenario start inputs

The dataset supplies [ScenarioStart](datasets.md#dataset-interface); attack selection metadata
is not a simulator input.

### Read prescribed motion

`start_replay` initializes and captures a recorded or recipe boundary with prescribed ego
motion. `advance_replay` advances that motion and captures without stage hooks. Replay rejects
bound hooks; advancement also requires an initialized runtime and replay capability.

| `Simulator` | Prescribed-motion advancement | Past observations |
|---|---|---|
| CARLA | Supported | Unsupported |
| Recorded playback | Supported | Unsupported |
| NuRec | Unsupported | Supported |

Services reuse forward scans where advancement is supported. NuRec instead resets prescribed
motion at each requested frame.

## Inserted world objects

`insert_object` registers an `ObjectAsset` under a unique ID and rigid world pose;
`remove_object` removes it. The physical `state().objects` changes immediately, while images
change at the next capture. Inserted objects stay separate from native actors, have zero
velocity and do not participate in native physics or collisions.

Shared camera composition projects the asset, resolves nearest surfaces and applies visibility.
The returned camera view contains composed RGB. Existing depth becomes the pixelwise minimum of
native and object depth; a view without native depth keeps `depth=None`.

| `Simulator` | Occlusion |
|---|---|
| CARLA | Aligned scene depth for `depth_camera`; actor cuboids for other cameras |
| NuRec | Reconstruction visibility from rendered probes, or mesh/actor geometry |
| Recorded playback | Actor cuboids; **Not yet implemented.** Static-scene occlusion; see [Recorded static occlusion](../development/TODO.md#recorded-static-occlusion). |

`SensorView.composition` records the unmodified background and its `CompositePlan`.
`plan.apply(background)` reproduces the capture. Differentiable composition follows the same
calculation where all harmonizers support it; see [Optimization](optimization.md) and the
[compositing API](../reference/python-api.md).

Occlusion tolerance keeps scene surfaces within the configured tolerance in front of an object
from hiding it. The same rule applies to geometry and reconstruction probes. Option defaults
are listed in the [configuration reference](../reference/configuration.md#simulator).

`object_visibility` describes the latest capture. A projected but fully hidden object has a
camera entry with zero visible pixels; an object outside a camera projection has no entry.
`SensorView.objects` holds the corresponding immutable placed objects. Captures retain their own
object membership until the next capture.

See [Writing attacks](attacks.md#insert-a-world-object) for insertion actions and
[Runtime records](../reference/runtime.md#runlog) for saved evidence.

**Not yet implemented.** Inserted objects in non-camera sensors.
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
wrapper's `flipy` setting; see [LiDAR records](../reference/runtime.md#carla-lidar-data).

## NuRec

NuRec requires a `ScenarioStart` from its dataset. The runtime initializes the recorded pose,
timeline and actors, then advances ego with the selected dynamics. Camera names remain native
unless `camera_aliases` maps them to observation names; aliases also apply to views
and calibration.

**Not yet implemented.** NuRec and recorded-playback traffic-light state recording; see
[Traffic-light states](../development/TODO.md#traffic-light-state-recording).

Constructing mesh-depth or ground-contact support requires the `nurec` extra.
Reset checks required archive meshes before connecting to NRE; missing resources fail preflight.
A manually started NRE must enable `--enable-editing-actors`.

The decision step comes from `dataset.time_step` through configuration loading. A driving system
can require a specific step. See [Configuration](../reference/configuration.md#simulator) for
the injection and consistency rules.

NRE is required for camera rendering; see
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

### Recorded past observations

After reset, `observe_past(count)` returns actual recorded observations before the selected
start, oldest first, with negative run-relative timestamps and frames. It rejects insufficient
history and preserves the current world, clock and capture. Start eligibility uses the larger
history requirement of the simulator and `DrivingAdapter`.

Past captures use recorded ego poses and actors, including the same short-track filtering as
current captures. Objects installed during reset or setup are composed using those recorded
camera poses. Later insertions and actor overrides are excluded. Past captures bypass stage
hooks and sensor attack actions. The runner applies sensor corruptions at each absolute dataset
position, then passes history directly to `AVStack.reset`; it bypasses localization and the
observation pipeline. See [initialization records](../reference/runtime.md#initialization) and
[environment randomness](environments.md#randomness).

### Navigation

Map navigation matches the recorded ego path to driving lanes and extends the route through
successors. Every capture projects the current physical ego origin onto that route and provides
forward points in ego coordinates. Navigation expresses intent; it does not constrain the
produced or executed command.

Recorded-route mode uses the recording without extension; disabled navigation supplies no route.
A map route needs driving lanes. Folded or inconsistent routes are rejected by `validate_start`
so selection can record the reason. With loaded metadata, an invalid segment fails preparation.
Algorithm parameters and constraints are in the [navigation API](../reference/python-api.md).

### Ground contact

The runtime retains a three-dimensional ego pose. Ground contact fits the predicted box bottom
to the selected scene mesh after each dynamics step. Rejected corrections retain the prediction;
termination events count failures by reason. Disabling physical updates retains the
dynamics prediction.

The full scene mesh may include roadside objects. A ground-only mesh avoids using those surfaces
for ego contact where that asset is available. Ground contact is an approximate fit rather than
a penetration solver; it does not correct traffic actors.

### Camera resolution

The renderer probes each configured camera when loading a scene and scales its lens to the image
dimensions actually returned by NRE. NRE preserves aspect ratio, so returned width can differ
from the request. Subsequent captures must retain those dimensions.

Captures render a single instant.

**Not yet implemented.** Rolling-shutter exposure;
see [TODO](../development/TODO.md#rolling-shutter-rendering).

The lens includes its f-theta model and recorded windshield distortion. Camera projection,
mesh depth and object placement share the resulting geometry. Camera coordinate conventions
and models are specified in the [Python API](../reference/python-api.md).

### Move recorded actors

Only controllable tracks can be moved by NRE. Parked tracks stay at recorded poses and
`set_actor_state` rejects them. Short controllable tracks are filtered by `min_track_duration_s`;
zero disables filtering.

Filtered controllable tracks are absent from world snapshots, sensor actors, depth and metrics;
all captures and probes hide their recorded render instances below the scene. Parked tracks are
never duration-filtered.

An accepted override changes world state immediately and camera state at the next capture. Its
explicit world velocity advances the actor at fixed orientation until reset or another override.
`control_actor` is unsupported. Without traffic simulation, checkpoints preserve actor
overrides; restoring a checkpoint creates a new capture of the restored state.

### Reactive traffic

Enable `AlpaSimTraffic` through `simulator.options.traffic`; YAML, timing and history follow
[Reactive traffic configuration](../reference/configuration.md#reactive-traffic).

World actors begin with unfiltered recorded states, then use returned controllable-session
poses, then explicit overrides. Parked tracks stay at recorded poses within recorded support.
Sessions include duration-qualified tracks with nonempty clipped history. A first override
backfills constant-velocity history and subsequent requests resend its complete history.

Trajectories are not smoothed. Offline candidate rollouts also require the service.

**Not yet implemented.** Fitting traffic actors to the selected ground mesh, checkpointing
traffic sessions and controlling CATK sampling through the caller seed; see
[Traffic actor ground contact](../development/TODO.md#traffic-actor-ground-contact).

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
origin. The controller tracks the full forecast; it need not execute its first planned
acceleration exactly. There is no automatic command conversion.

A custom dynamics component supplies `step` and may supply `reset` and `command_modes`. Each
runtime receives its own copy. Constructor defaults and trajectory constraints are listed in the
[dynamics API](../reference/python-api.md).

### Mesh depth

Mesh depth combines archived static surfaces and current actor cuboids. Calibrated camera rays
keep the nearest hit, expressed as optical-axis depth; missing hits are infinite. Actor extents
must be known. Depth is computed lazily per camera and cached for the capture.

Mesh content is not classified or erased when an actor moves. Archived vehicle geometry can
therefore continue to occlude its recorded location. Cuboid silhouettes can differ from rendered
vehicles. Rendered probes instead measure the reconstruction's visibility directly.

### Rendered-probe occlusion

Opaque white and black probe copies measure whether scene content hides inserted surfaces. Only
sufficiently opaque asset pixels contribute probes. The renderer requires a client probe
directory visible inside the NRE service; path and mount rules are
in [Configuration](../reference/configuration.md#nurec-closed-loop-settings).

**Not yet implemented.** Published asset conventions for all probe axes, scaling and padding;
see [NuRec rendered-probe conventions](../development/TODO.md#nurec-rendered-probe-conventions).

A captured `composition.plan` already contains this measured visibility. An arbitrary
replacement object list needs a fresh render request; `SensorView.composite_plan` cannot obtain
new reconstruction probes from a detached view. Geometry mode can build plans from its supplied
depth and actors.

## Recorded playback

`RecordedReplaySimulator` captures supported dataset frames while ego and other actors follow
their recorded motion. Commands do not change that trajectory. It supports camera aliases and
shared world-object composition. Environment support follows
[Configuration](../reference/configuration.md#environment).

## Write a custom simulator

See the [simulator base module](../reference/python-api.md#avsectesterclosed_loopsimulatorsbase)
for its Extension contract.
[Architecture](../concepts/architecture.md#world-ownership) describes the world-claim and
operation-isolation rules.

Subclass `Simulator` for configuration and `SimulatorRuntime` for native world operations.
Declare dataset kinds, timing, command modes, stages, world settings, capture capabilities and
identity.

Declare every view-capable sensor in `sensors` and cameras in `camera_sensors`. Native views
supply capture data and sensor geometry; the shared runtime owns camera composition. A
lightweight `sensor_geometry` implementation avoids image/depth work for projection-only
requests. Past-capable implementations supply recorded observations and recorded camera views
for start-object composition.

## Next steps

-  Configure timing and rendering in [Configuration](../reference/configuration.md#simulator).
-  Bind a [driving system](driving-systems.md).
-  Use [world insertion actions](attacks.md#insert-a-world-object).
