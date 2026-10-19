# Prepare scenarios

> **Audience:** Users who define experiment origins, insert objects or select scenarios that meet requirements.  
> **Outcome:** Prepare a scenario from a configuration, specify insertions, compose filters and select qualifying scenarios.  
> **Prerequisites:** [Architecture](../concepts/architecture.md) and [Run experiments](experiments.md).

A [prepared scenario](../concepts/architecture.md#terminology) is the origin of an experiment, and
every treatment starts from it. Selection is optional. Without it, a configuration becomes one
prepared scenario directly. With it, a scenario source evaluates
[filters](../concepts/architecture.md#terminology) on candidate initial states and returns only
qualifying scenarios, with fixed role bindings and insertions. Filters run during preparation,
never inside a run, and do not promise attack success.

The CLI and examples below retain the initial-selection compatibility API. New source readers
can use the [stateless data interface](#stateless-datasets), which separates reading data
from constructing simulator resources.

## Configured scenarios

`scenario_from_config` turns an experiment mapping without a `selection` key into one
`ScenarioInstance`. It does not open a simulator, render images or evaluate filters. The backend
is created when the scenario runs. This example runs offline with the NuRec stub renderer:

<!-- runnable -->
```python
from avsectester.closed_loop.avstack import AVStack
from avsectester.closed_loop.datatypes import Control
from avsectester.runner import run_prepared
from avsectester.scenarios import scenario_from_config


class Cruise(AVStack):
    """A minimal driving system that always applies light throttle."""

    def __call__(self, observation):
        return Control(throttle=0.3)


config = {"backend": {"type": "NuRecSimulator", "renderer": "StubRenderer"}}
scenario = scenario_from_config(config, provenance={"config_path": "inline"})
print(scenario.match)       # None: no selection, so no roles or insertions.
print(scenario.provenance)  # Supplied provenance plus the resolved conditions.

results = run_prepared(scenario, Cruise, frames=10)
print(results["clean"].provenance["treatment"])  # clean
```

`backend.type` selects `CarlaSimulator` (the default) or `NuRecSimulator`; see
[Backend configuration](simulators.md#backend-configuration). The backend factory receives a
detached scene view without experiment and intervention keys such as `frames`, `stack`,
`interventions`, `conditions` and `selection` (see
[File layout](../reference/configuration.md#file-layout)). For a custom simulator, pass
`backend_factory=make_backend`, where `make_backend(scene_config)` returns a `WorldBackend` and
receives a fresh copy on every call. See
[Write a custom simulator adapter](simulators.md#write-a-custom-simulator-adapter).
`source_from_config(config)` returns the same scenario through a `ConfiguredScenarioSource`.

## The prepared scenario

A `ScenarioInstance` holds a backend factory, the optional selection match, a detached scene
configuration, conditions, provenance and selection evidence. The field list is in
[Scenarios](../reference/python-api.md#scenarios).

**Create backends with `scenario.make_backend()`.** It calls `scenario.factory()` and then
configures the scenario's world conditions, closing the backend if that fails. `run_prepared` and
`run_configured` use it. `scenario.factory()` alone returns a backend without world conditions, so
call it only inside a replacement factory.

**The match is separate from the scenario.** Intervention preparation receives the complete
`ScenarioInstance` (see [Use the selected scenario](attacks.md#use-the-selected-scenario)). Runtime
handlers receive only the match, as `context.case`. `run_prepared` supplies `scenario.match` as the
case to every Runtime that has none. Without selection both are `None` and no roles or insertions
are inferred. A method that needs a selected role must report the missing selection clearly.
Explicit world positions and native actor names still work without selection.

**Fields cannot be reassigned.** Build a modified scenario with `dataclasses.replace`:

```python
from dataclasses import replace

# scenario is a ScenarioInstance whose factory returns a NuRecSimulator.
def coarse_backend():
    backend = scenario.factory()  # The original factory, without conditions.
    backend.dt = 0.1              # Use a longer simulation time step.
    return backend

coarse = replace(scenario, factory=coarse_backend)
```

`replace` validates the result and refreshes `provenance["conditions"]`. `config` and `provenance`
are detached copies but stay mutable. Editing `scenario.config` does not rebuild a factory that has
already captured its settings. Replacing `conditions` does not rerun filters (see
[Conditions during selection](conditions.md#conditions-during-selection)).

## Specify inserted objects

An [insertion](../concepts/architecture.md#terminology) has an explicit asset, placement and
orientation. Positions are never searched or adjusted to make a filter pass, and a missing host is
never replaced by another actor. Coordinates are metres on right-handed axes. Actor-local axes are
X forward, Y left and Z up. World coordinates are the provider's absolute scene frame. The CARLA
adapter converts Unreal coordinates at its boundary.

<!-- runnable -->
```python
import numpy as np
from avsectester.attacks.object_insertion.sign_spoof import STOP_SIGN
from avsectester.insertion import (
    AttachedPlacement, Insertion, Orientation, PlaneAsset, WorldPlacement,
)

# Replace this RGBA array with a texture path or your own uint8 RGBA array.
texture = np.full((32, 32, 4), [255, 0, 255, 255], dtype=np.uint8)

patch = Insertion(
    id="rear_patch",
    asset=PlaneAsset(texture, width_m=0.6, height_m=0.4),
    placement=AttachedPlacement(host="attacker", anchor="rear_center", offset_m=(-0.05, 0.0, 0.15)),
    orientation=Orientation("follow_host", rotation_deg=(0, 0, 180)),
)

sign = Insertion(
    id="roadside_sign",
    asset=PlaneAsset(STOP_SIGN, width_m=0.8, height_m=0.8),  # A PNG path, loaded with Pillow.
    placement=WorldPlacement(position_m=(20, -3, 1.8)),        # Illustrative world coordinates.
    orientation=Orientation("face_victim"),
)
```

`WorldPlacement` takes absolute world coordinates, not a distance ahead of the ego.
`AttachedPlacement` measures from the host's bounding-box centre. `host` is a role name, an indexed
role alias such as `hosts[0]`, or a native actor ID. The anchors are `center`, `front_center`,
`rear_center`, `left_center`, `right_center`, `top_center` and `bottom_center`. They are
bounding-box faces, not body panels. `offset_m` uses the host's local axes and rotates with it.

| Orientation mode | Rotation |
|---|---|
| `follow_host` | Host rotation multiplied by `rotation_deg`. Requires an `AttachedPlacement` |
| `fixed_world` (default) | `rotation_deg` is an absolute world rotation. An attached centre still follows its host |
| `face_victim` | The asset front points at the victim's current position plus `target_offset_m` (victim frame), keeping the asset top near world up. `rotation_deg` must be zero |

Angles are `(roll, pitch, yaw)` in degrees, applied as `Rz(yaw) @ Ry(pitch) @ Rx(roll)`. The asset
front is +X and its top is +Z, so a rear patch normally uses a local yaw of 180 degrees. A host can
carry several insertions, and several hosts can carry different ones. In paired runs, bindings and
world orientations stay the same, while `face_victim` follows each run's current ego pose.

`PlaneAsset` is an RGBA rectangle whose transparent pixels are not part of its silhouette. A custom
asset implements `planes()` and returns two-sided `PlaneSurface(corners, texture)` rectangles in
one asset-local frame, with corners ordered top-left, top-right, bottom-right, bottom-left and
`uint8` RGBA textures. It needs no registration, but dictionary serialization covers only
`PlaneAsset`. See [Insertion specification](../reference/python-api.md#insertion-specification).

## Compose a requirement

A `ScenarioRequirement` combines filters, [roles](../concepts/architecture.md#terminology),
insertions and an initial window. This requirement selects a vehicle ahead whose rear can carry
the patch from the previous section:

```python
from avsectester.scenarios import InitialWindow, RoleSpec, ScenarioRequirement
from avsectester.scenarios.filters import DistanceRange, InView, MinVisibility, ViewpointRear

requirement = ScenarioRequirement(
    name="rear_patch_case",
    roles={"attacker": RoleSpec(category="vehicle", count=1, select="nearest_ahead")},
    insertions=(patch,),
    camera="front",
    window=InitialWindow(frames=1),
    constraints=[
        DistanceRange(5, 25),
        ViewpointRear(max_deg=35, camera="front"),
        InView("front"),
        MinVisibility(0.5, camera="front"),
    ],
)
```

`requirement.evaluate(context)` takes a `FilterContext` or a bare `SceneGT` and returns a
`SelectionResult` with `status`, `match`, `candidates` (each binding's checks) and `reason`.
`requirement.match(context)` returns only the `ScenarioMatch` or `None`. Scenario sources call
these for you.

### Filters

A filter implements `evaluate(context)` and returns `FilterResult(status, reason)`, where status is
`pass`, `fail` or `unknown`. Missing evidence gives `unknown`: no visibility measurement, a missing
bound actor, missing calibration or an incomplete initial window. Programming errors propagate.
A candidate is selected only if every top-level filter passes. Combine filters for other logic:

| Combination | Result |
|---|---|
| `All(filters)` | Fail if any child fails, otherwise unknown if any is unknown, otherwise pass |
| `Any(filters)` | Pass if any child passes, otherwise unknown if any is unknown, otherwise fail |
| `Not(filter)` | Exchange pass and fail, preserving unknown |

Importing `Any` from `avsectester.scenarios` shadows `typing.Any`, so import it as `Any as AnyOf`
when you need both. The built-in filters are in `avsectester.scenarios.filters`:

| Filter | Measurement | Unknown when |
|---|---|---|
| `DistanceRange(min_m, max_m)` | Horizontal distance from the ego to each `attacker` member, inclusive bounds in metres | No `attacker` role is bound |
| `ViewpointRear(max_deg=35.0, camera="front", role="attacker")` | Horizontal angle between each role member's rear direction and its bearing to the camera, at most `max_deg` | No camera-to-ego transform, no role member, or camera and member coincide horizontally |
| `InView(camera, subjects=())` | At least one opaque insertion pixel projects inside the image, regardless of occlusion. Without insertions, the subject's annotated box overlaps the image | No calibration, or a named subject is missing |
| `ImageAreaFrac(min_frac, max_frac, camera="front", subjects=())` | Image-clipped projected bounding-box area divided by image area. Not a silhouette measurement. Fails without a projected box | No calibration, or a named subject is missing |
| `MinVisibility(min_vis=0.7, camera="front", subjects=())` | [Visible fraction](#visibility) of each subject is at least `min_vis` | No visibility evidence, or evidence from another camera |
| `EgoMoving(min_speed=1.0)` | Ego speed is at least `min_speed` m/s | Never |
| `ClearLaneAhead(corridor_half_width=1.5)` | No other object ahead is nearer than the attacker within the lateral half-width. A centre-based test, not lane geometry or occlusion | No attacker is bound |

With insertions, the visual filters (`InView`, `ImageAreaFrac`, `MinVisibility`) check the
inserted objects, never the host's own visibility. All insertions are checked unless `subjects`
names some, for example `subjects=("rear_patch",)`. A requirement with insertions also always
checks that every insertion overlaps the image of its camera: `camera`, otherwise the target
camera, otherwise `front`, otherwise the scene's only camera. If none applies, that check is
unknown.

`TargetSpec` is a single-object shorthand, passed as `ScenarioRequirement(target=...)`, that
queries existing annotations and binds the chosen object as `attacker`. It cannot be combined with
`roles`. The preset `PHYSICAL_PATCH_HIDE_VEHICLE` in
`avsectester.scenarios.presets` uses it, so its visual filters query the vehicle annotation, not a
patch. For an inserted patch, declare roles and insertions as above.

### Roles and bindings

`RoleSpec(ids=None, count=1, category="vehicle", select="nearest_ahead", camera="front")` describes
one named actor slot. `select` ranks candidates: `nearest_ahead` (objects ahead, by distance),
`nearest` (by distance) or `largest` (by projected box area in `camera`). The ego is the victim and
cannot be a role.

Selection evaluates every filter for each candidate assignment and takes the first passing
assignment in rank order, so a nearer vehicle that fails does not exclude a farther one that
passes. `RoleSpec(ids=("track-17",))` binds that exact actor and is never substituted. If it is
absent, selection fails. For two hosts, use `RoleSpec(count=2)` or `RoleSpec(ids=("a", "b"),
count=2)` and attach insertions to the aliases `hosts[0]` and `hosts[1]`. A single-member role name
is itself an alias.

`max_bindings` (default 1000) bounds the assignments examined. If the budget runs out before any
assignment passes, the result is unknown. A bounded search cannot promise the best assignment.
Bindings stay fixed through the initial window and every run. `scenario.match.binding_ids` maps
each role to a tuple of actor IDs.

## Select a scenario

A scenario source yields one `ScenarioInstance` per accepted candidate. `limit` counts accepted
scenarios. The source closes its preview resources before yielding each scenario.

```python
# source and requirement are supplied by the caller.
scenario = next(source.scenarios(requirement, limit=1), None)
if scenario is None:
    raise RuntimeError(f"No qualifying scenario: {source.selection_log}")

print(scenario.match.binding_ids)    # Role names mapped to fixed actor IDs.
print(scenario.selection["status"])  # pass
```

`scenario.match` holds the selected initial `scene`, `bindings`, `insertions`, `camera`, `window`
and an optional `target`. `scenario.selection` holds each candidate's checks and reasons; read them
to verify a selection. `source.selection_log` has one entry per candidate, accepted or not, and
`source.save_selection(path)` writes it as JSON without native clients or images. A complete
offline example is in [Support another dataset or provider](#support-another-dataset-or-provider).

### CARLA

Start a dedicated CARLA server matching the client version (see
[Start a CARLA server](../getting-started/installation.md#start-a-carla-server)). Then create
`source = CarlaScenarioBuilder(base_scenario=config)` from a loaded scene configuration, such as
`configs/carla_patch_scenario.yaml`, which contains an ego camera and a stationary lead vehicle.
Set `config["client"]["connect_port"]` if the server does not use port 2000. Without
`base_scenario`, the builder loads that file.

The builder creates each candidate world and reads its actual geometry. It drops the
configuration's physical `patches`, so insertion visibility is measured before any attack. Fixed
scene positions are kept. `sample_scene=True` samples `samples` lead positions with `seed`: a gap
within the requirement's `DistanceRange` (otherwise up to 20 m), never below `min_gap=6.0`, and a
lateral offset within `lateral_range=(0.0, 3.0)`. Sampling never moves insertions. The ego starts
stationary, so `ego_speed` must be 0.

Candidates use runtimes of `CarlaSimulator(scenario, depth_camera="front")`. It replaces stock RGB
cameras with undistorted `PinholeRgbCamera` sensors, adds an aligned depth camera and advances one
stationary setup frame before the first observation. The selected scenario's factory uses the same
configuration, so paired resets match the selection. The selected RGB camera must produce a frame at
every driving step: set `sensor_tick` to 0 or to the simulator step. `analytic_preview=True`
enumerates approximate scenes offline instead. It validates no CARLA spawn and measures no rendered
visibility.

### NuRec datasets

Point a `NuRecDataset` at local USDZ files and wrap it in a `DatasetFilter`:

<!-- runnable -->
```python
from avsectester.scenarios import DatasetFilter
from avsectester.scenarios.datasets.nurec import NuRecDataset

dataset = NuRecDataset.from_glob("/path/to/scenes/*.usdz", keyframe=0.5, endpoint="127.0.0.1:50051")
source = DatasetFilter(dataset)
```

Selection reads the recorded annotations and estimates visibility from cuboids, without a renderer
connection. Running the selected scenario, or rendering from a custom filter, needs an `nre-ga`
service serving that scene (see
[NuRec renderer and scenes](../getting-started/installation.md#nurec-renderer-and-scenes)).
`NuRecDataset.make_backend` uses actuator dynamics, while Alpamayo needs trajectory dynamics. See
[Insert into the model input](../examples/image-insertion.md#insert-into-the-model-input) and
[Dynamics and time step](simulators.md#dynamics-and-time-step).

`NuScenesDataset(dataroot, version="v1.0-trainval", camera="CAM_FRONT")` in
`avsectester.scenarios.datasets.nuscenes` selects nuScenes keyframes through the devkit.
`scenes()` iterates `nusc.sample[:max_samples]` in SDK order, numbers frames globally and keeps
sample-annotation tokens as object IDs, as before. Its backend advances through linked recorded
frames and ignores commands. N decisions require N+1
frames, including the terminal frame; requests past the sequence end fail.

### Use a selected scenario in an experiment

To run configured interventions and a configured stack on a Python-selected scenario:

```python
from avsectester.experiment import run_configured

# config enables an attack. scenario comes from a selection source.
results = run_configured(config, frames=30, scenario=scenario, mode="pair", treatment="attack")
```

The scenario's factory, match and conditions are used as they are. Scene settings, `selection` and
`conditions` in `config` are not reapplied. Supply a stack compatible with the selected backend, in
`config` or as `stack_factory=`. See [Configured runs](experiments.md#configured-runs). To build
Runtimes yourself, use `run_prepared(scenario, stack_factory, frames, runtimes=...)`, as in
[Custom treatments](experiments.md#custom-treatments).

Insertions in the match are specifications. Nothing renders them until a handler does; see
[Insert into the model input](../examples/image-insertion.md#insert-into-the-model-input). In a pair, the clean run
uses the same scenario with the insertion disabled, and both runs start from one origin (see
[Paired runs on one server](simulators.md#paired-runs-on-one-server)). Trajectories can diverge
once the runs start.

## Selection in YAML

A `selection` key makes `scenario_from_config` select instead of wrapping the configuration.
`requirement_to_dict(requirement)` produces the `selection.requirement` mapping from Python. Written
by hand, a selection section looks like this, and omitted fields take their Python defaults:

```yaml
selection:
  requirement:
    name: lead_vehicle_ahead
    roles:
      attacker: {category: vehicle, select: nearest_ahead}
    constraints:
      - {kind: DistanceRange, min_m: 5, max_m: 25}
      - {kind: Not, filter: {kind: EgoMoving, min_speed: 1.0}}
    window: {frames: 1}
  source:            # CarlaScenarioBuilder options.
    sample_scene: true
    samples: 4
```

`scenario_from_config` returns the first accepted scenario or raises `ValueError`.
`source_from_config(config).scenarios(limit=N)` enumerates accepted scenarios with the stored
requirement. YAML selection without a dataset requires a CARLA backend. For a recorded dataset,
pass `dataset=` to either function in Python and leave `selection.source` empty. Serialization
covers combinations, roles, windows and `PlaneAsset` insertions (texture path or inline `uint8`
array). Register a custom filter dataclass with `@register_constraint` to serialize it. All keys
are in [Selection keys](../reference/configuration.md#selection-keys).

## Initial windows and providers

`InitialWindow(frames=N)` requires the filters to pass on N consecutive prepared states, starting
at the selected origin. Bindings stay fixed: if an actor disappears, no replacement is selected
and the check is unknown. A source with fewer than N states cannot satisfy the window by repeating
one. No driving model or attack feedback produces the window, so passing it does not guarantee
that the filters still hold once the driving system responds.

**NuRec.** `DatasetFilter` checks the requirement on `dataset.initial_sequence(scene, N)`: recorded
camera frames with interpolated actor poses. The backend starts at the first frame. `keyframe` is
a fraction in `[0, 1]` of the camera's frame indices and gives one candidate start per USDZ file.
To search more starts, override `scenes()` and call `scene_at_frame(path, frame)`.

**CARLA.** A one-frame window needs no callback. For longer windows, give the provider a prescribed
`advance_initial(backend, index)` callback that advances one frame without a driving model:

```python
from dataclasses import replace
from avsectester.closed_loop.datatypes import Control
from avsectester.scenarios import CarlaScenarioBuilder, InitialWindow
from avsectester.scenarios.carla_provider import CarlaCandidateProvider

def advance_initial(backend, index):
    return backend.step(Control(brake=1.0))  # A prescribed command, not a driving model.

# config and requirement are supplied by the caller.
requirement = replace(requirement, window=InitialWindow(frames=10))
provider = CarlaCandidateProvider(initial_frames=10, advance_initial=advance_initial)
source = CarlaScenarioBuilder(base_scenario=config, candidate_provider=provider)
scenario = next(source.scenarios(requirement, limit=1), None)
```

Each frame's filters run before the world advances, so native CARLA state matches `context.scene`.

## Visibility

For known insertion geometry, visibility is a binary geometric measure:

```text
visible opaque target pixels inside the camera image
----------------------------------------------------
complete projected opaque target pixels from that viewpoint
```

The denominator includes the part outside the image. The host occludes its own patch, and
insertions occlude one another. Texture samples with alpha above 127 are opaque. Selection and
rendering share projection, silhouettes and pose resolution, and the compositor reuses the
estimator's samples. A target entirely behind the camera has visibility 0. A target crossing the
camera plane has unknown visibility.

| Provider | Evidence |
|---|---|
| CARLA | Aligned scene depth compared with insertion surface depth (`DepthVisibilityEstimator`, `z` depth) |
| NuRec | Ray intersections with annotated 3D cuboids and other insertion surfaces (`CuboidVisibilityEstimator`) |
| nuScenes | Dataset visibility tokens, for existing annotated objects only |
| Custom | `FilterContext.visibility_provider(context, insertion_id, camera)` |

Depth evidence needs the same camera pose, lens and frame as the geometry. Invalid depth at target
pixels gives unknown. Cuboid estimates (source `cuboid_estimate`) ignore unannotated buildings,
vegetation and exact mesh boundaries. nuScenes tokens map to coarse values `1 → 0.2`, `2 → 0.5`,
`3 → 0.7` and `4 → 0.9`. They are not camera-specific and never describe an inserted patch. For
native complex assets, `visibility_from_masks` and `visibility_from_depth` compare against a
complete unoccluded reference at the same viewpoint. Fog attenuation and recognition confidence are
not part of geometric visibility. See [Visibility](../reference/python-api.md#visibility).

## Custom filters and native access

A custom filter subclasses `Constraint` and returns a `FilterResult`:

```python
from dataclasses import dataclass, replace
from avsectester.scenarios import Constraint, FilterResult

@dataclass
class AllowedWeather(Constraint):
    max_rain: float

    def evaluate(self, context):
        world = context.native.get("world")
        if world is None:
            return FilterResult("unknown", "CARLA world unavailable")
        rain = world.get_weather().precipitation
        return FilterResult.from_bool(rain <= self.max_rain, f"Precipitation {rain}")

# requirement is a CARLA requirement supplied by the caller.
requirement = replace(requirement, constraints=[*requirement.constraints, AllowedWeather(20)])
```

`FilterContext` exposes `scene`, `bindings`, `insertions`, `metadata`, `dataset`, `backend`,
`renderer` and `native`, with no method whitelist. `context.actors` gives world poses by actor ID
and role alias, and `context.victim` gives the ego's bounding-box pose.

| Provider | Native access |
|---|---|
| CARLA | `context.native["world"]`, `["actors"]`, `["client"]`, `["depth"][camera]`, `["observation"]` |
| NuRec | `context.metadata` holds the raw JSON records. `context.native["open_archive"]()` opens the USDZ. `context.native["actor_poses"](timestamp_us)` interpolates actor poses. `context.renderer` is a lazy renderer |
| nuScenes | `context.native["nusc"]` is the devkit. `context.metadata` holds the source records |

Normalized geometry uses 4×4 rigid transforms, and extents are full length, width and height:

| Value | Transform |
|---|---|
| `scene.ego.pose` | Ego frame to world |
| `ObjectGT.pose` | Object bounding-box centre frame to ego |
| `CameraCalib.cam_to_ego` | Optical camera frame (X right, Y down, Z forward) to ego |
| `ActorPose.transform` | Actor bounding-box centre frame to world |

Native APIs keep their provider's conventions, so convert before comparing with normalized
geometry. Return `unknown` when information is unavailable. Native APIs can modify the world, so do
not change a candidate inside a filter. `register_constraint` is needed only for dictionary
serialization and the natural-language vocabulary.

## Stateless datasets

`avsectester.datasets.Dataset` reads recorded scenarios or recipes. It has no backend or renderer
factory. `avsectester.datasets.nurec.NuRecDataset(usdz_paths, camera_id=..., camera_alias=...,
time_step=0.1)` reads USDZ annotations. Each file is one scenario, identified by NuRec's scene ID
(`clipgt-<uuid>`, the ID the NuRec renderer loads); `camera_id` is NuRec's camera ID and
`camera_alias` the sensor name used in observations. `avsectester.datasets.nuscenes.NuScenesDataset`
reads linked samples within each recorded scene; its timelines always cover the complete scene and
its actor IDs are stable instance tokens. Their older `avsectester.scenarios.datasets` imports
retain factories, keyframe iteration and the original `sensor`/`camera` parameters for existing
callers.

| Method | Source contract |
|---|---|
| `scenarios()` | Scenario descriptors with stable source identities |
| `frames(scenario_id)` | Ordered `FrameReference` records: position, original ID and source timestamp |
| `frame_data(scenario_id, frame)` | Detached `FrameData` (scene ground truth, actor states, sensor data), or an explicit `Unavailable` result |
| `native(scenario_id)` | Dataset-specific data, without simulator resources |
| `scenario_start(scenario_id, frame)` | Detached `ScenarioStart`: what a simulator needs to start the scenario at `frame` |
| `decision_window(scenario_id, start, decision_steps)` | N+1 actual frames for N decisions; the last is the terminal state |

`decision_steps` has no default. An oversized request fails instead of repeating the last frame
or shortening the run. nuScenes replay preserves recorded timestamps. NuRec generates a decision timeline at its configured
`time_step` (default 0.1 s), with linear position interpolation and quaternion SLERP within recorded
support. Actor tracks are absent outside their own support; nothing is extrapolated. Source
positions and timestamps remain distinct from native
simulator frame IDs and clocks.

`CarlaDataset` exposes only a `CarlaRecipe`: source configuration, decision horizon, time step
and explicit controls or trajectories for every actor. A stationary actor still needs an explicit
zero-motion sequence. Traffic Manager is not a prescribed-motion source. Existing Traffic Manager
CLI scenarios keep their original from-start execution path.

Reading a CARLA recipe never generates its frame data:

<!-- runnable -->
```python
from avsectester.datasets import Unavailable
from avsectester.datasets.carla import CarlaDataset, CarlaRecipe, PrescribedMotion
from avsectester.closed_loop.datatypes import Control

recipe = CarlaRecipe(
    scenario={
        "client": {"synchronous": True, "rate": 20},
        "ego": {"autopilot": False},
        "npcs": [],
    },
    decision_steps=3,
    time_step=0.05,
    motion={"ego": PrescribedMotion(controls=tuple(Control(brake=1) for _ in range(3)))},
)
dataset = CarlaDataset({"example": recipe})
start = dataset.frames("example")[0]
assert len(dataset.decision_window("example", start, decision_steps=3)) == 4
assert dataset.frame_data("example", start) == Unavailable("recipe_only")
assert dataset.scenario_start("example", start).kind == "CarlaDataset"
```

Generated CARLA frame data belongs to the preparation layer's `FrameDataCache`; running a
simulator does not change these Dataset methods. The cache identity includes the recipe,
simulator configuration/version and seed. Failed generation is not cached.

## Support another dataset or provider

For the existing initial-selection API, subclass `avsectester.scenarios.Dataset`:

| Method | Contract |
|---|---|
| `scenes()` | Required. Yield candidate initial `SceneGT` records with stable actor IDs |
| `make_backend(scene)` | Required. Create a fresh `WorldBackend` at that origin |
| `initial_sequence(scene, frames)` | Consecutive states beginning with `scene`, without padding. Default `(scene,)` |
| `context(scene)` | A `FilterContext` for the scene and any native resources. Default: the scene and the dataset |

This dataset runs offline. It reuses `requirement` from
[Compose a requirement](#compose-a-requirement) and `Cruise` from
[Configured scenarios](#configured-scenarios):

```python
import numpy as np
from avsectester.rendering.visibility import CuboidVisibilityEstimator
from avsectester.runner import run_prepared
from avsectester.scenarios import (
    CameraCalib, Dataset, DatasetFilter, EgoState, FilterContext, ObjectGT, SceneGT,
)
from avsectester.closed_loop.simulators.nurec import NuRecSimulator, StubRenderer

# Optical camera axes to ego axes, with the camera 1.5 m above the ego origin.
CAM_TO_EGO = [[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 1.5], [0, 0, 0, 1]]
CAR = (4.5, 1.9, 1.5)


class ToyDataset(Dataset):
    def scenes(self):
        yield SceneGT(
            frame=0, t=0.0, ego=EgoState(pose=np.eye(4).tolist(), extent=CAR),
            cameras={"front": CameraCalib("front", 640, 480, model=(400.0, 320.0, 240.0),
                                          cam_to_ego=CAM_TO_EGO)},
            objects=[ObjectGT("car-1", "vehicle", center=(12.0, 0.0, 0.75), extent=CAR),
                     ObjectGT("car-2", "vehicle", center=(30.0, 4.0, 0.75), extent=CAR)],
            source={"dataset": "toy", "frame": 0},
        )

    def context(self, scene):
        return FilterContext(scene, dataset=self, visibility_estimator=CuboidVisibilityEstimator())

    def make_backend(self, scene):
        return NuRecSimulator({"camera_aliases": {"camera_front": "front"}},
                              renderer=StubRenderer()).create_runtime()


source = DatasetFilter(ToyDataset())
scenario = next(source.scenarios(requirement, limit=1))
print(scenario.match.binding_ids)  # {'attacker': ('car-1',)}. car-2 fails DistanceRange.
results = run_prepared(scenario, Cruise, frames=3)
```

**Resources.** `DatasetFilter(dataset, prepare_context=...)` can add resources to each context.
Register a created resource with `context.own(resource)` so its `close()` runs after selection,
including for rejected candidates and errors. Assigning `renderer`, `backend` or `native` does not
transfer ownership, so filters can borrow shared clients. Use `dataset.context(scene)` as a context
manager when calling it directly. A backend factory must create fresh resources and must not
capture preview clients.

**Simulator providers.** A live provider is a callable `config -> context manager[FilterContext]`,
passed as `CarlaScenarioBuilder(candidate_provider=...)`. Put a factory that rebuilds the selected
origin in `context.native["make_backend"]`, and set `context.initial_contexts` for multi-frame
windows. To support sensor conditions during selection, provide `context.native["observation"]` at
the scene's frame and time. See [Conditions during selection](conditions.md#conditions-during-selection).

## Natural-language requirements

`interpret` converts a description into a `ScenarioRequirement` through a completion function that
you supply:

```python
from avsectester.scenarios.nl import interpret

# complete(prompt) is supplied by the caller and returns text containing a requirement JSON object.
requirement = interpret("a rear-facing vehicle within 20 metres", complete)
```

The prompt lists the registered filter vocabulary. The response is parsed by the same deserializer
as YAML, so unknown filters or fields raise. The framework supplies no model provider, credentials
or agent. Use the result with any scenario source.

## Next steps

- [Apply operating conditions](conditions.md) to the prepared scenario.
- [Use the selected scenario](attacks.md#use-the-selected-scenario) in attack preparation.
- [Insert into the model input](../examples/image-insertion.md#insert-into-the-model-input) to render insertions.
- [Robustness across corruption settings](evaluation.md#robustness-across-corruption-settings) for
  selected scenarios.
- [Scenarios](../reference/python-api.md#scenarios) for the complete API.

## Shared closed-loop services

`avsectester.closed_loop.services.ClosedLoopServices` is the closed-loop entry point of the
metabuilder and the offline phase: sensor captures, sensor views with attack effects, driving
components, scorers and candidate rollouts. A `ScenarioView` exposes the same services to
`Attack.build_metadata` and resolves frame data through `frame_data(frame)`. Recorded frame data
comes from the Dataset. CARLA frame data is generated from explicit motion recipes into the
preparation-owned `FrameDataCache`. Calling the simulator never changes `Dataset.frame_data()` or
attaches records to a Dataset.

Construct services with a pure Dataset, a configured `Simulator` and an optional `DrivingAdapter`.
Each operation creates its own runtime with `simulator.create_runtime(seed)` (the session seed),
resets it inside a reserved operation and closes it when the operation ends; one rollout uses one
runtime for all its decisions. Simulator identity, dataset kinds, rendering and decision time step
are declared by the simulator. Driving bindings live in
`avsectester.closed_loop.driving_adapters`.

| Simulator | Dataset | Behavior |
|---|---|---|
| `CarlaSimulator` | CARLA recipe | Explicit-motion replay and selected-origin initialization; sensor views with measured depth when `depth_camera` is set |
| `NuRecSimulator` | NuRec USDZ | Interpolated-origin initialization, renderer, sensor views with recorded actors, configurable ego dynamics |
| `RecordedReplaySimulator` | nuScenes | Bounded open-loop replay of recorded frames; commands do not change recorded motion |

CARLA uses the configured client's `connect_ip`, `connect_port` and explicit `simulator_version`.
NuRec declares its renderer version via `simulator_version`, endpoint and decision `dt`.
The simulator's decision step must equal the NuRec Dataset's `time_step`; a mismatch fails before
world or model initialization. No per-window cadence tolerance is applied.
`TrajectoryFollower` remains the compatible dynamics choice for Alpamayo trajectory commands.

```python
from avsectester.closed_loop.services import ClosedLoopServices
from avsectester.closed_loop.simulators.nurec import (
    NuRecRenderer, NuRecSimulator, TrajectoryFollower,
)

# dataset is a NuRecDataset; its camera ID and alias configure the simulator.
simulator = NuRecSimulator(
    {"dt": 0.1, "simulator_version": "26.04",
     "camera_aliases": {dataset.camera_id: dataset.camera_alias}},
    NuRecRenderer(endpoint="127.0.0.1:50051", cameras=[dataset.camera_id]),
    TrajectoryFollower(),
)
services = ClosedLoopServices(dataset, simulator, driving_adapter)
```

### A user request

The following example assumes `dataset` and `services` have been constructed for the same
source and `scenario_id` identifies it. The positions and seed are example choices.

```python
from avsectester.closed_loop.runner import RecordingPolicy
from avsectester.attack import AttackSegment
from avsectester.closed_loop.services import CandidateRequest, SensorRequest

frames = dataset.frames(scenario_id)
view = services.scenario(scenario_id, seed=7)
reference = view.frame_data(frames[5])
segment = AttackSegment(scenario_id, frames[5], frames[7])
assert segment.decision_steps == 2  # decisions at positions 5 and 6; terminal state at 7

with services.session(seed=7) as session:
    sample = session.sensor_data(SensorRequest(scenario_id, frames[5], (camera_id,)))
    log = session.rollout(CandidateRequest(
        segment=segment, seed=7, decision_steps=2, environment={},
        recording=RecordingPolicy((camera_id,), ("raw", "model_input")),
    ))
```

`AttackSegment.roles` may be empty. Role bindings map names to tuples of stable actor IDs;
placement is optional. A candidate has no selection score or rank. Its start and stop must be
actual source boundaries, and `stop` must exist. A missing terminal frame is never silently
corrected. The framework passes only `ScenarioStart` and replay
initialization evidence to the simulator, never `AttackSegment`.

To render attack effects, call `session.render(RenderRequest(scenario_id, frame, camera_id,
effects))`. World attack effects are applied at the simulator's `simulator.setup` within the
services' deployment plan and appear in the capture; sensor attack effects then change the
captured data in the simulator's `SensorView`. `RenderResult` contains the changed observation
and the camera's detached `SensorView` before its sensor effects. A new operation uses new
simulator state, so different candidates cannot reuse another candidate's result.

Requests use original source poses and environment unless supported overrides are explicit.
CARLA currently accepts `weather`; other overrides fail. Source, environment, simulator identity
and seed are retained in result provenance and candidate RunLog metadata. A candidate attack is
an `Attack`: each rollout calls `attack.create_runtime(segment, artifacts)` and stops its own
`AttackRuntime`. Metabuilder orchestration, the offline phase and treatment orchestration remain
separate from this candidate service.

### Driving components and gradients

`DrivingAdapter(factory, components=..., scorers=...)` receives a factory called with the session
seed. It must return a fresh driving adapter and configure any model-specific random seed.
Component bindings pair a `ComponentDescriptor` with `(stack, inputs) -> value`; scorer bindings
pair a `ScorerDescriptor` with `(stack, inputs, target) -> value`. Descriptors declare input and
output schemas, codecs and differentiable input names. Scorers also declare output shape and
optimization direction. The user supplies the objective and target.

`modular_driving_adapter` binds explicitly requested native components and the existing MMDet
inference/RPN-objectness scorers. Component arguments follow their native signatures, passed as
`{"args": (...), "kwargs": {...}}`. `alpamayo_driving_adapter` exposes the complete `policy` call
on an `Observation`, with a fresh policy history. Neither helper claims gradients for components
that do not provide them.

`score(..., require_grad=True)` fails before constructing the driver if the descriptor declares
no differentiable inputs. A declared differentiable scorer returning a detached value also fails.
Successful differentiable scores retain their computation graph; they are not converted into log
snapshots before the user can differentiate them.

### Resource and replay guarantees

Use sessions as context managers. Each independent operation creates fresh simulator and driving
state and releases owned resources on success, failure or cancellation. Factories returning an
active or previously used instance are rejected. The same session cannot execute overlapping
operations. Read-only model weights may be shared behind independent adapters only when the
adapter author can guarantee isolation of all mutable model state.

CARLA operations reserve their endpoint until resource cleanup finishes. Another operation or
`ClosedLoopRunner` in the same process
cannot reset that world while it is reserved. Independent processes must still use dedicated
servers or coordinate their ownership externally. This process-local guard does not lock other
CARLA clients. NuRec sessions use independent simulator, renderer connection and driving histories.

CARLA pre-roll uses the cached resolved initial spawns and explicit motion, and does not count
as driving decisions. The actual state at the selected origin is recorded in the initialization
report; it is not compared against the reference. Every configured sensor must deliver a capture
for the current world frame; a missing or misaligned capture raises `SensorAlignmentError`,
whose report the runner records before failing.

Reference cache identity includes the full recipe, simulator adapter identity and seed; failed
generations are not cached. `AttackRuntime.setup` runs inside the deployment window, so its world
attack effects appear in the selected capture; it cannot advance time beyond its declared
`setup_ticks`. Later world effects apply before a world step.

Explicit CARLA trajectories hold actors during sensor setup, then use one additional matched
setup tick to establish native velocity before source time zero. During prescribed motion,
physics remains enabled so the engine reports velocity; gravity is disabled because the path
already specifies vertical motion. Each tick places the actor one integration interval before
the requested pose and applies the prescribed velocity. References contain the **measured**
result, including any contact or integration deviations. Releasing a trajectory-controlled ego
to the driving system restores gravity. The extra setup tick is recorded separately and is not
a scored decision. Explicit-control recipes use ordinary gravity and physics during replay
and driving. Base-world preparation and p=0 deployment temporarily hold their resolved poses.

### Selected-origin deployment and reset ownership

`SimulatorRuntime.reset(start, initialization, deployment=DeploymentPlan(k), environment=...)`
recomputes runtime state from configured defaults and the optional start. A subsequent plain reset
returns to those defaults. It does not retain the preceding start's scene, origin or bounds.

CARLA resets from a `ScenarioStart` include a fixed sensor preparation tick with actors held at
their resolved origin, before scenario time zero. Frame data generation and runs use the same
preparation; `start_sensor_setup_ticks` is recorded in frame-data provenance and simulator identity.

CARLA renders on ticks, so its first capture needs one tick after `simulator.setup`
(`capture_ticks = 1`). The deployment window is therefore k participant ticks plus that capture
tick. CARLA initializes the base world without hooks, replays to `p-k-1`, dispatches existing
`simulator.setup` deployments and continues recipe motion for k+1 ticks, then captures at p. The
window is taken from the pre-roll, so the selected boundary never moves. `setup_tick()` calls by
`simulator.setup` hooks and world attack effects consume the k-tick budget; the capture tick is not
available to them. Clean
treatments use the same plan. An origin with `0 < p < k+1` fails; at p=0 the window holds actors
before source time zero. Reference generation uses the same plan, whose tick budget
participates in the reference cache key. Inserted actors are logged separately.

A plain CARLA reset (no `ScenarioStart`) keeps the original order: `simulator.setup`, k deployment
ticks, one capture tick, then the first capture. A `simulator.setup` change that declares no ticks
is therefore visible in the first capture on both paths. NuRec renders at capture time and
needs no capture tick.

`AttackRuntime.setup` runs at `simulator.setup`, before the selected capture, with the live state at
that point; its world attack effects are visible in the selected capture. NuRec needs no physical
deployment ticks or pre-roll (k=0); its closed-loop vehicle dynamics still advance during driving
decisions. A session holds no server reservation between operations. Concurrent operations on the
same server are rejected; completion, failure and cancellation release each operation's resources.
