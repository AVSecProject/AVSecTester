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

## Configured scenarios

`scenario_from_config` turns an experiment mapping without a `selection` key into one
`ScenarioInstance`. It does not open a simulator, render images or evaluate filters. The backend
is created when the scenario runs. This example runs offline with the NuRec stub renderer:

<!-- runnable -->
```python
from avsectester.backend import AVStack
from avsectester.plane import Control
from avsectester.runner import run_prepared
from avsectester.scenarios import scenario_from_config


class Cruise(AVStack):
    """A minimal driving system that always applies light throttle."""

    def __call__(self, observation):
        return Control(throttle=0.3)


config = {"backend": {"type": "NuRecBackend", "renderer": "stub"}}
scenario = scenario_from_config(config, provenance={"config_path": "inline"})
print(scenario.match)       # None: no selection, so no roles or insertions.
print(scenario.provenance)  # Supplied provenance plus the resolved conditions.

results = run_prepared(scenario, Cruise, frames=10)
print(results["clean"].provenance["treatment"])  # clean
```

`backend.type` selects `CarlaBackend` (the default) or `NuRecBackend`; see
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

# scenario is a ScenarioInstance whose factory returns a NuRecBackend.
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

Candidates use `CarlaSelectionBackend`. It replaces stock RGB cameras with undistorted
`PinholeRgbCamera` sensors, adds an aligned depth camera and advances one stationary setup frame
before the first observation. The selected scenario's factory uses the same backend, so paired
resets match the selection. The selected RGB camera must produce a frame at every driving step:
set `sensor_tick` to 0 or to the simulator step. `analytic_preview=True` enumerates approximate
scenes offline instead. It validates no CARLA spawn and measures no rendered visibility.

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
`avsectester.scenarios.datasets.nuscenes` selects nuScenes keyframes through the devkit. Its
backend serves the one recorded image at every step and ignores commands.

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
from avsectester.plane import Control
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

## Support another dataset or provider

Subclass `Dataset` to select from another recorded source:

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
from avsectester.simulators.nurec import NuRecBackend, StubRenderer

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
        return NuRecBackend({"camera_aliases": {"camera_front": "front"}}, renderer=StubRenderer())


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
