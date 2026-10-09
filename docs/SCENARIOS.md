# Scenario selection and insertion

Select an initial case before running an experiment. A case contains the initial scene, stable
actor bindings and explicit insertion specifications. Clean and attacked runs use this same case.
Filters are not installed in the driving loop and do not promise attack success.

The workflow is:

1. Specify inserted assets and their exact world or host-local positions.
2. Combine filters into a `ScenarioRequirement`.
3. Ask a source to prepare candidates and return qualifying cases.
4. Create a backend from a selected case and run clean and attacked experiments from that origin.

Selection is a Python API. The `avsectester run` YAML command runs a configured CARLA experiment,
but does not parse the selection API described here. See [SETUP.md](SETUP.md) for dependencies.

## Interfaces

| API | Responsibility |
|---|---|
| `SceneGT`, `CameraCalib` | Normalized ground truth and camera calibration |
| `FilterContext` | Scene, selected roles, insertions, raw metadata and native provider APIs |
| `Constraint.evaluate(context)` | Return `FilterResult(status, reason)` |
| `ScenarioRequirement` | Compose conditions and select a fixed role assignment |
| `InitialWindow(frames=N)` | Require conditions on the first N prepared frames |
| `DatasetFilter`, `CarlaScenarioBuilder` | Prepare candidates and yield selected cases |
| `ScenarioInstance.make_backend()` | Create a backend starting at the selected origin |
| `Insertion`, `resolve_insertion` | Specify and resolve object placement and orientation |

A filter returns `pass`, `fail` or `unknown`. Missing visibility, missing bound actors and an
incomplete initial window produce unknown checks. Only passing cases are selected. Programming
errors propagate. All top-level constraints must pass. Use `All`, `Any` and `Not` for other
combinations:

| Combination | Result |
|---|---|
| `All(filters)` | Fail if any child fails, otherwise unknown if any is unknown, otherwise pass |
| `Any(filters)` | Pass if any child passes, otherwise unknown if any is unknown, otherwise fail |
| `Not(filter)` | Exchange pass and fail, preserving unknown |

`ScenarioRequirement.evaluate(context)` returns detailed candidate checks. `match(context)` returns
the selected `ScenarioMatch` or `None`. The match retains `binding_ids`, `insertions` and `window`.
Sources retain accepted, rejected and unknown checks in `selection_log`. `save_selection(path)`
writes these checks as JSON, without copying native clients or image arrays.

## Specify inserted objects

Positions are explicit. The framework does not search for a patch location or move an insertion
to make a filter pass. World and object coordinates use metres and right-handed axes: X forward,
Y left, Z up. The CARLA adapter converts Unreal coordinates at the boundary.

```python
import numpy as np
from avsectester.insertion import (
    AttachedPlacement, Insertion, Orientation, PlaneAsset, WorldPlacement,
)
from avsectester.attacks.object_insertion.sign_spoof import STOP_SIGN

# Replace this RGBA array with a texture path or your own array.
texture = np.full((32, 32, 4), [255, 0, 255, 255], dtype=np.uint8)

patch = Insertion(
    id="rear_patch",
    asset=PlaneAsset(texture, width_m=0.6, height_m=0.4),
    placement=AttachedPlacement(
        host="attacker", anchor="rear_center", offset_m=(-0.05, 0.0, 0.15),
    ),
    orientation=Orientation("follow_host", rotation_deg=(0, 0, 180)),
)

sign = Insertion(
    id="roadside_sign",
    asset=PlaneAsset(STOP_SIGN, width_m=0.8, height_m=0.8),
    placement=WorldPlacement(position_m=(20, -3, 1.8)),
    orientation=Orientation("face_victim"),
)
```

The sign coordinates are illustrative. `WorldPlacement` takes absolute coordinates in the
provider's world frame, not distance ahead of the ego. Choose a position in your selected map or
reconstruction. Neither placement searches for an alternative when its filters fail.

`AttachedPlacement` uses the host's bounding-box centre as its local origin. Named anchors are
`center`, `front_center`, `rear_center`, `left_center`, `right_center`, `top_center` and
`bottom_center`. Offsets use the host's local axes and rotate with it. These anchors describe
bounding-box faces, not exact vehicle body panels. Choose offsets appropriate to the actual asset.
Use `anchor="center"` with `offset_m=(x, y, z)` for a custom local point.

Position and orientation are independent:

| Orientation | Rotation behavior |
|---|---|
| `follow_host` | Host rotation multiplied by the specified local rotation |
| `fixed_world` | Absolute rotation remains constant while an attached centre follows its host |
| `face_victim` | Asset front continuously points toward the current victim position |

Angles are `(roll, pitch, yaw)` in degrees, with rotation `Rz(yaw) @ Ry(pitch) @ Rx(roll)`.
The asset front is +X and its top is +Z. A rear patch therefore normally uses local yaw 180°.
`face_victim` uses `target_offset_m` in the victim frame and keeps world up where possible.
A coincident insertion and target is invalid. `follow_host` requires an attached placement.

Each insertion has its own asset, dimensions, offset and orientation. A host can carry multiple
insertions. Multiple hosts can carry different insertions. A missing host is never replaced with
a new nearby actor. In paired runs, world orientation and host bindings remain the same, while
`face_victim` follows each run's current ego pose.

`PlaneAsset` supports RGBA textures. Transparent pixels do not belong to its silhouette. Custom
assets implement `planes()` and return local `PlaneSurface(corners, texture)` rectangles. This
supports multi-surface objects using the same resolver and compositor. Surfaces are currently
two-sided. General mesh rendering requires a provider-specific renderer and reference silhouette,
not a bounding-box substitute.

Each surface uses four `(x, y, z)` corners in top-left, top-right, bottom-right, bottom-left texture
order and a `uint8` RGBA texture. All surfaces are expressed in one asset-local frame. A custom
asset can be passed directly to `Insertion` without registration. Custom asset serialization
requires an application-defined loader, since the built-in dictionary format covers `PlaneAsset`.

## Compose a requirement

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

Automatic selection checks candidate bindings before choosing among eligible ones. A nearer car
that fails the conditions does not exclude a farther suitable car. `RoleSpec(ids=("track-17",))`
selects an explicit identity without substitution. For two hosts, use `RoleSpec(count=2)` or
`RoleSpec(ids=("track-17", "track-42"), count=2)`. Attach individual objects with role aliases
such as `hosts[0]` and `hosts[1]`. `max_bindings` bounds the number of role assignments examined.
If none of the examined bindings passes and the budget is exhausted, the result is unknown.
An accepted binding satisfies the filters, but a bounded search cannot promise the best choice
among unexamined assignments.

For insertion cases, visual filters evaluate the inserted objects only. Patch visibility does
not use the host vehicle's visibility label. With multiple insertions, all are checked unless a
filter specifies `subjects=("rear_patch",)`. Insertion requirements always check that every
insertion overlaps the victim image. `camera="front"` sets this mandatory check's camera.
Without an explicit setting, the target camera, `front`, or the sole available camera is used.
Ambiguous camera selection is unknown. Scene-only or nonvisual requirements need no visual filter.

| Filter | Measurement |
|---|---|
| `DistanceRange` | Horizontal attacker-to-victim distance in metres, inclusive bounds |
| `ViewpointRear` | Host rear direction versus host-to-camera bearing in the horizontal plane |
| `InView` | At least one opaque insertion pixel projects inside the image, independently of occlusion. Existing annotations use their projected boxes |
| `ImageAreaFrac` | Clipped projected bounding-box area divided by image area |
| `MinVisibility` | Visible fraction of the insertion, or an existing object's documented annotation |
| `EgoMoving` | Ego speed threshold |
| `ClearLaneAhead` | Existing centre-based corridor test, independent of optical occlusion |

Distance requires an `attacker` role. It never silently switches to patch-to-camera distance.
`TargetSpec` remains a shorthand for existing-object annotation queries. The named
`physical_patch_hide_vehicle` preset queries vehicle annotations. For inserted patch
selection, provide the actual patch specification as shown above.

## Select a case

The following examples use `requirement` and `patch` from above. Choose one source.

For CARLA, start a dedicated server matching your CARLA client version. The example configuration
contains an ego camera and a lead vehicle. The builder applies its own selection backend and
removes the configuration's physical `patches` so insertion visibility is measured before attack:

```python
from pathlib import Path
import yaml
from avsectester.scenarios import CarlaScenarioBuilder

config = yaml.safe_load(Path("configs/carla_patch_scenario.yaml").read_text())
# Set config["client"]["connect_port"] if your server does not use port 2000.
source = CarlaScenarioBuilder(base_scenario=config)
```

For NuRec, point the dataset at local USDZ files. Metadata-based selection does not require a
renderer connection. Running the selected backend or rendering from a custom filter does require
an `nre-ga` server serving the corresponding scene:

```python
from avsectester.scenarios import DatasetFilter
from avsectester.scenarios.datasets.nurec import NuRecDataset

dataset = NuRecDataset.from_glob(
    "/path/to/scenes/*.usdz", keyframe=0.5, endpoint="127.0.0.1:50051",
)
source = DatasetFilter(dataset)
```

Both sources use the same selection call:

```python
case = next(source.scenarios(requirement, limit=1), None)
if case is None:
    raise RuntimeError(f"No qualifying case: {source.selection_log}")

print(case.match.binding_ids)  # Role names mapped to fixed actor IDs.
print(case.selection)          # Filter decisions and reasons.
```

`limit` counts accepted cases. `case.match.scene` is the selected initial ground truth and
`case.match.insertions` holds the insertion specifications. `case.provenance` describes the
source. Preview clients are already closed when the case is returned. Call `case.make_backend()`
to construct experiment resources and close that backend when finished.

## Initial windows and providers

A window starts at the selected experiment origin and uses consecutive recorded or prescribed
states. Bindings remain fixed throughout it. If an actor disappears, the window does not select a
replacement. A source with one frame cannot establish a ten-frame condition by repeating it.
No driving-policy rollout or attack feedback is used to construct the window.
Passing an initial window does not guarantee that the same conditions persist once either
experiment starts responding to its driving model.

**NuRec** uses recorded camera timestamps and interpolated full actor poses. The dataset exposes
all labeled actors, including nonvehicle occluders. `DatasetFilter(dataset)` checks the same
requirement across `dataset.initial_sequence(scene, N)` and starts the backend at the first frame.
`NuRecDataset.from_glob(pattern, keyframe=0.5, endpoint="127.0.0.1:50051")` selects the candidate
start within each clip. It provides one candidate start per USDZ, not a sliding search over every
frame. `keyframe` is a fraction in `[0, 1]` of that camera's frame indices. To enumerate additional
starts, override `scenes()` and use `scene_at_frame(path, frame)` for each desired candidate.

**CARLA** creates a candidate world before checking its actual geometry. Fixed scene positions
are respected. `sample_scene=True` explicitly enables bounded lead-position sampling for new
scene candidates, independently of user-specified insertion locations.

```python
from dataclasses import replace
from avsectester.plane import Control
from avsectester.scenarios import CarlaScenarioBuilder
from avsectester.scenarios.carla_provider import CarlaCandidateProvider

def advance_initial(backend, index):
    # A prescribed control sequence, without querying a driving model.
    return backend.step(Control(brake=1.0))

requirement = replace(requirement, window=InitialWindow(frames=10))
provider = CarlaCandidateProvider(initial_frames=10, advance_initial=advance_initial)
source = CarlaScenarioBuilder(base_scenario=config, candidate_provider=provider)
case = next(source.scenarios(requirement, limit=1), None)
```

`advance_initial(backend, index)` advances one prescribed frame and obtains its observation.
For a one-frame requirement, the default provider needs no callback. For longer windows, provide
this callback or a custom provider. Live contexts are evaluated before advancing the world, so
custom filters see matching native state. Candidate resources close before the case is yielded.

The default `CarlaSelectionBackend` uses calibrated pinhole RGB cameras and adds matching depth.
It has a dedicated sensor setup step before returning the initial observation. Selected cases and
paired resets use this same preparation path. Ordinary `CarlaBackend` behavior is unchanged.
The selected RGB camera must produce a frame at every experiment step. Configure
`sensor_tick=0` or a period matching the simulator step. Slower camera streams do not satisfy
the same-frame insertion contract.
`analytic_preview=True` explicitly requests approximate offline enumeration. It does not establish
rendered visibility or validate an actual CARLA spawn.

Clean uses the attack-selected case with the insertion disabled. Resolve random scene choices
once per pair through `prepare_clean_attack_pair()`. Neither run selects its own preferred host,
position or start time. Trajectories may diverge after the experiment starts.

## Visibility

For known insertion geometry, visibility is:

```text
visible opaque target pixels inside the camera image
----------------------------------------------------
complete projected opaque target pixels from that viewpoint
```

The denominator includes the portion outside the image. The host remains an occluder for its
patch. Other inserted objects can also occlude one another. Projection, alpha silhouette and
pose resolution are shared between selection and rendering.

Planar assets use bilinearly sampled RGBA textures. Samples with alpha greater than 127 count
as opaque target pixels. Texture prefiltering uses the full projected surface size, including
the off-image part. Cropping the image therefore does not change the texture's sampling scale.
The built-in estimators retain these samples in `VisibilityEvidence.sampled_rgba`, and the
compositor reuses them. Harmonization changes appearance after visibility is determined.
This is a binary geometric visibility measure, not a model of transmission through translucent
materials. A target crossing the camera plane has an unavailable full projection and returns
unknown visibility.

| Provider | Evidence |
|---|---|
| CARLA | Aligned scene depth compared with insertion surface depth |
| NuRec | Ray intersections against known 3D cuboids and other insertion surfaces |
| nuScenes | Original coarse visibility labels for existing annotated objects |
| Custom | `visibility_provider(context, insertion_id, camera)` callback |

`DepthVisibilityEstimator` requires the same camera pose, lens and frame for depth and geometry.
Its depth convention is explicit: `z` or `range`. CARLA uses `z`. Invalid depth at target pixels
produces unknown. `CuboidVisibilityEstimator` reports source `cuboid_estimate`. It cannot account
for unannotated buildings, vegetation, holes in a vehicle, or precise mesh boundaries. NuRec's
estimate must not be interpreted as exact rendered occlusion.

`Visibility(fraction, source, camera=None, label=None)` retains provenance. nuScenes tokens retain
the coarse representatives `1 → 0.2`, `2 → 0.5`, `3 → 0.7`, `4 → 0.9`. They are dataset-wide labels,
not exact camera-specific measurements and not labels for a newly inserted patch.

For native complex assets, `visibility_from_masks` and `visibility_from_depth` accept a complete
unoccluded reference at the same viewpoint. A larger reference canvas must preserve focal length
in pixels. `image_bounds=(left, top, right, bottom)` restricts the numerator to the real image.
An empty reference is unknown. An opaque target entirely outside the image has visibility zero.
Fog attenuation and recognition confidence are separate from geometric visibility.

## Custom filters and native access

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

# For a CARLA requirement, append the custom filter to the same constraint list.
requirement = replace(requirement, constraints=[*requirement.constraints, AllowedWeather(20)])
```

`FilterContext` exposes `scene`, `bindings`, `insertions`, `metadata`, `dataset`, `backend`,
`renderer` and `native`. There is no method whitelist. NuRec provides raw track and rig records,
archive access and a lazy renderer. CARLA provides the live world, actors, client and aligned
depth. nuScenes provides its native SDK and source records. Custom filters can call these APIs
or query information not represented by `SceneGT`.

| Provider | Native access examples |
|---|---|
| CARLA | `context.native["world"]`, `["actors"]`, `["client"]`, `["depth"][camera]` |
| NuRec | `context.metadata` contains raw JSON records, `context.native["open_archive"]()` opens the USDZ, `["actor_poses"](timestamp_us)` interpolates actor poses |
| nuScenes | `context.native["nusc"]` exposes the SDK, `context.metadata` contains source records |

These capabilities are provider-specific. A filter that needs unavailable information should
return `FilterResult("unknown", reason)`. Filters return decisions, not modified observations.
They run on prepared initial frames, where native world state and `context.scene` agree.

Normalized geometry uses these transforms, all as 4×4 rigid matrices:

| Value | Transform |
|---|---|
| `scene.ego.pose` | Ego frame to world |
| `ObjectGT.pose` | Object bounding-box centre frame to ego |
| `CameraCalib.cam_to_ego` | Optical camera frame to ego, with X right, Y down and Z forward |
| `ActorPose.transform` | Actor bounding-box centre frame to world |

`FilterContext.actors` resolves world poses and role aliases. `context.victim` is the victim's
bounding-box centre pose. Extents are full length, width and height, not half-extents. Native APIs
retain their provider's conventions, so convert them before constructing normalized geometry.

To support another dataset, subclass `Dataset`:

| Method | Contract |
|---|---|
| `scenes()` | Yield candidate initial `SceneGT` records with stable actor IDs |
| `make_backend(scene)` | Create a fresh `WorldBackend` configured at that selected origin |
| `initial_sequence(scene, frames)` | Return consecutive prepared states beginning with `scene`, without padding a short sequence |
| `context(scene)` | Return a `FilterContext` exposing the scene and any native resources |

`scenes()` and `make_backend()` are required. The default sequence contains one frame, and the
default context exposes the dataset itself. A simulator provider instead implements
`config -> context manager[FilterContext]` for `CarlaScenarioBuilder(candidate_provider=...)`.
Provide `context.native["make_backend"]` as a factory that reconstructs the selected origin.

Use `DatasetFilter(..., prepare_context=...)` or a custom candidate provider to add resources.
A context's native resources are valid within its provider scope. Register newly created resources
with `context.own(resource)` to have their `close()` called after selection, including rejected
candidates and exceptions. Merely assigning `renderer`, `backend` or `native` does not transfer
ownership. This allows filters to borrow shared clients without closing them.

```python
def prepare_context(context):
    context.renderer = context.own(MyRenderer())  # Must provide close().
    return context

source = DatasetFilter(dataset, prepare_context=prepare_context)
```

`DatasetFilter` and `CarlaScenarioBuilder` release owned preview resources before yielding a
selected case. Derived contexts share the original resource scope. When using an adapter context
directly, use `with dataset.context(scene) as context:` or call `context.close()` in `finally`.
NuRec's context owns its lazy renderer. Selected backend factories create fresh resources and must
not capture preview clients. `register_constraint` is needed only for dictionary serialization and the optional
natural-language vocabulary. Direct Python composition does not require registration.
`requirement_to_dict` and `requirement_from_dict` support groups, roles, windows and built-in assets.

## Feed insertions to the driving model

Insertion rendering must modify the observation passed to the stack. A visualization-only overlay
does not constitute an input attack. `InsertionRenderer` resolves stable hosts each frame and
uses the same visibility evidence to mask the composite. It never reruns selection filters.

The following helper runs a selected case twice and returns both `Trace` objects. Supply a
`stack_factory()` returning a fresh `AVStack` compatible with the case's sensors. `run()` resets
the backend at the start of each run. The attack adapter is constructed after the clean run so
camera calibration is available, then follows the backend through the attacked reset:

```python
from avsectester.backend import run

def run_pair(case, stack_factory, make_perturb, frames=30):
    backend = case.make_backend()
    try:
        backend.prepare_clean_attack_pair()
        clean = run(backend, stack_factory(), frames=frames)
        perturb = make_perturb(case, backend)
        attacked = run(backend, stack_factory(), frames=frames, perturb=perturb)
        return clean, attacked
    finally:
        backend.close()
```

For a case selected by `CarlaScenarioBuilder` using `CarlaSelectionBackend`:

```python
from avsectester.simulators.carla import insertion_perturbation

def make_carla_perturb(case, backend):
    return insertion_perturbation(
        backend, case.match.insertions, bindings=case.match.binding_ids,
        camera=case.match.camera,
    )

# clean, attacked = run_pair(case, stack_factory, make_carla_perturb)
```

CARLA logical host names `lead` and `npc:<index>` survive paired resets. For a case selected from
the `NuRecDataset` instance named `dataset`:

```python
from avsectester.simulators.patch_insertion import frame_perturbation

def make_nurec_perturb(case, backend):
    insert = dataset.insertion_renderer(
        case.match.scene, backend, case.match.insertions,
        bindings=case.match.binding_ids,
    )
    return frame_perturbation(insert, camera=dataset.camera)

# clean, attacked = run_pair(case, stack_factory, make_nurec_perturb)
```

Use [component logging](COMPONENT_LOGGING.md) to capture stack outputs, or
[robustness evaluation](AUGMENTATION.md) to run selected cases across corruption conditions.

### Custom rendering adapters

Use `InsertionRenderer(insertions, camera, geometry, evidence_provider=...)`. The camera satisfies
`rendering.cameras.Camera`: `width`, `height`, `project(points)` and `unproject(pixels)`.
Projection maps `(N, 3)` optical-frame points to `(N, 2)` pixels. Unprojection returns `(N, 3)`
unit rays. Optical axes are X right, Y down, Z forward. Pixel centres are `(column + 0.5, row + 0.5)`.
An optional `max_angle` limits the lens field of view in radians. Custom cameras need not inherit
from our concrete classes.

`geometry(observation)` returns `InsertionGeometry`:

| Field | Contract |
|---|---|
| `actors` | Mapping from stable IDs or bound role aliases to world-frame `ActorPose` objects |
| `victim` | Victim `ActorPose` in the same world frame |
| `cam_from_world` | `4×4` transform from world coordinates to optical camera coordinates |

All geometry belongs to the current observation. Distances use metres. This callback updates
placement during a drive and does not run selection filters.

In this example, implement `read_native_geometry` using your simulator or dataset's native API:

```python
from avsectester.rendering.types import InsertionGeometry
from avsectester.rendering.visibility import CuboidVisibilityEstimator
from avsectester.simulators.patch_insertion import InsertionRenderer

def geometry(observation) -> InsertionGeometry:
    actors, victim, cam_from_world = read_native_geometry(observation)
    return InsertionGeometry(actors=actors, victim=victim, cam_from_world=cam_from_world)

estimator = CuboidVisibilityEstimator()

def evidence_provider(observation, resolved, geometry):
    return {
        item.id: estimator.estimate(
            item, camera, geometry.cam_from_world,
            occluders=geometry.actors, other_insertions=resolved, camera_name="front",
        )
        for item in resolved
    }

insert = InsertionRenderer(insertions, camera, geometry, evidence_provider=evidence_provider)
```

The evidence callback returns `VisibilityEvidence` keyed by every insertion ID. Masks are
image-sized, and `target_depth` uses optical-axis Z in metres. The data must match the current
camera, pose and frame. Custom providers may omit `sampled_rgba`, in which case composition
uses the shared surface sampler. Unknown visibility raises an error during composition rather
than painting an insertion with missing evidence. A scalar `Visibility` can answer a filter but
cannot mask a rendered image.

`GeometryProvider`, `EvidenceProvider` and `GeometryVisibilityEstimator` in `rendering.types`
describe these extension contracts. They do not restrict access to native SDKs or metadata.
Without an evidence provider, only surfaces within each insertion occlude one another.
`render_resolved` supports direct composition with the same contracts.

### Compatibility

Use `case.match` for the selected `ScenarioMatch`. `case.target` and the `target=` constructor
keyword remain compatibility aliases. `case.match.target` is the optional single-object target,
not the whole selected case. `InsertionGeometry` supports tuple unpacking, and geometry callbacks
returning `(actors, victim, cam_from_world)` remain supported.

Filters are defined in `scenarios.filters`. Camera models, visibility estimators and harmonizers
are defined in `rendering.cameras`, `rendering.visibility` and `rendering.harmonizers`.
Their previous imports through `scenarios.requirement`, `scenarios.estimators`,
`scenarios.visibility`, `simulators.camera_models` and `simulators.patch_insertion` remain available.
