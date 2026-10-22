# Datasets and attack segments

> **Audience:** Attack and dataset authors.
>
> **Outcome:** Read scene timelines and select explicit decision intervals.
>
> **Prerequisites:** [Architecture](../concepts/architecture.md).

## Dataset interface

Dataset provides `scenarios()`, `frames(scenario_id)`, `frame_data(scenario_id, frame)`,
`native(scenario_id)`, `scenario_start(scenario_id, frame)` and `scenario_map(scenario_id)`.
It never creates or owns a simulator. Returned mutable data is independent of cached dataset reads.

| Type | Meaning |
|---|---|
| ScenarioDescriptor | Stable id, dataset class name and provider payload |
| FrameReference | Scenario id, timeline position, native frame id and timestamp |
| FrameData | One frame's ego, actors, recorded sensor references and provenance |
| ActorState | Canonical world pose, velocity, category, full extent and local box center |
| SensorDataReference | Sensor name, data location, timestamp and decoding metadata |
| ScenarioStart | Source payload needed to reset the simulator at the chosen position |
| Unavailable | Requested evidence is unavailable, with an explicit reason |

`ActorState.pose` locates the actor/rig origin; box_center is the local bounding-box offset. The
world box center is pose applied to box_center. Extent is full length, width and height in metres.
World axes are right-handed x forward, y left, z up.

## CARLA recipes

CarlaDataset holds named CarlaRecipe records. A recipe defines map, ego and other actor spawns,
sensors, time_step, decision_steps and prescribed actor motion. The simulator's connection, tick
rate and synchronous settings are separate runtime configuration.

Each actor's PrescribedMotion chooses exactly one of constant_control, a per-decision control
sequence or an N+1 ActorState trajectory. Traffic Manager policies are not recorded motion. The
recipe horizon covers N decisions and the terminal state. Formal execution releases ego from
prescribed motion for AVStack commands; frame data generation follows the entire recipe.

The dataset returns Unavailable for generated frame data. Experiment preparation uses the simulator
to generate and cache it; it does not attach generated data to the Dataset.

## NuRec

NuRecDataset reads USDZ records. Configure camera_id for the recorded camera and camera_alias for
its observation name. Scene identifiers are native clip identifiers, available through
`dataset.scene_id(path)`; they are not filesystem paths.

The timeline starts at the first supported timestamp and advances at time_step. For a 0.1 s
experiment, positions count 0.1 s decisions even when source cameras were recorded at 30 Hz. Ego and
actor trajectories are interpolated at requested times without extrapolation. An actor outside its
recorded support is absent. Use camera_timestamps(path) for recorded video-frame times and
camera_geometry(path) for the optical camera model and rig-from-camera transform. These timestamps
differ from the decision timeline and must be used when overlaying video labels.
The camera model retains the archive's native forward and inverse lens polynomials. No inverse is
fitted: if the source supplies none, pixeldist_to_angle stays empty and ray unprojection raises
NotImplementedError.

The simulator decision time step must match the dataset before a world starts. Provenance records
the time step and interpolation.

NuRecDataset.scene_mesh(scenario_id) delegates to the public read_scene_mesh(usdz_path) function in
datasets/nurec.py. It merges mesh.ply with mesh_ground.ply into world-coordinate vertices and
triangular faces, without selecting a camera or reading camera calibration. The simulator uses
the same reader. Static mesh depth is a simulator option, not an attack-specific data source; see
[Mesh depth](simulators.md#mesh-depth).

## nuScenes recorded playback

NuScenesDataset exposes scene tokens, keyframe timelines and stable instance actor IDs across
frames. It retains all annotated categories and full extents. RecordedReplaySimulator decodes sensor
references and advances through frames; driving commands do not change recorded motion. A window of
N decisions needs N+1 available frames, including the terminal frame.

## Semantic maps

`Dataset.scenario_map(scenario_id)` returns ScenarioMap or Unavailable(reason). The base method
returns Unavailable("This dataset provides no map"). Selection uses the same interface through
`ScenarioView.scenario_map()`. A missing map does not prevent other dataset operations; attacks
that require it decide how to record or handle the missing evidence.

The public types are exported by avsectester.maps. All four records are frozen dataclasses.
Geometry uses the same canonical right-handed world frame as ActorState, in metres. Array inputs
are copied on construction and made read-only; a caller cannot change map geometry by modifying
an input array.

| Record | Geometry and attributes |
|---|---|
| ScenarioMap | Lanes, drivable polygons, road boundaries, crosswalks, intersections, traffic controls, stop lines and unavailable_layers |
| Lane | N×3 centerline and boundaries; lane type, predecessor/successor and neighbor IDs, optional speed limit in m/s |
| TrafficControl | ID, kind, native category, world pose and its reference point, full thickness/width/height, lane IDs and optional stop-line ID |
| StopLine | ID, N×3 points, native category, kind, lane IDs and optional control ID |

TrafficControl.pose has +X pointing out of the control's front face, toward approaching traffic.
extent is full (thickness, width, height) along the local X/Y/Z axes. The required pose_reference
states what the pose origin is:

| Source and control | pose_reference | Pose origin and extent |
|---|---|---|
| NuRec signs and lights | face_center | Center of the labeled box; extent from its dimensions |
| CARLA landmarks with positive height, except traffic lights | face_center | Signal position raised by half the face height; thickness unavailable |
| CARLA traffic lights | signal_position | OpenDRIVE signal position; it does not locate the light head and can lie below the road |
| CARLA landmarks with zero height (speed limits) | signal_position | OpenDRIVE signal position; the source gives no face size |

Only a face_center pose has its extent centered on the origin; place assets relative to
signal_position poses only with additional evidence. Lane types are
driving, shoulder, bike, parking or other. Traffic controls distinguish stop_sign, yield_sign,
speed_limit, traffic_light and other. Stop-line kinds are stop_sign, traffic_light, yield_sign and
unknown. STOP_LINE_KIND_BY_CONTROL_KIND, exported from avsectester.maps, maps the three stopping-control kinds to
their matching stop-line kinds; unrecognized control associations leave unknown.
StopLine.native_category preserves the source paint or reference category independently of that
derived kind. unavailable_layers maps missing evidence to reasons; it does not fill gaps with
invented geometry or associations. A key is one of:

- a layer name (lanes, drivable_area, road_boundaries, crosswalks, intersections,
  traffic_controls, stop_lines): the whole layer is unavailable;
- `<layer>.partial`: some records of the layer could not be read;
- `<layer>.<record field>`, such as lanes.speed_limit_mps or traffic_controls.extent: that field
  is unavailable for some records;
- `traffic_controls.<kind>`, such as traffic_controls.traffic_light: the source has no valid
  records of that control kind.

Other keys raise ValueError.

### Data-source coverage

| Provider | Map source and coverage |
|---|---|
| CARLA | Exported OpenDRIVE: lane geometry/connectivity, driving-lane polygons, landmarks, crosswalks and generated stop lines |
| NuRec | USDZ clipgt tables: lane rails, drivable_space, road_boundary, crosswalk, intersection_area, traffic_sign, traffic_light and wait_line, where present |
| nuScenes | Unavailable; map adaptation is not implemented |

CarlaDataset's optional map_directory contains `<map_name>.xodr` files. Missing configuration,
files or the carla package produces Unavailable with instructions for exporting maps. See
[Export CARLA maps](../getting-started/installation.md#export-carla-maps). Reading an exported map
uses the CARLA client package's parser without connecting to a server.

OpenDRIVE samples every lane type at intervals of at most two metres, including lanes absent from
the native driving-lane generator. The same maximum spacing applies to the centerline and both
boundaries, accounting for the longer outer edge on curves. Boundaries use each waypoint's lane
width. Continuous segments are refined through native midpoint queries. At a source discontinuity
the SDK cannot refine further, collinear points subdivide the existing straight polyline connector
while preserving its shape and native endpoints. Left and right
neighbors refer only to lanes traveling in the same direction; a neighboring opposing lane is not
recorded as a neighbor. The drivable area is the union of the driving-lane polygons. Each
OpenDRIVE junction with driving lanes gives one intersection polygon, the convex hull of those
lanes, matching NuRec's one intersection_area polygon per intersection. CARLA coordinates are
converted by flipping the y axis.

Landmark types 206, 205, 274 and 1000001 map to stop signs, yield signs, speed limits and traffic
lights. Each signal reference is resolved on its own road and governed lanes. A separate stop line
is generated for each reference and controlled lane, spanning that lane's width perpendicular to
its local heading. Its ID is `stop-line:<signal_id>:<road_id>:<lane_id>`. The TrafficControl's lane_ids
contains the union across references. stop_line_id is filled only when exactly one generated line
belongs to the control; with several lines, use each StopLine.control_id to find them. These lines
approximate the reference's longitudinal position; they are not surveyed painted road markings.
Their native_category is opendrive_signal_reference.
For CARLA non-light landmarks with positive height, the signal position is shifted upward by half
the height along world Z and pose_reference is face_center. Traffic lights and zero-height
landmarks retain the signal position with pose_reference=signal_position. Landmark orientation
faces against the mean normalized governed-lane direction when its horizontal magnitude is at least
0.5. Otherwise, it uses the converted landmark heading rotated 180° about local Z. CARLA extent is
(0, width, height); traffic_controls.extent reports "CARLA landmarks provide width and height but no
thickness". Unconfirmed lane speed limits are
reported under lanes.speed_limit_mps rather than inferred from an unlabeled number.

NuRec keeps the recorded left/right rails as lane boundaries. To form a centerline, it resamples
both rails at matching normalized arc-length fractions, using the larger rail's point count, then
takes the midpoint of each pair. Different point counts therefore do not misalign the centerline.
Association records supply lane connectivity and control/stop-line relations when the source
identifies them. Absent or ambiguous associations are not guessed. Lane types follow these rules,
in order:

| NuRec use_types / vehicle_types | Lane.lane_type |
|---|---|
| use_types contains SHOULDER_LANE | shoulder |
| use_types contains PARKING_LANE | parking |
| vehicle_types contains CAR (including CAR/BICYCLE, HOV_LANE and SERVICE_ROAD lanes) | driving |
| vehicle_types is only BICYCLE | bike |
| Anything else, such as bus-only lanes | other |

NuRecDataset.usdz_path(scenario_id) resolves a
scene ID to its archive path. NuRec controls preserve the raw center and XYZ dimensions as
pose_reference=face_center and extent=(thickness, width, height); no reference-point shift is applied.

Traffic-sign categories use exact complete names; native_category always retains the source text:

| NuRec native category | TrafficControl.kind |
|---|---|
| TRAFFIC_SIGN_REGULATORY_R1_STOP | stop_sign |
| TRAFFIC_SIGN_REGULATORY_R1_YIELD | yield_sign |
| TRAFFIC_SIGN_REGULATORY_R2_SPEED_LIMIT | speed_limit |
| Every other category | other |

For example, NO_STOPPING and STOP_HERE_ON_RED signs are not stop signs, and a sign asking turning
vehicles to yield to pedestrians is not a yield sign. NuRec retains only wait-line rows explicitly
marked is_implicit=False. An archive whose wait lines are all implicit has a known empty stop_lines
layer, rather than unavailable evidence. native_category retains the original category, such as
STOP; source subtypes such as ENTRY, NOT_APPLICABLE and CROSSWALK_ENTRY are not categories or filters.

A wait line's kind comes only from control and lane associations. It is not inferred from paint
category or nearest geometry. Multiple associated controls of one recognized kind can determine
the line's kind; control_id is filled only for a unique associated control. Missing or ambiguous
associations leave kind=unknown. A source with many such lines can consequently have a high unknown
count even when its geometry and paint labels are available. A STOP paint category does not
establish stop_sign control. A selection that needs the controlling sign or light of an unknown line
can compare the line with nearby TrafficControl poses and their lane_ids itself.

A canonical layer with no usable geometry appears in unavailable_layers, whether its source tables
are missing or contain only placeholder rows. An empty traffic_light placeholder supplies no evidence
about whether traffic lights exist, so it is reported under traffic_controls.traffic_light even when
valid traffic signs keep the traffic_controls layer nonempty. Source table names such as
drivable_space, road_boundary, traffic_sign and wait_line are not keys. Inspect these reasons
instead of assuming that every USDZ supplies every layer.

clipgt geometry uses the recorded rig trajectory's world frame. For archives without clipgt,
map.xodr can only be used after its transform into that frame has been established. A geographic
reference alone is insufficient; an unconfirmed transform produces Unavailable instead of a
silently misplaced map.

### Custom datasets and saved results

A custom Dataset can implement scenario_map and construct the four shared map records. For an
OpenDRIVE source already aligned with the dataset's canonical world frame, reuse
`read_opendrive(xodr_text, name=scenario_id)` from avsectester.maps.opendrive. It returns
ScenarioMap or Unavailable when the optional carla package is absent.

The online attack context does not expose maps. Copy only the map information your attack needs
into the selected segment's JSON-like or numpy user_data. This change does not add map input to
the AVStack or runner. Dynamic signal states belong to WorldSnapshot.traffic_lights: red, yellow,
green, off or unknown, keyed by TrafficControl.id. CARLA uses each signal's OpenDRIVE ID; NuRec and
recorded replay currently return an empty mapping.

Experiment.execute reads each selected scene's map once. With an output directory it saves maps
under `maps/<scenario_id>.json`, using the prepared-metadata encoding, including Unavailable and
its reason. In-memory execution retains them in ExperimentResult.scenario_maps. ExperimentResult.load
restores the saved maps, and evaluation passes the relevant value as ComparisonGroup.scenario_map.
Evaluation never calls Dataset to obtain a map.

## Select segments

Attack authors implement build_metadata and decide their own eligibility criteria. Read
`ScenarioView.frame_data` for actor states, use service sensor_data/render for visual evidence, and
attach user explanations through record_reason.

`split_segments` only partitions consecutive eligible decision frames. It does not choose hosts,
score visibility or impose fixed defaults. N decisions span `[start, stop)` and require the
reference at stop. Adjacent segments share terminal/input boundary states without repeating a
decision. Keep or discard short tails explicitly.

AttackSegment contains scenario_id, start, stop and user_data. Store selected actor IDs, placement
parameters, scores and ranks as attack-specific user_data using JSON-like values or numpy arrays.
The framework checks segment boundaries, while the attack checks actors when it uses them. The
framework does not rerun selection under each seed or after optimization.

## Specify inserted objects

Placement types belong to the core attack/placement.py module, not to Dataset. The attack/insertion/
package provides reusable framework tools for assets, visibility and compositing. A sensor view
supplies camera projection, calibration, canonical actor poses and available depth. The attack's
SensorAttackAction uses those inputs for insertion rendering. Physical objects use
WorldAttackAction.

See [Image insertion](../examples/image-insertion.md) and [Physical
patch](../examples/physical-patch.md).

## Next steps

Read [Attacks](attacks.md), [Simulators](simulators.md) and [Python
API](../reference/python-api.md#dataset-api).
