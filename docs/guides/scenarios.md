# Datasets and attack segments

> **Audience:** Attack and dataset authors.
>
> **Outcome:** Read scene timelines and select explicit decision intervals.
>
> **Prerequisites:** [Architecture](../concepts/architecture.md).

## Dataset interface

Dataset provides `scenarios()`, `frames(scenario_id)`, `frame_data(scenario_id, frame)`,
`native(scenario_id)` and `scenario_start(scenario_id, frame)`. It never creates or owns a
simulator. Returned mutable data is independent of cached dataset reads.

| Type | Meaning |
|---|---|
| ScenarioDescriptor | Stable id, dataset class name and provider payload |
| FrameReference | Scenario id, timeline position, native frame id and timestamp |
| FrameData | One frame's ego, actors, recorded sensor references and provenance |
| ActorState | Canonical world pose, velocity, category, full extent and local box center |
| SensorDataReference | Sensor name, data location, timestamp and decoding metadata |
| ScenarioStart | Source payload needed to reset the simulator at the chosen position |
| Unavailable | Data requires generation rather than a recorded dataset read |

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

The simulator decision time step must match the dataset before a world starts. Provenance records
the time step and interpolation.

## nuScenes recorded playback

NuScenesDataset exposes scene tokens, keyframe timelines and stable instance actor IDs across
frames. It retains all annotated categories and full extents. RecordedReplaySimulator decodes sensor
references and advances through frames; driving commands do not change recorded motion. A window of
N decisions needs N+1 available frames, including the terminal frame.

## Select segments

Attack authors implement build_metadata and decide their own eligibility criteria. Read
`ScenarioView.frame_data` for actor states, use service sensor_data/render for visual evidence, and
attach user explanations through record_reason.

`split_segments` only partitions consecutive eligible decision frames. It does not choose hosts,
score visibility or impose fixed defaults. N decisions span `[start, stop)` and require the
reference at stop. Adjacent segments share terminal/input boundary states without repeating a
decision. Keep or discard short tails explicitly.

Roles bind attack-defined names to stable actor IDs. Bind zero or many actors, and include a
placement only when the attack needs one. Optional selection scores and ranks remain user-defined.
The framework checks metadata structure and actor existence; it does not rerun selection under each
seed or after optimization.

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
