# TODO

This file lists open development work. Entries describe planned capabilities, not implemented features.

## Driving-model input fidelity

- [ ] Accumulate the state history actually received by `AlpamayoAVStack` instead of rebuilding
  every history from the current pose and constant speed. Keep model-visible estimates separate
  from physical ground truth so localization attacks retain their temporal effects. Define
  startup padding and per-run reset behavior. Review the fixed acceleration, straight command
  and absent route defaults when suitable input data is available.

## Vehicle dynamics

- [ ] Add a trajectory-to-actuator controller with a dynamics model for the NuRec/Alpamayo path.
  Retain `TrajectoryFollower` as an explicitly selectable ideal follower. Document the limits
  of each execution model so ideal trajectory execution is not interpreted as realistic vehicle
  motion.

## Driving-impact metrics

- [ ] Define and detect induced stopping events across a run, including a stop followed by
  resumed driving. Specify the prior-motion condition, stopping threshold and minimum duration,
  and compare the corresponding clean time window. Do not count initial stationary frames as
  induced stops.
- [ ] Allow experiments to select the success criterion used by `evaluate_robustness`, rather
  than always scoring induced or suppressed stops through `metric.impact`.

## CLI run modes

- [ ] Add configured and CLI three-arm comparisons (clean, attack, attack+defense) from one
  origin to measure defense effectiveness against an attack-only reference. `run_prepared`
  already supports named multi-treatment runs. Keep intervention preparation reusable and
  world/sensor conditions shared across all three treatments.

## Unify insertion geometry

- [ ] Replace detector-box placement in `scripts/optimization/optimize_patch.py` and the
  axis-aligned rendered-footprint approximation in `optimize_patch_physical.py`. Derive
  optimization placement from the same `Insertion`, host pose, `resolve_insertion` and camera
  geometry used for deployment. Preserve the optimization algorithms and differentiability.
  Match visibility, texture sampling and camera distortion where applicable. Projecting four
  corners alone does not cover distorted cameras or occlusion.
- [ ] Update `attacks/optim/geometry.py`, `PatchPerturbation`, sample-context documentation and
  tests to use the shared geometry. Remove `rear_panel_homography`. Retain mathematical
  homography utilities only where needed. Keep scorer regions separate from patch placement.
  For vehicle-hiding attacks, derive the scoring region from the bound vehicle's geometry,
  rather than an unrelated highest-scoring detection or the patch's own region.
- [ ] Replace the misleading `physical_patch_hide_vehicle` annotation-only preset with an
  insertion-aware requirement builder accepting the user's patch specification. Preserve
  existing-object annotation queries under accurate names. Visual filters must test the
  inserted patch, not substitute its host's visibility. The current filters already support
  insertion subjects when an `Insertion` is supplied.
- [ ] Consolidate `PlacementCandidate`, `SceneGT.placements` and `TargetSpec(kind="placement")`
  into the explicit `Insertion` path. Update their callers, exports, documentation and tests
  while retaining custom-filter access to scene and provider data.
- [ ] Remove the unused `rendering/compositor.py::warp_patch` wrapper and its dedicated test
  and documentation references. Keep arbitrary image modifications available through runtime
  handlers. Preserve 2D boxes used for scoring, driving policies, annotation queries,
  visualization and review-image cropping.

## Reactive traffic in NuRec

- [ ] Make other actors in `NuRecSimulator` react to the ego. Actors currently follow the recorded
  tracks because the adapter uses only the NuRec renderer, without AlpaSim's runtime, controller,
  physics or traffic services. Evaluate AlpaSim's traffic service or add a traffic model, and
  confirm the renderer accepts per-frame actor poses.

## Actor-aware adaptive attack (TTC phantom)

- [ ] Add a reference adaptive attack that estimates other actors from information available
  during the run and changes its behavior accordingly. `SpeedFeedbackAttack` does not satisfy
  this item: it estimates no other actor. This example must estimate other actors from model
  outputs rather than backend truth (`context.native`, other CARLA handles or `carla_scene_gt`).
  `context.ground_truth` contains only the ego snapshot. The method must reset per-run state,
  record its decisions with `context.record_output`, and have offline tests for its decisions
  and reset behavior.
  This is an example method that uses the existing interfaces (`Intervention`, `Plugin.reset`,
  `Plugin.after_step`, `StageContext`, `record_output`). It requires no framework change.
  Proposed design: a timing-triggered phantom for the CARLA modular path, registered at
  `perception.post`. It estimates the in-lane lead vehicle's distance and closing speed from the
  perception output across steps, which assumes gray-box access to that output. It injects a
  phantom only while the estimated time to collision is within a configured interval, in
  contrast to the static `PhantomInjection`. `prepare` validates its options, and `hooks`
  creates a fresh plugin for each run. Deliver it with a YAML example and a page in
  `docs/examples/`. The NuRec/Alpamayo path has no perception stage. Supporting it would require
  the attack to run its own detector on camera images, which is out of scope for this item.

## Defense examples and output metrics

- [ ] Add reference defense algorithms using the existing Intervention, Hook and Plugin contracts.
- [ ] Add evaluation metrics for recorded intervention outputs, such as defense alarms.
