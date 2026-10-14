# TODO

This file tracks deferred development work. Unchecked entries are not implemented features.

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
  induced stops. Keep this work separate from the current interface changes.

## CLI run modes

- [x] Support single and paired runs, with clean, attack, defense and combined treatments.
  Empty default attack comparisons explicitly compare baselines. Missing methods in single
  attack runs or defense/combined modes are configuration errors. Runtime hooks, native attacks,
  physical panels and prepared methods share the execution path.
- [ ] Add configured and CLI three-arm comparisons (clean, attack, attack+defense) from one
  origin to measure defense effectiveness against an attack-only reference. `run_prepared`
  already supports named multi-treatment runs. Keep intervention preparation reusable and
  world/sensor conditions shared across all three treatments.

## Unify insertion geometry

Deferred while development focuses on other work.

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
  existing-object annotation queries under accurate names. Visual conditions must test the
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

- [ ] Make other actors in `NuRecBackend` react to the ego. Actors currently follow the recorded
  tracks because the adapter uses only the NuRec renderer, without AlpaSim's runtime, controller,
  physics or traffic services. Evaluate AlpaSim's traffic service or add a traffic model, and
  confirm the renderer accepts per-frame actor poses.

## Unified scenario preparation path

- [x] Add `ConfiguredScenarioSource` for unfiltered configurations, without preview, inferred
  roles or constraints. It yields a `ScenarioInstance` with `match=None`.
- [x] Route CLI, CARLA convenience functions, driving demos and robustness evaluation through
  prepared scenarios and `run_prepared`. Single and comparison runs share execution and cleanup.
  Direct `run` and `run_pair` remain public for callers managing their own assembly.
- [x] Separate scene configuration from model/intervention settings. Default Runtime cases receive
  the optional selected match, while explicit Python cases remain supported.
- [x] Connect NuRec configuration to the CLI. Choose its renderer and command-compatible dynamics
  explicitly. Recorded dataset selection remains a Python source API.

## Scenario conditions

- [x] Carry `ScenarioConditions` in each prepared scenario. Apply shared world settings before
  reset and fresh sensor handlers to every treatment, with independent condition RNG streams.
- [x] Support CARLA weather and image corruptions. Custom sensor-condition factories use the
  existing handler/Plugin contract. Backends reject world settings they cannot apply.
- [x] Distinguish recorded properties selected by filters from imposed conditions. Custom filters
  can read conditioned measurements when providers supply aligned observations.
- [x] Define before/after ordering relative to same-stage interventions and record conditions,
  options and seeds in trace provenance. Robustness evaluation uses this shared mechanism.

Traffic-density controls and non-image condition algorithms are not supplied as built-ins.
Custom implementations can enter through supported world adapters or sensor-condition factories.

## Intervention preparation interface

- [x] Add `Intervention.prepare(scenario: ScenarioInstance) -> artifacts` and
  `hooks(artifacts) -> Sequence[Hook]` for attacks and defenses. Preparation receives the whole
  scenario and its optional match. It does not require an active formal backend.
- [x] Preserve Runtime, Hook, Plugin, stage names and handler signatures. Raw Runtime factory
  entries still accept arbitrary handlers without an Intervention class.
- [x] Add explicit artifact export/import and JSON save/load with provenance. Users control
  reuse across cases, conditions and runs. No optimization or automatic cache is introduced.
- [x] Adapt `attacks:` and `patches:` into prepared methods while retaining their YAML form.
  `PhysicalPatchPlugin` resolves roles from `context.case.binding_ids` or explicit logical actors.
- [x] Prepare enabled configured methods once per scenario. Robustness evaluation supports this
  interface and retains its legacy backend-dependent callback as an adapter.
- [x] Preserve explicit stage order across attacks and defenses. Convenience entries precede
  raw runtime hooks, followed by the new intervention list in its definition order.

## Adaptive attacks

- [x] Add `SpeedFeedbackAttack` as a small reference method using model-visible speed and
  post-step ground-truth feedback. Each run resets its bias/history independently. It illustrates
  the API, not a validated physical attack or a reactive actor policy.
- [ ] Add a reference adaptive attack that estimates other actors from information available
  during the run and changes its behavior accordingly. `SpeedFeedbackAttack` does not satisfy
  this item: it estimates no other actor and reads `context.ground_truth` for feedback. The
  reference attack must not read other actors from `context.ground_truth`, CARLA native handles
  or `carla_scene_gt`. It must reset per-run state, record its decisions with
  `context.record_output`, and have offline tests for its decisions and reset behavior.
  This is an example method that uses the existing interfaces (`Intervention`, `Plugin.reset`,
  `Plugin.after_step`, `StageContext`, `record_output`). It requires no framework change.
  Proposed design: a timing-triggered phantom for the CARLA modular path, registered at
  `perception.post`. It estimates the in-lane lead vehicle's distance and closing speed from the
  perception output across steps, which assumes gray-box access to that output. It injects a
  phantom only while the estimated time to collision is within a configured interval, in
  contrast to the static `PhantomInjection`. `prepare` validates its options, and `hooks`
  creates a fresh plugin for each run. Deliver it with a YAML example and a section in
  `INTERVENTIONS.md`. The NuRec/Alpamayo path has no perception stage. Supporting it would require
  the attack to run its own detector on camera images, which is out of scope for this item.

## Intervention outputs in run records

- [x] Add `StageContext.record_output(name, value)` and independent `InterventionOutput` snapshots
  in `Trace.outputs`, with step, stage, sensor, frame and timestamp. Attack, defense and condition
  handlers use the same output API. Defense algorithms and output-specific metrics remain deferred.

## Documentation accuracy

- [x] Correct backend descriptions and visualization references. Describe driving outputs as
  commands, distinguishing actuator and trajectory modes. Document actual backend compatibility
  and keep NuRec's recorded traffic and ideal follower limits explicit.
