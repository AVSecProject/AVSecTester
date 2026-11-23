# TODO

This page lists capabilities to implement.

## Recorded camera phase

Initialize camera clocks from each recorded camera's phase rather than ending every first frame
at the run origin. Preserve AlpaSim's frame intervals and exposure windows.

## Decisions across dataset grid points

Support a decision period spanning several dataset grid intervals with an explicit decision
stride. The current decision period must equal the dataset time grid. Segment boundaries,
decision counts, selection tools and frame-data caching retain their current meaning.

## Reactive traffic

- [ ] Fit AlpaSim traffic actors to the selected ground mesh instead of the height returned by
  the traffic service.
- [ ] Save and restore AlpaSim traffic sessions, including their sampling state, in simulator
  checkpoints.
- [ ] Draw CATK samples from a per-session generator seeded by the run seed. The AlpaSim service
  currently ignores the seed and shares one random state across batched sessions.

See [reactive traffic](../guides/simulators.md#reactive-traffic) for supported behavior.

## Actor-aware adaptive attack

- [ ] Supply a reference attack that adapts during a run to other actors estimated from
  perception outputs, such as a phantom timed by time to collision.

## Defenses

- [ ] Define defense treatments, their lifecycle and the comparison of unattacked, attacked and
  attacked-with-defense runs.
- [ ] Supply reference defenses and defense-specific output metrics.

## Inserted objects in non-camera sensors

- [ ] Support scene insertion in CARLA.
- [ ] Include inserted `ObjectAsset` geometry in non-camera captures with composite insertion.

NuRec scene insertion includes LiDAR. See [Simulators](../guides/simulators.md#scene-insertion).

## Ground and traffic interaction for inserted objects

- [ ] Fit inserted object poses to the ground surface.
- [ ] Make inserted objects visible to the traffic model.

## Remove real objects from reconstructed scenes

- [ ] Remove or replace real objects in reconstructed scenes, instead of only inserting assets.

## Mesh composite insertion

- [ ] Composite mesh assets into camera images and support white-box texture optimization.

## CARLA mesh insertion

- [ ] Insert user-supplied mesh assets in CARLA.

## FBX and USD assets

- [ ] Load FBX and USD mesh files.

## Gaussian assets

- [ ] Load 3DGS Gaussian PLY assets without interpreting them as triangle meshes.

## Shape optimization of inserted objects

- [ ] Optimize the geometry of inserted objects in white-box attacks. Differentiable composition
  covers surface colors only, and NRE returns no gradients. Camera attacks need an attack-side
  differentiable renderer, and LiDAR attacks need a differentiable LiDAR surrogate.

## Localization attack reference frames

- [ ] Establish the scope of localization attacks when CARLA/avstack perception uses physical
  sensor reference chains while tracking and planning use model-visible `vehicle_state`.

See [physical and model-visible records](../reference/runtime.md#physical-and-model-visible-records).

## NuRec LiDAR simulation

- [ ] Simulate LiDAR intensity for scenes that do not provide it.
- [ ] Support no-return-ray simulation and its filtering controls.

See [NuRec LiDAR](../guides/simulators.md#lidar) for supported capture behavior.

## NuRec offline LiDAR sampling

- [ ] Offline `session.render` at dataset frame 0 fails when NuRec collects LiDAR because the
  scan needs a preceding recorded decision step. The stop-sign demo's `d_max` search includes
  that frame, so configurations with LiDAR cannot complete its offline optimization.

## NuRec checkpoint restoration with active render hooks

- [ ] With an activated `HookSet` session containing a `render.post` handler, restoring a NuRec
  checkpoint after reset, checkpoint creation and a simulator step raises
  `RuntimeError('Reset the simulator before requesting its state')`. This prevents
  checkpoint-based reruns while those render hooks are active.

## CARLA physical patches

- [ ] Scale the CARLA patch prop to the declared `width_m` and `height_m`, or reject dimensions
  that differ from its mesh.
- [ ] Validate CARLA's texture sampling on the prop against the differentiable optimization
  model.
- [ ] Account for the default IronPlank mesh's right-edge UV cropping in the patch optimization
  model and evaluation, using the region actually displayed.

## CARLA 1.0 validation

- [ ] Validate detector performance on CARLA 1.0 camera and LiDAR data before using it to
  compare attack impact. Retrain the detector if needed.
- [ ] Change CI's Python client pin from `carla==0.9.16` to `carla==1.0` when CARLA 1.0 is
  published on PyPI.

## Sim-to-real fidelity

- [ ] Composite photographs of printed stop-sign patches into NuRec scenes and compare detection
  and driving results with digitally composited patches.

## Closed-loop rollouts on recorded datasets

- [ ] Reconstruct recorded datasets such as nuScenes into NuRec scenes that NRE can render, so
  they run closed-loop rollouts instead of recorded playback.
- [ ] Support reconstruction engines other than NRE for the same purpose.

## Apollo and Autoware driving systems

- [ ] Supply an `AVStack` adapter for Baidu Apollo.
- [x] Host Autoware in LiDAR mode on CARLA through native ROS 2.
- [ ] Add Autoware camera fusion, including forwarding the required image topics.
- [ ] Generate Autoware maps for additional CARLA towns.
- [x] Evaluate repeated Autoware runs and the distribution of their outcomes.

## Native ROS 2 sensor attacks

- [x] Apply sensor attacks through a ROS 2 relay process that forwards the modified messages to
  the topics subscribed to by the external driving system.

## Native ROS 2 runs after the origin

- [ ] Support starts after the scene origin by placing ego on the prepared trajectory at each
  pre-roll tick.

## External benchmark integrations

- [ ] Supply reference evaluation workflows and published benchmark adapters.
