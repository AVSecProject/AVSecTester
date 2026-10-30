# TODO

This page lists capabilities to implement.

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

- [ ] Include inserted `ObjectAsset` geometry in LiDAR and other non-camera captures.

## CARLA physical patches

- [ ] Scale the CARLA patch prop to the declared `width_m` and `height_m`, or reject dimensions
  that differ from its mesh.
- [ ] Validate CARLA's texture sampling on the prop against the differentiable optimization
  model.

## Sim-to-real fidelity

- [ ] Composite photographs of printed stop-sign patches into NuRec scenes and compare detection
  and driving results with digitally composited patches.

## Closed-loop rollouts on recorded datasets

- [ ] Reconstruct recorded datasets such as nuScenes into NuRec scenes that NRE can render, so
  they run closed-loop rollouts instead of recorded playback.
- [ ] Support reconstruction engines other than NRE for the same purpose.

## Apollo and Autoware driving systems

- [ ] Supply `AVStack` adapters for Baidu Apollo or Autoware that run their modules synchronously
  with each decision, starting with CARLA.

## External benchmark integrations

- [ ] Supply reference evaluation workflows and published benchmark adapters.
