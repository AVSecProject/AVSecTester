# TODO

This page lists capabilities to implement.

## Optional-component tests in CI

- [ ] Run AlpaSim traffic, controller and Alpamayo policy tests in CI.
- [ ] Run nuScenes dataset, recorded-playback and insertion tests in CI.
- [ ] Run OpenMMLab detector-configuration, scorer and camera forward-collision tests in CI.

## Traffic actor ground contact

- [ ] Fit trafficsim actors to the selected ground mesh instead of the service's
  nearest-lane height.
- [ ] Support traffic-session checkpoint/restore and caller-controlled CATK sampling seeds.

See [reactive traffic](../guides/simulators.md#reactive-traffic) for supported behavior.

## Rolling-shutter rendering

- [ ] Support `shutter_duration_us` when requesting camera captures.

## Driving-impact metrics

- [ ] Detect induced stopping events, including resumed driving, with matched unattacked windows and
  explicit prior-motion and duration requirements.

## Actor-aware adaptive attack

- [ ] Supply a reference TTC phantom attack that estimates actors from perception outputs and
  records its runtime decisions.

## Defenses

- [ ] Define defense lifecycle and comparison semantics.
- [ ] Supply reference defenses and output metrics.

## Recorded static occlusion

- [ ] Supply aligned static-scene depth or geometry for `RecordedReplaySimulator`.

## NuRec rendered-probe conventions

- [ ] Use published NRE asset conventions for PLY axes, scaling and track padding.

## Inserted objects in non-camera sensors

- [ ] Include inserted `ObjectAsset` geometry in LiDAR and other non-camera captures.

## Physical patch geometry

- [ ] Make CARLA native patch dimensions explicit or support native mesh scaling.
- [ ] Validate physical material sampling against the differentiable optimization model.

## Attack impact

- [ ] Measure supplied attacks against intended driving models and established baselines.

See [experiment interpretation](../examples/README.md#attack-impact).

## Cross-simulator driving configurations

- [ ] Provide `ModularAVStack` on NuRec and `AlpamayoAVStack` or `RuleBasedAVStack` on CARLA
  with compatible sensors and commands.

## Shared runs across different user data

- [ ] Reuse treatments without attacks for units sharing a scenario and window but differing
  in `user_data`.

## Environment transfer

- [ ] Evaluate artifacts trained under one set of environment conditions on testing segments
  under different conditions.

## nuScenes semantic maps

- [ ] Read nuScenes semantic maps through `ScenarioMap`.

## Traffic-light state recording

- [ ] Provide traffic-light states in NuRec and recorded playback.

## External benchmark integrations

- [ ] Supply reference evaluation workflows and published benchmark adapters.

## Versioned documentation

- [ ] Publish the site documentation for each release beside the latest `main`, with a version
  selector.

## Project authors and citation

- [ ] Fill the homepage author list in `website/mkdocs.yml` and add citation information.
