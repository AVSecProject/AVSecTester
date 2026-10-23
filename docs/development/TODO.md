# TODO

This file lists open work. Items are planned capabilities, not implemented features.

## Reactive traffic in NuRec

- [ ] Make other actors react to ego motion. Confirm renderer support for per-frame actor poses
  and integrate AlpaSim's trafficsim for reactive traffic.

## Rolling-shutter rendering

- [ ] Support AlpaSim's shutter_duration_us when requesting camera captures. Current requests
  render instantaneous frames without rolling-shutter timing.

## Driving-impact metrics

- [ ] Detect induced stopping events over time, including a stop followed by resumed driving.
  Specify prior motion, duration and matched clean windows; exclude initially stationary frames.

## Actor-aware adaptive attack

- [ ] Add a reference TTC phantom Attack whose runtime estimates another actor from perception
  outputs and adapts its actions. Keep history in AttackRuntime, reset each run and record decisions
  through OnlineContext.record_output. SpeedFeedbackAttack estimates no other actor and does not
  satisfy this example. Add offline tests and a supplied YAML configuration.

## Defenses

- [ ] Define defense lifecycle and comparison semantics, including clean, attack and
  attack-plus-defense treatments with reusable preparation.
- [ ] Supply reference defense algorithms and metrics for outputs such as alarms.

## Recorded static occlusion

- [ ] Supply aligned static-scene depth or geometry for RecordedReplaySimulator. Its image
  insertion currently uses actor cuboids without static mesh occlusion.

## NuRec rendered-probe conventions

- [ ] Replace the measured NRE asset conventions (PLY axes, object_size scaling, track padding)
  with documented ones once NVIDIA publishes them, and recheck probe alignment for each NRE release.
  Probe silhouettes differ from composited surfaces by about one pixel at edges.

## Inserted objects in non-camera sensors

- [ ] Include framework-inserted ObjectAsset geometry in LiDAR and other non-camera captures.
  Current object composition changes camera images and WorldSnapshot.objects only.

## Physical patch geometry

- [ ] Make CARLA native patch dimensions explicit and verifiable against the requested planar
  geometry, or support native mesh scaling. Current width_m/height_m do not rescale a prop. Measure
  physical materials/texture sampling against the differentiable optimization surrogate.

## Attack impact

- [ ] Measure migrated attacks' impact against the intended driving models and compare
  with established baselines. Successful preparation, execution and evaluation alone do not establish
  equivalent attack strength.

## Cross-simulator driving configurations

- [ ] Evaluate ModularAVStack on NuRec and AlpamayoAVStack on CARLA, including sensor formats,
  calibration and command compatibility. Current reuse guarantees cover supported scenarios within the
  same simulator.

## Multiple attacks on shared metadata

- [ ] Compare different attacks within one comparison group when their segments coincide, running
  treatments without attack_name once. Today each attack forms its own groups, and those treatments
  repeat for every attack.
