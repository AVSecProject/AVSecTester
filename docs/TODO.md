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

- [ ] Support explicit single-run and paired-run modes, including clean-only, attack-only,
  defense-only and attack-with-defense runs. Define the behavior when a comparison has no
  intervention configured. Determine active interventions from the unified runtime configuration,
  rather than only the legacy `attacks` list.
