# Speed-feedback attack

> **Audience:** Users studying per-run feedback state.
>
> **Outcome:** Inspect a model-visible speed bias separately from physical motion.
>
> **Prerequisites:** [Installation](../getting-started/installation.md) for NuRec.

## Configuration

```bash
docker compose run --rm nurec-experiment run configs/nurec_speed_feedback.yaml
```

The supplied configuration uses:

```yaml
type: SpeedFeedbackAttack
options:
  decision_steps: 20
  max_segments: 2
  target_speed: 12.0
  gain: 0.1
  max_bias: 5.0
```

Parameter defaults and constraints are in the [Python API](../reference/python-api.md).

## Offline phase

The attack selects decision windows. Its offline phase returns no artifact. See
[Writing attacks](../guides/attacks.md) for the common lifecycle.

## Online phase

A fresh runtime maintains a bounded nonnegative speed bias from physical-state feedback and
modifies `Observation.ego_speed` at the observation stage. It records `speed_bias` in each
decision's outputs. The physical state remains separate in
the [RunLog](../reference/runtime.md#runlog).

Alpamayo derives its speed inputs from pose history, so this observation-field change does not
alter those inputs. Driving-system inputs are described in
[Driving systems](../guides/driving-systems.md).

Feedback driven by another actor's perception history is **Not yet implemented.** See
[Actor-aware adaptive attack](../development/TODO.md#actor-aware-adaptive-attack).

## Evidence

Compare the recorded `speed_bias`, model observation and physical motion with the configured
[Evaluation](../guides/evaluation.md). The [example overview](README.md) explains
impact assessment.

## Next steps

Read [Runtime](../reference/runtime.md) and [Writing attacks](../guides/attacks.md).
