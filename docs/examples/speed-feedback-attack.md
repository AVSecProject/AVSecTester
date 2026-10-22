# Speed-feedback attack

> **Audience:** Users studying stateful online attacks.
>
> **Outcome:** Understand per-run feedback history and model-visible speed changes.
>
> **Prerequisites:** [Attacks](../guides/attacks.md).

## Configuration

```bash
avsectester run configs/nurec_speed_feedback.yaml
```

SpeedFeedbackAttack is defined in
avsectester.implemented_attacks.speed_feedback_attack.attack. It requires decision_steps and
target_speed. Options include gain=0.1, max_bias=5.0, initial_bias=0.0, max_segments and keep_tail.
It has no asset or host requirement.

```yaml
type: SpeedFeedbackAttack
options:
  decision_steps: 20
  target_speed: 3.0
  gain: 0.1
  max_bias: 5.0
```

## Workflow

The metabuilder partitions eligible decisions. Offline preparation is trivial. create_runtime
creates a fresh feedback state for each treatment run. The runtime subscribes to physical state,
updates its bounded bias and returns an AVStackAttackAction changing Observation.ego_speed. The
supplied NuRec/Alpamayo configuration consumes that model-visible field. It is not configured with a
Modular pipeline whose control path ignores it. Physical state remains separately logged. Online
histories are not shared through Attack.

The runtime records each step's bias through OnlineContext.record_output. Inspect
StepRecord.outputs, model_observation, physical_pre and executed_command to see the full causal
path.

This attack does not estimate another actor and is not a TTC phantom example. An adaptive attack
based on perception-output history remains in
[TODO](../development/TODO.md#actor-aware-adaptive-attack).

## Next steps

Read [Runtime](../reference/runtime.md) and [Evaluation](../guides/evaluation.md).
