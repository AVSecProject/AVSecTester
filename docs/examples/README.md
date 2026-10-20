# Attack examples

> **Audience:** Users studying supplied attacks.
>
> **Outcome:** Choose an Attack implementation and a runnable configuration.
>
> **Prerequisites:** [Attacks](../guides/attacks.md).

Every example implements the same Attack workflow: metabuilder, optional offline preparation, and a
new AttackRuntime for each run. Run it through the supplied configuration with `avsectester run
CONFIG.yaml`.

| Example | Implementation | Configuration |
|---|---|---|
| [Phantom injection](phantom-injection.md) | PhantomInjectionAttack | configs/carla_phantom.yaml |
| [Physical patch](physical-patch.md) | PatchAttack | configs/carla_patch.yaml, configs/carla_patch_optimized.yaml, configs/nurec_patch.yaml |
| [Image insertion](image-insertion.md) | ObjectInsertionAttack | configs/carla_object_insertion.yaml, configs/nurec_object_insertion.yaml, configs/nuscenes_object_insertion.yaml |
| [Speed feedback](speed-feedback-attack.md) | SpeedFeedbackAttack | configs/nurec_speed_feedback.yaml |

Paths in the table are relative to the repository root. All effect levels and model outcomes must be
measured in the target environment. Passing workflow tests establishes preparation, execution and
evaluation, not attack strength.
