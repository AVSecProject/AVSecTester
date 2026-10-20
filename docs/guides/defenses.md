# Defenses

> **Audience:** Researchers planning defense evaluation.
>
> **Outcome:** Understand the supported experiment scope.
>
> **Prerequisites:** [Experiments](experiments.md).

**Not yet implemented.** Defense treatments, reference defense algorithms and defense-specific
output metrics. See [TODO](../development/TODO.md#defenses).

A treatment currently enables at most one Attack and supplies environment overrides. Hooks are
available for explicit stage processing, but configuring a hook does not define a reusable defense
protocol or a clean/attack/attack-plus-defense comparison.

Do not present a speed-spoofing Attack as a defense implementation. A future defense interface must
define its lifecycle, saved parameters and comparison semantics before examples rely on it.

## Next steps

Use [Attacks](attacks.md) for the implemented workflow and [Evaluation](evaluation.md) for saved-run
metrics.
