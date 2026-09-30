"""Attack-robustness harness — does the attack still succeed under sensor corruption?

The scenario layer yields *which* scenes qualify for an attack; the augmentation layer yields *under
which conditions* to test it. This harness runs the grid: for every qualifying
:class:`~avsectester.scenarios.source.ScenarioInstance`, and every corruption condition (a clean baseline
plus each :class:`~avsectester.simulators.augment.AugmentationPipeline`), it drives the scenario twice —

* **clean-under-condition**   : ``run(backend, stack, frames, perturb=aug)``            (corruption only)
* **attacked-under-condition**: ``run(backend, stack, frames, perturb=compose(attack, aug))``

and scores the pair with :func:`avsectester.metric.impact`. The *same* corruption (same pipeline seed)
is applied to both runs, so ``impact`` measures only the attack under that condition. Aggregating
``impact.attack_succeeded`` over scenarios gives an **attack success rate (ASR)** per condition, and each
corruption's **resilience** = ``ASR(corruption) / ASR(clean)`` — the attack analogue of a robustness
retention rate (cf. Robo3D's mRR): 1.0 means the corruption did not blunt the attack, 0.0 means it
neutralised it.

The attack itself is injected as ``attack_for(match, backend) -> perturb`` so the harness is
attack-agnostic; the corruption suite defaults to :func:`~avsectester.simulators.augment.common_corruptions`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from avsectester.backend import run as _run_loop
from avsectester.metric import impact
from avsectester.simulators.augment import (
    AugmentationPipeline,
    common_corruptions,
    compose,
    sensor_augmentation,
)

if TYPE_CHECKING:
    from avsectester.backend import AVStack
    from avsectester.plane import Observation, Trace
    from avsectester.scenarios.requirement import ScenarioMatch, ScenarioRequirement
    from avsectester.scenarios.source import ScenarioSource

Perturb = Callable[["Observation"], "Observation"]
AttackFor = Callable[["ScenarioMatch", object], Perturb | None]
_CLEAN = "clean"


@dataclass
class ConditionResult:
    """Aggregated attack outcome for one corruption condition across all scenarios."""

    condition: str
    n: int = 0
    successes: int = 0

    @property
    def asr(self) -> float:
        """Attack success rate — fraction of scenarios where the attack succeeded under this condition."""
        return self.successes / self.n if self.n else 0.0


@dataclass
class RobustnessReport:
    """The robustness grid: attack success per (scenario, condition), aggregated per condition.

    ``rows`` are the raw ``(scenario_id, condition, succeeded)`` outcomes; ``per_condition`` the ASR per
    condition; ``baseline`` names the clean condition the resiliences are relative to."""

    per_condition: dict[str, ConditionResult] = field(default_factory=dict)
    rows: list[tuple[str, str, bool]] = field(default_factory=list)
    baseline: str = _CLEAN

    def record(self, scenario_id: str, condition: str, succeeded: bool) -> None:
        self.rows.append((scenario_id, condition, succeeded))
        cr = self.per_condition.setdefault(condition, ConditionResult(condition))
        cr.n += 1
        cr.successes += int(succeeded)

    @property
    def baseline_asr(self) -> float:
        cr = self.per_condition.get(self.baseline)
        return cr.asr if cr else 0.0

    def resilience(self, condition: str) -> float:
        """``ASR(condition) / ASR(baseline)`` clamped to [0, 1] — how much of the attack survives the
        corruption. Undefined (returns 0.0) when the attack never succeeded even clean."""
        base = self.baseline_asr
        if base <= 0.0:
            return 0.0
        return min(1.0, self.per_condition[condition].asr / base)

    @property
    def mean_resilience(self) -> float:
        """Mean resilience over the non-baseline (corruption) conditions — a single robustness number."""
        corruptions = [c for c in self.per_condition if c != self.baseline]
        return sum(self.resilience(c) for c in corruptions) / len(corruptions) if corruptions else 0.0

    def summary(self) -> str:
        n = self.per_condition.get(self.baseline, ConditionResult("")).n
        lines = [f"Attack robustness over {n} scenario(s) — baseline ASR = {self.baseline_asr:.0%}",
                 f"{'condition':<28} {'ASR':>6} {'resilience':>11}"]
        for name, cr in sorted(self.per_condition.items(), key=lambda kv: (kv[0] != self.baseline, kv[0])):
            res = "  (baseline)" if name == self.baseline else f"{self.resilience(name):>10.0%}"
            lines.append(f"{name:<28} {cr.asr:>6.0%} {res:>11}")
        lines.append(f"{'mean resilience (corruptions)':<28} {'':>6} {self.mean_resilience:>10.0%}")
        return "\n".join(lines)


def evaluate_robustness(
    source: ScenarioSource,
    req: ScenarioRequirement,
    attack_for: AttackFor,
    stack: Callable[[], AVStack],
    frames: int,
    *,
    conditions: Iterable[AugmentationPipeline] | None = None,
    severity: float = 0.5,
    limit: int | None = None,
    run_fn: Callable[..., Trace] = _run_loop,
) -> RobustnessReport:
    """Run the robustness grid and return a :class:`RobustnessReport`.

    For each :class:`ScenarioInstance` from ``source.scenarios(req, limit)`` and each condition (a clean
    baseline first, then each corruption pipeline — default :func:`common_corruptions` at ``severity``),
    build a fresh backend, run clean-vs-attacked under that condition, and record
    ``impact(...).attack_succeeded``. ``stack`` is a factory (a fresh AV box per run); ``attack_for`` maps
    a scenario's target + its backend to the attack ``perturb`` (or None to skip). ``run_fn`` is injectable
    for testing.
    """
    pipelines = list(conditions) if conditions is not None else common_corruptions(severity)
    labelled: list[tuple[str, AugmentationPipeline | None]] = [(_CLEAN, None)]
    labelled += [(p.name, p) for p in pipelines]

    report = RobustnessReport()
    for i, instance in enumerate(source.scenarios(req, limit=limit)):
        match = instance.target
        scenario_id = str(instance.provenance.get("scene_id")
                          or instance.provenance.get("sample_token")
                          or instance.provenance.get("image_path")
                          or i)
        for name, pipeline in labelled:
            backend = instance.make_backend()
            try:
                aug = sensor_augmentation(pipeline, camera=match.camera) if pipeline is not None else None
                attack = attack_for(match, backend)
                clean = run_fn(backend, stack(), frames, perturb=aug)
                attacked = run_fn(backend, stack(), frames, perturb=compose(attack, aug))
            finally:
                backend.close()
            report.record(scenario_id, name, impact(clean, attacked).attack_succeeded)
    return report
