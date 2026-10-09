"""Attack-robustness harness — does the attack still succeed under sensor corruption?

The scenario layer yields *which* scenes qualify for an attack. The augmentation layer yields *under
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
attack-agnostic. The corruption suite defaults to :func:`~avsectester.simulators.augment.common_corruptions`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace
from math import isnan
from typing import TYPE_CHECKING, Literal

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
OutcomeStatus = Literal["success", "failure", "skipped", "inconclusive"]


@dataclass(frozen=True)
class ScenarioResult:
    """One attempted pair, including why it was excluded from the valid-experiment count."""

    scenario_id: str
    condition: str
    status: OutcomeStatus
    reason: str = ""


@dataclass
class ConditionResult:
    """Aggregated attack outcome for one corruption condition across all scenarios."""

    condition: str
    successes: int = 0
    failures: int = 0
    skipped: int = 0
    inconclusive: int = 0

    @property
    def n(self) -> int:
        """Number of valid pairs. Skipped and inconclusive attempts are reported separately."""
        return self.successes + self.failures

    @property
    def asr(self) -> float:
        """Success fraction among valid pairs. NaN when there are no valid pairs."""
        return self.successes / self.n if self.n else float("nan")


@dataclass
class RobustnessReport:
    """The robustness grid: attack success per (scenario, condition), aggregated per condition.

    ``rows`` retain each attempt's status and reason. ``per_condition`` aggregates valid outcomes and
    excluded attempts separately. ``baseline`` names the condition the resiliences are relative to."""

    per_condition: dict[str, ConditionResult] = field(default_factory=dict)
    rows: list[ScenarioResult] = field(default_factory=list)
    baseline: str = _CLEAN

    def record(self, scenario_id: str, condition: str, status: OutcomeStatus,
               reason: str = "") -> None:
        counter = {"success": "successes", "failure": "failures",
                   "skipped": "skipped", "inconclusive": "inconclusive"}[status]
        self.rows.append(ScenarioResult(scenario_id, condition, status, reason))
        cr = self.per_condition.setdefault(condition, ConditionResult(condition))
        setattr(cr, counter, getattr(cr, counter) + 1)

    @property
    def baseline_asr(self) -> float:
        cr = self.per_condition.get(self.baseline)
        return cr.asr if cr else float("nan")

    def resilience(self, condition: str) -> float:
        """``ASR(condition) / ASR(baseline)`` clamped to [0, 1] — how much of the attack survives the
        corruption. Returns NaN when either condition has no valid pairs. The existing zero-baseline
        convention (0.0 when the attack never succeeded clean) is retained."""
        base = self.baseline_asr
        current = self.per_condition[condition].asr
        if isnan(base) or isnan(current):
            return float("nan")
        if base <= 0.0:
            return 0.0
        return min(1.0, current / base)

    @property
    def mean_resilience(self) -> float:
        """Mean resilience over the non-baseline (corruption) conditions — a single robustness number."""
        corruptions = [c for c in self.per_condition if c != self.baseline]
        return sum(self.resilience(c) for c in corruptions) / len(corruptions) if corruptions else 0.0

    def summary(self) -> str:
        def percent(value: float) -> str:
            return "N/A" if isnan(value) else f"{value:.0%}"

        n = self.per_condition.get(self.baseline, ConditionResult("")).n
        lines = [f"Attack robustness — {n} valid baseline pair(s). "
                 f"baseline ASR = {percent(self.baseline_asr)}",
                 f"{'condition':<28} {'valid':>6} {'skipped':>8} {'inconclusive':>12} "
                 f"{'ASR':>6} {'resilience':>11}"]
        for name, cr in sorted(self.per_condition.items(), key=lambda kv: (kv[0] != self.baseline, kv[0])):
            res = "(baseline)" if name == self.baseline else percent(self.resilience(name))
            lines.append(f"{name:<28} {cr.n:>6} {cr.skipped:>8} {cr.inconclusive:>12} "
                         f"{percent(cr.asr):>6} {res:>11}")
        lines.append(f"mean resilience (corruptions): {percent(self.mean_resilience)}")
        return "\n".join(lines)


def _condition_perturbation(pipeline: AugmentationPipeline | None, camera: str) -> Perturb | None:
    """Seed corruption by relative frame, since simulator frame IDs can differ after a reset."""
    if pipeline is None:
        return None
    if camera is None:
        raise ValueError("Image corruptions require a camera. Use conditions=[] for scene-only cases.")
    augment = sensor_augmentation(pipeline, camera=camera)
    first_frame = None

    def perturb(obs: Observation) -> Observation:
        nonlocal first_frame
        if first_frame is None:
            first_frame = obs.frame
        augmented = augment(replace(obs, frame=obs.frame - first_frame))
        return replace(augmented, frame=obs.frame)

    return perturb


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
    build a fresh backend, prepare a repeatable pair, and run clean-vs-attacked under that condition.
    Skipped attacks and inconclusive driving baselines are excluded from the valid-pair count.
    ``stack`` is a factory (a fresh AV box per run). ``attack_for`` maps
    a scenario's target + its backend to the attack ``perturb`` (or None to skip). ``run_fn`` is injectable
    for testing.
    """
    pipelines = list(conditions) if conditions is not None else common_corruptions(severity)
    labelled: list[tuple[str, AugmentationPipeline | None]] = [(_CLEAN, None)]
    labelled += [(p.name, p) for p in pipelines]

    report = RobustnessReport()
    for i, instance in enumerate(source.scenarios(req, limit=limit)):
        match = instance.match
        scenario_id = str(instance.provenance.get("scene_id")
                          or instance.provenance.get("sample_token")
                          or instance.provenance.get("image_path")
                          or i)
        for name, pipeline in labelled:
            backend = instance.make_backend()
            try:
                attack = attack_for(match, backend)
                if attack is None:
                    report.record(scenario_id, name, "skipped", "Attack factory returned None")
                    continue
                backend.prepare_clean_attack_pair()
                clean = run_fn(backend, stack(), frames,
                               perturb=_condition_perturbation(pipeline, match.camera))
                attacked = run_fn(
                    backend, stack(), frames,
                    perturb=compose(attack, _condition_perturbation(pipeline, match.camera)),
                )
            finally:
                backend.close()
            # TODO: Select success criteria by experiment/benchmark. impact currently only scores
            # induced/suppressed stops. It is not a universal definition of attack success.
            result = impact(clean, attacked)
            if not result.clean_drove:
                report.record(scenario_id, name, "inconclusive", result.verdict)
            else:
                status = "success" if result.attack_succeeded else "failure"
                report.record(scenario_id, name, status, result.verdict)
    return report
