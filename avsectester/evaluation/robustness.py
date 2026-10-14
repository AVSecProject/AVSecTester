"""Compare an attack across sensor conditions using the common prepared-scenario runner.

Each condition preserves the scenario's world and sensor settings and adds its image corruption
after digital interventions. Clean and attacked treatments use independent condition handlers.
The attack can be a method, reusable prepared artifacts or the legacy backend-dependent callback.
The existing stop-based impact and success-rate aggregation remain unchanged.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from copy import deepcopy
from dataclasses import dataclass, field, replace
from math import isnan
from typing import TYPE_CHECKING, Literal

from avsectester.backend import run as _run_loop
from avsectester.conditions import ScenarioConditions, SensorCondition
from avsectester.preparation import Intervention, PreparedIntervention, Treatment, prepare_intervention
from avsectester.runner import _run_prepared
from avsectester.runtime import Runtime, Hook
from avsectester.interventions import ObservationTransform
from avsectester.metric import impact
from avsectester.perturbations.image import (
    AugmentationPipeline,
    common_corruptions,
)

if TYPE_CHECKING:
    from avsectester.backend import AVStack, WorldBackend
    from avsectester.plane import Observation, Trace
    from avsectester.scenarios.requirement import ScenarioMatch, ScenarioRequirement
    from avsectester.scenarios.source import ScenarioSource

Perturb = Callable[["Observation"], "Observation"]
AttackFor = Callable[["ScenarioMatch | None", "WorldBackend"], Runtime | Perturb | None]
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


def evaluate_robustness(
    source: ScenarioSource,
    req: ScenarioRequirement,
    attack_for: AttackFor | Intervention | PreparedIntervention,
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
    ``stack`` is a factory for a fresh driver per run. An Intervention prepares once per selected
    scenario and creates handlers for each condition. A PreparedIntervention explicitly reuses
    existing artifacts. The legacy ``attack_for(match, backend)`` callback remains supported and
    is resolved once per pair after world-condition configuration and before origin preparation.
    It returns Runtime, a one-argument
    perturbation or None to skip. ``run_fn`` accepts runtime and returns the driving Trace.
    """
    if not isinstance(frames, int) or isinstance(frames, bool) or frames < 0:
        raise ValueError("frames must be a nonnegative integer")
    if isinstance(attack_for, PreparedIntervention) and attack_for.kind != "attack":
        raise ValueError("Robustness evaluation requires an attack intervention")
    if not isinstance(attack_for, (Intervention, PreparedIntervention)) and not callable(attack_for):
        raise TypeError("Expected Intervention, PreparedIntervention or an attack callback")
    pipelines = list(conditions) if conditions is not None else common_corruptions(severity)
    if any(not isinstance(p, AugmentationPipeline) for p in pipelines):
        raise TypeError("Robustness conditions must be AugmentationPipeline instances")
    labelled: list[tuple[str, AugmentationPipeline | None]] = [(_CLEAN, None)]
    labelled += [(p.name, p) for p in pipelines]

    report = RobustnessReport()
    for i, instance in enumerate(source.scenarios(req, limit=limit)):
        match = instance.match
        scenario_id = str(instance.provenance.get("scene_id")
                          or instance.provenance.get("sample_token")
                          or instance.provenance.get("image_path")
                          or i)
        prepared = (
            prepare_intervention(attack_for, instance) if isinstance(attack_for, Intervention)
            else attack_for if isinstance(attack_for, PreparedIntervention) else None
        )
        for name, pipeline in labelled:
            base = instance.conditions
            sensors = base.sensors
            if pipeline is not None:
                sensors += (SensorCondition.image(pipeline, camera=getattr(match, "camera", None)),)
            scenario = replace(instance, conditions=ScenarioConditions(base.world, sensors, base.seed))
            backend_ready = None
            if prepared is not None:
                runtimes = {"clean": Runtime(), "attack": Treatment((prepared,)).runtime(case=match)}
            else:
                resolved: list[Runtime] = []

                def resolve_attack(backend):
                    attack = attack_for(match, backend)
                    if attack is None:
                        return False
                    runtime = attack if isinstance(attack, Runtime) else Runtime([
                        Hook("observation", ObservationTransform(attack)),
                    ], case=match)
                    resolved.append(runtime)
                    return True

                backend_ready = resolve_attack
                runtimes = {
                    "clean": lambda backend: Runtime(
                        localizer=resolved[0].localizer, seed=resolved[0].seed, case=match,
                    ),
                    "attack": lambda backend: resolved[0],
                }
            # The shared runner configures conditions before the callback and owns cleanup
            # on skip, callback failure and execution failure. No scenario fields are replaced.
            results = _run_prepared(
                scenario, stack, frames, runtimes=runtimes, run_fn=run_fn,
                backend_ready=backend_ready,
            )
            if not results:
                report.record(scenario_id, name, "skipped", "Attack factory returned None")
                continue
            clean, attacked = results["clean"], results["attack"]
            if prepared is not None:
                clean.provenance["interventions"] = []
                attacked.provenance["interventions"] = [{
                    "name": prepared.name, "kind": prepared.kind,
                    "provenance": deepcopy(prepared.provenance),
                }]
            # TODO: Select success criteria by experiment/benchmark. impact currently only scores
            # induced/suppressed stops. It is not a universal definition of attack success.
            result = impact(clean, attacked)
            if not result.clean_drove:
                report.record(scenario_id, name, "inconclusive", result.verdict)
            else:
                status = "success" if result.attack_succeeded else "failure"
                report.record(scenario_id, name, status, result.verdict)
    return report
