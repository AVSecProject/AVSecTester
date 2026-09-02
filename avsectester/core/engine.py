"""Experiment engine: run an ExperimentSpec and produce metrics + the escalation DAG.

The runner is backend-agnostic (MockBackend / CarlaBackend / ...). It executes paired
passes on the same scenario — a **clean** baseline and an **attacked** run (attack attached
at the backend's perception-input seam) — then scores them with the EscalationMetric. If the
spec declares a defense, it also runs an **attacked+defended** pass to measure mitigation.

Everything imported here is lightweight; the heavy avstack/CARLA stack is pulled in lazily by
the backend's ``build()``, so importing the engine never requires the full stack.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..config import ATTACKS, BACKENDS, DEFENSES
from ..monitors.trace import Trace, build_trace
from .experiment import ExperimentSpec


@dataclass
class ExperimentResult:
    name: str
    metrics: dict[str, Any]
    dag: Any  # EscalationDAG
    clean_trace: Trace
    attacked_trace: Trace
    defended_trace: Trace | None = None
    defended_metrics: dict[str, Any] | None = None
    mitigated: bool | None = None


class ExperimentRunner:
    def __init__(self, spec: ExperimentSpec) -> None:
        self.spec = spec

    # -- build plugins from registries -----------------------------------------
    def _backend(self):
        return BACKENDS.build(dict(self.spec.scenario.backend))

    def _attack(self):
        if self.spec.attack is None:
            return None
        attack = ATTACKS.build(dict(self.spec.attack.spec))
        if hasattr(attack, "validate"):
            attack.validate(self.spec)
        return attack

    def _defense(self):
        if self.spec.defense is None:
            return None
        return DEFENSES.build(dict(self.spec.defense.spec))

    # -- one pass --------------------------------------------------------------
    def _attach(self, backend, plugin) -> tuple[str, ...]:
        """Attach ``plugin`` at every seam it declares (after a compatibility check).

        Returns the tuple of seams it hooked into.
        """
        plugin.check(backend.profile())  # loud-fail if the stack can't support its seams/caps
        for seam in plugin.seams:
            backend.attach(plugin, seam)
        return plugin.seams

    def _run(self, run_id: str, attack=None, defense=None) -> Trace:
        from .binding import seams_downstream_of

        backend = self._backend()
        backend.build(self.spec)
        attack_seams: tuple[str, ...] = ()
        if attack is not None:
            attack.reset()
            attack_seams = self._attach(backend, attack)  # inject first
        if defense is not None:
            # ... then sanitize at/downstream of the attack surface (an upstream sanitizer
            # can never see the injected payload).
            defense_seams = self._attach(backend, defense)
            if attack_seams:
                allowed = set().union(*(seams_downstream_of(s) for s in attack_seams))
                bad = [s for s in defense_seams if s not in allowed]
                if bad:
                    raise ValueError(
                        f"defense hooks at {bad}, upstream of every attack seam "
                        f"{list(attack_seams)}; it cannot observe the injected payload"
                    )
        try:
            records = list(backend.run())
        finally:
            backend.close()
        return build_trace(records, run_id)

    # -- full experiment -------------------------------------------------------
    def run(self) -> ExperimentResult:
        from ..metrics.escalation import EscalationMetric

        metric = EscalationMetric()
        clean = self._run("clean")

        attack = self._attack()
        attacked = self._run("attacked", attack=attack) if attack is not None else clean
        out = metric.compute(clean, attacked)
        result = ExperimentResult(
            name=self.spec.name,
            metrics=out["metrics"],
            dag=out["dag"],
            clean_trace=clean,
            attacked_trace=attacked,
        )

        defense = self._defense()
        if attack is not None and defense is not None:
            defended = self._run("defended", attack=self._attack(), defense=defense)
            dout = metric.compute(clean, defended)
            result.defended_trace = defended
            result.defended_metrics = dout["metrics"]
            result.mitigated = result.metrics["escalated"] and not dout["metrics"]["escalated"]

        return result
