"""Security metrics (PLAN.md Phase 6, PROJECT.md 4.2).

Each metric implements core.interfaces.MetricBase and registers with the METRICS registry.
Dimensions to implement (one module each): activation, targeting, persistence (propagation),
safety, detectability, mitigability, practicality.

``EscalationMetric`` covers activation + propagation + safety today and emits the
attack-escalation DAG. Remaining dimensions are TODO(phase6).

Safety metrics should reuse avstack's RSS metric + CARLA collision/infraction signals
rather than reimplementing them.
"""

from .escalation import EscalationMetric

__all__ = ["EscalationMetric"]
