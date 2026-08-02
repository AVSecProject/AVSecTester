"""Security metrics (PLAN.md Phase 6, PROJECT.md 4.2).

Each metric implements core.interfaces.MetricBase and registers with the METRICS registry.
Dimensions to implement (one module each):
  activation, targeting, persistence (propagation), safety, detectability,
  mitigability, practicality.

Safety metrics should reuse avstack's RSS metric + CARLA collision/infraction signals
rather than reimplementing them.

TODO(phase6): implement the seven metric modules + a statistical aggregation harness.
"""

__all__: list = []
