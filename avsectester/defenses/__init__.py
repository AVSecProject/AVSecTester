"""Defense / monitoring plugins (PLAN.md Phase 5).

Defenses register with the DEFENSES registry and are hook-shaped (core.interfaces.
DefenseBase). Initial categories (PROJECT.md 12.4):
  - input-level anomaly detection
  - temporal-consistency checking
  - cross-sensor-consistency checking
  - runtime safety monitoring (tie to avstack RSS metric)
  - attack-aware fallback / mitigation

TODO(phase5): implement TemporalConsistencyDefense as the first reference defense.
"""

__all__: list = []
