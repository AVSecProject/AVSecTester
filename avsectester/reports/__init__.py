"""Root-cause analysis + audit reporting (PLAN.md Phase 8, proposal Thrust 4).

Consumes the Attack Escalation DAG + traces to produce an audit report: feasibility,
impact, uncertainty, scenario dependence, component-level root cause, and recommended
mitigations.

``render_report`` is the baseline markdown renderer; rootcause.py (DAG attribution) and
audit.py (provenance capture) are TODO(phase8).
"""

from .report import render_report

__all__ = ["render_report"]
