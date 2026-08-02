"""AI-agent layer (PLAN.md Phase 9, PROJECT.md 8).

Principle: *the agent proposes; the framework validates and executes.*

Planned agents: repository inspection, adapter generation (AV model / attack -> plugin),
configuration generation (NL -> validated ExperimentSpec), integration debugging,
root-cause analysis, report generation. Each runs behind deterministic validation gates
(schema/static/unit/interface/clean-consistency + assumption preservation, PROJECT.md 9)
with full provenance logging.

TODO(phase9): define agent roles, tool interfaces, and the validation loop.
"""

__all__: list = []
