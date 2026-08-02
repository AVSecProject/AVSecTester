"""Runtime instrumentation + trace capture (PLAN.md Phase 4)."""

from .trace import ComponentIO, Trace, diff_traces

__all__ = ["ComponentIO", "Trace", "diff_traces"]
