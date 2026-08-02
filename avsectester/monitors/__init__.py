"""Runtime instrumentation + trace capture (PLAN.md Phase 4)."""

from .trace import ComponentIO, Trace, TraceMonitor, build_trace, diff_traces, record_to_ios

__all__ = [
    "ComponentIO",
    "Trace",
    "TraceMonitor",
    "build_trace",
    "diff_traces",
    "record_to_ios",
]
