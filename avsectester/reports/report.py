"""Render an experiment Result to a human-readable markdown report."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.runner import Result


def _table(metrics: dict) -> list[str]:
    return [f"| `{k}` | {v} |" for k, v in metrics.items()]


def render_report(name: str, result: Result) -> str:
    lines = [f"# Security test report — {name}", ""]
    lines.append(f"**Impacted:** {result.metrics.get('impacted')}  "
                 f"(clean {len(result.clean)} frames, attacked {len(result.attacked)} frames)")
    lines += ["", "## Attacked vs clean", "", "| metric | value |", "|---|---|", *_table(result.metrics)]
    if result.defended_metrics is not None:
        lines += ["", "## Defended vs clean", "", "| metric | value |", "|---|---|",
                  *_table(result.defended_metrics)]
        mitigated = result.metrics.get("impacted") and not result.defended_metrics.get("impacted")
        lines += ["", f"**Mitigated:** {bool(mitigated)}"]
    return "\n".join(lines) + "\n"
