"""Render an ExperimentResult to a human-readable markdown report (PLAN.md Phase 8)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.engine import ExperimentResult

_METRIC_ORDER = [
    "activated",
    "activation_frame",
    "stages_reached",
    "propagation_depth",
    "reached_consequence",
    "persistence_frames",
    "brake_frames_clean",
    "brake_frames_attacked",
    "min_speed_clean",
    "min_speed_attacked",
    "speed_suppression",
    "final_speed_attacked",
    "stopped",
    "escalated",
]


def render_report(result: ExperimentResult) -> str:
    m = result.metrics
    lines: list[str] = []
    lines.append(f"# AVSecTester report — {result.name}")
    if m["escalated"]:
        verdict = "🔴 ESCALATED — attack reached an unsafe driving consequence"
    elif m["activated"]:
        verdict = "🟠 ACTIVATED — component error induced, no unsafe consequence"
    else:
        verdict = "🟢 NO ACTIVATION — attack had no measurable effect"
    lines += [f"**Verdict:** {verdict}", ""]

    lines.append("## Metrics")
    for k in _METRIC_ORDER:
        if k in m:
            lines.append(f"- `{k}`: {m[k]}")
    lines.append("")

    lines.append("## Attack-escalation path")
    g = result.dag.graph
    if g.number_of_nodes() == 0:
        lines.append("_(no divergence from the clean run — attack did not activate)_")
    else:
        nodes = [g.nodes[n]["data"] for n in g.nodes]
        chain = " → ".join(nd.stage.value for nd in nodes)
        lines.append(f"`{chain}`")
        lines.append("")
        for nd in nodes:
            frame = nd.evidence.get("frame")
            ev = {k: v for k, v in nd.evidence.items() if k != "frame"}
            lines.append(f"- **{nd.stage.value}** (f{frame}, `{nd.component}`) — {nd.description}")
            if ev:
                lines.append(f"    - evidence: {ev}")
        root = result.dag.root_cause()
        if root is not None:
            lines += ["", f"**Root cause:** {root.stage.value} — {root.description}"]
    lines.append("")

    if result.defended_metrics is not None:
        lines.append("## Defense")
        dm = result.defended_metrics
        outcome = "✅ mitigated" if result.mitigated else "❌ not mitigated"
        lines += [
            f"- outcome: {outcome}",
            (
                f"- defended: escalated={dm['escalated']}, activated={dm['activated']}, "
                f"stages_reached={dm['stages_reached']}"
            ),
        ]
        lines.append("")

    return "\n".join(lines)
