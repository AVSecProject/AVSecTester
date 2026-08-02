"""Attack Escalation DAG (PROJECT.md 10, proposal Thrust 1).

The central abstraction: trace an attack signal from a constrained perturbation through
component-level errors to a closed-loop driving consequence. Nodes are attack-induced
system errors; edges encode the *conditions* under which an error propagates across
perception -> localization -> prediction -> planning -> control -> safeguards.

This is populated from paired clean-vs-attacked traces (see avsectester.monitors) and is
consumed by root-cause analysis (see avsectester.reports).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import networkx as nx


class Stage(str, Enum):
    """AV-stack stage where an error manifests."""

    ATTACK_SURFACE = "attack_surface"
    SENSOR = "sensor"
    PERCEPTION = "perception"
    LOCALIZATION = "localization"
    TRACKING = "tracking"
    FUSION = "fusion"
    PREDICTION = "prediction"
    PLANNING = "planning"
    CONTROL = "control"
    SAFEGUARD = "safeguard"
    CONSEQUENCE = "consequence"


@dataclass
class EscalationNode:
    """An attack-induced error observed at one stage/component."""

    id: str
    stage: Stage
    component: str
    description: str = ""
    # Quantified divergence from the clean run (metric name -> value).
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass
class EscalationEdge:
    """A propagation link: error at ``src`` induced error at ``dst`` under ``condition``."""

    src: str
    dst: str
    condition: str = ""
    # e.g. "amplified" | "persisted" | "absorbed" | "triggered".
    kind: str = "propagated"


class EscalationDAG:
    """Thin wrapper over a networkx DiGraph with escalation-specific helpers."""

    def __init__(self) -> None:
        self._g = nx.DiGraph()

    def add_node(self, node: EscalationNode) -> None:
        self._g.add_node(node.id, data=node)

    def add_edge(self, edge: EscalationEdge) -> None:
        self._g.add_edge(edge.src, edge.dst, data=edge)

    @property
    def graph(self) -> nx.DiGraph:
        return self._g

    def root_cause(self) -> EscalationNode | None:
        """First (earliest, in-degree 0) diverging node — candidate root cause.

        TODO(phase4/8): use ordering by stage + first-divergence timestamp rather than a
        pure topological heuristic.
        """
        roots = [n for n, d in self._g.in_degree() if d == 0]
        if not roots:
            return None
        return self._g.nodes[roots[0]]["data"]

    def consequence_paths(self) -> list[list[str]]:
        """All paths from any attack-surface node to any consequence node."""
        surfaces = [
            n for n, d in self._g.nodes(data="data")
            if d is not None and d.stage == Stage.ATTACK_SURFACE
        ]
        sinks = [
            n for n, d in self._g.nodes(data="data")
            if d is not None and d.stage == Stage.CONSEQUENCE
        ]
        paths: list[list[str]] = []
        for s in surfaces:
            for t in sinks:
                paths.extend(nx.all_simple_paths(self._g, s, t))
        return paths
