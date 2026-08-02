"""AVSecTester core: the security-experiment contract and shared abstractions."""

from .escalation import EscalationDAG, EscalationEdge, EscalationNode
from .experiment import (
    AttackConfig,
    DefenseConfig,
    EvaluationConfig,
    ExperimentSpec,
    ReproducibilityInfo,
    ScenarioSpec,
    SystemSpec,
)
from .interfaces import (
    AttackBase,
    Backend,
    DefenseBase,
    MetricBase,
    MonitorBase,
    SearchStrategy,
)
from .threat_model import ThreatModel

__all__ = [
    "AttackBase",
    "AttackConfig",
    "Backend",
    "DefenseBase",
    "DefenseConfig",
    "EscalationDAG",
    "EscalationEdge",
    "EscalationNode",
    "EvaluationConfig",
    "ExperimentSpec",
    "MetricBase",
    "MonitorBase",
    "ReproducibilityInfo",
    "ScenarioSpec",
    "SearchStrategy",
    "SystemSpec",
    "ThreatModel",
]
