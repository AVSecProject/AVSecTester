"""Public contracts, algorithms, parameterizations and session scoring for user attacks."""
from .interface import (
    AdversarialSample, DataSource, GradientOptimizer, HideBelowThreshold, HideObject,
    OptimizationObjective, OptimizationResult, Perturbation, Scorer, SpoofObject,
)
from .blackbox import NES
from .scorers import SessionScorer


def __getattr__(name: str):
    """Load tensor-based algorithms only when requested by the user."""
    if name == "PGD":
        from .whitebox import PGD
        return PGD
    if name in {"LinfImage", "PatchPerturbation"}:
        from . import perturbations
        return getattr(perturbations, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    'AdversarialSample',
    'DataSource',
    'GradientOptimizer',
    'HideObject',
    'HideBelowThreshold',
    'OptimizationObjective',
    'OptimizationResult',
    'Perturbation',
    'Scorer',
    'SpoofObject',
    'NES',
    'PGD',
    'LinfImage',
    'PatchPerturbation',
    'SessionScorer',
]
