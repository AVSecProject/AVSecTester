"""Gradient-based adversarial optimization — PGD & friends over any threat model.

The algorithm-facing contract is in :mod:`~avsectester.attacks.optim.interface` (pure, no torch):
``GradientAttack``/``PGD`` (algorithm) is orthogonal to ``Perturbation`` (threat model:
``LinfImage`` / ``PatchPerturbation``), ``Scorer`` (model access) and ``AttackObjective`` (goal).

Concrete pieces are torch-backed and added incrementally: ``perturbations.py`` (LinfImage,
PatchPerturbation with the differentiable EoT paste), ``scorers.py`` (white-box mmdet detector), and
``attacks.py`` (the PGD loop). A ``PatchPerturbation``'s ``export`` yields the texture that
:class:`avsectester.attacks.physical_patch.PhysicalPatch` deploys in CARLA.
"""

from .interface import (
    AdvSample,
    AttackObjective,
    AttackResult,
    DataSource,
    GradientAttack,
    HideObject,
    Perturbation,
    Scorer,
    SpoofObject,
    TargetSpec,
)

__all__ = [
    "AdvSample",
    "AttackObjective",
    "AttackResult",
    "DataSource",
    "GradientAttack",
    "HideObject",
    "Perturbation",
    "Scorer",
    "SpoofObject",
    "TargetSpec",
]
