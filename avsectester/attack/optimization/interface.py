"""Contracts for gradient-based and sampling-based optimization over perturbation models.

The optimization algorithm, the threat model, the model access, and the goal are **four orthogonal
axes**, so PGD (or FGSM/MI-FGSM) is *not* patch-specific — a physical patch is just one
``Perturbation``:

    DataSource.sample(batch)          -> [AdversarialSample(x, target, context)]   (EoT falls out here)
    Perturbation.apply(sample, delta) -> x_adv                             (differentiable)
    Perturbation.project(delta)       -> delta                             (feasible set: Lp / box / mask)
    Scorer.target_score(x_adv, tgt)   -> confidence in [0,1]               (grad iff white-box)
    OptimizationObjective.loss(confidence)  -> scalar to MINIMIZE                (lower == attack succeeds)
    GradientOptimizer.run(...)           -> OptimizationResult

Axes:
  * **Algorithm** (``GradientOptimizer``): PGD, FGSM, ... — iterate; call only ``apply``/``project`` +
    scorer/objective, so they work for every threat model unchanged.
  * **Threat model** (``Perturbation``): owns the variable δ, how it enters the input, and the
    projection (the Lp ball / [0,1] box / patch mask live HERE, not in the algorithm). ``LinfImage``
    (additive ε-perturbation) and ``PatchPerturbation`` (EoT warp+composite) are two instances.
  * **Model** (``Scorer``) and **goal** (``OptimizationObjective``): unchanged across algorithms/threat models.

Fixed conventions: images and δ are RGB float in [0,1], channels-first ``(C, H, W)`` torch tensors on
the scorer's device; **lower objective loss == more successful attack**; a threat model that has a
physical realization exports it via ``Perturbation.export`` (a patch -> ``(Hp,Wp,4)`` uint8 for
``PatchAttack``; ``LinfImage`` exports ``None``). Only numpy/stdlib import here (``Tensor = Any``);
torch is lazy in the implementations, so this contract imports/tests without torch or CARLA.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

import avsectester.data_types as data_types

Tensor = Any  # torch.Tensor at runtime; kept abstract so the contract imports without torch


@dataclass
class AdversarialSample:
    """One clean input to attack. ``context`` carries threat-model specifics — e.g. a patch's calibrated
    sampling grid and visibility mask — and is empty for a whole-image perturbation."""

    x: Tensor  # (3, H, W) in [0, 1]
    target: data_types.TargetSpec
    context: dict = field(default_factory=dict)


class DataSource(ABC):
    """Yields batches of clean inputs. A single fixed image for an L-inf attack; sampled backgrounds
    + placements + photometric jitter (Expectation-over-Transformation) for a physical patch."""

    @abstractmethod
    def sample(self, batch: int) -> list[AdversarialSample]:
        """Return ``batch`` samples (with any per-sample randomness freshly drawn)."""


class Perturbation(ABC):
    """A threat model: the adversarial variable δ, how it enters the input, and its feasible set.

    The Lp ball / [0,1] box / patch mask are the threat model's concern (``project``), NOT the
    algorithm's — which is exactly why the same ``GradientOptimizer`` drives every threat model.
    """

    @abstractmethod
    def init(self) -> Tensor:
        """Return an initial δ (zeros / random) of this threat model's shape."""

    @abstractmethod
    def apply(self, sample: AdversarialSample, delta: Tensor) -> Tensor:
        """Form the adversarial input from a clean ``sample`` and δ (differentiable in δ)."""

    @abstractmethod
    def project(self, delta: Tensor) -> Tensor:
        """Project δ back onto the feasible set (in place or returned)."""

    def export(self, delta: Tensor) -> np.ndarray | None:
        """Physical realization of δ, if any — a patch returns ``(Hp,Wp,4)`` uint8; default None."""
        return None

    def smoothness(self, delta: Tensor) -> Tensor:
        """Differentiable total variation of δ, weighted by an optimizer's smoothness term.

        The default is the mean absolute difference between vertically and horizontally adjacent
        values of the CHW variable. A threat model that realizes only part of δ overrides it.
        """
        vertical = (delta[:, 1:, :] - delta[:, :-1, :]).abs().mean()
        horizontal = (delta[:, :, 1:] - delta[:, :, :-1]).abs().mean()
        return vertical + horizontal


class Scorer(ABC):
    """Access to the perception under attack: image -> target detection confidence in [0, 1].

    White-box scorers return a **differentiable** score; black-box scorers a detached scalar.
    ``differentiable`` tells the algorithm which regime it is in (PGD needs True).
    """

    differentiable: bool = True
    'Whether input-image gradients are supported by this component.'

    @abstractmethod
    def target_score(self, image: Tensor, target: data_types.TargetSpec) -> Tensor:
        """Return a scalar tensor in [0, 1] for detecting ``target`` in ``image``."""


class OptimizationObjective(ABC):
    """Shape the target confidence into a scalar loss to MINIMIZE (lower == attack succeeds)."""

    @abstractmethod
    def loss(self, score: Tensor) -> Tensor:
        """Map a target confidence in [0, 1] to the scalar loss to minimize."""


class HideObject(OptimizationObjective):
    """Suppress the target's detection by minimizing ``loss = score`` toward zero."""

    def loss(self, score: Tensor) -> Tensor:
        return score


class HideBelowThreshold(OptimizationObjective):
    """Suppress a detection until its score is ``margin`` below the decision ``threshold``.

    ``loss = max(0, score - (threshold - margin))``. A detector reports the target when its
    score reaches ``threshold``; samples already below ``threshold - margin`` contribute no loss or
    gradient, so updates concentrate on samples that would still be detected. The margin absorbs
    differences between optimization samples and deployment.
    """

    def __init__(self, threshold: float, margin: float = .1) -> None:
        if not (isinstance(threshold, (int, float)) and not isinstance(threshold, bool)
                and 0. <= threshold <= 1.):
            raise ValueError("threshold must be a number in [0, 1]")
        if not (isinstance(margin, (int, float)) and not isinstance(margin, bool)
                and 0. <= margin <= 1.):
            raise ValueError("margin must be a number in [0, 1]")
        self.threshold = float(threshold)
        self.margin = float(margin)

    def loss(self, score: Tensor) -> Tensor:
        excess = score - (self.threshold - self.margin)
        return (excess + abs(excess)) / 2  # max(0, excess) for tensors, arrays and numbers


class SpoofObject(OptimizationObjective):
    """Fabricate / keep a detection where the target box is: ``loss = 1 - score``."""

    def loss(self, score: Tensor) -> Tensor:
        return 1.0 - score


@dataclass
class OptimizationResult:
    """Output of an optimization run."""

    delta: Tensor  # the optimized adversarial variable
    export: np.ndarray | None  # physical realization (patch -> (Hp,Wp,4) uint8), else None
    history: list[float]  # objective value per step
    final_score: float  # mean target confidence at the end (lower == better hide)


class GradientOptimizer(ABC):
    """Optimization algorithm over a data source, perturbation, scorer and objective.

    Implementations sample inputs, evaluate the objective and project candidate perturbations.
    Gradient-based implementations require a differentiable scorer; sampling-based ones do not.
    """

    @abstractmethod
    def run(
        self,
        data: DataSource,
        perturbation: Perturbation,
        scorer: Scorer,
        objective: OptimizationObjective,
        *,
        monitor: Callable[[int, Tensor], None] | None = None,
    ) -> OptimizationResult:
        """Optimize δ with exact input gradients in PGD or sampled gradient estimates in NES.

        ``monitor(step, delta)``, when given, receives the number of completed updates and the
        current projected δ before the first update and after every update. It observes only and
        must not modify δ.
        """


__all__ = [
    'AdversarialSample',
    'DataSource',
    'Perturbation',
    'Scorer',
    'OptimizationObjective',
    'HideObject',
    'HideBelowThreshold',
    'SpoofObject',
    'OptimizationResult',
    'GradientOptimizer',
]
