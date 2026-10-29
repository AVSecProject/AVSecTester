"""Optimization contracts for map-style datasets, perturbations and batched scoring.

Four orthogonal axes separate the algorithm (``GradientOptimizer``), threat model
(``Perturbation``), scoring model (``Scorer``) and goal (``OptimizationObjective``). Algorithms
own iteration and updates; perturbations own projection and physical realization. Lower objective
loss means a more successful attack.

``DataSource[index]`` yields one ``AdversarialSample``. ``Perturbation.apply`` forms one adversarial
input; ``Scorer.target_scores`` evaluates a batch and ``OptimizationObjective.loss`` preserves its
shape. Differentiable image batches are BCHW float RGB tensors in [0, 1]. Other scorers declare
their representation and return one score per input, including physical quantities such as meters.

This interface depends only on NumPy, the standard library and framework data records. Torch is
loaded lazily through implementation modules; importing the contracts does not import torch.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
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
    """A map-style dataset whose random augmentations are redrawn on each index access.

    EOT belongs in ``__getitem__``, like ordinary dataset augmentation. Algorithms own traversal
    and batching.
    """

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of samples in one epoch."""

    @abstractmethod
    def __getitem__(self, index: int) -> AdversarialSample:
        """Return one sample; an out-of-range index raises ``IndexError``."""


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
    """Score a batch with one score per input; descriptors or objectives define the direction.

    Differentiable inputs are finite BCHW float RGB tensors in [0, 1], returning a (B,) tensor
    differentiable with respect to the images. Other formats use sequences of B inputs and
    return B scores. Detector scores are confidences in [0, 1]; other scorers may return physical
    quantities such as distances.
    """

    differentiable: bool = True
    'Whether input-image gradients are supported by this component.'

    @abstractmethod
    def target_scores(self, images: Tensor, targets: Sequence[data_types.TargetSpec]) -> Tensor:
        """Return one score per input; mismatched input and target lengths raise ``ValueError``."""


class OptimizationObjective(ABC):
    """Transform scores elementwise into losses to minimize."""

    @abstractmethod
    def loss(self, score: Tensor) -> Tensor:
        """Map (B,) scores to (B,) losses with elementwise operations."""


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
    """Output of optimization: PGD history is mean sample loss per epoch; NES history is loss per step."""

    delta: Tensor  # the optimized adversarial variable
    export: np.ndarray | None  # physical realization (patch -> (Hp,Wp,4) uint8), else None
    history: list[float]  # PGD epoch mean (entry i = epoch i+1); NES per-step loss
    final_score: float  # final scorer output: PGD mean over all samples; NES on its fixed sample


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

        ``monitor(index, delta)`` observes the initial projected variable at index 0, then completed
        epochs for PGD or completed steps for NES. It must not modify δ.
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
