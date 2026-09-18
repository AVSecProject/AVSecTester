"""Contract for gradient-based adversarial optimization — PGD and friends, over any threat model.

The optimization algorithm, the threat model, the model access, and the goal are **four orthogonal
axes**, so PGD (or FGSM/MI-FGSM) is *not* patch-specific — a physical patch is just one
``Perturbation``:

    DataSource.sample(batch)          -> [AdvSample(x, target, context)]   (EoT falls out here)
    Perturbation.apply(sample, delta) -> x_adv                             (differentiable)
    Perturbation.project(delta)       -> delta                             (feasible set: Lp / box / mask)
    Scorer.target_score(x_adv, tgt)   -> confidence in [0,1]               (grad iff white-box)
    AttackObjective.loss(confidence)  -> scalar to MINIMIZE                (lower == attack succeeds)
    GradientAttack.run(...)           -> AttackResult

Axes:
  * **Algorithm** (``GradientAttack``): PGD, FGSM, ... — iterate; call only ``apply``/``project`` +
    scorer/objective, so they work for every threat model unchanged.
  * **Threat model** (``Perturbation``): owns the variable δ, how it enters the input, and the
    projection (the Lp ball / [0,1] box / patch mask live HERE, not in the algorithm). ``LinfImage``
    (additive ε-perturbation) and ``PatchPerturbation`` (EoT warp+composite) are two instances.
  * **Model** (``Scorer``) and **goal** (``AttackObjective``): unchanged across algorithms/threat models.

Fixed conventions: images and δ are RGB float in [0,1], channels-first ``(C, H, W)`` torch tensors on
the scorer's device; **lower objective loss == more successful attack**; a threat model that has a
physical realization exports it via ``Perturbation.export`` (a patch -> ``(Hp,Wp,4)`` uint8 for
``PhysicalPatch``; ``LinfImage`` exports ``None``). Only numpy/stdlib import here (``Tensor = Any``);
torch is lazy in the implementations, so this contract imports/tests without torch or CARLA.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

Tensor = Any  # torch.Tensor at runtime; kept abstract so the contract imports without torch


@dataclass
class TargetSpec:
    """The object to attack, in image-space pixels (``box`` = xyxy) plus an optional class label."""

    box: tuple[float, float, float, float]
    label: str | None = None


@dataclass
class AdvSample:
    """One clean input to attack. ``context`` carries threat-model specifics — e.g. a patch's
    ``{"placement": H}`` homography — and is empty for a whole-image perturbation."""

    x: Tensor  # (3, H, W) in [0, 1]
    target: TargetSpec
    context: dict = field(default_factory=dict)


class DataSource(ABC):
    """Yields batches of clean inputs. A single fixed image for an L-inf attack; sampled backgrounds
    + placements + photometric jitter (Expectation-over-Transformation) for a physical patch."""

    @abstractmethod
    def sample(self, batch: int) -> list[AdvSample]:
        """Return ``batch`` samples (with any per-sample randomness freshly drawn)."""


class Perturbation(ABC):
    """A threat model: the adversarial variable δ, how it enters the input, and its feasible set.

    The Lp ball / [0,1] box / patch mask are the threat model's concern (``project``), NOT the
    algorithm's — which is exactly why the same ``GradientAttack`` drives every threat model.
    """

    @abstractmethod
    def init(self) -> Tensor:
        """Return an initial δ (zeros / random) of this threat model's shape."""

    @abstractmethod
    def apply(self, sample: AdvSample, delta: Tensor) -> Tensor:
        """Form the adversarial input from a clean ``sample`` and δ (differentiable in δ)."""

    @abstractmethod
    def project(self, delta: Tensor) -> Tensor:
        """Project δ back onto the feasible set (in place or returned)."""

    def export(self, delta: Tensor) -> np.ndarray | None:
        """Physical realization of δ, if any — a patch returns ``(Hp,Wp,4)`` uint8; default None."""
        return None


class Scorer(ABC):
    """Access to the perception under attack: image -> target detection confidence in [0, 1].

    White-box scorers return a **differentiable** score; black-box scorers a detached scalar.
    ``differentiable`` tells the algorithm which regime it is in (PGD needs True).
    """

    differentiable: bool = True

    @abstractmethod
    def target_score(self, image: Tensor, target: TargetSpec) -> Tensor:
        """Confidence in [0, 1] that ``target`` is detected in ``image`` (scalar tensor)."""


class AttackObjective(ABC):
    """Shape the target confidence into a scalar loss to MINIMIZE (lower == attack succeeds)."""

    @abstractmethod
    def loss(self, score: Tensor) -> Tensor:
        """Map a target confidence in [0, 1] to the scalar loss to minimize."""


class HideObject(AttackObjective):
    """Suppress the target's detection: ``loss = score`` (minimizing drives its confidence to 0)."""

    def loss(self, score: Tensor) -> Tensor:
        return score


class SpoofObject(AttackObjective):
    """Fabricate / keep a detection where the target box is: ``loss = 1 - score``."""

    def loss(self, score: Tensor) -> Tensor:
        return 1.0 - score


@dataclass
class AttackResult:
    """Output of an optimization run."""

    delta: Tensor  # the optimized adversarial variable
    export: np.ndarray | None  # physical realization (patch -> (Hp,Wp,4) uint8), else None
    history: list[float]  # objective value per step
    final_score: float  # mean target confidence at the end (lower == better hide)


class GradientAttack(ABC):
    """A gradient-based attack algorithm (PGD, FGSM, ...), generic over the threat model.

    ``run`` iterates: draw an EoT batch from ``data``, ``perturbation.apply`` δ into each input,
    take ``objective.loss(scorer.target_score(...))`` averaged over the batch, backprop to δ, take an
    algorithm-specific step, then ``perturbation.project``. Requires ``scorer.differentiable``.
    """

    @abstractmethod
    def run(
        self,
        data: DataSource,
        perturbation: Perturbation,
        scorer: Scorer,
        objective: AttackObjective,
    ) -> AttackResult:
        """Optimize δ and return the result. Concrete algorithms (PGD, ...) live in ``attacks.py``."""
