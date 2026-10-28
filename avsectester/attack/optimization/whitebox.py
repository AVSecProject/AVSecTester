"""Gradient-based optimization algorithms (torch). PGD here; FGSM / MI-FGSM would sit beside it against the contract.

PGD is threat-model-agnostic: it only draws samples, calls ``perturbation.apply``/``project`` and the
scorer/objective. The budget (Lp ball / [0,1] box / patch mask) lives in the ``Perturbation``. The
gradient-free black-box algorithms (NES) live in ``blackbox.py`` (numpy, no torch).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from avsectester.validation import require_nonnegative_integer, require_positive_integer

from .interface import (
    OptimizationObjective,
    OptimizationResult,
    DataSource,
    GradientOptimizer,
    Perturbation,
    Scorer,
)


@dataclass
class PGD(GradientOptimizer):
    """Projected Gradient Descent (sign-gradient descent on the loss, minimizing it).

    Hyper-parameters are the algorithm's only; the perturbation budget lives in the ``Perturbation``.
    ``tv_weight`` adds that weight times ``perturbation.smoothness(delta)`` to the loss, favoring
    smooth artifacts that survive printing and camera sampling. ``verbose`` prints the objective
    every ``log_every`` steps.
    """

    steps: int = 200
    step_size: float = 4.0 / 255.0
    batch: int = 4
    tv_weight: float = 0.0
    seed: int = 0
    verbose: bool = False
    log_every: int = 25

    def __post_init__(self) -> None:
        for name in ("steps", "batch", "log_every"):
            require_positive_integer(getattr(self, name), name)
        require_nonnegative_integer(self.seed, "seed")
        for name in ("step_size", "tv_weight"):
            value = getattr(self, name)
            if not torch.isfinite(torch.tensor(value)) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not isinstance(self.verbose, bool):
            raise TypeError("verbose must be a bool")

    def run(
        self,
        data: DataSource,
        perturbation: Perturbation,
        scorer: Scorer,
        objective: OptimizationObjective,
        *,
        monitor: Callable[[int, torch.Tensor], None] | None = None,
    ) -> OptimizationResult:
        if not scorer.differentiable:
            raise ValueError("PGD needs a differentiable (white-box) scorer; use NES for black-box")
        torch.manual_seed(self.seed)
        delta = perturbation.init()
        history: list[float] = []
        if monitor is not None:
            monitor(0, delta.detach())
        for step in range(self.steps):
            samples = data.sample(self.batch)
            loss = torch.zeros((), device=delta.device)
            for s in samples:
                score = scorer.target_score(perturbation.apply(s, delta), s.target)
                loss = loss + objective.loss(score)
            loss = loss / max(len(samples), 1)
            history.append(float(loss))
            if self.tv_weight:
                loss = loss + self.tv_weight * perturbation.smoothness(delta)
            (grad,) = torch.autograd.grad(loss, delta)
            with torch.no_grad():
                delta = perturbation.project(delta - self.step_size * grad.sign())
            delta.requires_grad_(True)
            if monitor is not None:
                monitor(step + 1, delta.detach())
            if self.verbose and (step % self.log_every == 0 or step == self.steps - 1):
                print(f"  step {step:4d}  objective {history[-1]:.4f}")

        with torch.no_grad():  # final target confidence, averaged over a fresh batch
            samples = data.sample(self.batch)
            final = float(
                sum(float(scorer.target_score(perturbation.apply(s, delta), s.target))
                    for s in samples) / max(len(samples), 1)
            )
        return OptimizationResult(
            delta=delta.detach(),
            export=perturbation.export(delta),
            history=history,
            final_score=final,
        )


__all__ = [
    'PGD',
]
