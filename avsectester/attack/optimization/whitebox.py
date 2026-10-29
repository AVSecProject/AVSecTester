"""Epoch-based projected gradient descent over map-style perturbation samples.

PGD is independent of the threat model: it uses ``Perturbation.apply`` and ``Perturbation.project``
with the scorer and objective. Budgets such as Lp balls, [0, 1] boxes and patch masks belong to
``Perturbation``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
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
    """Projected sign-gradient descent with one batched forward, backward and update per batch.

    Each epoch shuffles the samples with ``seed`` and keeps the final smaller batch. Images within
    a batch must have equal dimensions. ``tv_weight`` adds that weight times
    ``perturbation.smoothness(delta)`` to the loss, favoring smooth artifacts that survive printing
    and camera sampling; history contains only the mean sample objective per epoch, without it.
    ``verbose`` prints the objective every ``log_every`` epochs.
    """

    epochs: int = 10
    batch_size: int = 4
    step_size: float = 4.0 / 255.0
    tv_weight: float = 0.0
    seed: int = 0
    verbose: bool = False
    log_every: int = 1

    def __post_init__(self) -> None:
        for name in ("epochs", "batch_size", "log_every"):
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
        if len(data) == 0:
            raise ValueError("PGD requires a nonempty data source")
        rng = np.random.default_rng(self.seed)
        torch.manual_seed(self.seed)
        delta = perturbation.init()
        history: list[float] = []
        if monitor is not None:
            monitor(0, delta.detach())
        for epoch in range(self.epochs):
            order = rng.permutation(len(data))
            total = 0.
            for start in range(0, len(order), self.batch_size):
                samples = [data[int(i)] for i in order[start:start + self.batch_size]]
                images = _images(samples, perturbation, delta)
                losses = objective.loss(scorer.target_scores(images, [s.target for s in samples]))
                if not torch.is_tensor(losses) or losses.shape != (len(samples),):
                    raise ValueError("PGD objective must return one loss per image, shape (B,)")
                loss = losses.mean()
                total += float(losses.detach().sum())
                if self.tv_weight:
                    loss = loss + self.tv_weight * perturbation.smoothness(delta)
                grad = _gradient(loss, delta)
                with torch.no_grad():
                    delta = perturbation.project(delta - self.step_size * grad.sign())
                delta.requires_grad_(True)
            history.append(total / len(data))
            if monitor is not None:
                monitor(epoch + 1, delta.detach())
            if self.verbose and ((epoch + 1) % self.log_every == 0 or epoch + 1 == self.epochs):
                print(f"  epoch {epoch + 1:4d}  objective {history[-1]:.4f}")

        total_score = 0.
        with torch.no_grad():
            for start in range(0, len(data), self.batch_size):
                samples = [data[i] for i in range(start, min(start + self.batch_size, len(data)))]
                scores = scorer.target_scores(_images(samples, perturbation, delta),
                                              [s.target for s in samples])
                if not torch.is_tensor(scores) or scores.shape != (len(samples),):
                    raise ValueError("PGD scorer must return one score per image, shape (B,)")
                total_score += float(scores.sum())
        return OptimizationResult(delta=delta.detach(), export=perturbation.export(delta),
                                  history=history, final_score=total_score / len(data))


def _images(samples, perturbation, delta):
    images = [perturbation.apply(sample, delta) for sample in samples]
    if any(image.shape != images[0].shape for image in images):
        raise ValueError("Images in the same batch must have equal dimensions")
    return torch.stack(images)


def _gradient(loss: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """Return the input gradient, or zero for an input-independent objective."""
    if not loss.requires_grad:
        return torch.zeros_like(delta)
    (grad,) = torch.autograd.grad(loss, delta, allow_unused=True)
    return torch.zeros_like(delta) if grad is None else grad


__all__ = [
    'PGD',
]
