"""Natural Evolution Strategies with batched candidate queries and deterministic sampling.

Each query calls the perturbation and scorer; the algorithm has no rendering assumptions.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

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
class NES(GradientOptimizer):
    """Estimate gradients over one seeded sample, querying candidates in batches per step.

    Each step uses ``population_size // 2`` antithetic Gaussian sampling pairs.
    """

    steps: int = 40
    population_size: int = 20
    sigma: float = 0.08
    learning_rate: float = 0.03
    seed: int = 0
    verbose: bool = False
    log_every: int = 5
    batch_size: int = 4

    def __post_init__(self) -> None:
        for name in ("steps", "population_size", "log_every", "batch_size"):
            require_positive_integer(getattr(self, name), name)
        if self.population_size < 2 or self.population_size % 2:
            raise ValueError("population_size must be an even integer of at least two")
        require_nonnegative_integer(self.seed, "seed")
        for name in ("sigma", "learning_rate"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not isinstance(self.verbose, bool):
            raise TypeError("verbose must be a bool")

    def run(
        self,
        data: DataSource,
        perturbation: Perturbation,
        scorer: Scorer,
        objective: OptimizationObjective,
        *,
        monitor: Callable[[int, np.ndarray], None] | None = None,
    ) -> OptimizationResult:
        if len(data) == 0:
            raise ValueError("NES requires a nonempty data source")
        rng = np.random.default_rng(self.seed)
        sample = data[int(rng.integers(len(data)))]

        def evaluate(candidates):
            """Return the scores and losses of the candidates, querying in batches."""
            images = [perturbation.apply(sample, candidate) for candidate in candidates]
            scores, losses = [], []
            for start in range(0, len(images), self.batch_size):
                batch = images[start:start + self.batch_size]
                values = np.asarray(scorer.target_scores(batch, [sample.target] * len(batch)))
                if values.shape != (len(batch),):
                    raise ValueError("NES scorer must return one score per input, shape (B,)")
                batch_losses = np.asarray(objective.loss(values))
                if batch_losses.shape != values.shape:
                    raise ValueError("NES objective must return one loss per input, shape (B,)")
                scores.extend(values.tolist())
                losses.extend(batch_losses.tolist())
            return scores, losses

        delta = perturbation.init()
        (final_score,), (loss,) = evaluate([delta])
        history = [float(loss)]
        if monitor is not None:
            monitor(0, delta.copy())
        pairs = self.population_size // 2
        for step in range(self.steps):
            noises = []
            candidates = []
            for _ in range(pairs):
                eps = rng.standard_normal(delta.shape).astype(delta.dtype)
                noises.append(eps)
                candidates.extend([perturbation.project(delta + self.sigma * eps),
                                   perturbation.project(delta - self.sigma * eps)])
            _, losses = evaluate(candidates)
            grad = np.zeros_like(delta)
            for index, eps in enumerate(noises):
                grad += (losses[2 * index] - losses[2 * index + 1]) * eps
            grad /= 2 * self.sigma * pairs
            delta = perturbation.project(delta - self.learning_rate * grad)
            (final_score,), (loss,) = evaluate([delta])
            history.append(float(loss))
            if monitor is not None:
                monitor(step + 1, delta.copy())
            if self.verbose and (step % self.log_every == 0 or step == self.steps - 1):
                print(f"  step {step:3d}  score {history[-1]:.4f}")
        return OptimizationResult(
            delta=delta,
            export=perturbation.export(delta),
            history=history,
            final_score=float(final_score),
        )


__all__ = [
    'NES',
]
