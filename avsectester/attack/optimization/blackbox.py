"""Natural Evolution Strategies over arbitrary user-defined samples and perturbations.

Each query calls the perturbation and scorer; the algorithm has no rendering assumptions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

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
    """Natural Evolution Strategies — gradient-free; estimates a gradient by Gaussian sampling."""

    steps: int = 40
    population_size: int = 20  # samples per step (antithetic pairs = population_size // 2)
    sigma: float = 0.08
    learning_rate: float = 0.03
    seed: int = 0
    verbose: bool = False
    log_every: int = 5

    def __post_init__(self) -> None:
        for name in ("steps", "population_size", "log_every"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.population_size < 2 or self.population_size % 2:
            raise ValueError("population_size must be an even integer of at least two")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
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
    ) -> OptimizationResult:
        rng = np.random.default_rng(self.seed)
        sample = data.sample(1)[0]

        def loss_at(d: np.ndarray) -> float:
            return float(objective.loss(scorer.target_score(perturbation.apply(sample, d), sample.target)))

        delta = perturbation.init()
        history = [loss_at(delta)]
        pairs = max(1, self.population_size // 2)
        for step in range(self.steps):
            grad = np.zeros_like(delta)
            for _ in range(pairs):
                eps = rng.standard_normal(delta.shape).astype(delta.dtype)
                lp = loss_at(perturbation.project(delta + self.sigma * eps))
                lm = loss_at(perturbation.project(delta - self.sigma * eps))
                grad += (lp - lm) * eps
            grad /= 2 * self.sigma * pairs
            delta = perturbation.project(delta - self.learning_rate * grad)  # minimize the loss
            history.append(loss_at(delta))
            if self.verbose and (step % self.log_every == 0 or step == self.steps - 1):
                print(f"  step {step:3d}  score {history[-1]:.4f}")
        return OptimizationResult(
            delta=delta,
            export=perturbation.export(delta),
            history=history,
            final_score=history[-1],
        )
