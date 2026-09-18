"""Gradient-free (black-box) attacks — numpy only, so they import/test without torch.

For a non-differentiable ``Scorer`` (e.g. the CARLA-in-the-loop physical render, where no autograd
crosses the renderer). NES estimates the gradient of the expected loss by antithetic Gaussian
sampling and descends it — every evaluation calls ``perturbation.apply`` (which, for the physical
threat model in ``render.py``, PAINTS + RENDERS in CARLA), so it optimizes exactly what the camera
sees. Query-limited in high dimensions; the white-box ``PGD`` in ``attacks.py`` is preferred where a
differentiable surrogate is available.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .interface import (
    AttackObjective,
    AttackResult,
    DataSource,
    GradientAttack,
    Perturbation,
    Scorer,
)


@dataclass
class NES(GradientAttack):
    """Natural Evolution Strategies — gradient-free; estimates a gradient by Gaussian sampling."""

    steps: int = 40
    popsize: int = 20  # samples per step (antithetic pairs = popsize // 2)
    sigma: float = 0.08
    lr: float = 0.03
    seed: int = 0
    verbose: bool = False
    log_every: int = 5

    def run(
        self,
        data: DataSource,
        perturbation: Perturbation,
        scorer: Scorer,
        objective: AttackObjective,
    ) -> AttackResult:
        rng = np.random.default_rng(self.seed)
        sample = data.sample(1)[0]

        def loss_at(d: np.ndarray) -> float:
            return float(objective.loss(scorer.target_score(perturbation.apply(sample, d), sample.target)))

        delta = perturbation.init()
        history = [loss_at(delta)]
        pairs = max(1, self.popsize // 2)
        for step in range(self.steps):
            grad = np.zeros_like(delta)
            for _ in range(pairs):
                eps = rng.standard_normal(delta.shape).astype(delta.dtype)
                lp = loss_at(perturbation.project(delta + self.sigma * eps))
                lm = loss_at(perturbation.project(delta - self.sigma * eps))
                grad += (lp - lm) * eps
            grad /= 2 * self.sigma * pairs
            delta = perturbation.project(delta - self.lr * grad)  # minimize the loss
            history.append(loss_at(delta))
            if self.verbose and (step % self.log_every == 0 or step == self.steps - 1):
                print(f"  step {step:3d}  score {history[-1]:.4f}")
        return AttackResult(
            delta=delta,
            export=perturbation.export(delta),
            history=history,
            final_score=history[-1],
        )
