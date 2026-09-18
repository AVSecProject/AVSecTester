"""Gradient-based attacks (torch). PGD here; FGSM / MI-FGSM would sit beside it against the contract.

PGD is threat-model-agnostic: it only draws samples, calls ``perturbation.apply``/``project`` and the
scorer/objective. The budget (Lp ball / [0,1] box / patch mask) lives in the ``Perturbation``. The
gradient-FREE black-box attacks (NES) live in ``blackbox.py`` (numpy, no torch).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .interface import (
    AttackObjective,
    AttackResult,
    DataSource,
    GradientAttack,
    Perturbation,
    Scorer,
)


def _tv(delta: torch.Tensor) -> torch.Tensor:
    """Total variation — a smoothness regularizer for physical realizability."""
    dh = (delta[:, 1:, :] - delta[:, :-1, :]).abs().mean()
    dw = (delta[:, :, 1:] - delta[:, :, :-1]).abs().mean()
    return dh + dw


@dataclass
class PGD(GradientAttack):
    """Projected Gradient Descent (sign-gradient descent on the loss, minimizing it).

    Hyper-parameters are the algorithm's only; the perturbation budget lives in the ``Perturbation``.
    ``verbose`` prints the objective every ``log_every`` steps.
    """

    steps: int = 200
    step_size: float = 4.0 / 255.0
    batch: int = 4
    tv_weight: float = 0.0
    seed: int = 0
    verbose: bool = False
    log_every: int = 25

    def run(
        self,
        data: DataSource,
        perturbation: Perturbation,
        scorer: Scorer,
        objective: AttackObjective,
    ) -> AttackResult:
        if not scorer.differentiable:
            raise ValueError("PGD needs a differentiable (white-box) scorer; use NES for black-box")
        torch.manual_seed(self.seed)
        delta = perturbation.init()
        history: list[float] = []
        for step in range(self.steps):
            samples = data.sample(self.batch)
            loss = torch.zeros((), device=delta.device)
            for s in samples:
                score = scorer.target_score(perturbation.apply(s, delta), s.target)
                loss = loss + objective.loss(score)
            loss = loss / max(len(samples), 1)
            if self.tv_weight:
                loss = loss + self.tv_weight * _tv(delta)
            (grad,) = torch.autograd.grad(loss, delta)
            with torch.no_grad():
                delta = perturbation.project(delta - self.step_size * grad.sign())
            delta.requires_grad_(True)
            history.append(float(loss))
            if self.verbose and (step % self.log_every == 0 or step == self.steps - 1):
                print(f"  step {step:4d}  loss {history[-1]:.4f}")

        with torch.no_grad():  # final target confidence, averaged over a fresh batch
            samples = data.sample(self.batch)
            final = float(
                sum(float(scorer.target_score(perturbation.apply(s, delta), s.target))
                    for s in samples) / max(len(samples), 1)
            )
        return AttackResult(
            delta=delta.detach(),
            export=perturbation.export(delta),
            history=history,
            final_score=final,
        )
