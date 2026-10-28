"""Optimization algorithms share user-defined samples, parameters and objectives."""

import numpy as np
import pytest
import torch

from avsectester.attack.optimization import (
    AdversarialSample, DataSource, HideBelowThreshold, HideObject, NES, PGD, Perturbation, Scorer, SpoofObject,
)
from avsectester.data_types import TargetSpec


class ImageSource(DataSource):
    def sample(self, batch):
        return [AdversarialSample(None, TargetSpec((0, 0, 4, 4))) for _ in range(batch)]


class TensorImage(Perturbation):
    def init(self):
        return torch.full((3, 4, 4), .5, requires_grad=True)

    def apply(self, sample, delta):
        return delta

    def project(self, delta):
        return delta.clamp(0, 1)

    def export(self, delta):
        return (delta.detach().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)


class ArrayImage(Perturbation):
    def init(self):
        return np.full((3, 4, 4), .5, np.float32)

    def apply(self, sample, delta):
        return delta

    def project(self, delta):
        return np.clip(delta, 0, 1)

    def export(self, delta):
        return (delta.transpose(1, 2, 0) * 255).astype(np.uint8)


class BrightnessScore(Scorer):
    def target_score(self, image, target):
        return image.mean()


@pytest.mark.parametrize("optimizer,perturbation", [
    pytest.param(PGD(steps=8, step_size=.1, batch=1),
                 TensorImage(), id="PGD"),
    pytest.param(NES(steps=12, population_size=10, learning_rate=.3, sigma=.1, seed=0),
                 ArrayImage(), id="NES"),
])
@pytest.mark.parametrize("objective", [HideObject(), SpoofObject()], ids=["hide", "spoof"])
def test_optimize(optimizer, perturbation, objective):
    observed = []
    result = optimizer.run(ImageSource(), perturbation, BrightnessScore(), objective,
                           monitor=lambda step, delta: observed.append((step, delta.shape)))
    assert result.history[-1] < result.history[0]
    assert observed == [(step, (3, 4, 4)) for step in range(optimizer.steps + 1)]
    assert result.export.shape == (4, 4, 3) and result.export.dtype == np.uint8
    with pytest.raises(ValueError, match="positive integer"):
        type(optimizer)(steps=0)


class CheckerImage(TensorImage):
    def init(self):
        image = torch.zeros(3, 4, 4)
        image[:, ::2, ::2] = image[:, 1::2, 1::2] = 1.
        return image.requires_grad_(True)


class ConstantScore(Scorer):
    def target_score(self, image, target):
        return image.sum() * 0 + .5


def test_pgd_smoothness():
    checker = CheckerImage()
    assert checker.smoothness(checker.init()).item() == pytest.approx(2.)
    plain = PGD(steps=4, step_size=.1, batch=1).run(ImageSource(), checker, ConstantScore(), HideObject())
    smooth = PGD(steps=4, step_size=.1, batch=1, tv_weight=1.).run(
        ImageSource(), checker, ConstantScore(), HideObject())
    torch.testing.assert_close(plain.delta, checker.init().detach())
    assert checker.smoothness(smooth.delta) < checker.smoothness(plain.delta)
    # The history records the attack objective only, not the weighted smoothness term.
    assert smooth.history == plain.history == [.5] * 4


def test_hide_below_threshold():
    objective = HideBelowThreshold(.5, margin=.1)
    scores = torch.tensor([.2, .35, .45, .9], requires_grad=True)
    losses = objective.loss(scores)
    torch.testing.assert_close(losses, torch.tensor([0., 0., .05, .5]))
    losses.sum().backward()
    # Scores already below threshold - margin receive no gradient, so updates go to detected samples.
    torch.testing.assert_close(scores.grad, torch.tensor([0., 0., 1., 1.]))
    assert objective.loss(np.float64(.7)) == pytest.approx(.3)
    for threshold, margin in ((1.5, .1), (.5, -.1), (True, .1)):
        with pytest.raises(ValueError, match="threshold|margin"):
            HideBelowThreshold(threshold, margin)
