"""Optimization algorithms share user-defined samples, parameters and objectives."""

import numpy as np
import pytest
import torch

from avsectester.attack.optimization import AdversarialSample, DataSource, HideObject, NES, PGD, Perturbation, Scorer, SpoofObject
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
    result = optimizer.run(ImageSource(), perturbation, BrightnessScore(), objective)
    assert result.history[-1] < result.history[0]
    assert result.export.shape == (4, 4, 3) and result.export.dtype == np.uint8
    with pytest.raises(ValueError, match="positive integer"):
        type(optimizer)(steps=0)
