"""Adversarial optimization: objective/geometry and CPU PGD/NES on small synthetic inputs.

Small synthetic inputs verify optimization without models, simulators or GPU access.
"""

import numpy as np
import pytest
from avsectester.attack.optimization import HideObject, SpoofObject


def test_objectives_sign_convention():
    # lower loss == more successful attack: HideObject minimizes the score, SpoofObject maximizes it
    assert HideObject().loss(0.9) == 0.9
    assert HideObject().loss(0.1) == 0.1
    assert SpoofObject().loss(0.1) == pytest.approx(0.9)


def test_pgd_reduces_the_objective_on_a_synthetic_scorer():
    import torch
    from avsectester.attack.optimization import PGD
    from avsectester.attack.optimization import AdversarialSample, DataSource, Scorer, TargetSpec
    from avsectester.attack.optimization import PatchPerturbation

    class BrightnessScorer(Scorer):  # differentiable, no model: "confidence" = brightness in the box
        differentiable = True

        def target_score(self, image, target):
            x0, y0, x1, y1 = (int(v) for v in target.box)
            return image[:, y0:y1, x0:x1].mean()

    class OneFrame(DataSource):
        def __init__(self, sample):
            self.sample_ = sample

        def sample(self, batch):
            return [self.sample_] * batch

    box = (10.0, 10.0, 70.0, 54.0)
    grid = np.full((64, 80, 2), 2., np.float32)
    mask = np.zeros((1, 64, 80), np.float32)
    yy, xx = np.meshgrid(np.linspace(-.9, .9, 44), np.linspace(-.9, .9, 60), indexing="ij")
    grid[10:54, 10:70] = np.stack((xx, yy), axis=-1)
    mask[:, 10:54, 10:70] = 1
    x = torch.full((3, 64, 80), 0.5)
    data = OneFrame(AdversarialSample(x=x, target=TargetSpec(box=box), context={"sampling_grid": grid, "mask": mask}))
    pert = PatchPerturbation(size=(32, 32), device="cpu", init="gray")

    result = PGD(steps=25, step_size=0.1, batch=1).run(data, pert, BrightnessScorer(), HideObject())

    assert result.history[-1] < result.history[0]  # HideObject minimizes -> patch darkens the region
    assert result.export.shape == (32, 32, 4) and result.export.dtype == np.uint8
    assert float(result.delta.mean()) < 0.5  # started at gray 0.5, optimized darker
    assert torch.all((result.delta >= 0) & (result.delta <= 1))


def test_nes_reduces_objective_black_box():
    # gradient-free NES on a non-differentiable synthetic scorer (numpy only, no torch/CARLA)
    from avsectester.attack.optimization import NES
    from avsectester.attack.optimization import (
        AdversarialSample,
        DataSource,
        Perturbation,
        Scorer,
        TargetSpec,
    )

    class AddPatch(Perturbation):  # the "image" is just delta. Project to [0,1]
        def init(self):
            return np.full((3, 4, 4), 0.8, np.float32)

        def apply(self, sample, delta):
            return delta

        def project(self, delta):
            return np.clip(delta, 0, 1)

        def export(self, delta):
            return (np.clip(delta, 0, 1) * 255).astype(np.uint8)

    class MeanBrightness(Scorer):
        differentiable = False

        def target_score(self, image, target):
            return float(np.mean(image))

    class One(DataSource):
        def sample(self, batch):
            return [AdversarialSample(x=None, target=TargetSpec((0, 0, 1, 1)))] * batch

    result = NES(steps=15, population_size=10, sigma=0.1, learning_rate=0.3, seed=0).run(
        One(), AddPatch(), MeanBrightness(), HideObject()
    )
    assert result.history[-1] < result.history[0]  # HideObject minimizes -> brightness falls
    assert result.export.shape == (3, 4, 4)
    assert result.export.dtype == np.uint8
    assert np.all((result.delta >= 0) & (result.delta <= 1))


def test_patch_changes_only_its_placed_pixels():
    import torch
    from avsectester.attack.optimization import AdversarialSample, TargetSpec
    from avsectester.attack.optimization import PatchPerturbation

    # An integer translation makes the expected footprint independent of the warp implementation.
    grid = np.full((8, 10, 2), 2., np.float32)
    grid[2:4, 3:6] = 0
    mask = np.zeros((1, 8, 10), np.float32)
    mask[:, 2:4, 3:6] = 1
    background = torch.full((3, 8, 10), 0.25)
    sample = AdversarialSample(background, TargetSpec((3, 2, 6, 4)), {"sampling_grid": grid, "mask": mask})
    patch = PatchPerturbation(size=(2, 3), device="cpu")

    attacked = patch.apply(sample, torch.ones((3, 2, 3)))

    expected = background.clone()
    expected[:, 2:4, 3:6] = 1.0
    torch.testing.assert_close(attacked, expected)
    torch.testing.assert_close(background, torch.full_like(background, 0.25))


def test_linf_attack_respects_budget_and_valid_image_range():
    import torch
    from avsectester.attack.optimization import AdversarialSample, TargetSpec
    from avsectester.attack.optimization import LinfImage

    perturb = LinfImage(epsilon=0.1, image_hw=(2, 3), device="cpu")
    background = torch.tensor([[[0.0, 0.5, 1.0], [1.0, 0.5, 0.0]]]).expand(3, -1, -1)
    proposed = torch.tensor([[[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]]).expand(3, -1, -1)
    sample = AdversarialSample(background, TargetSpec((0, 0, 3, 2)))

    attacked = perturb.apply(sample, perturb.project(proposed))

    expected = torch.tensor([[[0.0, 0.4, 0.9], [1.0, 0.6, 0.1]]]).expand(3, -1, -1)
    torch.testing.assert_close(attacked, expected)
    assert float((attacked - background).abs().max()) <= 0.1 + 1e-7


@pytest.mark.parametrize("algorithm, options", [
    ("PGD", {"steps": 0}), ("PGD", {"batch": True}), ("PGD", {"step_size": float("nan")}),
    ("PGD", {"seed": -1}), ("NES", {"sigma": 0}), ("NES", {"learning_rate": float("inf")}),
    ("NES", {"population_size": 1}), ("NES", {"seed": True}),
])
def test_algorithm_parameters_are_checked_at_construction(algorithm, options):
    from avsectester.attack.optimization import NES
    from avsectester.attack.optimization import PGD
    with pytest.raises((TypeError, ValueError)):
        {"PGD": PGD, "NES": NES}[algorithm](**options)


@pytest.mark.parametrize("options", [{"init": "typo"}, {"size": (0, 4)},
                                     {"alpha": np.zeros((4, 4), dtype=float)}])
def test_patch_perturbation_rejects_unknown_or_invalid_construction(options):
    from avsectester.attack.optimization import PatchPerturbation
    with pytest.raises(ValueError):
        PatchPerturbation(device="cpu", **options)
