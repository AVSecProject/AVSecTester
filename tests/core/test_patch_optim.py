"""Adversarial optimization: objective/geometry and CPU PGD/NES on small synthetic inputs.

The white-box detector + CARLA capture live in scripts/optimize_patch.py (needs GPU + a server); here
we cover the parts that run without them.
"""

import numpy as np
import pytest
from avsectester.attacks.optim.interface import HideObject, SpoofObject


def test_objectives_sign_convention():
    # lower loss == more successful attack: HideObject minimizes the score, SpoofObject maximizes it
    assert HideObject().loss(0.9) == 0.9
    assert HideObject().loss(0.1) == 0.1
    assert SpoofObject().loss(0.1) == pytest.approx(0.9)


def test_rear_panel_homography_maps_patch_into_the_box():
    from avsectester.attacks.optim.geometry import rear_panel_homography

    box = (100.0, 100.0, 300.0, 400.0)  # a 200x300 target box
    h = rear_panel_homography(box, (128, 128), wfrac=0.7, hfrac=0.5, ycenter=0.55)

    def apply(px, py):
        v = h @ np.array([px, py, 1.0])
        return v[:2] / v[2]

    corners = [apply(0, 0), apply(128, 0), apply(128, 128), apply(0, 128)]
    for cx, cy in corners:  # every patch corner lands inside the target box
        assert box[0] <= cx <= box[2] and box[1] <= cy <= box[3]
    center = apply(64, 64)
    assert center[0] == pytest.approx((box[0] + box[2]) / 2, abs=1.0)  # centered horizontally
    assert center[1] == pytest.approx(box[1] + 300 * 0.55, abs=1.0)  # at the rear-panel height


def test_pgd_reduces_the_objective_on_a_synthetic_scorer():
    import torch
    from avsectester.attacks.optim.attacks import PGD
    from avsectester.attacks.optim.interface import AdvSample, DataSource, Scorer, TargetSpec
    from avsectester.attacks.optim.perturbations import PatchPerturbation, rear_panel_homography

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
    h = torch.as_tensor(rear_panel_homography(box, (32, 32))).float()
    x = torch.full((3, 64, 80), 0.5)
    data = OneFrame(AdvSample(x=x, target=TargetSpec(box=box), context={"placement": h}))
    pert = PatchPerturbation(size=(32, 32), device="cpu", init="gray")

    result = PGD(steps=25, step_size=0.1, batch=1).run(data, pert, BrightnessScorer(), HideObject())

    assert result.history[-1] < result.history[0]  # HideObject minimizes -> patch darkens the region
    assert result.export.shape == (32, 32, 4) and result.export.dtype == np.uint8
    assert float(result.delta.mean()) < 0.5  # started at gray 0.5, optimized darker
    assert torch.all((result.delta >= 0) & (result.delta <= 1))


def test_nes_reduces_objective_black_box():
    # gradient-free NES on a non-differentiable synthetic scorer (numpy only, no torch/CARLA)
    from avsectester.attacks.optim.blackbox import NES
    from avsectester.attacks.optim.interface import (
        AdvSample,
        DataSource,
        Perturbation,
        Scorer,
        TargetSpec,
    )

    class AddPatch(Perturbation):  # the "image" is just delta; project to [0,1]
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
            return [AdvSample(x=None, target=TargetSpec((0, 0, 1, 1)))] * batch

    result = NES(steps=15, popsize=10, sigma=0.1, lr=0.3, seed=0).run(
        One(), AddPatch(), MeanBrightness(), HideObject()
    )
    assert result.history[-1] < result.history[0]  # HideObject minimizes -> brightness falls
    assert result.export.shape == (3, 4, 4)
    assert result.export.dtype == np.uint8
    assert np.all((result.delta >= 0) & (result.delta <= 1))


def test_patch_changes_only_its_placed_pixels():
    import torch
    from avsectester.attacks.optim.interface import AdvSample, TargetSpec
    from avsectester.attacks.optim.perturbations import PatchPerturbation

    # An integer translation makes the expected footprint independent of the warp implementation.
    placement = torch.tensor([[1, 0, 3], [0, 1, 2], [0, 0, 1]], dtype=torch.float32)
    background = torch.full((3, 8, 10), 0.25)
    sample = AdvSample(background, TargetSpec((3, 2, 6, 4)), {"placement": placement})
    patch = PatchPerturbation(size=(2, 3), device="cpu")

    attacked = patch.apply(sample, torch.ones((3, 2, 3)))

    expected = background.clone()
    expected[:, 2:4, 3:6] = 1.0
    torch.testing.assert_close(attacked, expected)
    torch.testing.assert_close(background, torch.full_like(background, 0.25))


def test_linf_attack_respects_budget_and_valid_image_range():
    import torch
    from avsectester.attacks.optim.interface import AdvSample, TargetSpec
    from avsectester.attacks.optim.perturbations import LinfImage

    perturb = LinfImage(epsilon=0.1, image_hw=(2, 3), device="cpu")
    background = torch.tensor([[[0.0, 0.5, 1.0], [1.0, 0.5, 0.0]]]).expand(3, -1, -1)
    proposed = torch.tensor([[[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]]).expand(3, -1, -1)
    sample = AdvSample(background, TargetSpec((0, 0, 3, 2)))

    attacked = perturb.apply(sample, perturb.project(proposed))

    expected = torch.tensor([[[0.0, 0.4, 0.9], [1.0, 0.6, 0.1]]]).expand(3, -1, -1)
    torch.testing.assert_close(attacked, expected)
    assert float((attacked - background).abs().max()) <= 0.1 + 1e-7
