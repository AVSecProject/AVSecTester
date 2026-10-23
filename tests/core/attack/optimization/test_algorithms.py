"""Optimization algorithms share user-defined samples, parameters and objectives."""

import numpy as np
import pytest

from avsectester.attack.optimization import AdversarialSample, DataSource, HideObject, NES, Perturbation, Scorer, SpoofObject
from avsectester.data_types import TargetSpec


class ImageSource(DataSource):
    def sample(self, batch):
        return [AdversarialSample(None, TargetSpec((0, 0, 4, 4))) for _ in range(batch)]


class ImagePerturbation(Perturbation):
    def __init__(self, *, tensor):
        self.tensor = tensor

    def init(self):
        if self.tensor:
            import torch
            return torch.full((3, 4, 4), .5, requires_grad=True)
        return np.full((3, 4, 4), .5, np.float32)

    def apply(self, sample, delta):
        return delta

    def project(self, delta):
        return delta.clamp(0, 1) if self.tensor else np.clip(delta, 0, 1)

    def export(self, delta):
        value = delta.detach().numpy() if self.tensor else delta
        return (value.transpose(1, 2, 0) * 255).astype(np.uint8)


class BrightnessScore(Scorer):
    def target_score(self, image, target):
        return image.mean()


def make_pgd(**options):
    pytest.importorskip("torch")
    from avsectester.attack.optimization import PGD
    return PGD(**options)


@pytest.mark.parametrize("make_optimizer,options,tensor", [
    pytest.param(make_pgd, {"steps": 8, "step_size": .1, "batch": 1}, True, id="PGD"),
    pytest.param(NES, {"steps": 12, "population_size": 10, "learning_rate": .3,
                       "sigma": .1, "seed": 0}, False, id="NES"),
])
@pytest.mark.parametrize("objective", [HideObject(), SpoofObject()], ids=["hide", "spoof"])
def test_optimize(make_optimizer, options, tensor, objective):
    optimizer = make_optimizer(**options)
    result = optimizer.run(ImageSource(), ImagePerturbation(tensor=tensor),
                           BrightnessScore(), objective)
    assert result.history[-1] < result.history[0]
    assert result.export.shape == (4, 4, 3) and result.export.dtype == np.uint8
    with pytest.raises(ValueError, match="positive integer"):
        type(optimizer)(steps=0)


def test_linf_image_perturbation():
    torch = pytest.importorskip("torch")
    from avsectester.attack.optimization import LinfImage

    image = torch.full((3, 4, 5), .25)
    perturbation = LinfImage(epsilon=.1, image_hw=(4, 5), device="cpu")
    sample = AdversarialSample(image, TargetSpec((0, 0, 5, 4)))
    changed = perturbation.apply(sample, perturbation.project(torch.ones_like(image)))
    torch.testing.assert_close(changed, torch.full_like(image, .35))
    torch.testing.assert_close(image, torch.full_like(image, .25))


def test_patch_perturbation():
    torch = pytest.importorskip("torch")
    from avsectester.attack.optimization import PatchPerturbation

    image = torch.full((3, 4, 5), .25)
    perturbation = PatchPerturbation(size=(2, 2), device="cpu")
    grid = np.full((4, 5, 2), 2., np.float32)
    grid[1:3, 2:4] = 0
    mask = np.zeros((1, 4, 5), np.float32)
    mask[:, 1:3, 2:4] = 1
    sample = AdversarialSample(image, TargetSpec((2, 1, 4, 3)),
                               {"sampling_grid": grid, "mask": mask})
    changed = perturbation.apply(sample, torch.ones((3, 2, 2)))
    expected = image.clone()
    expected[:, 1:3, 2:4] = 1.
    torch.testing.assert_close(changed, expected)
    assert perturbation.export(torch.ones((3, 2, 2))).shape == (2, 2, 4)
    torch.testing.assert_close(image, torch.full_like(image, .25))


def test_patch_perturbation_matches_deployment():
    torch = pytest.importorskip("torch")
    from avsectester.attack.optimization import PatchPerturbation
    from avsectester.geometry.cameras import PinholeCamera
    from avsectester.rendering import PlaneSurface, blend_rgba, render_surfaces
    from avsectester.rendering.surfaces import plane_sampling, prepare_surface

    camera = PinholeCamera(np.array([[50., 0., 50.], [0., 50., 40.], [0., 0., 1.]]), 100, 80)
    perturbation = PatchPerturbation(size=(16, 16), device="cpu", seed=3)
    delta = perturbation.init()
    texture = perturbation.export(delta)
    surface = PlaneSurface(np.array([[-1., -1., 12.], [1., -1., 12.], [1., 1., 12.], [-1., 1., 12.]]),
                           texture)
    sampler = prepare_surface(surface, camera, np.eye(4))
    sampling = plane_sampling(sampler, camera, sampler.image_bounds(camera))
    background = np.full((80, 100, 3), 90, np.uint8)
    rgba, depth = render_surfaces((surface,), camera, np.eye(4))
    deployed = blend_rgba(background, rgba, np.isfinite(depth))
    visible = np.isfinite(depth).astype(np.float32)[None]
    x = torch.as_tensor(background).permute(2, 0, 1).float() / 255
    optimized = perturbation.apply(AdversarialSample(x, TargetSpec((0, 0, 1, 1)),
                                                     {"plane_sampling": sampling, "mask": visible}),
                                   torch.as_tensor(texture[..., :3]).permute(2, 0, 1).float() / 255)
    # Optimization and deployment use one sampler; only 8-bit quantization differs.
    np.testing.assert_allclose(optimized.permute(1, 2, 0).detach().numpy() * 255, deployed, atol=1.5)
    delta.grad = None
    perturbation.apply(AdversarialSample(x, TargetSpec((0, 0, 1, 1)),
                                         {"plane_sampling": sampling, "mask": visible}), delta).sum().backward()
    assert (delta.grad.abs() > 0).all()
