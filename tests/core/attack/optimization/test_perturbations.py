"""Image and patch perturbations preserve their public constraints and deployment sampling."""
import numpy as np
import torch

from avsectester.attack.optimization import AdversarialSample
from avsectester.data_types import TargetSpec


def test_linf_image_perturbation():
    from avsectester.attack.optimization import LinfImage

    image = torch.full((3, 4, 5), .25)
    perturbation = LinfImage(epsilon=.1, image_hw=(4, 5), device="cpu")
    sample = AdversarialSample(image, TargetSpec((0, 0, 5, 4)))
    changed = perturbation.apply(sample, perturbation.project(torch.ones_like(image)))
    torch.testing.assert_close(changed, torch.full_like(image, .35))
    torch.testing.assert_close(image, torch.full_like(image, .25))


def test_patch_perturbation():
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


def test_patch_perturbation_matches_deployment(camera):
    from avsectester.attack.optimization import PatchPerturbation
    from avsectester.rendering import PlaneSurface, blend_rgba, render_surfaces
    from avsectester.rendering.surfaces import plane_sampling, prepare_surface

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
