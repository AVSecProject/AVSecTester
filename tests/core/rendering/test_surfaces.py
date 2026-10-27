"""Planar sampling filters textures over pixel footprints for rendering and optimization."""
import numpy as np
import pytest

from avsectester.rendering import PlaneSurface, render_surfaces
from avsectester.rendering.surfaces import plane_sampling, prepare_surface, sample_texture


def square(distance, half, texture):
    return PlaneSurface(np.array([[-half, -half, distance], [half, -half, distance],
                                  [half, half, distance], [-half, half, distance]]), texture)


def test_surface_sample(camera):
    surface = square(5., 1., np.full((8, 8, 4), [255, 0, 0, 255], np.uint8))
    sampler = prepare_surface(surface, camera, np.eye(4))
    rays = np.array([[0., 0., 1.], [1., 0., 1.]])
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    distances, colors = sampler.sample(rays)
    assert distances[0] == pytest.approx(5.) and np.isinf(distances[1])
    np.testing.assert_array_equal(colors[0], [255, 0, 0, 255])
    image, depth = render_surfaces((surface,), camera, np.eye(4))
    np.testing.assert_array_equal(image[40, 50], [255, 0, 0, 255])
    assert depth[40, 50] == pytest.approx(5.)
    # Covered pixels form the antialiased silhouette: partial edge coverage keeps finite depth.
    image, depth = render_surfaces((square(5., 1.05, surface.texture),), camera, np.eye(4))
    edges = np.isfinite(depth) & (image[..., 3] < 255)
    assert edges.any() and (image[edges, 3] > 0).all()
    assert (image[~np.isfinite(depth)] == 0).all()


def test_distant_texture_is_filtered(camera):
    pixels = np.indices((64, 64)).sum(axis=0) % 2
    checker = np.zeros((64, 64, 4), np.uint8)
    checker[..., :3] = (pixels * 255)[..., None]
    checker[..., 3] = 255
    # About ten pixels across: each pixel covers roughly six texels per axis.
    image, depth = render_surfaces((square(20., 2., checker),), camera, np.eye(4))
    interior = image[np.isfinite(depth) & (image[..., 3] == 255)]
    assert len(interior) and np.abs(interior[:, :3].astype(float) - 127.5).max() < 12


def test_sample_texture_gradient(camera):
    import torch
    surface = square(20., 2., np.full((32, 32, 4), 255, np.uint8))
    sampler = prepare_surface(surface, camera, np.eye(4))
    sampling = plane_sampling(sampler, camera, sampler.image_bounds(camera))
    texture = (torch.tensor(surface.texture, dtype=torch.float32).permute(2, 0, 1) / 255).requires_grad_(True)
    sample_texture(texture, sampling).sum().backward()
    assert (texture.grad > 0).all()
    with torch.no_grad():
        filtered = sample_texture(texture, sampling)
    assert filtered.shape == (4, *sampling.depth.shape)
