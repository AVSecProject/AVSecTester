"""Relocated planar sampling retains its lens geometry and opaque-texture rule."""
import numpy as np
import pytest

from avsectester.rendering import PlaneSurface, render_surfaces
from avsectester.rendering.surfaces import prepare_surface


def test_surface_sample(camera):
    texture = np.full((8, 8, 4), [255, 0, 0, 255], np.uint8)
    surface = PlaneSurface(np.array([[-1., -1., 5.], [1., -1., 5.],
                                    [1., 1., 5.], [-1., 1., 5.]]), texture)
    sampler = prepare_surface(surface, camera, np.eye(4))
    rays = np.array([[0., 0., 1.], [1., 0., 1.]])
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    distances, colors = sampler.sample(rays)
    assert distances[0] == pytest.approx(5.) and np.isinf(distances[1])
    np.testing.assert_array_equal(colors[0], [255, 0, 0, 255])
    image, depth = render_surfaces((surface,), camera, np.eye(4))
    np.testing.assert_array_equal(image[40, 50], [255, 0, 0, 255])
    assert depth[40, 50] == pytest.approx(5.)
