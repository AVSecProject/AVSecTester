"""Rendering images preserve their RGB values and independent RGBA storage."""
import numpy as np
import pytest
from PIL import Image

from avsectester.rendering import load_rgba_image


def test_load_rgba_image(tmp_path):
    rgba = np.full((2, 3, 4), [50, 80, 100, 255], np.uint8)
    path = tmp_path / "texture.png"
    Image.fromarray(rgba).save(path)
    for source in (rgba, rgba[..., :3], str(path), path):
        loaded = load_rgba_image(source)
        np.testing.assert_array_equal(loaded, rgba)
        loaded[:] = 0
    np.testing.assert_array_equal(rgba[0, 0], [50, 80, 100, 255])
    with pytest.raises(ValueError, match="texture"):
        load_rgba_image(np.zeros((2, 3), np.uint8))

