"""Textures preserve explicit pixel values and independent RGBA storage."""
import numpy as np
import pytest
from PIL import Image

from avsectester.rendering.textures import (
    CheckerboardTexture, Texture, load_rgba_image,
)


def test_load_rgba_image(tmp_path):
    rgba = np.full((2, 3, 4), [50, 80, 100, 255], np.uint8)
    path = tmp_path / "texture.png"
    Image.fromarray(rgba).save(path)

    class UserTexture:
        def rgba(self):
            return rgba

    texture: Texture = UserTexture()
    for source in (rgba, rgba[..., :3], str(path), path, texture):
        loaded = load_rgba_image(source)
        np.testing.assert_array_equal(loaded, rgba)
        loaded[:] = 0
    np.testing.assert_array_equal(rgba[0, 0], [50, 80, 100, 255])
    with pytest.raises(ValueError, match="texture"):
        load_rgba_image(np.zeros((2, 3), np.uint8))


def test_checkerboard_texture():
    checker = CheckerboardTexture(size=4, cells=2, colors=((0, 0, 0), (255, 0, 0)))
    rgba = checker.rgba()
    np.testing.assert_array_equal(rgba[:, :, 0], [[0, 0, 255, 255], [0, 0, 255, 255],
                                                [255, 255, 0, 0], [255, 255, 0, 0]])
    assert np.all(rgba[..., 3] == 255)
    rgba[:] = 0
    assert checker.rgba().max() == 255
    with pytest.raises(ValueError, match="positive integer"):
        CheckerboardTexture(size=0)
