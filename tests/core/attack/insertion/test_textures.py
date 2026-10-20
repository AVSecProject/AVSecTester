"""Explicit texture types share one validated image-loading path."""
from pathlib import Path

import numpy as np
import pytest

from avsectester.attack.insertion.assets import (CheckerboardTexture, ImageTexture,
                                               PlaneAsset, load_rgba_image)


def test_checkerboard_preserves_default_patch_pixels_and_returns_independent_arrays():
    texture = CheckerboardTexture()
    image = texture.rgba()
    assert image.shape == (256, 256, 4)
    np.testing.assert_array_equal(image[0, 0], [255, 0, 0, 255])
    np.testing.assert_array_equal(image[0, 32], [255, 255, 255, 255])
    image[:] = 0
    assert texture.rgba()[0, 0, 0] == 255


def test_image_texture_loads_rgb_once_and_plane_uses_the_same_rgba_conversion(tmp_path):
    from PIL import Image
    path = tmp_path / "texture.png"
    rgb = np.full((3, 5, 3), [12, 34, 56], np.uint8)
    Image.fromarray(rgb).save(path)
    texture = ImageTexture(path)
    path.unlink()
    image = texture.rgba()
    assert image.shape == (3, 5, 4) and (image[..., 3] == 255).all()
    np.testing.assert_array_equal(image[..., :3], rgb)
    plane = PlaneAsset(texture, 1., 1.)
    np.testing.assert_array_equal(plane.texture, image)
    image[:] = 0
    assert texture.rgba()[0, 0, 0] == 12


@pytest.mark.parametrize("value", [{"image": "texture.png"}, {"pattern": "typo"}, None, 12])
def test_unknown_texture_format_cannot_fall_back_to_checkerboard(value):
    with pytest.raises(TypeError, match="texture must"):
        load_rgba_image(value)


@pytest.mark.parametrize("options", [{"size": 0}, {"size": True}, {"cells": 0},
                                     {"size": 4, "cells": 5}, {"colors": ((-1, 0, 0), (1, 2, 3))}])
def test_checkerboard_rejects_invalid_options_at_construction(options):
    with pytest.raises(ValueError):
        CheckerboardTexture(**options)


def test_image_texture_requires_path_and_loader_keeps_input_independent():
    with pytest.raises(TypeError, match="file path"):
        ImageTexture(np.zeros((2, 2, 4), np.uint8))
    rgb = np.ones((2, 3, 3), np.uint8)
    rgba = load_rgba_image(rgb)
    rgba[:] = 0
    assert rgb.all()
    with pytest.raises(FileNotFoundError):
        load_rgba_image(Path("missing-texture-for-test.png"))
