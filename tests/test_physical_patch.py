"""Pure (CARLA-free) parts of the physical-patch attack: texture generation + config parsing."""

import numpy as np
from avsectester.attacks.physical_patch import (
    DEFAULT_PROP,
    PhysicalPatch,
    build_patch,
    checkerboard_rgba,
    image_rgba,
)


def test_checkerboard_is_rgba_and_alternates():
    img = checkerboard_rgba(size=64, squares=4, color_a=(255, 0, 0), color_b=(255, 255, 255))
    assert img.shape == (64, 64, 4) and img.dtype == np.uint8
    assert (img[..., 3] == 255).all()  # fully opaque
    cell = 64 // 4
    assert tuple(img[0, 0, :3]) == (255, 0, 0)  # first cell = color_a
    assert tuple(img[0, cell, :3]) == (255, 255, 255)  # neighbor cell flips to color_b
    assert tuple(img[cell, cell, :3]) == (255, 0, 0)  # diagonal returns to color_a


def test_build_patch_defaults_and_strips_target_keys():
    p = build_patch({"target": "lead", "gap": 9.0, "lead_vehicle": "vehicle.tesla.model3"})
    assert isinstance(p, PhysicalPatch)
    assert p.prop == DEFAULT_PROP  # target/gap/lead_vehicle are the backend's concern, not the patch
    assert p.rotation == (90.0, 0.0, 0.0)  # stands the plank vertical, facing the ego


def test_build_patch_passes_through_a_patch_object():
    p = PhysicalPatch(prop="static.prop.box01")
    assert build_patch(p) is p


def test_image_rgba_resizes_to_square(tmp_path):
    import pytest

    Image = pytest.importorskip("PIL.Image")
    src = tmp_path / "patch.png"
    Image.fromarray(np.full((10, 20, 3), 128, np.uint8)).save(src)
    out = image_rgba(str(src), size=32)
    assert out.shape == (32, 32, 4) and out.dtype == np.uint8
