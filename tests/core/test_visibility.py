"""Visibility counts occlusion and image truncation against a full silhouette."""

import numpy as np
import pytest

from avsectester.scenarios.visibility import visibility_from_depth, visibility_from_masks


def test_depth_visibility_ignores_background_and_measures_partial_occlusion():
    footprint = np.zeros((4, 6), dtype=bool)
    footprint[:, 2:4] = True
    target = np.full((4, 6), 10.0)
    scene = np.full((4, 6), 20.0)
    assert visibility_from_depth(footprint, target, scene, camera="front").fraction == 1
    scene[:2] = 5  # Blocks half the silhouette, not half a containing bounding box.
    assert visibility_from_depth(footprint, target, scene, camera="front").fraction == 0.5
    scene[:] = 5
    assert visibility_from_depth(footprint, target, scene, camera="front").fraction == 0


def test_image_truncation_and_occlusion_share_the_full_denominator():
    reference = np.ones((2, 8), bool)
    target = np.full(reference.shape, 10.0)
    scene = np.full(reference.shape, np.nan)
    bounds = (2, 0, 6, 2)
    scene[:, 2:6] = 20
    assert (
        visibility_from_depth(
            reference, target, scene, camera="front", image_bounds=bounds
        ).fraction
        == 0.5
    )
    scene[:, 2:4] = 5
    assert (
        visibility_from_depth(
            reference, target, scene, camera="front", image_bounds=bounds
        ).fraction
        == 0.25
    )
    # The mask route has exactly the same camera boundary convention.
    assert (
        visibility_from_masks(
            reference, scene >= target, camera="front", image_bounds=bounds
        ).fraction
        == 0.25
    )


def test_fully_outside_image_is_zero_but_missing_reference_is_unknown():
    reference = np.zeros((2, 8), bool)
    reference[:, :2] = True
    missing = np.full(reference.shape, np.nan)
    assert (
        visibility_from_depth(
            reference, missing, missing, camera="front", image_bounds=(2, 0, 6, 2)
        ).fraction
        == 0
    )
    assert visibility_from_masks(reference & False, reference, camera="front") is None


@pytest.mark.parametrize("bounds", [(0, 0, 3, 2), (0, 0, 0, 2), (-1, 0, 2, 2), (0.0, 0, 2, 2)])
def test_invalid_camera_bounds_are_rejected(bounds):
    with pytest.raises(ValueError, match="bounds"):
        visibility_from_masks(np.ones((2, 2)), np.ones((2, 2)), camera="front", image_bounds=bounds)


def test_missing_depth_and_misaligned_masks_are_rejected():
    mask = np.ones((2, 2), bool)
    with pytest.raises(ValueError, match="aligned"):
        visibility_from_masks(mask, np.ones((3, 3)), camera="front")
    with pytest.raises(ValueError, match="missing or invalid"):
        visibility_from_depth(mask, np.ones((2, 2)), np.full((2, 2), np.nan), camera="front")


def test_depth_tolerance_and_mask_intersection():
    mask = np.ones((1, 4), bool)
    target = np.full((1, 4), 10.0)
    scene = np.array([[9.99, 9.9, 10, 12]])
    assert (
        visibility_from_depth(mask, target, scene, camera="front", tolerance_m=0.02).fraction
        == 0.75
    )
    assert visibility_from_masks([[1, 0]], [[0, 1]], camera="front").fraction == 0


def test_default_tolerance_does_not_hide_close_surface_occlusion():
    # A surface 5 mm in front of the patch is an occluder, not numerical depth noise.
    result = visibility_from_depth(
        [[True, True]], [[10.0, 10.0]], [[9.995, 9.9999]], camera="front"
    )
    assert result.fraction == 0.5
