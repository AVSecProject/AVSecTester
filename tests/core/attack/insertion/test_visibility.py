"""Visibility fractions retain their full silhouette denominator and explicit evidence."""
import numpy as np
import pytest
from avsectester.attack.insertion import (CuboidVisibilityEstimator, DepthVisibilityEstimator,
    Insertion, PlaneAsset, resolve_insertion, visibility_from_depth, visibility_from_masks)
from avsectester.attack.placement import Orientation, WorldPlacement
from avsectester.geometry.poses import ActorPose


def panel(position=(10, 0, 0)):
    return resolve_insertion(Insertion("panel", PlaneAsset(np.full((8, 8, 4), 255, np.uint8), 4, 2),
        WorldPlacement(position), Orientation("fixed_world", (0, 0, 180))), {}, ActorPose(np.eye(4)))


def test_visibility_from_masks_and_depth():
    reference = np.ones((2, 4), bool)
    visible = reference.copy()
    visible[:, :2] = False
    assert visibility_from_masks(reference, visible, camera="front").fraction == .5
    assert visibility_from_masks(reference, reference, camera="front", image_bounds=(0, 0, 2, 2)).fraction == .5
    assert visibility_from_masks(np.zeros((2, 4)), reference, camera="front") is None
    target, scene = np.full((2, 4), 5.), np.full((2, 4), 10.)
    scene[:, :2] = 2
    assert visibility_from_depth(reference, target, scene, camera="front").fraction == .5
    with pytest.raises(ValueError, match="invalid"):
        visibility_from_depth(reference, target, np.zeros((2, 4)), camera="front")


BLOCKER = np.eye(4)
BLOCKER[:3, 3] = (5, 1, 0)


@pytest.mark.parametrize("estimator,clear_options,blocked_options", [
    pytest.param(CuboidVisibilityEstimator, {"occluders": {}},
                 {"occluders": {"blocker": ActorPose(BLOCKER, (1, 2, 4))}}, id="cuboid"),
    pytest.param(DepthVisibilityEstimator,
                 {"scene_depth": np.full((80, 100), 100.), "depth_convention": "z"},
                 {"scene_depth": np.tile(np.r_[np.ones(50), np.full(50, 100.)], (80, 1)),
                  "depth_convention": "z"}, id="depth"),
])
def test_estimate(estimator, clear_options, blocked_options, camera_pair, optical_from_world):
    camera, cropped = camera_pair
    implementation = estimator()
    clear = implementation.estimate(panel(), camera, optical_from_world,
                                    camera_name="front", **clear_options)
    partial = implementation.estimate(panel(), camera, optical_from_world,
                                      camera_name="front", **blocked_options)
    assert clear.visibility.fraction == 1
    assert partial.visibility.fraction == pytest.approx(.5, abs=.01)
    assert not partial.visible_mask[:, :50].any()
    truncated = implementation.estimate(panel(), cropped, optical_from_world,
                                        camera_name="front", **clear_options)
    assert truncated.reference_pixels == clear.reference_pixels
    assert 0 < truncated.visibility.fraction < 1
    behind = implementation.estimate(panel((-10, 0, 0)), camera, optical_from_world,
                                     camera_name="front", **clear_options)
    assert behind.visibility.fraction == 0


def test_depth_estimate_with_invalid_depth(camera, optical_from_world):
    result = DepthVisibilityEstimator().estimate(
        panel(), camera, optical_from_world, camera_name="front",
        scene_depth=np.full((80, 100), np.nan), depth_convention="z")
    assert result.visibility is None and "Invalid depth" in result.reason
