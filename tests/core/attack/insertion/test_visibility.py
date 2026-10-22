"""Visibility fractions retain their full silhouette denominator and explicit evidence."""
import numpy as np
import pytest
from avsectester.attack.insertion import (CuboidVisibilityEstimator, DepthVisibilityEstimator,
    Insertion, PlaneAsset, resolve_insertion, visibility_from_depth, visibility_from_masks)
from avsectester.attack.insertion.placement import Orientation, WorldPlacement
from avsectester.data_types import ActorState
from avsectester.rendering import ObjectAsset, PlacedObject, PlaneSurface


def panel(position=(10, 0, 0)):
    return resolve_insertion(Insertion("panel", PlaneAsset(np.full((8, 8, 4), 255, np.uint8), 4, 2),
        WorldPlacement(position), Orientation("fixed_world", (0, 0, 180))), {}, ActorState(np.eye(4)))


def test_visibility_from_masks_and_depth():
    reference = np.ones((2, 4), bool)
    visible = reference.copy()
    visible[:, :2] = False
    masked = visibility_from_masks(reference, visible, camera="front")
    assert masked.fraction == .5 and masked.source == "rendered_masks"
    assert visibility_from_masks(reference, reference, camera="front", image_bounds=(0, 0, 2, 2)).fraction == .5
    assert visibility_from_masks(np.zeros((2, 4)), reference, camera="front") is None
    target, scene = np.full((2, 4), 5.), np.full((2, 4), 10.)
    scene[:, :2] = 2
    measured = visibility_from_depth(reference, target, scene, camera="front")
    assert measured.fraction == .5 and measured.source == "depth"
    assert visibility_from_depth(reference, target, np.full((2, 4), np.inf), camera="front").fraction == 1.
    with pytest.raises(ValueError, match="invalid"):
        visibility_from_depth(reference, target, np.zeros((2, 4)), camera="front")
    with pytest.raises(ValueError, match="invalid"):
        visibility_from_depth(reference, np.full((2, 4), np.inf), scene, camera="front")


BLOCKER = np.eye(4)
BLOCKER[:3, 3] = (5, 1, 0)


@pytest.mark.parametrize("estimator,clear_options,blocked_options,source", [
    pytest.param(CuboidVisibilityEstimator, {"occluders": {}},
                 {"occluders": {"blocker": ActorState(BLOCKER, extent=(1, 2, 4))}}, "actor_cuboids", id="cuboid"),
    pytest.param(DepthVisibilityEstimator,
                 {"scene_depth": np.full((80, 100), 100.), "depth_convention": "z"},
                 {"scene_depth": np.tile(np.r_[np.ones(50), np.full(50, 100.)], (80, 1)),
                  "depth_convention": "z"}, "depth", id="depth"),
])
def test_estimate(estimator, clear_options, blocked_options, source, camera_pair, optical_from_world):
    camera, cropped = camera_pair
    implementation = estimator()
    clear = implementation.estimate(panel(), camera, optical_from_world,
                                    camera_name="front", **clear_options)
    partial = implementation.estimate(panel(), camera, optical_from_world,
                                      camera_name="front", **blocked_options)
    assert clear.visibility.fraction == 1
    assert clear.visibility.source == source and clear.ignored_occluders == ()
    assert partial.visibility.fraction == pytest.approx(.5, abs=.01)
    assert not partial.visible_mask[:, :50].any()
    truncated = implementation.estimate(panel(), cropped, optical_from_world,
                                        camera_name="front", **clear_options)
    assert truncated.reference_pixels == clear.reference_pixels
    assert 0 < truncated.visibility.fraction < 1
    behind = implementation.estimate(panel((-10, 0, 0)), camera, optical_from_world,
                                     camera_name="front", **clear_options)
    assert behind.visibility.fraction == 0
    with pytest.raises(ValueError, match="max_reference_pixels must be a positive integer"):
        estimator(max_reference_pixels=True)


def test_cuboid_evidence(camera, optical_from_world):
    estimator = CuboidVisibilityEstimator()
    actor = ActorState(BLOCKER)
    clear = estimator.estimate(panel(), camera, optical_from_world, camera_name="front",
                               occluders={"unbounded": actor})
    assert clear.visibility.fraction == 1. and clear.ignored_occluders == ("unbounded",)
    behind = estimator.estimate(panel((-10, 0, 0)), camera, optical_from_world, camera_name="front",
                                occluders={"unbounded": actor})
    assert behind.visibility.fraction == 0. and behind.ignored_occluders == ("unbounded",)
    texture = np.full((16, 16, 4), 255, np.uint8)
    texture[:, :8, 3] = 0
    surface = PlaneSurface(np.array([[0., -1., 1.], [0., 1., 1.],
                                    [0., 1., 0.], [0., -1., 0.]]), texture)
    asset = ObjectAsset.from_surfaces((surface,), "sign")
    pose = np.eye(4)
    pose[:3, 3] = (5., 0., -.5)
    placed = PlacedObject("panel", asset, pose)
    partial = estimator.estimate(panel(), camera, optical_from_world, camera_name="front",
                                 occluders={}, world_objects=(placed,))
    assert partial.visibility.fraction == pytest.approx(.5, abs=.01)
    assert partial.visible_mask[:, 50:].any() and not partial.visible_mask[:, :50].any()
    alpha_hole = texture.copy()
    alpha_hole[4:12, 4:12, 3] = 0
    transparent = PlaneSurface(surface.corners, alpha_hole)
    perforated = PlacedObject("panel", ObjectAsset.from_surfaces((transparent,), "sign"), pose)
    through_hole = estimator.estimate(panel(), camera, optical_from_world, camera_name="front",
                                      occluders={}, world_objects=(perforated,))
    assert through_hole.visible_mask[40, 50]
    assert through_hole.visibility.fraction > partial.visibility.fraction


def test_depth_evidence(camera, optical_from_world):
    estimator = DepthVisibilityEstimator()
    for convention in ("z", "range"):
        clear = estimator.estimate(
            panel(), camera, optical_from_world, camera_name="front",
            scene_depth=np.full((80, 100), np.inf), depth_convention=convention)
        assert clear.visibility.fraction == 1.
    for invalid in (np.nan, 0., -np.inf):
        result = estimator.estimate(
            panel(), camera, optical_from_world, camera_name="front",
            scene_depth=np.full((80, 100), invalid), depth_convention="z")
        assert result.visibility is None and "Invalid depth" in result.reason
