"""World-object compositing shares lens sampling, occlusion and appearance semantics."""
from dataclasses import replace

import numpy as np
import pytest

from avsectester.data_types import ActorState
from avsectester.geometry.boxes import box_ray_distances
from avsectester.rendering import (
    ObjectAsset, ObjectVisibility, PlacedObject, PlaneSurface, blend_rgba, composite_objects, render_surfaces,
)


def test_blend_rgba():
    image = np.full((2, 2, 3), 20, np.uint8)
    rgba = np.full((2, 2, 4), [220, 0, 0, 64], np.uint8)
    visible = np.array([[True, False], [False, False]])
    blended = blend_rgba(image, rgba, visible)
    np.testing.assert_array_equal(blended[0, 0], [70, 14, 14])
    np.testing.assert_array_equal(blended[~visible], image[~visible])

    def green(composite, mask, background):
        composite[:] = [0, 255, 0]
        background[:] = 0
        return composite

    harmonized = blend_rgba(image, rgba, visible, harmonizer=green)
    np.testing.assert_array_equal(harmonized[0, 0], [0, 255, 0])
    np.testing.assert_array_equal(harmonized[~visible], image[~visible])
    assert np.all(image == 20)


def test_composite_objects(camera, optical_from_world):
    texture = np.full((16, 16, 4), [240, 20, 20, 255], np.uint8)
    surface = PlaneSurface(np.array([[0., -2., 2.], [0., 2., 2.],
                                    [0., 2., 0.], [0., -2., 0.]]), texture)
    asset = ObjectAsset.from_surfaces((surface,), "sign")
    pose = np.eye(4)
    pose[:3, 3] = (6., 0., -1.)
    placed = PlacedObject("sign", asset, pose)
    image = np.full((80, 100, 3), [20, 60, 80], np.uint8)
    rgba, target_depth = render_surfaces(placed.world_surfaces(), camera, optical_from_world)
    reference = np.isfinite(target_depth)
    result = composite_objects(image, camera, optical_from_world, (placed,))
    np.testing.assert_array_equal(result.image, blend_rgba(image, rgba, reference))
    np.testing.assert_array_equal(result.image[40, 50], [240, 20, 20])
    np.testing.assert_array_equal(result.image[~reference], image[~reference])
    np.testing.assert_array_equal(result.depth, target_depth)
    np.testing.assert_array_equal(result.visible_mask, reference)
    assert result.occlusion_source == "actor_cuboids"
    assert result.object_visibility == {"sign": ObjectVisibility("actor_cuboids", (), int(reference.sum()))}

    scene_depth = np.full((80, 100), 100.)
    scene_depth[:, :50] = 1.
    partial = composite_objects(image, camera, optical_from_world, (placed,), scene_depth=scene_depth)
    assert not partial.visible_mask[:, :50].any() and partial.visible_mask[:, 50:].any()
    np.testing.assert_array_equal(partial.image[~partial.visible_mask], image[~partial.visible_mask])
    np.testing.assert_array_equal(partial.depth, np.minimum(scene_depth, target_depth))
    assert partial.occlusion_source == "depth"
    assert partial.object_visibility["sign"] == ObjectVisibility("depth", (), int(partial.visible_mask.sum()))

    blocker_pose = np.eye(4)
    blocker_pose[0, 3] = 3.
    blocker = ActorState(blocker_pose, extent=(.5, 20., 20.))
    blocked = composite_objects(image, camera, optical_from_world, (placed,), occluders={"car": blocker})
    assert not blocked.visible_mask.any()
    assert blocked.object_visibility == {"sign": ObjectVisibility("actor_cuboids", (), 0)}
    np.testing.assert_array_equal(blocked.image, image)
    rows, columns = np.indices(image.shape[:2])
    rays = camera.unproject(np.column_stack((columns.ravel() + .5, rows.ravel() + .5)))
    distances = box_ray_distances(blocker.box_transform(), blocker.extent, rays, np.linalg.inv(optical_from_world))
    full_depth = (distances * rays[:, 2]).reshape(image.shape[:2])
    full_cast = composite_objects(image, camera, optical_from_world, (placed,), scene_depth=full_depth)
    np.testing.assert_array_equal(blocked.image, full_cast.image)
    np.testing.assert_array_equal(blocked.visible_mask, full_cast.visible_mask)
    projected_rows, projected_columns = np.nonzero(reference)
    candidate_region = np.zeros(image.shape[:2], dtype=bool)
    candidate_region[max(0, projected_rows.min() - 2):min(camera.height, projected_rows.max() + 3),
                     max(0, projected_columns.min() - 2):min(camera.width, projected_columns.max() + 3)] = True
    np.testing.assert_array_equal(blocked.depth[candidate_region], full_cast.depth[candidate_region])
    assert np.isinf(blocked.depth[~candidate_region]).all()
    unknown = ActorState(blocker_pose)
    unmeasured = composite_objects(image, camera, optical_from_world, (placed,),
                                   occluders={"unknown": unknown})
    np.testing.assert_array_equal(unmeasured.image, result.image)
    assert unmeasured.ignored_occluders == ("unknown",)
    skipped = composite_objects(image, camera, optical_from_world, (placed,),
                                occluders={"unknown-first": unknown, "car": blocker,
                                           "unknown-second": unknown})
    np.testing.assert_array_equal(skipped.image, blocked.image)
    assert skipped.ignored_occluders == ("unknown-first", "unknown-second")
    assert skipped.object_visibility["sign"] == ObjectVisibility(
        "actor_cuboids", ("unknown-first", "unknown-second"), 0)
    depth_only = composite_objects(image, camera, optical_from_world, (placed,),
                                   scene_depth=np.full((80, 100), np.inf),
                                   occluders={"car": blocker, "unknown": unknown})
    np.testing.assert_array_equal(depth_only.image, result.image)
    assert depth_only.ignored_occluders == ()

    def green(composite, mask, background):
        composite[:] = [0, 255, 0]
        return composite

    placed = replace(placed, harmonizer=green)
    colored = composite_objects(image, camera, optical_from_world, (placed,), scene_depth=scene_depth)
    assert np.all(colored.image[colored.visible_mask] == [0, 255, 0])
    np.testing.assert_array_equal(colored.image[~colored.visible_mask], image[~colored.visible_mask])

    far_pose = pose.copy()
    far_pose[:3, 3] = (9., 1.5, -1.)
    def yellow(composite, mask, background):
        composite[:] = [255, 255, 0]
        return composite

    farther = PlacedObject("farther", asset, far_pose, harmonizer=yellow)
    ordered = composite_objects(image, camera, optical_from_world, (placed, farther))
    reversed_order = composite_objects(image, camera, optical_from_world, (farther, placed))
    np.testing.assert_array_equal(ordered.image, reversed_order.image)
    np.testing.assert_array_equal(ordered.image[40, 50], [0, 255, 0])
    np.testing.assert_array_equal(ordered.image[40, 32], [255, 255, 0])
    assert ordered.object_visibility == reversed_order.object_visibility
    assert ordered.object_visibility["sign"].visible_pixels == np.count_nonzero(
        np.all(ordered.image == [0, 255, 0], axis=2))
    assert ordered.object_visibility["farther"].visible_pixels == np.count_nonzero(
        np.all(ordered.image == [255, 255, 0], axis=2))
    with pytest.raises(ValueError, match="uint8 RGB"):
        composite_objects(image.astype(float), camera, optical_from_world, (placed,))
