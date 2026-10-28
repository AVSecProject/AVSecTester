"""World-object compositing shares lens sampling, occlusion and appearance semantics."""
from dataclasses import replace

import numpy as np
import pytest

from avsectester.data_types import ActorState
from avsectester.geometry.boxes import box_ray_distances
from avsectester.rendering import (
    ClassicHarmonizer, ExposureHarmonizer, Harmonizer, ObjectAsset, ObjectVisibility, PlacedObject,
    PlaneSurface, blend_rgba, composite_objects, plan_composite, render_surfaces,
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
    # Composition blends filtered premultiplied colours in floating point and rounds once.
    assert np.abs(result.image.astype(int) - blend_rgba(image, rgba, reference).astype(int)).max() <= 1
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


def test_plan_composite_matches_deployment(camera, optical_from_world):
    import torch
    rng = np.random.default_rng(0)
    texture = np.concatenate((rng.integers(0, 256, (16, 16, 3)), np.full((16, 16, 1), 255)), axis=2)
    texture[:2, :2, 3] = 0
    surface = PlaneSurface(np.array([[0., -2., 2.], [0., 2., 2.],
                                    [0., 2., 0.], [0., -2., 0.]]), texture.astype(np.uint8))
    pose = np.eye(4)
    pose[:3, 3] = (6., 0., -1.)
    placed = PlacedObject("sign", ObjectAsset.from_surfaces((surface,), "sign"), pose)
    image = rng.integers(0, 256, (80, 100, 3)).astype(np.uint8)
    scene_depth = np.full((80, 100), 100.)
    scene_depth[:, :50] = 1.
    plan = plan_composite(camera, optical_from_world, (placed,), scene_depth=scene_depth)
    deployed = plan.apply(image)
    np.testing.assert_array_equal(
        deployed.image, composite_objects(image, camera, optical_from_world, (placed,), scene_depth=scene_depth).image)
    assert plan.object_visibility == deployed.object_visibility

    background = torch.as_tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255
    colors = (torch.as_tensor(texture[..., :3], dtype=torch.float32).permute(2, 0, 1) / 255).requires_grad_()
    composite = plan.composite(background, {("sign", 0): colors})
    rounded = np.rint(composite.detach().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    np.testing.assert_array_equal(rounded, deployed.image)
    composite.sum().backward()
    assert colors.grad is not None and colors.grad.abs().sum() > 0

    changed = plan.composite(background, {("sign", 0): torch.zeros_like(colors)}).detach()
    moved = (changed - composite.detach()).abs().amax(dim=0).numpy() > 0
    assert moved.any() and not moved[~deployed.visible_mask].any()
    with pytest.raises(KeyError, match="Unknown planned surface"):
        plan.composite(background, {("other", 0): colors})
    with pytest.raises(KeyError, match="Unknown planned surface"):
        plan.composite(background, {("sign", 1): colors})
    with pytest.raises(ValueError, match="must be a"):
        plan.composite(background, {("sign", 0): colors[:, :8]})
    harmonized = plan_composite(camera, optical_from_world, (replace(placed, harmonizer=lambda c, m, b: c),))
    with pytest.raises(ValueError, match="Harmonizers are not differentiable"):
        harmonized.composite(background)


def test_plan_composite_harmonizers(camera, optical_from_world):
    import torch
    rng = np.random.default_rng(7)
    texture = rng.integers(0, 256, (12, 12, 4), dtype=np.uint8)
    texture[..., 3] = 255
    texture[:2, :2, 3] = 0
    surface = PlaneSurface(np.array([[0., -.75, 2.], [0., .75, 2.],
                                    [0., .75, 0.], [0., -.75, 0.]]), texture)
    asset = ObjectAsset.from_surfaces((surface,), "sign")
    pose = np.eye(4)
    pose[:3, 3] = (6., -1., -1.)
    far_pose = pose.copy()
    far_pose[:3, 3] = (9., 1.5, -1.)
    near = PlacedObject("near", asset, pose, harmonizer=ExposureHarmonizer(reference="under"))
    far = PlacedObject("far", asset, far_pose, harmonizer=ExposureHarmonizer(
        reference="ring", ring_px=3, gain_range=(.4, .4), feather_sigma=0.))
    image = rng.integers(10, 100, (80, 100, 3), dtype=np.uint8)
    plan = plan_composite(camera, optical_from_world, (near, far))
    assert all(row.visible_pixels > 0 for row in plan.object_visibility.values())
    background = (torch.as_tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255).requires_grad_()
    colors = (torch.as_tensor(texture[..., :3], dtype=torch.float32).permute(2, 0, 1) / 255).requires_grad_()
    output = plan.composite(background, {("near", 0): colors, ("far", 0): colors})
    assert output.min() >= 0 and output.max() <= 1
    deployed = plan.apply(image)
    rounded = (output.detach() * 255).round().to(torch.uint8).permute(1, 2, 0).numpy()
    np.testing.assert_array_equal(deployed.image, rounded)
    np.testing.assert_array_equal(deployed.image[~plan.visible_mask], image[~plan.visible_mask])
    output.sum().backward()
    assert colors.grad is not None and torch.isfinite(colors.grad).all() and colors.grad.abs().sum() > 0
    assert background.grad is not None and torch.isfinite(background.grad).all()

    unadjusted = plan_composite(camera, optical_from_world, (
        replace(near, harmonizer=None), replace(far, harmonizer=None))).apply(image)
    assert np.any(deployed.image[plan.visible_mask] != unadjusted.image[plan.visible_mask])
    reversed_plan = plan_composite(camera, optical_from_world, (far, near))
    np.testing.assert_array_equal(deployed.image, reversed_plan.apply(image).image)
    np.testing.assert_array_equal(
        rounded, (reversed_plan.composite(background).detach() * 255)
        .round().to(torch.uint8).permute(1, 2, 0).numpy())

    classic = replace(far, harmonizer=ClassicHarmonizer(color_transfer=False, poisson=False))
    mixed = plan_composite(camera, optical_from_world, (near, classic))
    with pytest.raises(ValueError, match="object 'far' uses ClassicHarmonizer"):
        mixed.composite(background)
    near_only = plan_composite(camera, optical_from_world, (near,))
    np.testing.assert_array_equal(
        mixed.apply(image).image[near_only.visible_mask], deployed.image[near_only.visible_mask])

    class Overexposed(Harmonizer):
        """A user harmonizer whose output leaves [0, 1]."""
        differentiable = True

        def harmonize_tensor(self, composite, mask, background, nominal_composite):
            return composite * 4 - 1

        def __call__(self, composite_rgb, mask, background_rgb):
            return composite_rgb

    overexposed = plan_composite(camera, optical_from_world, (replace(near, harmonizer=Overexposed()),))
    clamped = overexposed.composite(background.detach(), {("near", 0): colors.detach()})
    assert clamped.min() == 0 and clamped.max() == 1
    np.testing.assert_array_equal(
        overexposed.apply(image).image,
        (clamped * 255).round().to(torch.uint8).permute(1, 2, 0).numpy())

    # A nominal texture fixes the harmonizer's statistics, so editing part of the texture leaves
    # the rest of the object unchanged; without one, the edit changes the whole object's gain.
    edited = texture.copy()
    edited[6:, :, :3] = 255
    edited_colors = torch.as_tensor(edited[..., :3], dtype=torch.float32).permute(2, 0, 1) / 255
    exposure = ExposureHarmonizer(reference="under", gain_range=(.01, 10.), feather_sigma=0.)
    nominal_asset = ObjectAsset.from_surfaces((PlaneSurface(surface.corners, edited, texture),), "sign")
    plans = {name: plan_composite(camera, optical_from_world, (PlacedObject("near", item, pose, exposure),))
             for name, item in (("nominal", nominal_asset), ("plain", asset))}
    unadjusted = plan_composite(camera, optical_from_world, (PlacedObject("near", asset, pose),))
    background = background.detach()
    with torch.no_grad():
        changed = (unadjusted.composite(background, {("near", 0): colors})
                   != unadjusted.composite(background, {("near", 0): edited_colors})).any(0)
        unchanged = torch.as_tensor(unadjusted.visible_mask) & ~changed
        assert changed.any() and unchanged.any()
        outputs = {name: [plan.composite(background, {("near", 0): value})[:, unchanged]
                          for value in (colors, edited_colors)] for name, plan in plans.items()}
        torch.testing.assert_close(*outputs["nominal"], atol=0, rtol=0)
        assert not torch.allclose(*outputs["plain"])
        rounded = (plans["nominal"].composite(background) * 255).round().to(torch.uint8)
    np.testing.assert_array_equal(plans["nominal"].apply(image).image, rounded.permute(1, 2, 0).numpy())


def test_plan_composite_with_scene_visibility(camera, optical_from_world):
    surface = PlaneSurface(np.array([[0., -2., 2.], [0., 2., 2.], [0., 2., 0.], [0., -2., 0.]]),
                           np.full((8, 8, 4), [200, 40, 40, 255], np.uint8))
    pose = np.eye(4)
    pose[:3, 3] = (6., 0., -1.)
    placed = PlacedObject("sign", ObjectAsset.from_surfaces((surface,), "sign"), pose)
    # Renderer-measured visibility replaces depth evidence: the scene hides the left half.
    scene_visibility = np.ones((80, 100), bool)
    scene_visibility[:, :50] = False
    plan = plan_composite(camera, optical_from_world, (placed,), scene_visibility=scene_visibility)
    unoccluded = plan_composite(camera, optical_from_world, (placed,), scene_visibility=np.ones((80, 100), bool))
    np.testing.assert_array_equal(plan.visible_mask, unoccluded.visible_mask & scene_visibility)
    assert plan.visible_mask.any() and unoccluded.visible_mask[:, :50].any()
    assert plan.occlusion_source == "scene_visibility"
    assert plan.object_visibility["sign"].visible_pixels == plan.visible_mask.sum()
    assert np.isposinf(plan.depth[~plan.visible_mask]).all() and np.isfinite(plan.depth[plan.visible_mask]).all()
    with pytest.raises(ValueError, match="excludes scene_depth"):
        plan_composite(camera, optical_from_world, (placed,), scene_visibility=scene_visibility,
                       scene_depth=np.ones((80, 100)))
    with pytest.raises(ValueError, match="boolean image"):
        plan_composite(camera, optical_from_world, (placed,), scene_visibility=scene_visibility[:, :10])
