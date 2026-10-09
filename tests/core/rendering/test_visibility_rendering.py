"""Rendered insertion pixels must agree with the visibility evidence used to clip them."""

import numpy as np
import pytest

from avsectester.insertion import (
    ActorPose,
    Insertion,
    Orientation,
    PlaneAsset,
    WorldPlacement,
    resolve_insertion,
)
from avsectester.rendering.cameras import FThetaCamera, PinholeCamera
from avsectester.rendering.visibility import CuboidVisibilityEstimator, DepthVisibilityEstimator
from avsectester.simulators.patch_insertion import render_plane, render_resolved


CAM_FROM_WORLD = np.array([[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1.0]])


@pytest.fixture(params=["pinhole", "fisheye"])
def camera(request):
    if request.param == "fisheye":
        return FThetaCamera(50, 40, (0, 50), 100, 80, (0, 1 / 50), max_angle=1.5)
    return PinholeCamera(np.array([[50.0, 0, 50], [0, 50, 40], [0, 0, 1]]), 100, 80)


def panel(texture, position=(10, 0, 0), width=4, height=2, id="panel"):
    return resolve_insertion(
        Insertion(
            id,
            PlaneAsset(texture, width, height),
            WorldPlacement(position),
            Orientation("fixed_world", (0, 0, 180)),
        ),
        {},
        ActorPose(np.eye(4)),
    )


@pytest.mark.parametrize("position", [(10, 0, 0), (20, 0, 0), (10, 10, 0)])
def test_minified_cutout_uses_the_rendered_silhouette(camera, position):
    texture = np.full((512, 512, 4), [240, 40, 20, 255], np.uint8)
    texture[:, :, 3] = (np.arange(512)[None, :] % 8 >= 4) * 255
    item = panel(texture, position)
    clean = np.zeros((camera.height, camera.width, 3), np.uint8)
    surface = item.planes()[0]
    rendered, mask = render_plane(clean, camera, CAM_FROM_WORLD, *surface)
    evidence = CuboidVisibilityEstimator().estimate(
        item,
        camera,
        CAM_FROM_WORLD,
        occluders={},
        camera_name="front",
    )
    np.testing.assert_array_equal(evidence.reference_mask, mask > 0)
    clipped, clipped_mask = render_resolved(clean, camera, CAM_FROM_WORLD, item, evidence)
    np.testing.assert_array_equal(clipped_mask, mask)
    np.testing.assert_array_equal(clipped[mask > 0], rendered[mask > 0])

    # Scene depth blocks exactly the left half of the rendered footprint.
    depth = np.full(clean.shape[:2], 100.0)
    depth[:, :50] = 1
    blocked = DepthVisibilityEstimator().estimate(
        item,
        camera,
        CAM_FROM_WORLD,
        scene_depth=depth,
        depth_convention="z",
        camera_name="front",
    )
    expected = mask > 0
    expected[:, :50] = False
    np.testing.assert_array_equal(blocked.visible_mask, expected)
    result, result_mask = render_resolved(clean, camera, CAM_FROM_WORLD, item, blocked)
    np.testing.assert_array_equal(result_mask > 0, expected)
    np.testing.assert_array_equal(result[~expected], clean[~expected])


def test_small_surface_close_to_camera_is_not_silently_dropped(camera):
    item = panel(np.full((16, 16, 4), 255, np.uint8), (0.03, 0, 0), 0.02, 0.01)
    evidence = CuboidVisibilityEstimator().estimate(
        item,
        camera,
        CAM_FROM_WORLD,
        occluders={},
        camera_name="front",
    )
    _, mask = render_resolved(
        np.zeros((80, 100, 3), np.uint8), camera, CAM_FROM_WORLD, item, evidence
    )
    assert evidence.visibility.fraction == 1
    np.testing.assert_array_equal(mask > 0, evidence.visible_mask)


def test_custom_evidence_can_omit_color_samples(camera):
    from dataclasses import replace

    item = panel(np.full((16, 16, 4), [255, 20, 40, 255], np.uint8))
    depth = np.full((80, 100), 100.0)
    depth[:, :50] = 1
    evidence = DepthVisibilityEstimator().estimate(
        item,
        camera,
        CAM_FROM_WORLD,
        scene_depth=depth,
        depth_convention="z",
        camera_name="front",
    )
    background = np.full((80, 100, 3), 30, np.uint8)
    image, mask = render_resolved(background, camera, CAM_FROM_WORLD, item, evidence)
    uncached, uncached_mask = render_resolved(
        background,
        camera,
        CAM_FROM_WORLD,
        item,
        replace(evidence, sampled_rgba=None),
    )
    np.testing.assert_array_equal(image, uncached)
    np.testing.assert_array_equal(mask, uncached_mask)
    with pytest.raises(ValueError, match="visibility is unknown"):
        render_resolved(
            background, camera, CAM_FROM_WORLD, item, replace(evidence, visibility=None)
        )


def test_camera_crop_does_not_change_texture_sampling(camera):
    from dataclasses import replace

    texture = np.full((512, 512, 4), [240, 40, 20, 255], np.uint8)
    texture[:, :, 3] = (np.arange(512)[None, :] % 8 >= 4) * 255
    item = panel(texture)
    if isinstance(camera, PinholeCamera):
        intrinsic = camera.K.copy()
        intrinsic[0, 2] -= 50
        cropped = replace(camera, K=intrinsic)
    else:
        cropped = replace(camera, cx=camera.cx - 50)
    estimator = CuboidVisibilityEstimator()
    full = estimator.estimate(item, camera, CAM_FROM_WORLD, occluders={}, camera_name="front")
    partial = estimator.estimate(item, cropped, CAM_FROM_WORLD, occluders={}, camera_name="front")
    assert partial.reference_pixels == full.reference_pixels
    assert 0 < partial.visibility.fraction < full.visibility.fraction
    np.testing.assert_array_equal(partial.reference_mask[:, :50], full.reference_mask[:, 50:])
    np.testing.assert_array_equal(partial.sampled_rgba[:, :50], full.sampled_rgba[:, 50:])


@pytest.mark.parametrize("method", ["depth", "cuboid"])
def test_independent_insertions_occlude_by_depth_and_keep_holes(camera, method):
    from avsectester.plane import Observation
    from avsectester.rendering.types import InsertionGeometry
    from avsectester.simulators.patch_insertion import InsertionRenderer, PatchCompositor

    red = np.full((16, 16, 4), [255, 0, 0, 255], np.uint8)
    red[6:10, 6:10, 3] = 0
    blue = np.full((16, 16, 4), [0, 0, 255, 255], np.uint8)
    items = (
        Insertion(
            "near",
            PlaneAsset(red, 2, 2),
            WorldPlacement((5, 0, 0)),
            Orientation("fixed_world", (0, 0, 180)),
        ),
        Insertion(
            "far",
            PlaneAsset(blue, 4, 4),
            WorldPlacement((10, 0, 0)),
            Orientation("fixed_world", (0, 0, 180)),
        ),
    )
    state = InsertionGeometry({}, ActorPose(np.eye(4)), CAM_FROM_WORLD)

    def evidence(obs, resolved, geometry):
        assert isinstance(geometry, InsertionGeometry)
        if method == "depth":
            estimator = DepthVisibilityEstimator()
            kwargs = dict(scene_depth=np.full((80, 100), 100.0), depth_convention="z")
        else:
            estimator = CuboidVisibilityEstimator()
            kwargs = dict(occluders=geometry.actors)
        return {
            item.id: estimator.estimate(
                item,
                camera,
                geometry.cam_from_world,
                camera_name="front",
                other_insertions=resolved,
                **kwargs,
            )
            for item in resolved
        }

    clean = np.full((80, 100, 3), [0, 60, 0], np.uint8)
    results = []
    # Insertion order must not change occlusion results.
    for order, geometry in [(items, state), (items[::-1], state)]:
        renderer = InsertionRenderer(
            order,
            camera,
            lambda _: geometry,
            compositor=PatchCompositor(lambda image, mask, background: image),
            evidence_provider=evidence,
        )
        results.append(renderer(Observation(0, 0), clean))
        assert renderer.evidence["near"].visibility.fraction == 1
        assert 0 < renderer.evidence["far"].visibility.fraction < 1
    np.testing.assert_array_equal(results[0], results[1])
    np.testing.assert_array_equal(results[0][40, 50], [0, 0, 255])
    np.testing.assert_array_equal(results[0][40, 56], [255, 0, 0])
    np.testing.assert_array_equal(results[0][0, 0], clean[0, 0])


def test_appearance_softening_preserves_visibility_and_foreground_pixels(camera):
    from avsectester.simulators.patch_insertion import PatchCompositor

    texture = np.full((32, 32, 4), [240, 20, 30, 255], np.uint8)
    texture[:, 16:, :3] = [20, 30, 240]
    item = panel(texture)
    clean = np.full((camera.height, camera.width, 3), 70, np.uint8)
    depth = np.full(clean.shape[:2], 100.0)
    depth[:, :50] = 1
    evidence = DepthVisibilityEstimator().estimate(
        item, camera, CAM_FROM_WORLD, scene_depth=depth, depth_convention="z", camera_name="front",
    )
    original_samples = evidence.sampled_rgba.copy()
    compositor = PatchCompositor(lambda image, mask, background: image, soften=.6)
    image, mask = render_resolved(clean, camera, CAM_FROM_WORLD, item, evidence, compositor=compositor)
    np.testing.assert_array_equal(mask > 0, evidence.visible_mask)
    np.testing.assert_array_equal(image[~evidence.visible_mask], clean[~evidence.visible_mask])
    np.testing.assert_array_equal(evidence.sampled_rgba, original_samples)
    assert np.any(image[evidence.visible_mask] != clean[evidence.visible_mask])
