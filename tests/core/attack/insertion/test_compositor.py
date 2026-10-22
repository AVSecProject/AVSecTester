"""Image compositing preserves transparent holes, surface depth and clipping."""
from dataclasses import replace
import numpy as np
import pytest
from avsectester.attack.insertion import (CuboidVisibilityEstimator, InsertionAttackAction,
    DepthVisibilityEstimator, Insertion, PatchCompositor, PlaneAsset,
    render_resolved, resolve_insertion, warp_patch)
from avsectester.attack.placement import Orientation, WorldPlacement
from avsectester.geometry.poses import ActorPose
from avsectester.closed_loop.simulator import SensorView
from avsectester.rendering import ClassicHarmonizer, ObjectAsset, PCTNetHarmonizer, PlacedObject, PlaneSurface


def insertion(identity, x, texture, size):
    return resolve_insertion(Insertion(identity, PlaneAsset(texture, size, size),
        WorldPlacement((x, 0, 0)), Orientation("fixed_world", (0, 0, 180))), {}, ActorPose(np.eye(4)))


def test_warp_patch():
    frame = np.full((10, 10, 3), 20, np.uint8)
    texture = np.full((4, 4, 4), [220, 0, 0, 128], np.uint8)
    image, mask = warp_patch(frame, np.array([[2, 2], [6, 2], [6, 6], [2, 6]]), texture)
    np.testing.assert_allclose(image[3, 3], [120, 9, 9], atol=1)
    np.testing.assert_array_equal(image[0, 0], frame[0, 0])
    assert mask[3, 3] == 255 and mask[0, 0] == 0


def test_render_resolved(camera, optical_from_world):
    red = np.full((16, 16, 4), [255, 0, 0, 255], np.uint8)
    red[6:10, 6:10, 3] = 0
    blue = np.full((16, 16, 4), [0, 0, 255, 255], np.uint8)
    near, far = insertion("near", 5, red, 2), insertion("far", 10, blue, 4)
    clean = np.full((80, 100, 3), [0, 60, 0], np.uint8)
    rendered = []
    for surfaces in ((near, far), (far, near)):
        image = clean.copy()
        for item in surfaces:
            evidence = CuboidVisibilityEstimator().estimate(item, camera, optical_from_world,
                occluders={}, camera_name="front", other_insertions=surfaces)
            image, mask = render_resolved(image, camera, optical_from_world, item, evidence)
            np.testing.assert_array_equal(mask > 0, evidence.visible_mask)
        rendered.append(image)
    np.testing.assert_array_equal(*rendered)
    np.testing.assert_array_equal(rendered[0][40, 50], [0, 0, 255])
    np.testing.assert_array_equal(rendered[0][40, 56], [255, 0, 0])
    np.testing.assert_array_equal(rendered[0][0, 0], clean[0, 0])


def test_compositor_and_harmonizer(camera, optical_from_world, tmp_path):
    item = insertion("panel", 10, np.full((16, 16, 4), [240, 20, 30, 255], np.uint8), 4)
    frame = np.full((80, 100, 3), 70, np.uint8)
    depth = np.full((80, 100), 100.)
    depth[:, :50] = 1
    evidence = DepthVisibilityEstimator().estimate(item, camera, optical_from_world,
                         scene_depth=depth, depth_convention="z", camera_name="front")
    def green(image, mask, background):
        result = image.copy()
        result[mask > 0] = [0, 255, 0]
        return result
    plain, plain_mask = render_resolved(frame, camera, optical_from_world, item, evidence,
                                        compositor=PatchCompositor())
    colored, mask = render_resolved(frame, camera, optical_from_world, item, evidence,
                                    compositor=PatchCompositor(green, soften=.5))
    np.testing.assert_array_equal(mask, plain_mask)
    np.testing.assert_array_equal(colored[mask == 0], frame[mask == 0])
    assert np.all(colored[mask > 0] == [0, 255, 0]) and np.any(plain != colored)
    classic = ClassicHarmonizer(color_transfer=False, poisson=False)
    np.testing.assert_array_equal(classic(plain, mask, frame), plain)
    with pytest.raises(ValueError, match="visibility is unknown"):
        render_resolved(frame, camera, optical_from_world, item, replace(evidence, visibility=None))
    # Explicitly absent model weights are an external boundary; strict mode must expose failure.
    with pytest.raises(RuntimeError, match="PCTNet harmonization failed"):
        PCTNetHarmonizer(weights=str(tmp_path / "missing.pth"), strict=True)(plain, mask, frame)


def test_insertion_attack_action(camera, optical_from_world):
    image = np.full((80, 100, 3), [20, 60, 80], np.uint8)
    texture = np.full((16, 16, 4), [220, 0, 0, 255], np.uint8)
    target = Insertion("target", PlaneAsset(texture, 4., 2.), WorldPlacement((10., 0., 0.)),
                       Orientation("fixed_world", (0., 0., 180.)))
    texture[:, :8, 3] = 0
    surface = PlaneSurface(np.array([[0., -1., 1.], [0., 1., 1.],
                                    [0., 1., 0.], [0., -1., 0.]]), texture)
    pose = np.eye(4)
    pose[:3, 3] = (5., 0., -.5)
    placed = PlacedObject("screen", ObjectAsset.from_surfaces((surface,), "screen"), pose)
    view = SensorView("front", image, camera, optical_from_world,
                      {"unbounded": ActorPose(np.eye(4))}, ActorPose(np.eye(4)), 0., 0,
                      objects={"screen": placed})
    action = InsertionAttackAction("front", target)
    rendered = action.apply(image, view)
    assert np.any(rendered[:, 50:] != image[:, 50:])
    np.testing.assert_array_equal(rendered[:, :50], image[:, :50])
    assert action.receipt_details() == {
        "rendered": True, "reason": "", "occlusion_source": "actor_cuboids",
        "ignored_occluders": ("unbounded",),
    }
    action.apply(image, replace(view, depth=np.full((80, 100), np.inf)))
    assert action.receipt_details()["ignored_occluders"] == ()
