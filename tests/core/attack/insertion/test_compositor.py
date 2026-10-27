"""Image compositing preserves transparent holes, surface depth and clipping."""
from dataclasses import replace
import numpy as np
import pytest
from avsectester.attack.insertion import (CuboidVisibilityEstimator,
    DepthVisibilityEstimator, Insertion, PatchCompositor, PlaneAsset,
    render_resolved, resolve_insertion)
from avsectester.attack.insertion.placement import Orientation, WorldPlacement
from avsectester.data_types import ActorState


def insertion(identity, x, texture, size):
    return resolve_insertion(Insertion(identity, PlaneAsset(texture, size, size),
        WorldPlacement((x, 0, 0)), Orientation("fixed_world", (0, 0, 180))), {}, ActorState(np.eye(4)))


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


def test_patch_compositor(camera, optical_from_world):
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
    with pytest.raises(ValueError, match="visibility is unknown"):
        render_resolved(frame, camera, optical_from_world, item, replace(evidence, visibility=None))
