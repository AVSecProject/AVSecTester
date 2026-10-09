"""Offline projection, compositing, target selection and camera perturbation tests."""

import numpy as np
import pytest


def test_image_warp_returns_an_alpha_composite_and_opaque_mask():
    from avsectester.simulators.patch_insertion import warp_patch

    frame = np.full((40, 60, 3), 100, np.uint8)
    patch = np.dstack([np.full((16, 16), 255, np.uint8)] * 3 + [np.full((16, 16), 255, np.uint8)])
    quad = np.array([[10, 10], [30, 10], [30, 28], [10, 28]], np.float32)
    comp, mask = warp_patch(frame, quad, patch)
    assert comp.shape == frame.shape and (mask > 0).any()  # patch landed in the quad
    np.testing.assert_array_equal(comp[15, 20], [255, 255, 255])
    np.testing.assert_array_equal(comp[0, 0], frame[0, 0])
    assert mask[15, 20] == 255 and mask[0, 0] == 0


def test_render_plane_lands_where_projected_and_keeps_red(ftheta_camera, cam_from_world):
    from avsectester.attacks.object_insertion.sign_spoof import SignAsset
    from avsectester.insertion import ActorPose, Insertion, WorldPlacement, Orientation, resolve_insertion
    from avsectester.rendering.harmonizers import ClassicHarmonizer
    from avsectester.simulators.patch_insertion import PatchCompositor, render_plane

    frame = np.full((1080, 1920, 3), 90, np.uint8)
    insertion = Insertion("sign", SignAsset(post=False), WorldPlacement((12, -4, 0)),
                          Orientation("fixed_world", (0, 0, 180)))
    resolved = resolve_insertion(insertion, {}, ActorPose(np.eye(4)))
    surface = resolved.planes()[0]
    corners, face = surface.corners, surface.texture
    _comp, mask = render_plane(frame, ftheta_camera, cam_from_world, corners, face)
    ys, xs = np.where(mask > 0)
    centre_world = corners.mean(axis=0)[None]
    c = (cam_from_world[:3, :3] @ centre_world.T).T + cam_from_world[:3, 3]
    u, v = ftheta_camera.project(c)[0]
    assert (
        abs(xs.mean() - u) < 3 and abs(ys.mean() - v) < 3
    )  # rendered footprint centred on the projection
    assert xs.mean() > ftheta_camera.cx  # right of centre (negative y)
    # chroma-preserving harmonization keeps the sign red. The default Lab transfer pulls it to grey
    from avsectester.simulators.patch_insertion import render_resolved

    kept, _ = render_resolved(frame, ftheta_camera, cam_from_world, resolved,
                             compositor=PatchCompositor(ClassicHarmonizer(preserve_chroma=True, blend="feather")))
    r, g, b = kept[mask > 0].astype(float).mean(axis=0)
    assert r > g + 30 and r > b + 30


def test_render_plane_behind_camera_is_noop(ftheta_camera, cam_from_world):
    from avsectester.attacks.object_insertion.sign_spoof import SignAsset
    from avsectester.insertion import ActorPose, Insertion, WorldPlacement, Orientation, resolve_insertion
    from avsectester.simulators.patch_insertion import render_plane

    frame = np.full((1080, 1920, 3), 90, np.uint8)
    insertion = Insertion("sign", SignAsset(post=False), WorldPlacement((-10, -4, 0)),
                          Orientation("fixed_world", (0, 0, 180)))
    surface = resolve_insertion(insertion, {}, ActorPose(np.eye(4))).planes()[0]
    corners, face = surface.corners, surface.texture
    comp, mask = render_plane(frame, ftheta_camera, cam_from_world, corners, face)
    assert not mask.any() and np.array_equal(comp, frame)


@pytest.mark.parametrize("custom_view", [False, True], ids=["default-view", "custom-view"])
def test_frame_perturbation_reads_and_rewrites_the_selected_camera(custom_view):
    from avsectester.plane import Observation
    from avsectester.simulators.patch_insertion import frame_perturbation

    other = np.full((4, 4, 3), 90, np.uint8)
    target = np.zeros((4, 4, 3), np.uint8)
    obs = Observation(t=0.0, frame=0, sensor_data={"other": other, "target": target})
    calls = []

    def view(observation):
        calls.append(observation)
        return observation.sensor_data["target"] + 3

    kwargs = {"base": view} if custom_view else {}
    seen = frame_perturbation(lambda o, rgb: rgb + 7, camera="target", **kwargs)(obs)

    assert (seen.sensor_data["target"] == (10 if custom_view else 7)).all()
    assert seen.sensor_data["other"] is other
    assert (other == 90).all() and (target == 0).all()
    if custom_view:
        assert len(calls) == 1 and calls[0] is obs


@pytest.mark.parametrize("reverse_surfaces", [False, True])
def test_insertion_renderer_depth_orders_surfaces_and_preserves_cutout_holes(
    reverse_surfaces,
):
    from avsectester.insertion import ActorPose, Insertion, PlaneAsset, PlaneSurface, WorldPlacement
    from avsectester.plane import Observation
    from avsectester.rendering.cameras import PinholeCamera
    from avsectester.rendering.types import InsertionGeometry
    from avsectester.simulators.patch_insertion import InsertionRenderer

    red = np.full((16, 16, 4), [255, 0, 0, 255], np.uint8)
    red[6:10, 6:10, 3] = 0
    blue = np.full((16, 16, 4), [0, 0, 255, 255], np.uint8)
    near = PlaneAsset(red, 2, 2).planes()[0]
    far = PlaneAsset(blue, 4, 4).planes()[0]
    surfaces = [
        PlaneSurface(near.corners + [5, 0, 0], near.texture),
        PlaneSurface(far.corners + [10, 0, 0], far.texture),
    ]
    if reverse_surfaces:
        surfaces.reverse()

    class LayeredAsset:
        def planes(self):
            return surfaces

    camera = PinholeCamera(np.array([[50, 0, 50], [0, 50, 40], [0, 0, 1]]), 100, 80)
    camera_from_world = np.array([[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]])
    render = InsertionRenderer(
        [Insertion("layered", LayeredAsset(), WorldPlacement((0, 0, 0)))],
        camera,
        lambda _: InsertionGeometry({}, ActorPose(np.eye(4)), camera_from_world),
    )
    image = render(Observation(0, 0), np.zeros((80, 100, 3), np.uint8))
    np.testing.assert_array_equal(image[40, 50], [0, 0, 255])
    np.testing.assert_array_equal(image[40, 56], [255, 0, 0])
