"""Offline projection, compositing, target selection and camera perturbation tests."""

import numpy as np
import pytest
from avsectester.rendering.cameras import cam_coords, carla_cam_coords, project_to_pixels


def test_projection():
    # pinhole K (f=50, principal point 50,50): optical axis -> principal point. +x shifts right by f*x/z
    k = np.array([[50.0, 0, 50], [0, 50.0, 50], [0, 0, 1]])
    assert project_to_pixels(np.array([[0, 0, 1.0]]), k)[0] == pytest.approx([50, 50])
    assert project_to_pixels(np.array([[1, 0, 1.0]]), k)[0] == pytest.approx([100, 50])


def test_extrinsics_and_carla_axis_swap():
    eye = np.eye(4)
    assert np.allclose(cam_coords(np.array([[1, 2, 3.0]]), eye), [[1, 2, 3]])
    # CARLA UE (x fwd, y right, z up) -> standard (x right, y down, z fwd) = [y, -z, x]
    assert np.allclose(carla_cam_coords(np.array([[1, 2, 3.0]]), eye), [[2, -3, 1]])


def test_composite_view_wraps_base_and_skips_when_no_quad():
    """The generic view wrapper composites only when quad_of yields a quad. Else passes the frame."""
    from avsectester.plane import Observation
    from avsectester.simulators.patch_insertion import composite_view

    frame = np.zeros((8, 8, 3), np.uint8)
    patched = np.ones((8, 8, 3), np.uint8)

    class StubCompositor:
        def apply(self, rgb, quad, patch):
            return patched

    obs = Observation(t=0.0, frame=0)
    quad = np.array([[1, 1], [5, 1], [5, 5], [1, 5]], np.float32)

    hit = composite_view(StubCompositor(), None, lambda _o: quad, base=lambda _o: frame)
    miss = composite_view(StubCompositor(), None, lambda _o: None, base=lambda _o: frame)
    none_base = composite_view(StubCompositor(), None, lambda _o: quad, base=lambda _o: None)
    assert np.array_equal(hit(obs), patched)  # quad present -> composited
    assert np.array_equal(miss(obs), frame)  # no quad -> clean frame unchanged
    assert none_base(obs) is None  # base view skipped -> skip


def test_warp_and_harmonize_under_cv2():
    from avsectester.rendering.harmonizers import ClassicHarmonizer
    from avsectester.simulators.patch_insertion import PatchCompositor, warp_patch

    frame = np.full((40, 60, 3), 100, np.uint8)
    patch = np.dstack([np.full((16, 16), 255, np.uint8)] * 3 + [np.full((16, 16), 255, np.uint8)])
    quad = np.array([[10, 10], [30, 10], [30, 28], [10, 28]], np.float32)
    comp, mask = warp_patch(frame, quad, patch)
    assert comp.shape == frame.shape and (mask > 0).any()  # patch landed in the quad
    out = PatchCompositor(ClassicHarmonizer()).apply(frame, quad, patch)
    assert out.shape == frame.shape and out.dtype == np.uint8


def test_order_quad_shared_helper():
    from avsectester.simulators.patch_insertion import order_quad

    pts = np.array([[5, 5], [1, 5], [1, 1], [5, 1]], float)  # BR, BL, TL, TR scrambled
    assert np.allclose(order_quad(pts), [[1, 1], [5, 1], [5, 5], [1, 5]])  # TL, TR, BR, BL


def test_box_to_quad_planar_target():
    """A detection box -> a centered planar-warp quad (TL,TR,BR,BL). Yaw makes it a trapezoid."""
    from avsectester.simulators.patch_insertion import box_to_quad

    q = box_to_quad([100, 100, 200, 200], width_frac=0.5, height_frac=0.5, v_center=0.5)
    assert np.allclose(
        q, [[125, 125], [175, 125], [175, 175], [125, 175]]
    )  # centered half-size box
    qy = box_to_quad([100, 100, 200, 200], width_frac=0.5, height_frac=0.5, yaw=0.3)
    assert (qy[0, 1] < qy[1, 1]) and (
        qy[3, 1] > qy[2, 1]
    )  # left edge taller -> foreshortened trapezoid


def test_render_plane_lands_where_projected_and_keeps_red(ftheta_camera, cam_from_world):
    from avsectester.attacks.object_insertion.sign_spoof import RoadsideSign
    from avsectester.rendering.harmonizers import ClassicHarmonizer
    from avsectester.simulators.patch_insertion import PatchCompositor, render_plane

    frame = np.full((1080, 1920, 3), 90, np.uint8)
    sign = RoadsideSign(x=12.0, y=-4.0, post=False)
    corners, face = sign.planes()[0]
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
    kept = PatchCompositor(ClassicHarmonizer(preserve_chroma=True, blend="feather")).apply_planes(
        frame, ftheta_camera, cam_from_world, [(corners, face)]
    )
    r, g, b = kept[mask > 0].astype(float).mean(axis=0)
    assert r > g + 30 and r > b + 30


def test_render_plane_behind_camera_is_noop(ftheta_camera, cam_from_world):
    from avsectester.attacks.object_insertion.sign_spoof import RoadsideSign
    from avsectester.simulators.patch_insertion import render_plane

    frame = np.full((1080, 1920, 3), 90, np.uint8)
    corners, face = RoadsideSign(x=-10.0, y=-4.0, post=False).planes()[0]  # behind the ego
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


def test_lane_pick_and_hold():
    from avsectester.simulators.patch_insertion import detector_quad, hold_quad

    frame = np.zeros((100, 200, 3), np.uint8)
    lead = ([90, 40, 130, 60], 0.9, 8)  # spans the centre column (x=100): the ego-lane lead
    big_side = ([10, 30, 95, 90], 0.9, 3)  # larger, nearer, but in the next lane
    boxes = [[lead, big_side], [], [], [], [], [lead]]
    detect = lambda _rgb: boxes.pop(0)
    quad_of = hold_quad(
        detector_quad(detect, base=lambda _o: frame, pick="lane", aspect=1.0), frames=3
    )
    q0 = quad_of(None)
    assert (
        q0 is not None and 90 < q0[:, 0].mean() < 130
    )  # took the in-lane lead, not the bigger car
    assert np.allclose(q0[2] - q0[1], [0, q0[1, 0] - q0[0, 0]])  # aspect=1: square
    held = [quad_of(None) for _ in range(4)]
    assert (
        all(np.array_equal(h, q0) for h in held[:3]) and held[3] is None
    )  # held 3 misses, then dropped
    assert quad_of(None) is not None  # re-acquired


@pytest.mark.parametrize("reverse_surfaces", [False, True])
@pytest.mark.parametrize("harmonize", [False, True])
def test_insertion_renderer_depth_orders_surfaces_and_preserves_cutout_holes(
    reverse_surfaces,
    harmonize,
):
    from avsectester.insertion import ActorPose, Insertion, PlaneAsset, PlaneSurface, WorldPlacement
    from avsectester.plane import Observation
    from avsectester.rendering.cameras import PinholeCamera
    from avsectester.rendering.types import InsertionGeometry
    from avsectester.simulators.patch_insertion import InsertionRenderer, PatchCompositor

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
    compositor = PatchCompositor(lambda image, mask, background: image) if harmonize else None
    render = InsertionRenderer(
        [Insertion("layered", LayeredAsset(), WorldPlacement((0, 0, 0)))],
        camera,
        lambda _: InsertionGeometry({}, ActorPose(np.eye(4)), camera_from_world),
        compositor=compositor,
    )
    image = render(Observation(0, 0), np.zeros((80, 100, 3), np.uint8))
    np.testing.assert_array_equal(image[40, 50], [0, 0, 255])
    np.testing.assert_array_equal(image[40, 56], [255, 0, 0])
