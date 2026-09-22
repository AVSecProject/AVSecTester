"""Patch compositor: pure projection/geometry offline; warp + harmonize under a cv2 guard."""

import numpy as np
import pytest
from avsectester.attacks.patch_composite import (
    cam_coords,
    carla_cam_coords,
    intrinsics_from_fov,
    project_to_pixels,
    rear_face_quad,
)


def test_intrinsics_and_projection():
    k = intrinsics_from_fov(100, 100, 90.0)
    assert k[0, 0] == pytest.approx(50.0) and k[0, 2] == 50.0 and k[1, 2] == 50.0
    # a point on the optical axis maps to the principal point; +x shifts right by f*x/z
    assert project_to_pixels(np.array([[0, 0, 1.0]]), k)[0] == pytest.approx([50, 50])
    assert project_to_pixels(np.array([[1, 0, 1.0]]), k)[0] == pytest.approx([100, 50])


def test_extrinsics_and_carla_axis_swap():
    eye = np.eye(4)
    assert np.allclose(cam_coords(np.array([[1, 2, 3.0]]), eye), [[1, 2, 3]])
    # CARLA UE (x fwd, y right, z up) -> standard (x right, y down, z fwd) = [y, -z, x]
    assert np.allclose(carla_cam_coords(np.array([[1, 2, 3.0]]), eye), [[2, -3, 1]])


def test_rear_face_quad_corners():
    q = rear_face_quad([0, 0, 0], [1, 0, 0], [0, 1, 0])
    assert np.allclose(q, [[-1, 1, 0], [1, 1, 0], [1, -1, 0], [-1, -1, 0]])  # TL, TR, BR, BL


def test_composite_view_wraps_base_and_skips_when_no_quad():
    """The generic view wrapper composites only when quad_of yields a quad; else passes the frame."""
    from avsectester.plane import Observation
    from avsectester.simulators.viz import composite_view

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
    pytest.importorskip("cv2")
    from avsectester.attacks.patch_composite import ClassicHarmonizer, PatchCompositor, warp_patch

    frame = np.full((40, 60, 3), 100, np.uint8)
    patch = np.dstack([np.full((16, 16), 255, np.uint8)] * 3 + [np.full((16, 16), 255, np.uint8)])
    quad = np.array([[10, 10], [30, 10], [30, 28], [10, 28]], np.float32)
    comp, mask = warp_patch(frame, quad, patch)
    assert comp.shape == frame.shape and (mask > 0).any()  # patch landed in the quad
    out = PatchCompositor(ClassicHarmonizer()).apply(frame, quad, patch)
    assert out.shape == frame.shape and out.dtype == np.uint8


def test_ftheta_projection_axis_and_center():
    """f-theta: a point on the optical axis maps to the principal point; +x lands to its right."""
    from avsectester.simulators.nurec import ftheta_project

    pp = (960.0, 540.0)
    poly = [0.0, 900.0]  # r(theta) = 900*theta
    # straight ahead (+z) -> principal point
    px, z = ftheta_project(np.array([[0.0, 0.0, 5.0]]), pp, poly)
    assert np.allclose(px[0], pp) and z[0] > 0
    # a point offset in +x at 45deg (x==z) -> radius 900*(pi/4) to the right of cx, same cy
    px2, _ = ftheta_project(np.array([[1.0, 0.0, 1.0]]), pp, poly)
    assert px2[0, 0] == pytest.approx(pp[0] + 900.0 * (np.pi / 4), rel=1e-6)
    assert px2[0, 1] == pytest.approx(pp[1], abs=1e-6)


def test_order_quad_shared_helper():
    from avsectester.attacks.patch_composite import order_quad

    pts = np.array([[5, 5], [1, 5], [1, 1], [5, 1]], float)  # BR, BL, TL, TR scrambled
    assert np.allclose(order_quad(pts), [[1, 1], [5, 1], [5, 5], [1, 5]])  # TL, TR, BR, BL


def test_decal_project_plane_and_occlusion():
    """Decal on a fronto-parallel plane paints a centered block; points off the slab are skipped."""
    from avsectester.attacks.patch_composite import DecalFrame, decal_project, pinhole_unproject

    H = W = 40
    K = np.array([[50.0, 0, 20], [0, 50.0, 20], [0, 0, 1]])
    depth = np.full((H, W), 5.0)                 # a flat wall 5 m ahead
    frame = np.zeros((H, W, 3), np.uint8)
    patch = np.dstack([np.full((16, 16), 200, np.uint8)] * 3 + [np.full((16, 16), 255, np.uint8)])
    decal = DecalFrame(origin=np.array([0, 0, 5.0]), u_axis=np.array([1.0, 0, 0]),
                       v_axis=np.array([0, -1.0, 0]), normal=np.array([0, 0, 1.0]),
                       size_u=1.0, size_v=1.0, thickness=0.2)
    comp, mask = decal_project(frame, patch, depth, pinhole_unproject(K), decal)
    assert (mask > 0).any() and (comp[mask > 0] == 200).all()   # painted where the plane is in range
    # push the wall outside the decal slab -> nothing lands
    _, mask_far = decal_project(frame, patch, np.full((H, W), 50.0), pinhole_unproject(K), decal)
    assert not (mask_far > 0).any()


def test_ftheta_rays_inverse():
    """f-theta rays: optical-axis pixel -> +z; ftheta_project should round-trip the ray back."""
    from avsectester.simulators.nurec import ftheta_rays

    rays = ftheta_rays(60, 40, (30, 20), [0.0, 0.002])
    assert np.allclose(rays[20, 30], [0, 0, 1], atol=1e-6)     # principal point -> optical axis
    assert np.all(np.abs(np.linalg.norm(rays, axis=-1) - 1) < 1e-6)  # unit rays
    assert rays[35, 30, 1] > 0 and rays[5, 30, 1] < 0          # lower pixels look down, upper look up
