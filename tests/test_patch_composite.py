"""Patch compositor: pure projection/geometry offline; warp + harmonize under a cv2 guard."""

import numpy as np
import pytest
from avsectester.attacks.patch_composite import (
    cam_coords,
    carla_cam_coords,
    project_to_pixels,
)


def test_projection():
    # pinhole K (f=50, principal point 50,50): optical axis -> principal point; +x shifts right by f*x/z
    k = np.array([[50.0, 0, 50], [0, 50.0, 50], [0, 0, 1]])
    assert project_to_pixels(np.array([[0, 0, 1.0]]), k)[0] == pytest.approx([50, 50])
    assert project_to_pixels(np.array([[1, 0, 1.0]]), k)[0] == pytest.approx([100, 50])


def test_extrinsics_and_carla_axis_swap():
    eye = np.eye(4)
    assert np.allclose(cam_coords(np.array([[1, 2, 3.0]]), eye), [[1, 2, 3]])
    # CARLA UE (x fwd, y right, z up) -> standard (x right, y down, z fwd) = [y, -z, x]
    assert np.allclose(carla_cam_coords(np.array([[1, 2, 3.0]]), eye), [[2, -3, 1]])


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


def test_order_quad_shared_helper():
    from avsectester.attacks.patch_composite import order_quad

    pts = np.array([[5, 5], [1, 5], [1, 1], [5, 1]], float)  # BR, BL, TL, TR scrambled
    assert np.allclose(order_quad(pts), [[1, 1], [5, 1], [5, 5], [1, 5]])  # TL, TR, BR, BL


def test_box_to_quad_planar_target():
    """A detection box -> a centered planar-warp quad (TL,TR,BR,BL); yaw makes it a trapezoid."""
    from avsectester.attacks.patch_composite import box_to_quad

    q = box_to_quad([100, 100, 200, 200], width_frac=0.5, height_frac=0.5, v_center=0.5)
    assert np.allclose(q, [[125, 125], [175, 125], [175, 175], [125, 175]])  # centered half-size box
    qy = box_to_quad([100, 100, 200, 200], width_frac=0.5, height_frac=0.5, yaw=0.3)
    assert (qy[0, 1] < qy[1, 1]) and (qy[3, 1] > qy[2, 1])  # left edge taller -> foreshortened trapezoid
