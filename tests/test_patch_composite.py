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
