"""Known-height sign ranging and camera-guided LiDAR vehicle-face estimates."""

import numpy as np
import pytest

from avsectester.closed_loop.avstacks.implemented.modules.ranging import (
    estimate_stop_sign_position, estimate_vehicle_face_points,
)
from avsectester.geometry import CameraCalibration, FThetaCamera, PinholeCamera
from avsectester.geometry.cameras import BODY_FROM_OPTICAL


@pytest.mark.parametrize("camera", [
    PinholeCamera(np.array([[400., 0., 400.], [0., 400., 300.], [0., 0., 1.]]), 800, 600),
    FThetaCamera(400., 300., (0., 400.), 800, 600, pixeldist_to_angle=(0., .0025)),
])
def test_estimate_stop_sign_position(camera):
    extrinsic = BODY_FROM_OPTICAL.copy()
    extrinsic[:3, 3] = [1., 0., 1.5]
    calibration = CameraCalibration(camera, extrinsic)
    height, distance = .762, 20.
    pixels = camera.project(np.array([[0., -height / 2, distance], [0., height / 2, distance]]))
    box = [pixels[0, 0] - 4., pixels[0, 1], pixels[1, 0] + 4., pixels[1, 1]]
    position = estimate_stop_sign_position(box, calibration, height)
    np.testing.assert_allclose(position, [21., 0., 1.5], rtol=.02, atol=.01)
    with pytest.raises(ValueError, match="nonempty"):
        estimate_stop_sign_position([1., 1., 1., 2.], calibration)
    invalid = extrinsic.copy()
    invalid[0, 0] = 2.
    with pytest.raises(ValueError, match="right-handed rigid"):
        estimate_stop_sign_position(box, CameraCalibration(camera, invalid), height)


def test_estimate_vehicle_face_points():
    camera = PinholeCamera(np.array([[400., 0., 400.], [0., 400., 300.], [0., 0., 1.]]), 800, 600)
    extrinsic = BODY_FROM_OPTICAL.copy()
    extrinsic[:3, 3] = [2., .4, 1.5]
    calibration = CameraCalibration(camera, extrinsic)
    box = np.array([300., 230., 500., 400.])
    # Unequal return density must not move the midpoint toward the denser side.
    ys = np.r_[np.linspace(-1., 1., 200), np.full(40, -1.), np.full(40, 1.), np.full(100, .6)]
    face_y, face_z = np.meshgrid(ys, [1.4, 1.7, 2.])
    face = np.column_stack((np.full(face_y.size, 20.), face_y.ravel(), face_z.ravel()))
    background = face.copy()
    background[:, 0] = 45.
    ground = np.column_stack((np.full(100, 10.), np.linspace(-1., 1., 100), np.zeros(100)))
    outside = np.tile([10., 10., 2.], (100, 1))
    too_close = np.tile([2.4, .4, 1.5], (600, 1))
    points = np.vstack((face, background, ground, outside, too_close))
    actual = estimate_vehicle_face_points(box, calibration, points)
    np.testing.assert_allclose(actual, [[20., 1., 0.], [20., 0., 0.], [20., -1., 0.]], atol=1e-12)
    with pytest.raises(ValueError, match="nonempty"):
        estimate_vehicle_face_points([1., 1., 1., 2.], calibration, points)
    with pytest.raises(ValueError, match=r"\(N, 3\)"):
        estimate_vehicle_face_points(box, calibration, np.zeros((5, 2)))
    with pytest.raises(TypeError, match="CameraCalibration"):
        estimate_vehicle_face_points(box, None, points)


def test_estimate_vehicle_face_points_insufficient_points():
    camera = PinholeCamera(np.array([[400., 0., 400.], [0., 400., 300.], [0., 0., 1.]]), 800, 600)
    extrinsic = BODY_FROM_OPTICAL.copy()
    extrinsic[:3, 3] = [2., .4, 1.5]
    calibration = CameraCalibration(camera, extrinsic)
    box = [300., 230., 500., 400.]
    with pytest.raises(ValueError, match="vehicle box region"):
        estimate_vehicle_face_points(box, calibration, np.tile([20., 0., 1.5], (4, 1)))
    separated = np.column_stack((np.arange(10., 60., 10.), np.full(5, .4), np.full(5, 1.5)))
    with pytest.raises(ValueError, match="nearest vehicle cluster"):
        estimate_vehicle_face_points(box, calibration, separated)
