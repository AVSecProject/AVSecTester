"""Known angular heights recover positions through both supported lens models."""

import numpy as np
import pytest

from avsectester.closed_loop.avstacks.implemented.rule_based.ranging import estimate_stop_sign_position
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
