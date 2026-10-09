"""Shared NuRec camera calibration for projection and compositor tests."""

import numpy as np
import pytest

from avsectester.rendering.cameras import FThetaCamera, make_pose, planar_rig_pose, quat_to_matrix


@pytest.fixture
def ftheta_camera():
    # camera_front_wide_120fov in scene clipgt-01d503d4.
    return FThetaCamera(
        cx=956.11, cy=754.85,
        angle_to_pixeldist=(0, 944.4935, -10.977355, 32.704746, -77.398132, 32.516323),
        pixeldist_to_angle=(0, 1.0570853e-3, 2.9625948e-08, -9.3092818e-11, 1.6680121e-13, -6.9934230e-17),
        width=1920, height=1080, max_angle=1.35,
    )


@pytest.fixture
def cam_from_world():
    rig_from_cam = make_pose(
        quat_to_matrix(-0.496408641, 0.504378, -0.495750666, 0.503401),
        (2.016, -0.061, 1.591),
    )
    return np.linalg.inv(planar_rig_pose(0.0, 0.0, 0.0) @ rig_from_cam)
