"""Camera projection, inverse projection and camera-to-rig axes."""

import numpy as np

from avsectester.geometry.cameras import PinholeCamera


def test_ftheta_project_unproject_roundtrip():
    from avsectester.geometry.cameras import FThetaCamera
    ftheta_camera = FThetaCamera(50, 40, (0, 50), 100, 80, (0, .02))
    rays = np.array([[0, 0, 1], [0.3, 0.1, 1], [-0.8, 0.3, 1], [1.2, -0.4, 1.0]])
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    uv = ftheta_camera.project(rays * 7.0)
    assert np.allclose(uv[0], [ftheta_camera.cx, ftheta_camera.cy])  # optical axis -> principal point
    back = ftheta_camera.unproject(uv)
    assert np.degrees(np.arccos(np.clip((back * rays).sum(1), -1, 1))).max() < 0.01


def test_pinhole_roundtrip():
    cam = PinholeCamera(np.array([[500.0, 0, 320], [0, 500.0, 240], [0, 0, 1]]), 640, 480)
    pts = np.array([[1.0, -0.5, 4.0], [0.0, 0.0, 2.0]])
    np.testing.assert_allclose(cam.project(pts), [[445, 177.5], [320, 240]])
    rays = cam.unproject(cam.project(pts))
    assert np.allclose(rays, pts / np.linalg.norm(pts, axis=1, keepdims=True))


def test_calibrated_pinhole_projects_the_same_world_geometry():
    lens = PinholeCamera(np.array([[50., 0., 50.], [0., 50., 40.], [0., 0., 1.]]), 100, 80)
    np.testing.assert_allclose(lens.project([[0, 0, 10]]), [[50, 40]])
    points = np.array([[10., -2., 1.], [10., 2., 1.], [10., 2., -1.], [10., -2., -1.]])
    camera_from_world = np.array([[0., -1., 0.], [0., 0., -1.], [1., 0., 0.]])
    pixels = lens.project(points @ camera_from_world.T)
    np.testing.assert_allclose(pixels.min(axis=0), [40, 35])
    np.testing.assert_allclose(pixels.max(axis=0), [60, 45])


def test_fisheye_model_projects_and_unprojects_the_same_rays():
    from avsectester.geometry.cameras import FThetaCamera

    camera = FThetaCamera(50, 40, (0, 50), 100, 80, (0, 1 / 50), max_angle=1.5)
    rays = np.array([[0, 0, 1], [0.3, -0.2, 1.0]])
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    np.testing.assert_allclose(camera.unproject(camera.project(rays)), rays, atol=1e-10)
