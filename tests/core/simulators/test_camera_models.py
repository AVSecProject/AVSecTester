"""Camera projection, inverse projection and camera-to-rig axes."""

import numpy as np

from avsectester.simulators.camera_models import PinholeCamera


def test_ftheta_project_unproject_roundtrip(ftheta_camera):
    rays = np.array([[0, 0, 1], [0.3, 0.1, 1], [-0.8, 0.3, 1], [1.2, -0.4, 1.0]])
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    uv = ftheta_camera.project(rays * 7.0)
    assert np.allclose(uv[0], [ftheta_camera.cx, ftheta_camera.cy])  # optical axis -> principal point
    back = ftheta_camera.unproject(uv)
    assert np.degrees(np.arccos(np.clip((back * rays).sum(1), -1, 1))).max() < 0.01


def test_pinhole_roundtrip():
    cam = PinholeCamera(np.array([[500.0, 0, 320], [0, 500.0, 240], [0, 0, 1]]), 640, 480)
    pts = np.array([[1.0, -0.5, 4.0], [0.0, 0.0, 2.0]])
    rays = cam.unproject(cam.project(pts))
    assert np.allclose(rays, pts / np.linalg.norm(pts, axis=1, keepdims=True))


def test_rig_extrinsic_is_forward_looking(ftheta_camera, cam_from_world):
    """NuRec camera axes (x right, y down, z fwd) map to rig (-y, -z, +x): a point ahead of the ego
    lands near the image centre column, below the principal point when it is on the ground."""
    pts = np.array([[20.0, 0.0, 0.0]])  # 20 m ahead on the ground
    cam = (cam_from_world[:3, :3] @ pts.T).T + cam_from_world[:3, 3]
    assert cam[0, 2] > 17  # in front
    u, v = ftheta_camera.project(cam)[0]
    assert abs(u - ftheta_camera.cx) < 20 and v > ftheta_camera.cy
