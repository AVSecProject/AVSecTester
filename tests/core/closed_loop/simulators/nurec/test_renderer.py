"""The NuRec RPC boundary retains recorded sensor pose, tilt, height and timing."""

import numpy as np
import pytest

from avsectester.closed_loop.simulators.nurec import NuRecRenderer
from avsectester.geometry import CameraCalibration, FThetaCamera
from tests.fakes import CAMERA, NuRecImageBoundary, NuRecRpcBoundary, proto_world_pose


def test_reset_origin(nurec_dataset):
    renderer = NuRecRenderer(cameras=["front"])
    start = nurec_dataset.scenario_start("scene", nurec_dataset.frames("scene")[-2])
    renderer.reset_origin(start)
    assert renderer.origin_timestamp_us == 400000
    assert renderer.origin_bounds == (0., .5)
    np.testing.assert_array_equal(renderer.origin_transform, start.payload["ego"].pose)
    renderer.reset_origin()
    assert renderer.origin_transform is None and renderer.origin_timestamp_us is None


def test_calibration(monkeypatch):
    assert NuRecImageBoundary(CAMERA).calibration == {}
    boundary = NuRecRpcBoundary().install(monkeypatch)
    renderer = NuRecRenderer(cameras=["front"])
    assert renderer.calibration == {}
    try:
        renderer.load_scene("scene")
        calibration = renderer.calibration["front"]
        assert isinstance(calibration, CameraCalibration)
        assert (calibration.camera.height, calibration.camera.width) == (12, 16)
        np.testing.assert_allclose(calibration.rig_from_camera, proto_world_pose(boundary.mount))
        np.testing.assert_allclose(
            calibration.rig_from_camera @ renderer.cam_from_world(np.eye(4), "front"), np.eye(4),
            atol=1e-12,
        )
        assert renderer.copy().calibration == {}
    finally:
        renderer.close()


def test_camera_model(monkeypatch):
    NuRecRpcBoundary().install(monkeypatch)
    renderer = NuRecRenderer(cameras=["front"])
    try:
        renderer.load_scene("scene")
        camera = renderer.camera_model("front")
        assert camera == FThetaCamera(8.5, 6.5, (0., 8.), 16, 12, (0., .125), 1.5)
    finally:
        renderer.close()


def test_windshield_spec(monkeypatch):
    from types import SimpleNamespace

    spec = NuRecRpcBoundary().spec
    windshield = SimpleNamespace(horizontal_poly=[0., 1., 0., 0., 0., 0.], vertical_poly=[0., 0., 0., 1., 0., 0.],
                                 horizontal_poly_inverse=[0., 1., 0., 0., 0., 0.],
                                 vertical_poly_inverse=[0., 0., 0., 1., 0., 0.])
    distorted = SimpleNamespace(**{**vars(spec), "bivariate_windshield_model_param": windshield,
                                   "WhichOneof": lambda group: "bivariate_windshield_model_param"})
    boundary = NuRecRpcBoundary().install(monkeypatch)
    boundary.spec = distorted
    renderer = NuRecRenderer(cameras=["front"])
    try:
        renderer.load_scene("scene")
        camera = renderer.camera_model("front")
        assert camera.windshield is not None
        assert camera.windshield.vertical_poly == (0., 0., 0., 1., 0., 0.)
        renderer.close()
        boundary.spec = spec
        renderer.load_scene("scene")
        assert renderer.camera_model("front").windshield is None
        renderer.close()
        boundary.spec = SimpleNamespace(**{**vars(spec), "WhichOneof": lambda group: "other_model"})
        with pytest.raises(ValueError, match="Unsupported NuRec external distortion"):
            renderer.load_scene("scene")
    finally:
        renderer.close()


def test_served_scenes(monkeypatch):
    boundary = NuRecRpcBoundary().install(monkeypatch)
    renderer = NuRecRenderer(cameras=["front"])
    assert renderer.served_scenes(timeout_s=.05) == ("scene",)
    assert boundary.scene_query_timeouts[-1] == .05
    assert renderer.calibration == {}
    renderer.load_scene("scene")
    try:
        assert renderer.served_scenes() == ("scene",)
        assert renderer.camera_model("front").width == boundary.spec.resolution_w
        for code in ("UNAVAILABLE", "DEADLINE_EXCEEDED"):
            boundary.scene_query_error = (code, "endpoint not ready")
            assert renderer.served_scenes() is None
        boundary.scene_query_error = ("PERMISSION_DENIED", "not authorized")
        with pytest.raises(RuntimeError, match="PERMISSION_DENIED.*not authorized"):
            renderer.served_scenes()
        with pytest.raises(ValueError, match="timeout_s"):
            renderer.served_scenes(timeout_s=0.)
        assert renderer.camera_model("front").width == boundary.spec.resolution_w
    finally:
        renderer.close()
