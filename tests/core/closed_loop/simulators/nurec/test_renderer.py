"""The NuRec RPC boundary retains recorded sensor pose, tilt, height and timing."""

import json
import zipfile

import numpy as np
import pytest

from avsectester.closed_loop.simulators.nurec import NuRecRenderer
from avsectester.geometry import CameraCalibration, FThetaCamera
from avsectester.geometry.lidars import LidarCalibration
from avsectester.geometry.poses import make_pose
from avsectester.data_types import PointCloud
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


def test_lidar_calibration(monkeypatch, nurec_usdz_path, tmp_path):
    NuRecRpcBoundary().install(monkeypatch)
    renderer = NuRecRenderer(cameras=["front"], lidar="lidar_gt_top_p128")
    scene = {"scene_id": "scene", "usdz_path": nurec_usdz_path}
    try:
        renderer.load_scene(scene)
        calibration = renderer.calibration["lidar_gt_top_p128"]
        assert isinstance(calibration, LidarCalibration)
        assert calibration.spin_period_s == .1
        np.testing.assert_array_equal(calibration.rig_from_lidar, [
            [1., 0., 0., 1.17], [0., 1., 0., 0.],
            [0., 0., 1., 1.95], [0., 0., 0., 1.]])
        assert renderer.copy().calibration == {}
        missing = NuRecRenderer(cameras=["front"], lidar="missing")
        try:
            with pytest.raises(ValueError, match="LiDAR 'missing' is not available"):
                missing.load_scene(scene)
        finally:
            missing.close()
        with zipfile.ZipFile(nurec_usdz_path) as archive:
            metadata = json.loads(archive.read("rig_trajectories.json"))
        path = tmp_path / "invalid-lidar.usdz"
        calibration_record = metadata["lidar_calibrations"]["lidar_gt_top_p128@scene"]
        calibration_record["lidar_model"]["parameters"]["spinning_frequency_hz"] = 1.09e-43
        rig = metadata["rig_trajectories"][0]
        rig["lidars_frame_timestamps_us"] = {
            "lidar_gt_top_p128@scene": [[0, 98000], [100000, 200000], [200000, 304000]]}
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("rig_trajectories.json", json.dumps(metadata))
        renderer.load_scene({**scene, "usdz_path": path})
        assert renderer.calibration["lidar_gt_top_p128"].spin_period_s == .1
        for scans in ([], [[0, 0]], [[100000, 99999]], [[0, float("inf")]]):
            rig["lidars_frame_timestamps_us"] = {"lidar_gt_top_p128@scene": scans}
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("rig_trajectories.json", json.dumps(metadata))
            with pytest.raises(ValueError, match="has no recorded LiDAR scans"):
                renderer.load_scene({**scene, "usdz_path": path})
        calibration_record["lidar_model"]["type"] = "solid-state"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("rig_trajectories.json", json.dumps(metadata))
        with pytest.raises(ValueError, match="requires a row-offset-spinning model"):
            renderer.load_scene({**scene, "usdz_path": path})
        metadata["lidar_calibrations"]["another"] = calibration_record
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("rig_trajectories.json", json.dumps(metadata))
        with pytest.raises(ValueError, match="supports scenes with only one LiDAR"):
            renderer.load_scene({**scene, "usdz_path": path})
    finally:
        renderer.close()


def test_render_lidar(monkeypatch, nurec_usdz_path):
    boundary = NuRecRpcBoundary().install(monkeypatch)
    renderer = NuRecRenderer(cameras=["front"], lidar="lidar_gt_top_p128")
    renderer.load_scene({"scene_id": "scene", "usdz_path": nurec_usdz_path})
    start = make_pose(np.eye(3), (3., -2., .3))
    end = make_pose(np.eye(3), (3.5, -1., .3))
    actor_start = make_pose(np.eye(3), (8., 1., .5))
    actor_end = make_pose(np.eye(3), (9., 2., .5))
    try:
        cloud = renderer.render_lidar(start, end, 100000, 200000,
            [("host", actor_start), ("ended", actor_start)],
            [("host", actor_end), ("started", actor_end)])
        assert isinstance(cloud, PointCloud)
        assert cloud.points.dtype == np.float32 and cloud.intensity is None
        np.testing.assert_array_equal(cloud.points,
            np.array([[4., 1., -1.95], [8., -2., -1.]], np.float32))
        request = boundary.lidar_requests[-1]
        assert request.scene_id == "scene"
        assert (request.frame_start_us, request.frame_end_us) == (100000, 200000)
        assert request.lidar_config.lidar_type == 0
        mount = renderer.calibration[renderer.lidar].rig_from_lidar
        np.testing.assert_allclose(proto_world_pose(request.sensor_pose.start_pose), start @ mount)
        np.testing.assert_allclose(proto_world_pose(request.sensor_pose.end_pose), end @ mount)
        assert [item.track_id for item in request.dynamic_objects] == ["host"]
        actor = request.dynamic_objects[0]
        np.testing.assert_allclose(proto_world_pose(actor.pose_pair.start_pose), actor_start)
        np.testing.assert_allclose(proto_world_pose(actor.pose_pair.end_pose), actor_end)
        boundary.lidar_response.point_xyzs_buffer = b""
        boundary.lidar_response.point_xyzs = [1., 2., 3., 4., 5., 6.]
        boundary.lidar_response.point_intensities = [.2, .8]
        repeated = renderer.render_lidar(start, end, 100000, 200000)
        np.testing.assert_array_equal(repeated.points, [[1., 2., 3.], [4., 5., 6.]])
        np.testing.assert_array_equal(repeated.intensity, np.array([.2, .8], np.float32))
        assert repeated.intensity.dtype == np.float32
        boundary.lidar_response.point_intensities_buffer = np.array([.3, .7], np.float32).tobytes()
        buffered = renderer.render_lidar(start, end, 100000, 200000)
        np.testing.assert_array_equal(buffered.intensity, np.array([.3, .7], np.float32))
        boundary.lidar_response.num_points = 3
        with pytest.raises(ValueError, match="point count does not match num_points"):
            renderer.render_lidar(start, end, 100000, 200000)
    finally:
        renderer.close()


def test_render_lidar_native_protocol(monkeypatch, nurec_usdz_path):
    common = pytest.importorskip("alpasim_grpc.v0.common_pb2")
    sensors = pytest.importorskip("alpasim_grpc.v0.sensorsim_pb2")
    boundary = NuRecRpcBoundary().install(monkeypatch)
    import alpasim_grpc.v0 as protocol

    monkeypatch.setattr(protocol, "common_pb2", common)
    monkeypatch.setattr(protocol, "sensorsim_pb2", sensors)
    renderer = NuRecRenderer(cameras=["front"], lidar="lidar_gt_top_p128")
    try:
        renderer.load_scene({"scene_id": "scene", "usdz_path": nurec_usdz_path})
        cloud = renderer.render_lidar(np.eye(4), np.eye(4), 100000, 200000)
        request = boundary.lidar_requests[-1]
        assert isinstance(request, sensors.LidarRenderRequest)
        decoded = sensors.LidarRenderRequest.FromString(request.SerializeToString())
        assert decoded.lidar_config.lidar_type == sensors.PANDAR128
        assert decoded.scene_id == "scene" and len(cloud.points) == 2
    finally:
        renderer.close()
