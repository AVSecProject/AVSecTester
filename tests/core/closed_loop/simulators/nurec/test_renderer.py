"""The NuRec RPC boundary retains recorded sensor pose, tilt, height and timing."""

from dataclasses import replace

import numpy as np
import pytest

from avsectester.closed_loop.simulators.nurec import NuRecRenderer, NuRecSimulator
from avsectester.closed_loop.simulators.nurec.renderer import ftheta_camera_from_spec, pose_from_proto
from avsectester.datasets.nurec import NuRecDataset
from avsectester.geometry.poses import rotation_matrix
from avsectester.data_types import Control
from tests.fakes import NuRecRpcBoundary


def test_recorded_support(nurec_usdz_path, monkeypatch):
    boundary = NuRecRpcBoundary().install(monkeypatch)
    dataset = NuRecDataset([nurec_usdz_path], time_step=.1)
    start = dataset.scenario_start("scene", dataset.frames("scene")[-2])
    origin = start.payload["ego"].pose.copy()
    origin[:3, :3] = rotation_matrix((8., -12., 30.))
    start = replace(start, payload={**start.payload, "ego": replace(start.payload["ego"], pose=origin)})
    with NuRecSimulator({"dt": .1}, NuRecRenderer(cameras=["front"]),
                        route_generator="none", physics_update_mode="none").create_runtime() as runtime:
        runtime.reset(start)
        assert runtime.native["renderer"].origin_timestamp_us == 400000
        assert runtime.native["renderer"].origin_bounds == (0., .5)
        np.testing.assert_allclose(runtime.state().ego.pose, origin, atol=1e-12)
        np.testing.assert_allclose(pose_from_proto(boundary.last_request.sensor_pose.start_pose),
                                   origin @ pose_from_proto(boundary.mount), atol=1e-12)
        runtime.step(Control())
        assert runtime.sensor_view("front").native["timestamp_us"] == 500000
        assert (boundary.last_request.frame_start_us, boundary.last_request.frame_end_us) == (500000, 500001)
        with pytest.raises(ValueError, match="terminal recorded support"):
            runtime.step(Control())
        runtime.reset()
        assert runtime.native["renderer"].origin_bounds is None
        assert runtime.observe().t == 0.


def test_resolution(nurec_dataset, monkeypatch):
    boundary = NuRecRpcBoundary().install(monkeypatch)
    start = nurec_dataset.scenario_start("scene", nurec_dataset.frames("scene")[0])
    renderer = NuRecRenderer(cameras=["front"], resolution_hw={"front": (6, 7)})
    simulator = NuRecSimulator({"dt": .1}, renderer, route_generator="none",
                               physics_update_mode="none")
    with simulator.create_runtime() as runtime:
        runtime.reset(start)
        view = runtime.sensor_view("front")
        geometry = runtime.sensor_geometry("front")
        native_camera = ftheta_camera_from_spec(boundary.spec)
        scale = .5
        assert (boundary.last_request.resolution_h, boundary.last_request.resolution_w) == (6, 7)
        assert (view.camera.height, view.camera.width) == (6, 8)
        assert view.data.shape == (6, 8, 3) and view.depth.shape == (6, 8)
        assert geometry.camera == view.camera
        points = np.array([[0., 0., 1.], [1., 2., 3.], [-2., 1., 4.]])
        projected = native_camera.project(points)
        np.testing.assert_allclose(view.camera.project(points), projected * scale, atol=1e-12)
        np.testing.assert_allclose(view.camera.unproject(projected * scale),
                                   native_camera.unproject(projected), atol=1e-12)
        boundary.returned_resolution_hw = (7, 9)
        with pytest.raises(ValueError, match="NRE image dimensions for camera 'front' changed"):
            runtime.step(Control())
    with pytest.raises(ValueError, match="resolution_hw names an unconfigured camera 'side'"):
        NuRecRenderer(cameras=["front"], resolution_hw={"side": (6, 8)})
    with pytest.raises(ValueError, match="height must be a positive integer"):
        NuRecRenderer(cameras=["front"], resolution_hw={"front": (0, 8)})


def test_windshield_spec():
    from types import SimpleNamespace

    spec = NuRecRpcBoundary().spec
    windshield = SimpleNamespace(horizontal_poly=[0., 1., 0., 0., 0., 0.], vertical_poly=[0., 0., 0., 1., 0., 0.],
                                 horizontal_poly_inverse=[0., 1., 0., 0., 0., 0.],
                                 vertical_poly_inverse=[0., 0., 0., 1., 0., 0.])
    distorted = SimpleNamespace(**{**vars(spec), "bivariate_windshield_model_param": windshield,
                                   "WhichOneof": lambda group: "bivariate_windshield_model_param"})
    camera = ftheta_camera_from_spec(distorted)
    assert camera.windshield is not None and camera.windshield.vertical_poly == (0., 0., 0., 1., 0., 0.)
    assert ftheta_camera_from_spec(spec).windshield is None
    with pytest.raises(ValueError, match="Unsupported NuRec external distortion"):
        ftheta_camera_from_spec(SimpleNamespace(**{**vars(spec), "WhichOneof": lambda group: "other_model"}))
