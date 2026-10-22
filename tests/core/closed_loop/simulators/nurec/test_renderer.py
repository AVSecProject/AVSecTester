"""The NuRec RPC boundary retains recorded sensor pose, tilt, height and timing."""

from dataclasses import replace

import numpy as np
import pytest

from avsectester.closed_loop.simulators.nurec import NuRecRenderer, NuRecSimulator
from avsectester.closed_loop.simulators.nurec.renderer import pose_from_proto
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
    with NuRecSimulator({"dt": .1}, NuRecRenderer(cameras=["front"])).create_runtime() as runtime:
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
