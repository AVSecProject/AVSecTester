"""Per-simulation scene visualization: camera_view + record_run save frames (no server/GPU needed)."""

import pytest
from avsectester.backend import AVStack
from avsectester.plane import Control, Observation
from avsectester.simulators.nurec import NuRecBackend, StubRenderer
from avsectester.simulators.viz import camera_view, lidar_bev, record_run


class Cruise(AVStack):
    def __call__(self, obs):
        return Control(throttle=1.0)


def test_camera_view_returns_the_frame_and_skips_non_images():
    import numpy as np

    img = np.dstack([np.zeros((4, 5), np.uint8), np.ones((4, 5), np.uint8), np.full((4, 5), 2, np.uint8)])
    obs = Observation(t=0.0, frame=0, sensor_data={"cam": img})
    out = camera_view(obs)  # normalized HWC-uint8 RGB, not necessarily the same object
    assert out.shape == (4, 5, 3) and out.dtype == np.uint8 and np.array_equal(out, img)
    assert camera_view(Observation(t=0.0, frame=0, sensor_data={"cam": {"stub": 1}})) is None
    assert camera_view(Observation(t=0.0, frame=0)) is None  # empty sensor_data


def test_camera_view_extracts_rgb_from_an_imagedata_like_payload():
    import numpy as np

    class FakeImageData:  # avstack ImageData exposes .rgb_image + .shape
        def __init__(self, arr):
            self.data = arr

        @property
        def rgb_image(self):
            return self.data

        @property
        def shape(self):
            return self.data.shape

    arr = np.zeros((3, 4, 3), np.uint8)
    out = camera_view(Observation(t=0.0, frame=0, sensor_data={"camera-0": FakeImageData(arr)}))
    assert out.shape == (3, 4, 3) and out.dtype == np.uint8


def test_lidar_bev_is_defensive_on_non_lidar_payloads():
    obs = Observation(t=0.0, frame=0, sensor_data={"cam": object()})
    assert lidar_bev(obs) is None  # can't parse -> None, not a crash


def test_record_run_saves_a_frame_per_step(tmp_path):
    pytest.importorskip("matplotlib")
    backend = NuRecBackend({"dt": 0.1}, renderer=StubRenderer(cameras=["camera_front"], height=16, width=24))
    trace = record_run(backend, Cruise(), frames=4, out_dir=tmp_path, visualize=camera_view)
    assert len(trace.records) == 4
    saved = sorted(tmp_path.glob("frame_*.png"))
    assert [p.name for p in saved] == ["frame_0000.png", "frame_0001.png", "frame_0002.png", "frame_0003.png"]
