"""Scene visualization: the generic viz interface + the CARLA view adapter (no server/GPU needed)."""

import pytest
from avsectester.backend import AVStack
from avsectester.plane import Control, Observation
from avsectester.simulators import carla as carla_view
from avsectester.simulators.nurec import NuRecBackend, StubRenderer
from avsectester.simulators.viz import camera_view, detections_view, record_run


class Cruise(AVStack):
    def __call__(self, obs):
        return Control(throttle=1.0)


def test_generic_camera_view_takes_raw_ndarray_and_skips_non_images():
    import numpy as np

    img = np.dstack([np.zeros((4, 5), np.uint8), np.ones((4, 5), np.uint8), np.full((4, 5), 2, np.uint8)])
    out = camera_view(Observation(t=0.0, frame=0, sensor_data={"cam": img}))
    assert out.shape == (4, 5, 3) and out.dtype == np.uint8 and np.array_equal(out, img)
    assert camera_view(Observation(t=0.0, frame=0, sensor_data={"cam": {"stub": 1}})) is None
    assert camera_view(Observation(t=0.0, frame=0)) is None  # empty sensor_data


class _FakeImageData:  # avstack ImageData exposes .rgb_image
    def __init__(self, arr):
        self._arr = arr

    @property
    def rgb_image(self):
        return self._arr


def test_carla_camera_view_unwraps_imagedata_and_falls_back_to_ndarray():
    import numpy as np

    arr = np.zeros((3, 4, 3), np.uint8)
    out = carla_view.camera_view(Observation(t=0.0, frame=0, sensor_data={"camera-0": _FakeImageData(arr)}))
    assert out.shape == (3, 4, 3) and out.dtype == np.uint8
    # also accepts a raw ndarray (generic fallback), unlike the generic view which ignores ImageData
    raw = np.zeros((2, 2, 3), np.uint8)
    assert carla_view.camera_view(Observation(t=0.0, frame=0, sensor_data={"c": raw})).shape == (2, 2, 3)
    assert camera_view(Observation(t=0.0, frame=0, sensor_data={"c": _FakeImageData(arr)})) is None


def test_carla_lidar_bev_is_defensive_on_non_lidar_payloads():
    obs = Observation(t=0.0, frame=0, sensor_data={"cam": object()})
    assert carla_view.lidar_bev(obs) is None  # can't parse -> None, not a crash


def test_detections_view_overlays_boxes_on_any_base_view():
    pytest.importorskip("PIL")
    import numpy as np

    img = np.zeros((20, 20, 3), np.uint8)
    obs = Observation(t=0.0, frame=0, sensor_data={"cam": img})
    view = detections_view(lambda rgb: [((1, 1, 8, 8), 0.9, "car")], base=camera_view)
    out = view(obs)
    assert out.shape == (20, 20, 3)  # box drawn in green -> pixels changed
    assert out[:, :, 1].sum() > img[:, :, 1].sum()


def test_record_run_saves_a_frame_per_step(tmp_path):
    pytest.importorskip("matplotlib")
    backend = NuRecBackend({"dt": 0.1}, renderer=StubRenderer(cameras=["camera_front"], height=16, width=24))
    trace = record_run(backend, Cruise(), frames=4, out_dir=tmp_path, visualize=camera_view)
    assert len(trace.records) == 4
    saved = sorted(tmp_path.glob("frame_*.png"))
    assert [p.name for p in saved] == ["frame_0000.png", "frame_0001.png", "frame_0002.png", "frame_0003.png"]
