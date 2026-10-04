"""Recorded-image playback preserves RGB pixels and remains static under controls."""

import numpy as np
from PIL import Image

from avsectester.plane import Control
from avsectester.scenarios.datasets.nuscenes import RecordedFrameBackend


def test_recorded_frame_backend_preserves_rgb_and_replays_the_same_image(tmp_path):
    rgb = np.zeros((9, 16, 3), np.uint8)
    rgb[:] = (200, 70, 10)
    rgb[0, 0] = (20, 30, 240)
    path = tmp_path / "frame.png"
    Image.fromarray(rgb).save(path)
    backend = RecordedFrameBackend(str(path), sensor_id="front")

    initial = backend.reset()
    advanced = backend.step(Control(throttle=1.0, steer=0.5))
    restarted = backend.reset()
    for obs in (initial, advanced, restarted):
        assert set(obs.sensor_data) == {"front"}
        np.testing.assert_array_equal(obs.sensor_data["front"], rgb)
        assert obs.frame == initial.frame and obs.t == initial.t
