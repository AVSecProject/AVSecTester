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


def test_recorded_observation_edits_do_not_change_source_or_true_clock(tmp_path):
    rgb = np.full((2, 3, 3), 100, np.uint8)
    path = tmp_path / "frame.png"
    Image.fromarray(rgb).save(path)
    backend = RecordedFrameBackend(str(path), frame=12, t=4.0)

    observation = backend.reset()
    observation.sensor_data["front"][:] = 0
    observation.frame, observation.t = -1, -2.0
    truth = backend.ground_truth()
    later = backend.step(Control())

    assert truth.frame == later.frame == 12
    assert truth.t == later.t == 4.0
    assert truth.vehicle_state is None
    np.testing.assert_array_equal(later.sensor_data["front"], rgb)
