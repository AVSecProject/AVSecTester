"""Recorded-image playback requires an image, not a complete nuScenes dataset."""

import numpy as np
from avsectester.scenarios.datasets.nuscenes import RecordedFrameBackend


def test_recorded_frame_backend_serves_the_image(tmp_path):
    import cv2

    p = tmp_path / "frame.png"
    cv2.imwrite(str(p), np.full((90, 160, 3), 128, np.uint8))
    backend = RecordedFrameBackend(str(p), sensor_id="front")
    assert backend.reset().sensor_data["front"].shape == (90, 160, 3)
    assert backend.step(None).sensor_data["front"].shape == (90, 160, 3)  # static: same frame
