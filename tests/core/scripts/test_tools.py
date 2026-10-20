"""Remaining tool entry points and timestamp-based video/frame matching."""

import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from scripts import REPO_ROOT
from scripts.visualization.visualize_scene_labels import recorded_video_frame


@pytest.mark.parametrize("module", [
    "validation.patch_hide_probe", "validation.validate_carla_visibility",
    "visualization.visualize_augmentations", "visualization.visualize_scene_labels",
])
def test_help_requires_no_simulator_weights_or_records(module):
    result = subprocess.run([sys.executable, "-m", f"scripts.{module}", "--help"],
                            cwd=REPO_ROOT, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


def test_video_matching_uses_recorded_time_not_decision_position(tmp_path, monkeypatch):
    timestamps = np.arange(10) / 30
    capture = Mock()
    bgr = np.full((2, 3, 3), [1, 2, 3], dtype=np.uint8)
    capture.read.return_value = (True, bgr)
    cv2 = SimpleNamespace(VideoCapture=Mock(return_value=capture), CAP_PROP_POS_FRAMES=1)
    monkeypatch.setitem(sys.modules, "cv2", cv2)
    rgb = recorded_video_frame(tmp_path / "front.mp4", timestamps, .1)
    capture.set.assert_called_once_with(1, 3)
    np.testing.assert_array_equal(rgb[0, 0], [3, 2, 1])
    capture.release.assert_called_once()


def test_video_matching_rejects_unsupported_time_before_opening_video(tmp_path, monkeypatch):
    cv2 = SimpleNamespace(VideoCapture=Mock())
    monkeypatch.setitem(sys.modules, "cv2", cv2)
    with pytest.raises(ValueError, match="support"):
        recorded_video_frame(tmp_path / "front.mp4", [0., .1], .2)
    cv2.VideoCapture.assert_not_called()


def test_validation_scripts_configure_distinct_rpc_and_traffic_manager_ports(monkeypatch):
    """Overriding RPC must not leave TM on an unrelated server's default port."""
    from scripts.validation.patch_hide_probe import probe_configuration
    from scripts.validation.validate_carla_visibility import validation_configuration

    for build in (lambda port, tm: probe_configuration(REPO_ROOT / "configs/carla_patch.yaml",
                                                       9., port, tm),
                  lambda port, tm: validation_configuration(REPO_ROOT / "configs/carla_patch.yaml",
                                                            port, tm)):
        _, simulator, _ = build(2300, None)
        assert simulator.client_config["connect_port"] == 2300
        assert simulator.client_config["traffic_manager_port"] == 8300
        _, simulator, _ = build(2300, 9300)
        assert simulator.client_config["traffic_manager_port"] == 9300


def test_visibility_validation_uses_simulator_capture_without_private_sensor_transport():
    """The script reuses sensor_view and contains no competing depth/camera implementation."""
    from pathlib import Path
    from scripts.validation import validate_carla_visibility

    source = Path(validate_carla_visibility.__file__).read_text()
    assert '.sensor_view("camera-0-0")' in source
    assert "sensor.camera." not in source
    assert "from queue import" not in source
    assert "raw_data" not in source
    assert "16777215" not in source
