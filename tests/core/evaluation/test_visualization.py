"""Recorded scene views use log detections and physical actors, never running models."""

import numpy as np
import pytest

from avsectester.closed_loop.datatypes import Detection, Observation
from avsectester.evaluation.visualization import (
    annotate, box3d_corners, camera_view, detections_view, draw_boxes3d, filmstrip,
    labels_view, lidar_bev, record_run, save_gif, save_sequence, sensor_view,
)
from avsectester.geometry import PinholeCamera
from avsectester.runlog import Recorder, RunLog

from avsectester.closed_loop.simulator import SensorGeometry


@pytest.fixture
def camera_log(log_factory, actor_factory):
    result = log_factory([1, 1], actors={"host": actor_factory(10, category="vehicle")},
                 detections=(Detection((10, 0, 0), (4, 2, 2), 0, "car", .9),))
    # Ego x-forward/y-left/z-up to optical x-right/y-down/z-forward.
    camera_from_ego = np.array([[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]])
    camera = PinholeCamera(np.array([[60, 0, 64], [0, 60, 48], [0, 0, 1]]), 128, 96)
    for step in result.steps:
        step.model_observation = Observation(step.step * .1, step.step,
                                            sensor_data={"camera": np.zeros((96, 128, 3), np.uint8)})
        step.sensor_geometry = {"camera": SensorGeometry(
            "camera", camera, camera_from_ego @ np.linalg.inv(step.physical_pre.ego.pose))}
    return result


def test_camera_view_unwraps_native_or_data_only_camera_payloads():
    pixels = np.zeros((4, 5, 3), np.uint8)
    pixels[:, :, 0] = 20
    class ImageData:
        rgb_image = pixels

    obs = Observation(0, 0, sensor_data={"lidar": object(), "camera": ImageData()})
    np.testing.assert_array_equal(camera_view(obs), pixels)
    decoded = Observation(0, 0, sensor_data={"camera": {"data": pixels[:, :, ::-1],
                                                      "calibration": {"channel_order": "bgr"}}})
    np.testing.assert_array_equal(camera_view(decoded), pixels)
    assert camera_view(obs, "missing") is None
    assert lidar_bev(obs) is None


def test_lidar_view_reads_recorded_points_and_ignores_nonfinite_values():
    points = np.array([[5, 0, 0], [10, 2, 1], [np.nan, 0, 0]])
    view = lidar_bev(Observation(0, 0, sensor_data={"lidar": points}), size=100)
    assert view.shape == (100, 100, 3) and view[:, :, 1].sum() > 0


def test_world_and_detection_overlays_use_recorded_data(camera_log):
    record = camera_log.steps[0]
    assert labels_view(record).sum() > 0
    assert detections_view(record).sum() > 0
    assert detections_view(record, threshold=1).sum() == 0
    record.components = {"perception": object()}  # not a detector input or inference entry point
    assert detections_view(record).sum() > 0
    record.detections = ()
    assert detections_view(record).sum() == 0
    geometry = sensor_view(record)
    assert draw_boxes3d(camera_view(record), geometry, max_distance=5).sum() == 0
    assert draw_boxes3d(camera_view(record), geometry, max_distance=5, target="host").sum() > 0


def test_geometry_is_not_invented_when_camera_calibration_is_missing(camera_log):
    record = camera_log.steps[0]
    record.sensor_geometry = {}
    assert sensor_view(record) is None
    np.testing.assert_array_equal(labels_view(record), camera_view(record))
    np.testing.assert_array_equal(detections_view(record), camera_view(record))


def test_persisted_overlay_and_export_work_after_all_live_objects_are_gone(tmp_path, camera_log):
    original = camera_log
    with Recorder(tmp_path / "run") as recorder:
        for step in original.steps:
            recorder.record_step(step)
    decoded = RunLog.read(tmp_path / "run")
    assert isinstance(decoded.steps[0].sensor_geometry["camera"].camera, PinholeCamera)
    np.testing.assert_array_equal(labels_view(original.steps[0]), labels_view(decoded.steps[0]))
    np.testing.assert_array_equal(detections_view(original.steps[0]), detections_view(decoded.steps[0]))
    paths = record_run(decoded, tmp_path / "frames", visualize=labels_view)
    assert [path.name for path in paths] == ["frame_0000.png", "frame_0001.png"]
    assert all(path.read_bytes().startswith(b"\x89PNG") for path in paths)
    strip, gif = save_sequence(decoded, tmp_path / "sequence")
    assert strip.endswith("_filmstrip.png") and gif.endswith(".gif")
    assert filmstrip(decoded, cols=2).shape == (108, 274, 3)


def test_empty_sequences_fail_clearly_or_skip_optional_export(tmp_path):
    assert save_sequence(RunLog(), tmp_path) is None
    with pytest.raises(ValueError, match="Filmstrip"):
        filmstrip([])
    with pytest.raises(ValueError, match="GIF"):
        save_gif([], tmp_path / "empty.gif")


def test_box_dimensions_and_explicit_labels():
    corners = box3d_corners((10, 0, 0), (4, 2, 1.5), 0)
    np.testing.assert_allclose(np.ptp(corners, axis=0), [4, 2, 1.5])
    image = np.zeros((30, 30, 3), np.uint8)
    labeled = annotate(image, [((3, 5, 20, 20), 1, "vehicle", (200, 0, 0))])
    assert labeled[:, :, 0].sum() > 0 and labeled[:, :, 1].sum() == 0


def test_calibration_fallback_uses_nuscenes_ego_from_sensor(camera_log):
    record = camera_log.steps[0]
    geometry = record.sensor_geometry["camera"]
    expected = geometry.cam_from_world.copy()
    record.model_observation.calibration = {"camera": {
        "camera": geometry.camera,
        "ego_from_sensor": np.linalg.inv(record.physical_pre.ego.pose) @ np.linalg.inv(expected),
    }}
    record.sensor_geometry = {}
    np.testing.assert_allclose(sensor_view(record).cam_from_world, expected)
    assert labels_view(record).sum() > 0


def test_persisted_ftheta_geometry_retains_its_camera_type(tmp_path, camera_log):
    from avsectester.geometry.cameras import FThetaCamera

    lens = FThetaCamera(64, 48, (0, 60), 128, 96, (0, 1 / 60))
    step = camera_log.steps[0]
    step.sensor_geometry = {"camera": SensorGeometry(
        "camera", lens, step.sensor_geometry["camera"].cam_from_world)}
    with Recorder(tmp_path / "run") as recorder:
        recorder.record_step(step)
    decoded = RunLog.read(tmp_path / "run")
    geometry = sensor_view(decoded.steps[0])
    assert isinstance(geometry.camera, FThetaCamera)
    np.testing.assert_array_equal(labels_view(step), labels_view(decoded.steps[0]))
