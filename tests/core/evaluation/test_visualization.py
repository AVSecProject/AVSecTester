"""Recorded visualization features use small images and physical camera geometry."""

from pathlib import Path
from dataclasses import replace

import numpy as np
import pytest

from avsectester.data_types import Detection, Observation
from avsectester.closed_loop.simulators.base import SensorGeometry
from avsectester.evaluation.visualization import (
    camera_view, detections_view, draw_boxes3d, filmstrip, labels_view, plot_group, record_run, save_sequence, sensor_view,
)
from avsectester.geometry import FThetaCamera, PinholeCamera
from avsectester.runlog import RunLog


@pytest.fixture(params=["pinhole", "fisheye"])
def camera_log(request, make_log, make_actor):
    log = make_log([1., 1.], actors={"lead": make_actor(10.)},
                      detections=[Detection((10., 0., 0.), (4., 2., 2.), 0., "car", .9)])
    camera = (PinholeCamera(np.array([[60., 0., 64.], [0., 60., 48.], [0., 0., 1.]]), 128, 96)
              if request.param == "pinhole"
              else FThetaCamera(64., 48., (0., 60.), 128, 96, (0., 1 / 60)))
    camera_from_ego = np.array([[0., -1., 0., 0.], [0., 0., -1., 0.],
                                [1., 0., 0., 0.], [0., 0., 0., 1.]])
    for step in log.steps:
        pixels = np.full((96, 128, 3), step.step * 20, np.uint8)
        step.model_observation = Observation(step.step * .1, step.step, {"front": pixels})
        step.sensor_geometry = {"front": SensorGeometry(
            "front", camera, camera_from_ego @ np.linalg.inv(step.physical_pre.ego.pose))}
    return log


def test_camera_view(camera_log):
    step = camera_log.steps[1]
    np.testing.assert_array_equal(camera_view(step, "front"), step.model_observation.sensor_data["front"])
    assert camera_view(step, "missing") is None


def test_sensor_view(camera_log):
    step = camera_log.steps[0]
    view = sensor_view(step, "front")
    assert view.sensor == "front"
    np.testing.assert_array_equal(view.cam_from_world, step.sensor_geometry["front"].cam_from_world)
    step.sensor_geometry = {}
    assert sensor_view(step) is None


def test_labels_view(camera_log):
    step = camera_log.steps[0]
    assert labels_view(step).sum() > step.model_observation.sensor_data["front"].sum()
    step.sensor_geometry = {}
    np.testing.assert_array_equal(labels_view(step), step.model_observation.sensor_data["front"])


def test_draw_boxes3d(camera_log):
    view = sensor_view(camera_log.steps[0], "front")
    drawn = draw_boxes3d(view.data, view)
    assert drawn.shape == view.data.shape and np.any(drawn != view.data)
    with pytest.raises(ValueError, match="requires camera geometry"):
        draw_boxes3d(view.data, replace(view, camera=None, cam_from_world=None))


def test_detections_view(camera_log):
    step = camera_log.steps[0]
    assert detections_view(step).sum() > step.model_observation.sensor_data["front"].sum()
    np.testing.assert_array_equal(detections_view(step, threshold=1.), step.model_observation.sensor_data["front"])


def test_record_run(tmp_path, camera_log):
    paths = record_run(camera_log, tmp_path / "frames", visualize=labels_view)
    assert [path.name for path in paths] == ["frame_0000.png", "frame_0001.png"]
    assert all(path.read_bytes().startswith(b"\x89PNG") for path in paths)


def test_save_sequence(tmp_path, camera_log):
    strip_path, gif_path = save_sequence(camera_log, tmp_path / "sequence", cols=2)
    assert Path(strip_path).read_bytes().startswith(b"\x89PNG")
    assert Path(gif_path).read_bytes().startswith((b"GIF87a", b"GIF89a"))
    assert save_sequence(RunLog(), tmp_path / "empty") is None


def test_filmstrip(camera_log):
    assert filmstrip(camera_log, cols=2).shape == (108, 274, 3)
    with pytest.raises(ValueError, match="Filmstrip"):
        filmstrip([])


def test_plot_group(tmp_path, make_log):
    path = plot_group({"clean": make_log([3., 2.]),
                       "attacked": make_log([2., 0.], [0., 1.])}, tmp_path / "plot.png",
                      title="Small recorded experiment")
    assert path.read_bytes().startswith(b"\x89PNG")
    empty = plot_group({}, tmp_path / "empty.png")
    assert empty.is_file()
