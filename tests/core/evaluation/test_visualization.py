"""Recorded visualization features use small images and physical camera geometry."""

from pathlib import Path
from dataclasses import replace

from PIL import Image

import numpy as np
import pytest

from avsectester.data_types import Detection, Observation
from avsectester.closed_loop.simulators.base import SensorGeometry
from avsectester.evaluation.visualization import (
    annotate, as_rgb, box3d_corners, camera_view, detections_view, draw_boxes3d, filmstrip,
    labels_view, lidar_bev, plot_group, record_run, save_gif, save_image, save_sequence, sensor_view,
)
from avsectester.runlog import RunLog


def image_log_with_geometry(make_log, make_actor, camera=None):
    log = make_log([1., 1.], actors={"lead": make_actor(10.)},
                   detections=[Detection((10., 0., 0.), (4., 2., 2.), 0., "car", .9)])
    height, width = (96, 128) if camera is None else (camera.height, camera.width)
    camera_from_ego = np.array([[0., -1., 0., 0.], [0., 0., -1., 0.],
                                [1., 0., 0., 0.], [0., 0., 0., 1.]])
    for step in log.steps:
        pixels = np.full((height, width, 3), step.step * 20, np.uint8)
        step.model_observation = Observation(step.step * .1, step.step, {"front": pixels})
        if camera is not None:
            step.sensor_geometry = {"front": SensorGeometry(
                "front", camera, camera_from_ego @ np.linalg.inv(step.physical_pre.ego.pose))}
    return log


@pytest.fixture
def camera_log(camera, make_log, make_actor):
    return image_log_with_geometry(make_log, make_actor, camera)


@pytest.fixture
def image_log(make_log, make_actor):
    return image_log_with_geometry(make_log, make_actor)


def test_camera_view(image_log):
    step = image_log.steps[1]
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


def test_record_run(tmp_path, image_log):
    paths = record_run(image_log, tmp_path / "frames", visualize=labels_view)
    assert [path.name for path in paths] == ["frame_0000.png", "frame_0001.png"]
    assert all(path.read_bytes().startswith(b"\x89PNG") for path in paths)


def test_save_sequence(tmp_path, image_log):
    strip_path, gif_path = save_sequence(image_log, tmp_path / "sequence", cols=2)
    assert Path(strip_path).read_bytes().startswith(b"\x89PNG")
    assert Path(gif_path).read_bytes().startswith((b"GIF87a", b"GIF89a"))
    assert save_sequence(RunLog(), tmp_path / "empty") is None


def test_filmstrip(image_log):
    assert filmstrip(image_log, cols=2).shape == (108, 274, 3)
    with pytest.raises(ValueError, match="Filmstrip"):
        filmstrip([])


def test_plot_group(tmp_path, make_log):
    from avsectester.data_types import Control

    path = plot_group({"clean": make_log([3., 2.]),
                       "attacked": make_log([2., 0.], [0., 1.])}, tmp_path / "plot.png",
                      title="Small recorded experiment")
    assert path.read_bytes().startswith(bytes.fromhex("89504e47"))
    trajectory = make_log([3., 2.])
    for step in trajectory.steps:
        step.executed_command = Control(trajectory=[])
    single = plot_group({"trajectory": trajectory}, tmp_path / "speed.png")
    with Image.open(path) as two_panels, Image.open(single) as one_panel:
        assert one_panel.width == two_panels.width
        assert one_panel.height < two_panels.height
    empty = plot_group({}, tmp_path / "empty.png")
    assert empty.is_file()


def test_as_rgb():
    rgb = np.array([[[20, 40, 60], [70, 80, 90]]], np.uint8)
    np.testing.assert_array_equal(as_rgb(rgb), rgb)
    native = {"data": rgb[..., ::-1], "calibration": {"channel_order": "bgr"}}
    np.testing.assert_array_equal(as_rgb(native), rgb)
    raw = np.concatenate((rgb[..., ::-1], np.full((1, 2, 1), 255, np.uint8)), axis=2)
    np.testing.assert_array_equal(as_rgb({
        "raw_bytes": raw.tobytes(), "metadata": {"width": 2, "height": 1},
    }), rgb)
    assert as_rgb(np.zeros((2, 3))) is None


def test_lidar_bev():
    cloud = np.array([[4., 2., 0.], [50., 50., 0.], [np.nan, 0., 0.]], np.float32)
    observation = Observation(0., 0, {"lidar": cloud})
    image = lidar_bev(observation, size=20, meters=10.)
    assert image.shape == (20, 20, 3) and image.dtype == np.uint8
    np.testing.assert_array_equal(image[6, 8], [0, 255, 0])
    np.testing.assert_array_equal(image[10, 10], [255, 80, 80])
    assert lidar_bev(Observation(0., 0)) is None
    with pytest.raises(ValueError, match="positive"):
        lidar_bev(observation, size=0)


def test_annotate():
    image = np.zeros((20, 24, 3), np.uint8)
    detection = ((2., 3., 12., 14.), .9, "vehicle", (255, 0, 0))
    drawn = annotate(image, [detection])
    np.testing.assert_array_equal(drawn[3, 2], [255, 0, 0])
    np.testing.assert_array_equal(image, 0)
    np.testing.assert_array_equal(annotate(image, [detection], threshold=1.), image)


def test_box3d_corners():
    corners = box3d_corners((3., 4., 1.), (2., 4., 2.), np.pi / 2)
    assert corners.shape == (8, 3) and len(np.unique(corners, axis=0)) == 8
    np.testing.assert_allclose(corners.min(axis=0), [1., 3., 0.])
    np.testing.assert_allclose(corners.max(axis=0), [5., 5., 2.])


def test_save_image(tmp_path):
    pixels = np.arange(36, dtype=np.uint8).reshape(3, 4, 3)
    path = tmp_path / "frames" / "image.png"
    save_image(pixels, path)
    with Image.open(path) as restored:
        assert restored.mode == "RGB"
        np.testing.assert_array_equal(np.asarray(restored), pixels)


def test_save_gif(tmp_path):
    frames = [np.zeros((3, 4, 3), np.uint8), np.full((3, 4, 3), 120, np.uint8)]
    path = tmp_path / "animations" / "drive.gif"
    save_gif(frames, path, fps=5)
    with Image.open(path) as image:
        assert image.n_frames == 2 and image.info["duration"] == 200
        image.seek(1)
        np.testing.assert_array_equal(np.asarray(image.convert("RGB")), frames[1])
    with pytest.raises(ValueError, match="frames and positive fps"):
        save_gif([], path)
