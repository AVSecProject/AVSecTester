"""Scene visualization: the generic viz interface + the CARLA view adapter (no server/GPU needed)."""

import pytest
from avsectester.backend import AVStack
from avsectester.plane import Control, Observation
from avsectester.simulators import carla as carla_view
from avsectester.simulators.nurec import NuRecBackend, StubRenderer
from avsectester.simulators.viz import (
    annotate,
    camera_view,
    detections_view,
    labels_view,
    record_run,
    scene_labels,
)


class _Obj:  # duck-types ObjectGT for the pure label-mapping logic (no dataset needed)
    def __init__(self, track_id, category, distance, box):
        self.track_id, self.category, self._d, self.box2d = track_id, category, distance, box

    @property
    def distance(self):
        return self._d


class _Scene:  # duck-types SceneGT
    def __init__(self, objects, cameras=("front",)):
        self.objects = objects
        self.cameras = {c: object() for c in cameras}


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


def test_scene_labels_maps_gt_to_annotate_tuples_and_highlights_target():
    target = _Obj("t2", "vehicle", 12.0, {"front": (5, 5, 9, 9)})
    scene = _Scene([
        _Obj("t1", "vehicle", 30.0, {"front": (1, 1, 4, 4)}),
        target,
        _Obj("t3", "vehicle", 8.0, {}),            # no box in this camera -> skipped
    ])
    tuples = scene_labels(scene, camera="front", target=target)
    assert len(tuples) == 2                          # the boxless object is dropped
    # score is 1.0 (GT), text carries category + distance, colour is explicit (4-tuple)
    assert all(len(t) == 4 and t[1] == 1.0 for t in tuples)
    assert tuples[0][2] == "vehicle 30m"
    # the target is emitted last (drawn on top) and in the highlight colour
    assert tuples[-1][0] == (5, 5, 9, 9) and tuples[-1][3] == (235, 64, 52)
    assert scene_labels(scene) == scene_labels(scene, camera="front")  # default = first camera


def test_annotate_honours_explicit_colour_and_hides_gt_score():
    import numpy as np

    img = np.zeros((16, 16, 3), np.uint8)
    out = annotate(img, [((2, 2, 10, 10), 1.0, "vehicle 12m", (200, 0, 0))])
    assert out.shape == (16, 16, 3) and out[:, :, 0].sum() > 0 and out[:, :, 1].sum() == 0  # red, not green


def test_labels_view_overlays_scene_gt_through_the_camera_path():
    import numpy as np

    img = np.zeros((20, 20, 3), np.uint8)
    obs = Observation(t=0.0, frame=0, sensor_data={"front": img})
    scene = _Scene([_Obj("t1", "vehicle", 10.0, {"front": (2, 2, 15, 15)})])
    out = labels_view(lambda _o: scene, camera="front")(obs)
    assert out.shape == (20, 20, 3) and out[:, :, 1].sum() > 0            # GT box drawn (green)
    assert labels_view(lambda _o: None)(obs) is not None                  # no scene -> plain frame, not None


def test_record_run_saves_a_frame_per_step(tmp_path):
    pytest.importorskip("matplotlib")
    backend = NuRecBackend({"dt": 0.1}, renderer=StubRenderer(cameras=["camera_front"], height=16, width=24))
    trace = record_run(backend, Cruise(), frames=4, out_dir=tmp_path, visualize=camera_view)
    assert len(trace.records) == 4
    saved = sorted(tmp_path.glob("frame_*.png"))
    assert [p.name for p in saved] == ["frame_0000.png", "frame_0001.png", "frame_0002.png", "frame_0003.png"]
