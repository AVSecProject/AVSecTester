"""Scene visualization: the generic viz interface + the CARLA view adapter (no server/GPU needed)."""

import pytest
from avsectester.backend import AVStack
from avsectester.plane import Control, Observation
from avsectester.simulators import carla as carla_view
from avsectester.simulators.nurec import NuRecBackend, StubRenderer
from avsectester.simulators.viz import (
    annotate,
    box3d_corners,
    camera_view,
    detections_view,
    draw_boxes3d,
    ego_projector,
    labels_view,
    record_run,
    scene_labels,
)


class _Cam:  # duck-types CameraCalib (only .model is read by the projector)
    def __init__(self, model=None):
        self.model = model


class _Obj:  # duck-types ObjectGT for the pure label/box logic (no dataset needed)
    def __init__(self, track_id, category, distance, box,
                 center=(10.0, 0.0, 0.0), extent=(4.0, 2.0, 1.5), yaw=0.0):
        self.track_id, self.category, self._d, self.box2d = track_id, category, distance, box
        self.center, self.extent, self.yaw = center, extent, yaw

    @property
    def distance(self):
        return self._d


class _Scene:  # duck-types SceneGT
    def __init__(self, objects, cameras=("front",), model=None):
        self.objects = objects
        self.cameras = {c: _Cam(model) for c in cameras}


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


def test_box3d_corners_and_pinhole_projector_are_geometric():
    import numpy as np

    corners = box3d_corners((10.0, 0.0, 0.0), (4.0, 2.0, 1.5), 0.0)  # length x, width y, height z
    assert corners.shape == (8, 3)
    assert np.isclose(np.ptp(corners[:, 0]), 4.0)   # extent spans match (length, width, height)
    assert np.isclose(np.ptp(corners[:, 1]), 2.0) and np.isclose(np.ptp(corners[:, 2]), 1.5)

    k = np.array([[1000.0, 0, 960.0], [0, 1000.0, 540.0], [0, 0, 1.0]])
    project = ego_projector(k)
    px, valid = project(np.array([[10.0, 0.0, 0.0]]))     # dead ahead -> principal point, in front
    assert valid[0] and np.allclose(px[0], [960.0, 540.0], atol=1e-6)
    _, behind = project(np.array([[-5.0, 0.0, 0.0]]))     # behind the camera -> invalid
    assert not behind[0]
    assert ego_projector(None) is None                    # no model -> no projector


def test_draw_boxes3d_projects_wireframe_and_falls_back_to_2d():
    import numpy as np

    k = np.array([[1000.0, 0, 960.0], [0, 1000.0, 540.0], [0, 0, 1.0]])
    obj = _Obj("t1", "vehicle", 10.0, {"front": (900, 500, 1020, 600)},
               center=(10.0, 0.0, 0.0), extent=(4.0, 2.0, 1.5))
    img = np.zeros((1080, 1920, 3), np.uint8)
    out = draw_boxes3d(img, _Scene([obj], model=k), camera="front")
    assert out.shape == img.shape and out[:, :, 1].sum() > 0        # 3-D wireframe drawn (green)
    # no projector (model=None) -> falls back to 2-D boxes via annotate(scene_labels)
    out2d = draw_boxes3d(img, _Scene([obj], model=None), camera="front")
    assert out2d.shape == img.shape and out2d[:, :, 1].sum() > 0


def test_draw_boxes3d_skips_out_of_view_and_far_boxes():
    import numpy as np

    k = np.array([[1000.0, 0, 960.0], [0, 1000.0, 540.0], [0, 0, 1.0]])
    img = np.zeros((1080, 1920, 3), np.uint8)
    # a box straddling the camera plane (center 1 m ahead, 4 m long -> rear corners behind) has corners
    # that do not all project -> the box is skipped rather than drawn as a frame-spanning mess
    behind = _Obj("t1", "vehicle", 1.0, {}, center=(1.0, 0.0, 0.0), extent=(4.0, 2.0, 1.5))
    assert draw_boxes3d(img, _Scene([behind], model=k), camera="front")[:, :, 1].sum() == 0
    # max_distance drops a far actor (60 m) but never the highlighted target
    far = _Obj("t2", "vehicle", 60.0, {}, center=(60.0, 0.0, 0.0), extent=(4.0, 2.0, 1.5))
    scene = _Scene([far], model=k)
    assert draw_boxes3d(img, scene, camera="front", max_distance=40.0)[:, :, 1].sum() == 0
    assert draw_boxes3d(img, scene, camera="front", target=far, max_distance=40.0).sum() > 0


def test_record_run_saves_a_frame_per_step(tmp_path):
    pytest.importorskip("matplotlib")
    backend = NuRecBackend({"dt": 0.1}, renderer=StubRenderer(cameras=["camera_front"], height=16, width=24))
    trace = record_run(backend, Cruise(), frames=4, out_dir=tmp_path, visualize=camera_view)
    assert len(trace.records) == 4
    saved = sorted(tmp_path.glob("frame_*.png"))
    assert [p.name for p in saved] == ["frame_0000.png", "frame_0001.png", "frame_0002.png", "frame_0003.png"]
