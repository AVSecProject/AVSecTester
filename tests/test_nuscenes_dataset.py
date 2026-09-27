"""nuScenes adapter: box->SceneGT conversion + the replay backend, tested without the dataset/devkit.

The real nuScenes GT is validated by feeding duck-typed camera-frame boxes (as the devkit's
``get_sample_data`` returns) into the pure ``scene_from_cam_boxes``; the requirement predicate then runs
exactly as it would on the real data.
"""

import numpy as np
from avsectester.scenarios.datasets.nuscenes import RecordedFrameBackend, scene_from_cam_boxes
from avsectester.scenarios.requirements import PHYSICAL_PATCH_HIDE_VEHICLE as REQ

# a nuScenes CAM_FRONT-like intrinsic (1600x900, f~1266)
K = np.array([[1266.0, 0, 800.0], [0, 1266.0, 450.0], [0, 0, 1.0]])
W, H = 1600, 900


class _FakeQuat:
    def __init__(self, fwd):
        self._fwd = np.asarray(fwd, float)

    def rotate(self, v):  # only the +x (vehicle forward) rotation is used
        return self._fwd


class _FakeBox:
    """Duck-types the nuScenes ``Box`` in the camera frame (x right, y down, z fwd)."""

    def __init__(self, name, token, center, fwd, size=(2.0, 4.5, 1.5)):
        self.name, self.token, self.center = name, token, np.asarray(center, float)
        self.orientation = _FakeQuat(fwd)
        self._w, self._l, self._h = size

    def corners(self):
        cx, cy, cz = self.center
        w, l, h = self._w / 2, self._l / 2, self._h / 2
        pts = [(cx + sx, cy + sy, cz + sz) for sx in (-w, w) for sy in (-h, h) for sz in (-l, l)]
        return np.array(pts).T  # (3, 8)


def _lead(z, fwd=(0, 0, 1), name="vehicle.car"):  # a vehicle z m ahead, centred
    return _FakeBox(name, f"t{z}", center=(0.0, 0.5, float(z)), fwd=fwd)


def test_scene_from_cam_boxes_builds_and_matches():
    scene = scene_from_cam_boxes([_lead(10)], K, W, H, visibility={"t10": 0.9})
    assert len(scene.objects) == 1
    o = scene.objects[0]
    assert abs(o.distance - 10.0) < 0.2 and o.ahead and abs(o.yaw) < 0.1  # 10 m ahead, rear-facing
    assert "front" in o.box2d and REQ.match(scene) is not None            # a close rear-facing lead qualifies


def test_scene_from_cam_boxes_filters_by_requirement():
    assert REQ.match(scene_from_cam_boxes([_lead(60)], K, W, H)) is None          # too far / tiny box
    assert REQ.match(scene_from_cam_boxes([_lead(10, fwd=(0, 0, -1))], K, W, H)) is None  # facing us
    # a bicycle is not a "vehicle" target -> dropped
    assert scene_from_cam_boxes([_lead(10, name="vehicle.bicycle")], K, W, H).objects == []


def test_recorded_frame_backend_serves_the_image(tmp_path):
    import cv2

    p = tmp_path / "frame.png"
    cv2.imwrite(str(p), np.full((90, 160, 3), 128, np.uint8))
    backend = RecordedFrameBackend(str(p), sensor_id="front")
    obs = backend.reset()
    rgb = obs.sensor_data["front"]
    assert rgb.shape == (90, 160, 3)
    assert backend.step(None).sensor_data["front"].shape == (90, 160, 3)  # static: same frame


def test_detector_labeler_builds_scenegt_and_filters(tmp_path):
    import cv2
    from avsectester.scenarios.datasets.detector import DetectorLabeler
    from avsectester.scenarios.datasets.frames import ImageFolder
    from avsectester.scenarios.source import DatasetFilter

    p = tmp_path / "f.jpg"
    cv2.imwrite(str(p), np.full((900, 1600, 3), 120, np.uint8))
    # a detector returning one large, centred vehicle box (height 400 px -> ~4.9 m at f=1266, H=1.55)
    def detect(rgb):
        return [((700.0, 300.0, 1000.0, 700.0), 0.9, "vehicle")]

    # the labeling strategy composes over a bare frame source (pixels only), not a sibling Dataset
    ds = DetectorLabeler(ImageFolder([str(p)], K), detect)
    scenes = list(ds.scenes())
    assert len(scenes) == 1 and len(scenes[0].objects) == 1
    o = scenes[0].objects[0]
    assert 4.0 < o.distance < 6.0 and "front" in o.box2d       # monocular distance estimate
    assert len(list(DatasetFilter(ds).scenarios(REQ))) == 1    # this real-image frame qualifies
    # the .over_images convenience is equivalent
    scene = next(DetectorLabeler.over_images([str(p)], detect, K).scenes())
    assert scene.objects[0].category == "vehicle"
