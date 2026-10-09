"""Selected-frame geometry and timing survive the transition into a backend."""

import math
from types import SimpleNamespace

import numpy as np
import pytest

from avsectester.scenarios import EgoState, SceneGT
from avsectester.scenarios.datasets.nurec import NuRecDataset
from avsectester.simulators.nurec import EgoPose, NuRecRenderer


def test_nurec_factory_preserves_selected_pose_time_camera_and_reset(monkeypatch):
    import avsectester.simulators.nurec as module
    from avsectester.rendering.cameras import planar_rig_pose

    pose = planar_rig_pose(12, -3, 0.7, 1.8)
    scene = SceneGT(
        42,
        123.4,
        EgoState(5, pose.tolist()),
        {},
        [],
        {"scene_id": "selected", "timestamp_us": 123400000},
    )
    requests = []

    class CaptureRenderer:
        def __init__(self, **kwargs):
            self.cameras = kwargs["cameras"]
            self.calibration = {self.cameras[0]: "calibration"}
            self.kwargs = kwargs

        def load_scene(self, scene_id):
            assert scene_id == "selected"

        def render(self, pose, camera):
            requests.append((pose.x, pose.y, pose.yaw, pose.speed, pose.t, camera))
            return np.zeros((2, 3, 3), np.uint8)

    monkeypatch.setattr(module, "NuRecRenderer", CaptureRenderer)
    backend = NuRecDataset([], sensor="native-camera", camera="front").make_backend(scene)
    initial = backend.reset()
    backend.reset()
    assert requests[0] == requests[1] == (12, -3, 0.7, 5, 0.0, "native-camera")
    assert backend.renderer.kwargs["start_timestamp_us"] == 123400000
    np.testing.assert_allclose(backend.renderer.kwargs["start_transform"], pose)
    assert set(initial.sensor_data) == {"front"} and initial.calibration == {"front": "calibration"}


def test_nurec_renderer_keeps_initial_tilt_and_height():
    from avsectester.rendering.cameras import planar_rig_pose

    anchor = planar_rig_pose(10, 20, 0.4, 2)
    tilt = np.array(
        [[1, 0, 0], [0, math.cos(0.1), -math.sin(0.1)], [0, math.sin(0.1), math.cos(0.1)]]
    )
    anchor[:3, :3] = anchor[:3, :3] @ tilt
    renderer = NuRecRenderer(start_transform=anchor)
    np.testing.assert_allclose(renderer.rig_transform(EgoPose(10, 20, 0.4)), anchor, atol=1e-12)
    advanced = renderer.rig_transform(EgoPose(11, 21, 0.5))
    expected = planar_rig_pose(11, 21, 0.5, 2)
    expected[:3, :3] = expected[:3, :3] @ tilt
    np.testing.assert_allclose(advanced, expected, atol=1e-12)


def test_nuscenes_geometry_uses_recorded_camera_extrinsic():
    from avsectester.scenarios.datasets.nuscenes import scene_from_cam_boxes
    from avsectester.scenarios.filters import ViewpointRear

    class Orientation:
        def rotate(self, _):
            return np.array([0, 0, 1])

    box = SimpleNamespace(
        name="vehicle.car",
        token="car",
        center=np.array([0, 0, 10]),
        wlh=[2, 4, 2],
        orientation=Orientation(),
        corners=lambda: np.array([[-1, 1, 1, -1], [-1, -1, 1, 1], [10, 10, 10, 10]]),
    )
    calibration = np.array([[0, 0, 1, 2], [-1, 0, 0, 3], [0, -1, 0, 1], [0, 0, 0, 1]])
    scene = scene_from_cam_boxes(
        [box], np.array([[100, 0, 100], [0, 100, 100], [0, 0, 1]]), 200, 200, cam_to_ego=calibration
    )
    target = scene.objects[0]
    assert target.center == (12, 3, 1) and target.visibility is None
    assert ViewpointRear(1).holds(scene, target)


def test_carla_live_gt_selects_rgb_camera_and_converts_unreal_yaw():
    from avsectester.scenarios.carla_gt import carla_scene_gt
    from avsectester.scenarios.filters import ViewpointRear

    class Transform:
        rotation = SimpleNamespace(yaw=0)

        def get_inverse_matrix(self):
            return np.eye(4)

    class Actors(list):
        def filter(self, _):
            return self

    target_transform = Transform()
    target_transform.rotation = SimpleNamespace(yaw=30)
    angle = math.radians(30)
    target_pose = np.array(
        [
            [math.cos(angle), -math.sin(angle), 0, 10],
            [math.sin(angle), math.cos(angle), 0, 10],
            [0, 0, 1, 1],
            [0, 0, 0, 1],
        ]
    )
    target_transform.get_inverse_matrix = lambda: np.linalg.inv(target_pose)
    corners = [
        SimpleNamespace(x=10 + x, y=10 + y, z=1 + z)
        for x in (-2, 2)
        for y in (-1, 1)
        for z in (-1, 1)
    ]
    target = SimpleNamespace(
        id=2,
        get_transform=lambda: target_transform,
        bounding_box=SimpleNamespace(
            extent=SimpleNamespace(x=2, y=1, z=1),
            get_world_vertices=lambda _: corners,
            location=SimpleNamespace(x=0, y=0, z=0),
        ),
    )
    ego = SimpleNamespace(
        id=1,
        get_transform=Transform,
        bounding_box=SimpleNamespace(
            extent=SimpleNamespace(x=2, y=1, z=1),
            location=SimpleNamespace(x=0, y=0, z=0),
        ),
    )
    rgb = SimpleNamespace(
        object=SimpleNamespace(type_id="sensor.camera.rgb", get_transform=Transform),
        P=np.array([[100, 0, 100, 0], [0, 100, 100, 0], [0, 0, 1, 0]]),
        imsize=(200, 200),
    )
    depth = SimpleNamespace(object=SimpleNamespace(type_id="sensor.camera.depth"))
    world = SimpleNamespace(
        get_actors=lambda: Actors([ego, target]),
        get_snapshot=lambda: SimpleNamespace(
            frame=44, timestamp=SimpleNamespace(elapsed_seconds=0.2)
        ),
    )
    backend = SimpleNamespace(
        client=SimpleNamespace(world=world),
        ego=SimpleNamespace(
            actor=ego,
            sensors={"depth": depth, "front": rgb},
            get_object_state=lambda: SimpleNamespace(velocity=SimpleNamespace(norm=lambda: 5)),
        ),
    )
    scene = carla_scene_gt(backend)
    assert scene.frame == 44 and scene.t == 0.2
    assert scene.objects[0].center == (10, -10, 1)
    assert scene.objects[0].yaw == pytest.approx(-math.pi / 6)
    assert scene.objects[0].visibility is None
    assert ViewpointRear(20).holds(scene, scene.objects[0])


def test_nurec_rpc_uses_selected_timestamp_and_initial_pose(monkeypatch):
    """Check our RPC request construction, without claiming compatibility with a live service."""
    import sys
    import cv2
    from scipy.spatial.transform import Rotation
    from avsectester.rendering.cameras import planar_rig_pose, pose_from_proto

    encoded = cv2.imencode(".jpg", np.zeros((2, 3, 3), np.uint8))[1].tobytes()
    common = SimpleNamespace(
        Empty=SimpleNamespace, Pose=SimpleNamespace, Vec3=SimpleNamespace, Quat=SimpleNamespace
    )
    identity = SimpleNamespace(
        vec=SimpleNamespace(x=0, y=0, z=0), quat=SimpleNamespace(w=1, x=0, y=0, z=0)
    )
    spec = SimpleNamespace(resolution_h=2, resolution_w=3)
    side_spec = SimpleNamespace(resolution_h=4, resolution_w=5)
    side_pose = SimpleNamespace(vec=SimpleNamespace(x=0, y=2, z=0), quat=identity.quat)
    requests = []

    class Service:
        def get_available_scenes(self, _):
            return SimpleNamespace(scene_ids=["chosen-scene"])

        def get_available_cameras(self, _):
            return SimpleNamespace(
                available_cameras=[
                    SimpleNamespace(logical_id="front", intrinsics=spec, rig_to_camera=identity),
                    SimpleNamespace(
                        logical_id="side", intrinsics=side_spec, rig_to_camera=side_pose
                    ),
                ]
            )

        def get_available_trajectories(self, _):
            return SimpleNamespace(
                available_trajectories=[
                    SimpleNamespace(
                        trajectory=SimpleNamespace(
                            poses=[SimpleNamespace(pose=identity, timestamp_us=1)]
                        )
                    )
                ]
            )

        def render_rgb(self, request):
            requests.append(request)
            return SimpleNamespace(image_bytes=encoded)

    pb = SimpleNamespace(
        AvailableCamerasRequest=SimpleNamespace,
        AvailableTrajectoriesRequest=SimpleNamespace,
        RGBRenderRequest=SimpleNamespace,
        PosePair=SimpleNamespace,
        ImageFormat=SimpleNamespace(JPEG="jpeg"),
    )
    grpc_pb = SimpleNamespace(SensorsimServiceStub=lambda _: Service())
    monkeypatch.setitem(sys.modules, "grpc", SimpleNamespace(insecure_channel=lambda _: None))
    monkeypatch.setitem(sys.modules, "alpasim_grpc", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "alpasim_grpc.v0",
        SimpleNamespace(common_pb2=common, sensorsim_pb2=pb, sensorsim_pb2_grpc=grpc_pb),
    )
    anchor = planar_rig_pose(7, 8, 0.2, 1.5)
    anchor[:3, :3] = Rotation.from_euler("ZYX", [0.2, 0.05, -0.1]).as_matrix()
    renderer = NuRecRenderer(
        cameras=["front", "side"], start_timestamp_us=1000000, start_transform=anchor
    )
    renderer.load_scene("chosen-scene")
    output = renderer.render(EgoPose(7, 8, 0.2, t=0.25), "front")
    assert output.shape == (2, 3, 3)
    assert requests[0].frame_start_us == 1250000
    np.testing.assert_allclose(
        pose_from_proto(requests[0].sensor_pose.start_pose), anchor, atol=1e-12
    )
    renderer.render_at(anchor, 1400000, "side")
    assert requests[1].camera_intrinsics is side_spec
    assert requests[1].frame_start_us == 1400000
    np.testing.assert_allclose(
        pose_from_proto(requests[1].sensor_pose.start_pose),
        anchor @ pose_from_proto(side_pose),
        atol=1e-12,
    )
    with pytest.raises(ValueError, match="exactly one"):
        renderer.load_scene("missing-scene")
