"""The RPC boundary receives the selected source time and complete rig/camera pose."""
from dataclasses import replace
from types import ModuleType, SimpleNamespace
import sys

import numpy as np

from avsectester.closed_loop.simulators.nurec import EgoPose, NuRecRenderer
from avsectester.datasets.base import ActorState, FrameReference, ScenarioStart
from avsectester.geometry.cameras import pose_from_proto
from avsectester.geometry.poses import rotation_matrix


def message(**values):
    return SimpleNamespace(**values)


def test_rpc_uses_selected_timestamp_height_tilt_and_sensor_mount(monkeypatch):
    recorded_origin = np.eye(4)
    recorded_origin[:3, :3] = rotation_matrix((8., -12., 35.))
    recorded_origin[:3, 3] = (12., 3., 2.)
    source = ScenarioStart('NuRecDataset', 'clip', FrameReference('clip', 5, 42, 100.5), {
        'ego': ActorState(recorded_origin, extent=(4.5, 2., 1.6)),
        'first_timestamp': 100., 'terminal_timestamp': 101.,
    })
    mount = message(vec=message(x=1., y=.2, z=1.4), quat=message(w=1., x=0., y=0., z=0.))
    spec = message(resolution_h=2, resolution_w=3)
    requests = []
    stub = message(
        get_available_scenes=lambda _: message(scene_ids=['clip']),
        get_available_cameras=lambda _: message(available_cameras=[message(
            logical_id='front', intrinsics=spec, rig_to_camera=mount)]),
        get_available_trajectories=lambda _: message(available_trajectories=[message(
            trajectory=message(poses=[message(timestamp_us=100_000_000,
                pose=message(vec=message(x=0., y=0., z=.1),
                             quat=message(w=1., x=0., y=0., z=0.)))]))]),
        render_rgb=lambda request: (requests.append(request) or message(image_bytes=b'image')),
    )
    common = message(Pose=message, Quat=message, Vec3=message, Empty=lambda: None)
    sensor = message(AvailableCamerasRequest=message, AvailableTrajectoriesRequest=message,
                     RGBRenderRequest=message, PosePair=message, ImageFormat=message(JPEG=1))
    modules = {
        'grpc': message(insecure_channel=lambda _: message(close=lambda: None)),
        'alpasim_grpc': {},
        'alpasim_grpc.v0': {'common_pb2': common, 'sensorsim_pb2': sensor,
                           'sensorsim_pb2_grpc': message(SensorsimServiceStub=lambda _: stub)},
        'cv2': {'IMREAD_COLOR': 1, 'COLOR_BGR2RGB': 2,
                'imdecode': lambda *_: np.zeros((2, 3, 3), np.uint8),
                'cvtColor': lambda value, _: value},
    }
    for name, values in modules.items():
        module = ModuleType(name)
        vars(module).update(vars(values) if isinstance(values, SimpleNamespace) else values)
        monkeypatch.setitem(sys.modules, name, module)
    renderer = NuRecRenderer(cameras=['front'])
    renderer.reset_origin(source)
    renderer.load_scene('clip')
    initial = EgoPose(x=12., y=3., yaw=np.deg2rad(35.))
    image = renderer.render(initial, 'front')
    assert image.shape == (2, 3, 3)
    first = requests[0]
    assert (first.frame_start_us, first.frame_end_us) == (100_500_000, 100_500_001)
    np.testing.assert_allclose(pose_from_proto(first.sensor_pose.start_pose),
                               recorded_origin @ pose_from_proto(mount), atol=1e-10)
    advanced = replace(initial, x=14., y=4., yaw=initial.yaw + .2, t=.1)
    renderer.render(advanced, 'front')
    assert requests[-1].frame_start_us == 100_600_000
    actual_rig = pose_from_proto(requests[-1].sensor_pose.start_pose) @ np.linalg.inv(pose_from_proto(mount))
    np.testing.assert_allclose(actual_rig[:3, 3], [14., 4., 2.], atol=1e-10)
    # Only the planar heading evolves; the source's body-frame roll/pitch are retained.
    np.testing.assert_allclose(actual_rig[:3, :3],
        rotation_matrix((0, 0, np.rad2deg(.2))) @ recorded_origin[:3, :3], atol=1e-10)
