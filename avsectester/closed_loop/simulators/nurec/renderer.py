"""NuRec RGB rendering and native camera/pose conversion at the RPC boundary."""

from __future__ import annotations

import copy
import dataclasses
import math
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, ClassVar, TYPE_CHECKING

from avsectester.closed_loop.simulators.ego_dynamics import EgoPose
from avsectester.geometry.poses import validated_transform

if TYPE_CHECKING:
    from avsectester.geometry.cameras import Camera, FThetaCamera
    import numpy as np


def ftheta_camera_from_spec(spec: Any) -> FThetaCamera:
    """Build an f-theta lens from a NuRec CameraSpec."""
    from avsectester.geometry.cameras import FThetaCamera

    parameters = spec.ftheta_param
    correction = parameters.linear_cde
    return FThetaCamera(
        cx=parameters.principal_point_x,
        cy=parameters.principal_point_y,
        angle_to_pixeldist=tuple(parameters.angle_to_pixeldist_poly),
        pixeldist_to_angle=tuple(parameters.pixeldist_to_angle_poly),
        width=spec.resolution_w,
        height=spec.resolution_h,
        max_angle=parameters.max_angle or math.pi / 2,
        linear_cde=(correction.linear_c or 1.0, correction.linear_d, correction.linear_e),
    )


def pose_from_proto(pose: Any) -> np.ndarray:
    """A NuRec/AlpaSim common.Pose (vec + quat) as a rigid transform."""
    from avsectester.geometry.poses import make_pose, quaternion_to_rotation

    quaternion, vector = pose.quat, pose.vec
    return make_pose(quaternion_to_rotation(
        quaternion.w, quaternion.x, quaternion.y, quaternion.z),
        (vector.x, vector.y, vector.z))


def planar_rig_pose(x: float, y: float, yaw: float, z: float = 0.0) -> np.ndarray:
    """World-from-rig transform for planar position and heading, retaining rig height."""
    import numpy as np
    from avsectester.geometry.poses import make_pose

    cosine, sine = math.cos(yaw), math.sin(yaw)
    return make_pose(np.array([[cosine, -sine, 0.0], [sine, cosine, 0.0],
                               [0.0, 0.0, 1.0]]), (x, y, z))


class Renderer(ABC):
    """Renders sensor observations from the reconstructed scene at a given ego pose."""

    cameras: ClassVar[list[str]] = ["camera_front"]
    calibration: ClassVar[dict] = {}

    def copy(self) -> Renderer:
        """A renderer with this configuration and no connection, loaded scene or run origin.

        Every simulator runtime renders through its own copy. Override this when a renderer
        holds resources that a shallow copy would share.
        """
        return copy.copy(self)

    def reset_origin(self, start=None):
        """Set the run origin from the dataset start; configured renderers have no origin."""
        self._origin_transform = None if start is None else deepcopy(start.payload["ego"].pose)
        self._origin_timestamp_us = None if start is None else round(start.frame.timestamp * 1e6)
        self._origin_bounds = None if start is None else (
            start.payload["first_timestamp"], start.payload["terminal_timestamp"])

    @property
    def origin_transform(self):
        return deepcopy(getattr(self, "_origin_transform", None))

    @property
    def origin_timestamp_us(self):
        return getattr(self, "_origin_timestamp_us", None)

    @property
    def origin_bounds(self):
        return getattr(self, "_origin_bounds", None)

    def load_scene(self, scene: Any) -> None:
        """Load a reconstruction once; renderers without a service may leave this empty."""

    @abstractmethod
    def render(self, pose: EgoPose, camera: str, dynamic_objects=()) -> Any:
        """Render one camera and the supplied track-ID/world-pose pairs at ``pose``."""

    def camera_model(self, camera: str | None = None) -> Camera | None:
        """Optical camera model, or None when this renderer supplies no geometry."""
        return None

    def cam_from_world(self, pose: EgoPose, camera: str | None = None) -> np.ndarray:
        """World-to-optical-camera transform for the rendered pose."""
        raise NotImplementedError("This renderer does not provide camera geometry")

    def close(self) -> None:
        pass


@dataclass
class NuRecRenderer(Renderer):
    """Real NuRec renderer: one stateless ``SensorsimService.render_rgb`` per frame to an nre server.

    The heavy neural rendering lives in the ``nre-ga`` server (like the CARLA server), so this stays a
    thin, stateless client — no runtime, no controller/physics/traffic microservices, no callback
    inversion. Lazily imports the gRPC stubs. Connect it to a running renderer and a loaded NuRec
    scene. ``service`` exposes the native RPC client after loading. ``render_at`` supports exact
    recorded poses for initial-sequence inspection without advancing a driving policy.
    """

    endpoint: str = "127.0.0.1:50051"
    scene_id: str | None = None
    cameras: list[str] = field(default_factory=lambda: ["camera_front_wide_120fov"])
    calibration: dict = field(default_factory=dict)
    image_quality: float = 95.0

    def copy(self) -> NuRecRenderer:
        """The same configuration without a connection, loaded scene or run origin."""
        return dataclasses.replace(self)

    def __post_init__(self) -> None:
        self._stub = None
        self._channel = None
        self._camera_specs = {}
        self._camera_poses = {}
        self._spec = None  # CameraSpec for our camera (intrinsics), queried from the scene
        self._rig_to_camera = None  # camera pose in the rig (extrinsic), composed onto the ego pose
        self._start_pose = None  # scene-frame ego start pose (from the recorded trajectory)
        self._t0 = (
            0  # Scene start timestamp in microseconds, corresponding to simulation time zero.
        )

    def load_scene(self, scene: Any) -> None:
        """Connect to the nre-ga renderer and read the scene id, camera spec, and start pose/time."""
        import grpc
        from alpasim_grpc.v0 import common_pb2, sensorsim_pb2, sensorsim_pb2_grpc

        self.close()
        self._channel = grpc.insecure_channel(self.endpoint)
        self._stub = sensorsim_pb2_grpc.SensorsimServiceStub(self._channel)
        available = list(self._stub.get_available_scenes(common_pb2.Empty()).scene_ids)
        want = scene or self.scene_id
        matches = [s for s in available if want and want in s]
        if want in available:
            self._loaded_scene_id = want
        elif want and len(matches) == 1:
            self._loaded_scene_id = matches[0]
        elif want or not available:
            raise ValueError(f"Scene {want!r} did not resolve to exactly one renderer scene")
        else:
            self._loaded_scene_id = available[0]
        cams = self._stub.get_available_cameras(
            sensorsim_pb2.AvailableCamerasRequest(scene_id=self._loaded_scene_id)
        )
        native_cameras = {c.logical_id: c for c in cams.available_cameras}
        for name in self.cameras:
            if name not in native_cameras:
                raise ValueError(f"Camera {name!r} is not available in {self._loaded_scene_id!r}")
        self._camera_specs = {name: native_cameras[name].intrinsics for name in self.cameras}
        self._camera_poses = {name: native_cameras[name].rig_to_camera for name in self.cameras}
        self._spec = self._camera_specs[self.cameras[0]]
        self._rig_to_camera = self._camera_poses[self.cameras[0]]
        poses = (
            self._stub.get_available_trajectories(
                sensorsim_pb2.AvailableTrajectoriesRequest(scene_id=self._loaded_scene_id)
            )
            .available_trajectories[0]
            .trajectory.poses
        )
        self._start_pose = poses[0].pose  # scene world-frame origin for the ego
        self._t0 = (
            poses[0].timestamp_us if self.origin_timestamp_us is None else self.origin_timestamp_us
        )

    @property
    def service(self):
        """Native SensorsimService client, including metadata APIs not normalized by this adapter."""
        if self._stub is None:
            raise RuntimeError("Call load_scene before accessing the NuRec service")
        return self._stub

    def close(self):
        if self._channel is not None:
            self._channel.close()
        self._channel = None
        self._stub = None

    def timestamp_us(self, pose: EgoPose) -> int:
        return self._t0 + round(pose.t * 1e6)

    def camera_model(self, camera: str | None = None) -> Camera:
        """The rendered camera's lens model (:class:`~avsectester.geometry.cameras.FThetaCamera`)
        for world-anchored insertion, available after :meth:`load_scene`."""
        return ftheta_camera_from_spec(self._camera_specs.get(camera, self._spec))

    def rig_transform(self, pose: EgoPose):
        """Preserve the selected frame's height/tilt while evolving planar position and heading."""
        import numpy as np

        if self.origin_transform is None:
            z = self._start_pose.vec.z if self._start_pose is not None else 0.0
            return planar_rig_pose(pose.x, pose.y, pose.yaw, z)
        anchor = np.asarray(self.origin_transform, dtype=float)
        anchor_yaw = math.atan2(anchor[1, 0], anchor[0, 0])
        current = planar_rig_pose(pose.x, pose.y, pose.yaw, anchor[2, 3])
        initial_flat = planar_rig_pose(0, 0, anchor_yaw)
        current[:3, :3] = current[:3, :3] @ initial_flat[:3, :3].T @ anchor[:3, :3]
        return current

    def cam_from_world(self, pose: EgoPose, camera: str | None = None) -> np.ndarray:
        """The same camera pose used by the render request, including selected-frame tilt."""
        import numpy as np

        extrinsic = self._camera_poses.get(camera, self._rig_to_camera)
        return np.linalg.inv(self.rig_transform(pose) @ pose_from_proto(extrinsic))

    def render(self, pose: EgoPose, camera: str, dynamic_objects=()):
        """Render one RGB frame at the ego pose via a single stateless render_rgb call."""
        return self.render_at(self.rig_transform(pose), self.timestamp_us(pose), camera, dynamic_objects)

    def render_at(self, world_from_rig, timestamp_us: int, camera: str, dynamic_objects=()):
        """Render an absolute rig pose/time and all present track-ID/world-pose pairs.

        Actor poses use the same reference point as sequence_tracks.json; no bounding-box
        offset is applied. Each pose pair repeats the pose to describe an instantaneous capture.
        """
        if self.origin_bounds is not None:
            first, last = self.origin_bounds
            if not first - 1e-6 <= timestamp_us / 1e6 <= last + 1e-6:
                raise ValueError("NuRec render timestamp exceeds recorded support")

        import cv2
        import numpy as np
        from alpasim_grpc.v0 import common_pb2, sensorsim_pb2

        from scipy.spatial.transform import Rotation

        if self._stub is None:
            raise RuntimeError("Call load_scene before rendering")
        if camera not in self._camera_specs:
            raise ValueError(f"Camera {camera!r} was not loaded")
        spec = self._camera_specs[camera]

        def protobuf_pose(world_from_object):
            transform = np.array(world_from_object, dtype=float, copy=True)
            qx, qy, qz, qw = Rotation.from_matrix(transform[:3, :3]).as_quat()
            return common_pb2.Pose(
                vec=common_pb2.Vec3(x=transform[0, 3], y=transform[1, 3], z=transform[2, 3]),
                quat=common_pb2.Quat(w=qw, x=qx, y=qy, z=qz),
            )

        rig = protobuf_pose(world_from_rig)
        from .simulator import _compose_pose

        cam = _compose_pose(rig, self._camera_poses[camera], common_pb2)
        objects = []
        for track_id, world_from_actor in dynamic_objects:
            actor_pose = protobuf_pose(validated_transform(world_from_actor))
            objects.append(sensorsim_pb2.DynamicObject(track_id=track_id,
                pose_pair=sensorsim_pb2.PosePair(start_pose=actor_pose, end_pose=actor_pose)))
        frame_us = int(timestamp_us)
        req = sensorsim_pb2.RGBRenderRequest(
            scene_id=self._loaded_scene_id,
            resolution_h=spec.resolution_h,
            resolution_w=spec.resolution_w,
            camera_intrinsics=spec,
            frame_start_us=frame_us,
            frame_end_us=frame_us + 1,  # render at an instant: a non-empty [t, t+1) interval
            sensor_pose=sensorsim_pb2.PosePair(start_pose=cam, end_pose=cam),
            dynamic_objects=objects,
            image_format=sensorsim_pb2.ImageFormat.JPEG,
            image_quality=self.image_quality,
            insert_ego_mask=False,
        )
        ret = self._stub.render_rgb(req)
        img = cv2.imdecode(np.frombuffer(ret.image_bytes, np.uint8), cv2.IMREAD_COLOR)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # HWC uint8 RGB for the AVStack
