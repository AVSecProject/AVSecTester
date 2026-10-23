"""NuRec RGB rendering and native camera/pose conversion at the RPC boundary."""

from __future__ import annotations

import copy
import dataclasses
import math
from abc import ABC, abstractmethod
from copy import deepcopy
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar, TYPE_CHECKING

from avsectester.geometry.poses import validated_transform
from avsectester.validation import require_positive_integer

if TYPE_CHECKING:
    from avsectester.geometry.cameras import Camera, FThetaCamera
    import numpy as np


def ftheta_camera_from_spec(spec: Any) -> FThetaCamera:
    """Build an f-theta lens from a NuRec CameraSpec, including its windshield distortion.

    NRE renders pixel ``i`` from the spec's image point ``i``, whereas ``Camera`` addresses pixel
    centres at ``i + 0.5``; the principal point is shifted by half a pixel accordingly. Probe
    objects inserted into NRE scenes land on the projected pixels within 0.05 px.
    """
    from avsectester.geometry.cameras import FThetaCamera, WindshieldDistortion

    parameters = spec.ftheta_param
    correction = parameters.linear_cde
    windshield = None
    if spec.WhichOneof("external_distortion") == "bivariate_windshield_model_param":
        model = spec.bivariate_windshield_model_param
        windshield = WindshieldDistortion(
            tuple(model.horizontal_poly), tuple(model.vertical_poly),
            tuple(model.horizontal_poly_inverse), tuple(model.vertical_poly_inverse))
    elif spec.WhichOneof("external_distortion") is not None:
        raise ValueError(f"Unsupported NuRec external distortion {spec.WhichOneof('external_distortion')!r}")
    return FThetaCamera(
        cx=parameters.principal_point_x + 0.5,
        cy=parameters.principal_point_y + 0.5,
        angle_to_pixeldist=tuple(parameters.angle_to_pixeldist_poly),
        pixeldist_to_angle=tuple(parameters.pixeldist_to_angle_poly),
        width=spec.resolution_w,
        height=spec.resolution_h,
        max_angle=parameters.max_angle or math.pi / 2,
        linear_cde=(correction.linear_c or 1.0, correction.linear_d, correction.linear_e),
        windshield=windshield,
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
    def render(
        self, world_from_rig: np.ndarray, timestamp_us: int, camera: str, dynamic_objects=(),
    ) -> Any:
        """Render one camera at the explicit three-dimensional rig pose and scene timestamp."""

    def camera_model(self, camera: str | None = None) -> Camera | None:
        """Optical camera model, or None when this renderer supplies no geometry."""
        return None

    def cam_from_world(
        self, world_from_rig: np.ndarray, camera: str | None = None,
    ) -> np.ndarray:
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
    resolution_hw: Mapping[str, tuple[int, int]] = field(default_factory=dict)

    def copy(self) -> NuRecRenderer:
        """The same configuration without a connection, loaded scene or run origin."""
        return dataclasses.replace(self)

    def __post_init__(self) -> None:
        if not isinstance(self.resolution_hw, Mapping):
            raise TypeError("resolution_hw must map camera names to (height, width)")
        resolutions = {}
        for camera, dimensions in self.resolution_hw.items():
            if camera not in self.cameras:
                raise ValueError(f"resolution_hw names an unconfigured camera {camera!r}")
            if not isinstance(dimensions, (tuple, list)) or len(dimensions) != 2:
                raise ValueError(f"resolution_hw[{camera!r}] must contain height and width")
            height, width = dimensions
            require_positive_integer(height, f"resolution_hw[{camera!r}] height")
            require_positive_integer(width, f"resolution_hw[{camera!r}] width")
            resolutions[camera] = (height, width)
        self.resolution_hw = resolutions
        self._camera_models = {}
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
        self._camera_models = {}
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
        world_from_rig = self.origin_transform
        if world_from_rig is None:
            world_from_rig = pose_from_proto(self._start_pose)
        for camera in self.cameras:
            image = self.render_at(world_from_rig, self._t0, camera)
            height, width = image.shape[:2]
            native = ftheta_camera_from_spec(self._camera_specs[camera])
            scale = height / native.height
            self._camera_models[camera] = dataclasses.replace(
                native, width=width, height=height, cx=native.cx * scale, cy=native.cy * scale,
                angle_to_pixeldist=tuple(value * scale for value in native.angle_to_pixeldist),
                pixeldist_to_angle=tuple(value * scale ** -power
                                         for power, value in enumerate(native.pixeldist_to_angle)),
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

    def camera_model(self, camera: str | None = None) -> Camera:
        """The lens scaled to the image dimensions actually returned by NRE during load_scene."""
        if self._stub is None:
            raise RuntimeError("Call load_scene before requesting a NuRec camera model")
        name = self.cameras[0] if camera is None else camera
        if name not in self._camera_models:
            raise ValueError(f"Camera {name!r} was not loaded")
        return deepcopy(self._camera_models[name])

    def cam_from_world(
        self, world_from_rig: np.ndarray, camera: str | None = None,
    ) -> np.ndarray:
        """World-to-optical-camera transform at the explicit three-dimensional rig pose."""
        import numpy as np

        extrinsic = self._camera_poses.get(camera, self._rig_to_camera)
        return np.linalg.inv(validated_transform(world_from_rig) @ pose_from_proto(extrinsic))

    def render(
        self, world_from_rig: np.ndarray, timestamp_us: int, camera: str, dynamic_objects=(),
    ):
        """Render one RGB frame at an explicit three-dimensional rig pose and scene timestamp."""
        return self.render_at(world_from_rig, timestamp_us, camera, dynamic_objects)

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
        height, width = self.resolution_hw.get(camera, (spec.resolution_h, spec.resolution_w))
        req = sensorsim_pb2.RGBRenderRequest(
            scene_id=self._loaded_scene_id,
            resolution_h=height,
            resolution_w=width,
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
        if img is None:
            raise ValueError(f"NRE returned an undecodable image for camera {camera!r}")
        model = self._camera_models.get(camera)
        if model is not None and img.shape[:2] != (model.height, model.width):
            raise ValueError(
                f"NRE image dimensions for camera {camera!r} changed: "
                f"expected {(model.height, model.width)}, got {img.shape[:2]}"
            )
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # HWC uint8 RGB for the AVStack
