"""NuRec camera and LiDAR rendering at the native RPC boundary."""

from __future__ import annotations

import copy
import dataclasses
import json
import math
import zipfile
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from collections.abc import Mapping
from typing import Any, ClassVar, TYPE_CHECKING

from avsectester.geometry.poses import validated_transform
from avsectester.validation import require_nonnegative_number, require_positive_number

if TYPE_CHECKING:
    from collections.abc import Sequence

    from avsectester.geometry.cameras import Camera, CameraCalibration, FThetaCamera
    from avsectester.geometry.lidars import LidarCalibration
    from avsectester.data_types import PointCloud
    from avsectester.rendering import PlacedObject
    from .gaussian_assets import GaussianAsset
    import numpy as np


_UNIFORM_COLORS = {"white": (1., 1., 1.), "black": (0., 0., 0.)}


@dataclass(frozen=True)
class _KnownTrack:
    asset: GaussianAsset
    filename: str


@dataclass(frozen=True)
class _CachedAsset:
    """A Gaussian asset with digests of its PLY bytes and of its bytes and NRE object size."""

    source: Any
    asset: GaussianAsset
    ply_digest: str
    digest: str

    @property
    def filename(self) -> str:
        return f"avsectester_asset_{self.ply_digest}.ply"


def _recorded_lidar_period(metadata: Mapping[str, Any], calibration: Mapping[str, Any]) -> float:
    """Median recorded scan duration in seconds, read under the sensor's ``name@sequence`` key."""
    import numpy as np

    lidar = calibration["logical_sensor_name"]
    sensor = f"{lidar}@{calibration['sequence_id']}"
    scans = [scan for rig in metadata.get("rig_trajectories", ())
             for scan in rig.get("lidars_frame_timestamps_us", {}).get(sensor, ())]
    error = f"LiDAR {lidar!r} has no recorded LiDAR scans"
    try:
        intervals = np.asarray(scans, dtype=np.float64)
    except (TypeError, ValueError) as failure:
        raise ValueError(error) from failure
    if (intervals.ndim != 2 or intervals.shape[1] != 2 or not len(intervals)
            or not np.isfinite(intervals).all() or not (intervals[:, 1] > intervals[:, 0]).all()):
        raise ValueError(error)
    return float(np.median(intervals[:, 1] - intervals[:, 0])) / 1e6


def _protobuf_pose(world_from_object: np.ndarray) -> Any:
    from alpasim_grpc.v0 import common_pb2
    from scipy.spatial.transform import Rotation

    transform = validated_transform(world_from_object)
    qx, qy, qz, qw = Rotation.from_matrix(transform[:3, :3]).as_quat()
    return common_pb2.Pose(
        vec=common_pb2.Vec3(x=transform[0, 3], y=transform[1, 3], z=transform[2, 3]),
        quat=common_pb2.Quat(w=qw, x=qx, y=qy, z=qz),
    )


def ftheta_camera_from_spec(spec: Any) -> FThetaCamera:
    """Build an f-theta lens from a NuRec CameraSpec, including its windshield distortion.

    NRE renders pixel ``i`` from the spec's image point ``i``, whereas ``Camera`` addresses pixel
    centers at ``i + 0.5``; the principal point is shifted by half a pixel accordingly. Probe
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
    'Camera names rendered by this renderer.'
    lidar: ClassVar[str | None] = None
    'LiDAR sensor name rendered by this renderer, or None without LiDAR.'
    @property
    def calibration(self) -> dict[str, CameraCalibration | LidarCalibration]:
        """Sensor calibrations supplied by a loaded renderer, empty by default."""
        return {}

    def copy(self) -> Renderer:
        """A renderer with this configuration and no connection, loaded scene or run origin.

        Every simulator runtime renders through its own copy. Override this when a renderer
        holds resources that a shallow copy would share.
        """
        return copy.copy(self)

    def reset_origin(self, start=None):
        """Set the run origin from the dataset start. Configured renderers have no origin."""
        self._origin_transform = None if start is None else deepcopy(start.payload["ego"].pose)
        self._origin_timestamp_us = None if start is None else round(start.frame.timestamp * 1e6)
        self._origin_bounds = None if start is None else (
            start.payload["first_timestamp"], start.payload["terminal_timestamp"])

    @property
    def origin_transform(self) -> np.ndarray | None:
        """World transform locating the loaded scene origin."""
        return deepcopy(getattr(self, "_origin_transform", None))

    @property
    def origin_timestamp_us(self) -> int | None:
        """Recorded timestamp in microseconds at the scene origin."""
        return getattr(self, "_origin_timestamp_us", None)

    @property
    def origin_bounds(self) -> tuple[float, float] | None:
        """First and terminal recorded timestamps, in seconds, of the loaded scene."""
        return getattr(self, "_origin_bounds", None)

    def load_scene(self, scene: Any) -> None:
        """Load a reconstruction once. Renderers without a service may leave this empty."""

    @abstractmethod
    def render(
        self, world_from_rig: np.ndarray, timestamp_us: int, camera: str, dynamic_objects=(),
    ) -> Any:
        """Render one camera at the explicit three-dimensional rig pose and scene timestamp."""

    def render_lidar(
        self, world_from_rig_start: np.ndarray, world_from_rig_end: np.ndarray,
        start_us: int, end_us: int,
        dynamic_objects_start: Sequence[tuple[str, np.ndarray]] = (),
        dynamic_objects_end: Sequence[tuple[str, np.ndarray]] = (),
    ) -> PointCloud:
        """Render one scan of ``lidar`` in the sensor frame at its ending pose and timestamp."""
        raise NotImplementedError(f"{type(self).__name__} cannot render LiDAR")

    def render_scene_visibility(
        self, world_from_rig: np.ndarray, timestamp_us: int, camera: str,
        objects: Sequence[PlacedObject], dynamic_objects=(), *, occlusion_tolerance_m: float = 0.,
    ) -> np.ndarray:
        """Pixels where the reconstructed scene does not hide ``objects`` in this capture.

        Scene content less than about ``occlusion_tolerance_m`` in front of an object does not
        hide it.
        """
        raise NotImplementedError(f"{type(self).__name__} cannot render object visibility; "
                                  "use object_occlusion='geometry'")

    def scene_object_tracks(
        self, objects: Sequence[PlacedObject],
    ) -> tuple[tuple[str, np.ndarray], ...]:
        """Register scene-rendered assets and return their track IDs and world poses."""
        raise NotImplementedError(f"{type(self).__name__} cannot insert scene objects")

    def render_object_coverage(
        self, world_from_rig: np.ndarray, timestamp_us: int, camera: str,
        placed: PlacedObject, dynamic_objects=(),
    ) -> np.ndarray:
        """Visible image coverage of one object, with values between zero and one."""
        raise NotImplementedError(f"{type(self).__name__} cannot render object coverage")

    def camera_model(self, camera: str | None = None) -> Camera | None:
        """Optical camera model, or None when this renderer supplies no geometry."""
        return None

    def cam_from_world(
        self, world_from_rig: np.ndarray, camera: str | None = None,
    ) -> np.ndarray:
        """World-to-optical-camera transform for the rendered pose."""
        raise NotImplementedError("This renderer does not provide camera geometry")

    def close(self) -> None:
        """Release renderer resources. The default implementation has no effect."""
        pass


@dataclass
class NuRecRenderer(Renderer):
    """Real NuRec renderer: one stateless ``SensorsimService`` render call per capture to an nre server.

    The heavy neural rendering lives in the ``nre-ga`` server (like the CARLA server), so this stays a
    thin, stateless client — no runtime, no controller/physics/traffic microservices, no callback
    inversion. Lazily imports the gRPC stubs. Connect it to a running renderer and a loaded NuRec
    scene. ``service`` exposes the native RPC client after loading. ``render_at`` supports exact
    recorded poses for initial-sequence inspection without advancing a driving policy.

    Cameras render at their native resolution. A different image size returned by NRE raises
    ``ValueError``. Set ``lidar`` to a scene logical sensor name to render LiDAR scans.
    Points use the sensor frame at the end of each scan.

    ``render_scene_visibility`` inserts probe copies of objects into the loaded scene. It writes
    their Gaussian PLY files to ``asset_directory``, which the NRE server must read at
    ``server_asset_directory`` (the same path by default). ``track_padding_m`` is the server's
    track padding, which sets the inserted probes' scale.
    """

    endpoint: str = "127.0.0.1:50051"
    scene_id: str | None = None
    cameras: list[str] = field(default_factory=lambda: ["camera_front_wide_120fov"])
    lidar: str | None = None
    image_quality: float = 95.0
    asset_directory: str | Path | None = None
    server_asset_directory: str | None = None
    track_padding_m: tuple[float, float, float] = (1.0, 1.0, 0.25)

    def copy(self) -> NuRecRenderer:
        """The same configuration without a connection, loaded scene or run origin."""
        return dataclasses.replace(self)

    def __post_init__(self) -> None:
        if self.lidar is not None and (not isinstance(self.lidar, str) or not self.lidar):
            raise ValueError("lidar must be a nonempty logical sensor name or None")
        self._lidar_calibration: LidarCalibration | None = None
        padding = tuple(float(value) for value in self.track_padding_m)
        if len(padding) != 3 or not all(math.isfinite(value) and value >= 0 for value in padding):
            raise ValueError("track_padding_m must contain three finite nonnegative values")
        self.track_padding_m = padding
        self._known_tracks: dict[str, _KnownTrack] = {}
        self._assets: dict[tuple[int, str, str], _CachedAsset] = {}
        self._trajectory_bounds_us = (0, 1)
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

    def served_scenes(self, timeout_s: float = 2.0) -> tuple[str, ...] | None:
        """Query supported scene IDs without changing this renderer's loaded scene.

        Unavailable endpoints and deadline expiry return None. Other gRPC errors
        retain their status and details in a RuntimeError.
        """
        require_positive_number(timeout_s, "timeout_s")
        import grpc
        from alpasim_grpc.v0 import common_pb2, sensorsim_pb2_grpc

        try:
            with grpc.insecure_channel(self.endpoint) as channel:
                response = sensorsim_pb2_grpc.SensorsimServiceStub(channel).get_available_scenes(
                    common_pb2.Empty(), timeout=timeout_s)
                return tuple(response.scene_ids)
        except grpc.RpcError as error:
            if error.code() in {grpc.StatusCode.UNAVAILABLE, grpc.StatusCode.DEADLINE_EXCEEDED}:
                return None
            raise RuntimeError(f"NuRec renderer RPC {error.code()}: {error.details()}") from error

    def load_scene(self, scene: Any) -> None:
        """Load sensor calibration and connect to a reconstructed scene.

        LiDAR requires a mapping with ``scene_id`` and ``usdz_path`` pointing to its archive.
        Camera-only scenes also accept a scene ID string.
        """
        import grpc
        from alpasim_grpc.v0 import common_pb2, sensorsim_pb2, sensorsim_pb2_grpc

        self.close()
        self._camera_models = {}
        self._lidar_calibration = None
        self._channel = grpc.insecure_channel(
            self.endpoint, options=[("grpc.max_receive_message_length", 64 * 1024 * 1024)])
        self._stub = sensorsim_pb2_grpc.SensorsimServiceStub(self._channel)
        available = list(self._stub.get_available_scenes(common_pb2.Empty()).scene_ids)
        source = scene if isinstance(scene, Mapping) else {"scene_id": scene or self.scene_id}
        want = source.get("scene_id", self.scene_id)
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
        self._trajectory_bounds_us = (int(poses[0].timestamp_us), max(int(poses[-1].timestamp_us),
                                                                      int(poses[0].timestamp_us) + 1))
        self._t0 = (
            poses[0].timestamp_us if self.origin_timestamp_us is None else self.origin_timestamp_us
        )
        for camera in self.cameras:
            self._camera_models[camera] = ftheta_camera_from_spec(self._camera_specs[camera])
        if self.lidar is not None:
            from avsectester.geometry.lidars import LidarCalibration

            if not source.get("usdz_path"):
                raise ValueError("NuRec LiDAR requires the scene archive usdz_path")
            with zipfile.ZipFile(source["usdz_path"]) as archive:
                metadata = json.loads(archive.read("rig_trajectories.json"))
            calibrations = list(metadata.get("lidar_calibrations", {}).values())
            if len(calibrations) > 1:
                raise ValueError("NRE supports scenes with only one LiDAR")
            matches = [entry for entry in calibrations
                       if entry.get("logical_sensor_name") == self.lidar]
            if len(matches) != 1:
                raise ValueError(f"LiDAR {self.lidar!r} is not available in the scene archive")
            calibration = matches[0]
            model = calibration.get("lidar_model", {})
            if model.get("type") != "row-offset-spinning":
                raise ValueError("NuRec LiDAR requires a row-offset-spinning model")
            # Use recorded durations because some releases have invalid spin frequencies.
            period = _recorded_lidar_period(metadata, calibration)
            self._lidar_calibration = LidarCalibration(
                validated_transform(calibration["T_sensor_rig"]), period)

    @property
    def calibration(self) -> dict[str, CameraCalibration | LidarCalibration]:
        """Loaded camera and LiDAR calibration, empty before loading."""
        from avsectester.geometry.cameras import CameraCalibration

        if self._stub is None:
            return {}
        calibrations: dict[str, CameraCalibration | LidarCalibration] = {name: CameraCalibration(
                    self.camera_model(name), validated_transform(pose_from_proto(self._camera_poses[name])))
                for name in self.cameras}
        if self.lidar is not None and self._lidar_calibration is not None:
            calibrations[self.lidar] = self._lidar_calibration
        return calibrations

    @property
    def service(self) -> Any:
        """Native SensorsimService client, including metadata APIs this renderer does not normalize."""
        if self._stub is None:
            raise RuntimeError("Call load_scene before accessing the NuRec service")
        return self._stub

    def close(self):
        if self._channel is not None:
            self._channel.close()
        self._channel = None
        self._stub = None
        self._lidar_calibration = None
        self._known_tracks.clear()

    def camera_model(self, camera: str | None = None) -> Camera:
        """The native lens read from the loaded scene's camera specification."""
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
        return self._render_image(world_from_rig, timestamp_us, camera, dynamic_objects, self.image_quality)

    def render_lidar(
        self, world_from_rig_start: np.ndarray, world_from_rig_end: np.ndarray,
        start_us: int, end_us: int,
        dynamic_objects_start: Sequence[tuple[str, np.ndarray]] = (),
        dynamic_objects_end: Sequence[tuple[str, np.ndarray]] = (),
    ) -> PointCloud:
        """Render a scan in the LiDAR frame at its ending pose and timestamp.

        Rig and actor poses describe the scan endpoints. Only actors present at both
        endpoints are included. Missing intensity is represented by None.
        """
        import numpy as np
        from alpasim_grpc.v0 import sensorsim_pb2
        from avsectester.data_types import PointCloud

        if self._stub is None or self._lidar_calibration is None:
            raise RuntimeError("Load a scene with LiDAR configured before rendering LiDAR")
        if end_us <= start_us:
            raise ValueError("LiDAR scan end must follow its start")
        calibration = self._lidar_calibration
        rig_from_lidar = validated_transform(calibration.rig_from_lidar)
        ends = dict(dynamic_objects_end)
        objects = [sensorsim_pb2.DynamicObject(
            track_id=track_id, pose_pair=sensorsim_pb2.PosePair(
                start_pose=_protobuf_pose(start_pose), end_pose=_protobuf_pose(ends[track_id])))
            for track_id, start_pose in dynamic_objects_start if track_id in ends]
        request = sensorsim_pb2.LidarRenderRequest(
            scene_id=self._loaded_scene_id,
            lidar_config=sensorsim_pb2.LidarSpec(lidar_type=sensorsim_pb2.PANDAR128),
            frame_start_us=int(start_us), frame_end_us=int(end_us),
            sensor_pose=sensorsim_pb2.PosePair(
                start_pose=_protobuf_pose(validated_transform(world_from_rig_start) @ rig_from_lidar),
                end_pose=_protobuf_pose(validated_transform(world_from_rig_end) @ rig_from_lidar)),
            dynamic_objects=objects,
        )
        response = self._render_request(self._stub.render_lidar, request)
        xyz = (np.frombuffer(response.point_xyzs_buffer, dtype=np.float32).copy()
               if response.point_xyzs_buffer else np.asarray(response.point_xyzs, dtype=np.float32))
        if xyz.ndim != 1 or xyz.size % 3:
            raise ValueError("NRE LiDAR coordinates must have shape (N, 3)")
        points = xyz.reshape(-1, 3)
        if len(points) != response.num_points:
            raise ValueError("NRE LiDAR point count does not match num_points")
        if not np.isfinite(points).all():
            raise ValueError("NRE LiDAR points must be finite float32 coordinates")
        intensity = (np.frombuffer(response.point_intensities_buffer, dtype=np.float32).copy()
                     if response.point_intensities_buffer
                     else np.asarray(response.point_intensities, dtype=np.float32))
        if intensity.ndim != 1 or intensity.size not in {0, len(points)}:
            raise ValueError("NRE LiDAR intensity must contain one value per point")
        if not np.isfinite(intensity).all() or ((intensity < 0) | (intensity > 1)).any():
            raise ValueError("NRE LiDAR intensity must be finite and in [0, 1]")
        return PointCloud(points, None if intensity.size == 0 else intensity)

    def render_scene_visibility(
        self, world_from_rig: np.ndarray, timestamp_us: int, camera: str,
        objects: Sequence[PlacedObject], dynamic_objects=(), *, occlusion_tolerance_m: float = 0.,
    ) -> np.ndarray:
        """Pixels where the reconstructed scene does not hide ``objects`` in an NRE render.

        Each object has two opaque Gaussian probes, white and black, under track IDs derived from
        their content and object ID, so distinct objects can share an asset without sharing tracks.
        Probes are inserted when the server reports them missing, on first use or after the server
        drops the scene's edits because it reloaded the scene or an edit failed.
        Both probes are rendered with ``dynamic_objects`` at this capture, as maximum-quality JPEG:
        server-side PNG encoding is about fifteen times slower, and the resulting visibility
        differs only in single edge pixels. A pixel is visible when the white render exceeds the
        black one by at least half the color range.
        Rejects probe pixels outside the objects' projected boxes, which indicate an asset scale
        or axis convention that differs from the measured NRE behavior.
        ``occlusion_tolerance_m`` moves each probe that distance toward the camera, along the ray
        to its bounding-box center and at most halfway there, so scene content less than about
        that distance in front of an object does not hide it. Composed objects keep their poses.
        The moved probe projects slightly larger, which the composition plan clips to the object.
        """
        import numpy as np

        from avsectester.geometry.projection import project_box

        if self.asset_directory is None:
            raise ValueError("Rendered-probe occlusion requires NuRecRenderer.asset_directory, readable by "
                             "the NRE server; or use object_occlusion='geometry'")
        require_nonnegative_number(occlusion_tolerance_m, "occlusion_tolerance_m")
        cam_from_world = self.cam_from_world(world_from_rig, camera)
        camera_center = np.linalg.inv(cam_from_world)[:3, 3]
        tracks: dict[str, list] = {color: [] for color in _UNIFORM_COLORS}
        boxes = []
        for placed in objects:
            world_from_probe = placed.world_from_object.copy()
            center = world_from_probe @ np.array([0., 0., placed.asset.extent[2] / 2, 1.])
            ray = center[:3] - camera_center
            distance = float(np.linalg.norm(ray))
            if occlusion_tolerance_m and distance > 0:
                world_from_probe[:3, 3] -= min(occlusion_tolerance_m, distance / 2) * ray / distance
            for color in _UNIFORM_COLORS:
                probe, track_id = self._probe(placed.asset, placed.object_id, color)
                world_from_track = world_from_probe @ probe.object_from_track
                tracks[color].append((track_id, world_from_track))
            boxes.append((world_from_track, placed.asset.extent))

        def render(color: str) -> np.ndarray:
            return self._render_image(world_from_rig, timestamp_us, camera,
                                      (*dynamic_objects, *tracks[color]), 100.0)

        white, black = render("white"), render("black")
        visible = (white.astype(np.int16) - black).mean(axis=2) >= 127.5
        camera_model = self.camera_model(camera)
        expected = np.zeros_like(visible)
        for world_from_box, extent in boxes:
            bounds = project_box(camera_model, cam_from_world, world_from_box, extent)
            if bounds is not None:
                x0, y0, x1, y1 = bounds
                expected[max(0, math.floor(y0) - 3):max(0, math.ceil(y1) + 3),
                         max(0, math.floor(x0) - 3):max(0, math.ceil(x1) + 3)] = True
        stray = int(np.count_nonzero(visible & ~expected))
        if stray > max(10, 0.02 * np.count_nonzero(visible)):
            raise ValueError(f"{stray} NRE probe pixels fall outside the objects' projections; "
                             "check track_padding_m and the NRE version")
        return visible

    def _parked_pose(self) -> np.ndarray:
        """Stored pose of inserted tracks, far below the scene.

        A request listing no dynamic objects renders every track at its stored pose.
        """
        import numpy as np

        pose = np.eye(4)
        pose[2, 3] = -10_000.
        return pose

    def _cached_asset(self, asset: Any, kind: str, color: str) -> _CachedAsset:
        """A probe or scene Gaussian copy of ``asset`` in one color, built once per asset."""
        import hashlib

        from .gaussian_assets import gaussian_asset, probe_asset

        key = (id(asset), kind, color)
        if key not in self._assets:
            if kind == "probe":
                gaussian = probe_asset(asset, _UNIFORM_COLORS[color],
                                       track_padding_m=self.track_padding_m)
            else:
                sources = (tuple(surface.texture for surface in asset.surfaces) if color == "texture"
                           else (_UNIFORM_COLORS[color],) * len(asset.surfaces))
                gaussian = gaussian_asset(asset, sources, spacing_m=.005, lateral_scale=1.,
                                          track_padding_m=self.track_padding_m)
            digest = hashlib.sha256(gaussian.ply)
            ply_digest = digest.hexdigest()
            digest.update(repr(gaussian.object_size).encode())
            # The cached source keeps its id from being reused while the entry exists.
            self._assets[key] = _CachedAsset(asset, gaussian, ply_digest, digest.hexdigest())
        return self._assets[key]

    def _register_object_track(self, prefix: str, cached: _CachedAsset, object_id: str) -> str:
        """Register a track specific to one asset and object, and return its ID."""
        import hashlib

        identity = json.dumps((cached.digest, object_id)).encode()
        track_id = f"{prefix}_{hashlib.sha256(identity).hexdigest()[:24]}"
        self._known_tracks[track_id] = _KnownTrack(cached.asset, cached.filename)
        return track_id

    def _probe(self, asset: Any, object_id: str, color: str) -> tuple[GaussianAsset, str]:
        """A single-color probe with a track ID specific to its content and object."""
        cached = self._cached_asset(asset, "probe", color)
        return cached.asset, self._register_object_track("avsectester_probe", cached, object_id)

    def _coverage_copy(self, asset: Any, color: str) -> tuple[GaussianAsset, str]:
        """A single-color scene copy for coverage measurement and its track ID."""
        cached = self._cached_asset(asset, "scene", color)
        track_id = f"avsectester_coverage_{cached.digest[:24]}"
        self._known_tracks[track_id] = _KnownTrack(cached.asset, cached.filename)
        return cached.asset, track_id

    def scene_object_tracks(
        self, objects: Sequence[PlacedObject],
    ) -> tuple[tuple[str, np.ndarray], ...]:
        """Register textured scene assets and return their requested world-pose track entries.

        Scene assets use a 5 mm grid and lateral Gaussian sizes equal to the grid spacing.
        Stored tracks remain underground. Requests place the current objects at their real poses.
        Each object reuses its track when moved. Only texture changes accumulate tracks until
        the NRE process restarts. Edits are retained rather than using
        ``restore_model_parameters``, which also removes other clients' edits.
        """
        if self.asset_directory is None:
            raise ValueError("Scene insertion requires NuRecRenderer.asset_directory")
        if self._stub is None:
            raise RuntimeError("Call load_scene before registering scene objects")
        result = []
        for placed in objects:
            cached = self._cached_asset(placed.asset, "scene", "texture")
            track_id = self._register_object_track("avsectester_scene", cached, placed.object_id)
            self._write_asset(cached.asset, cached.filename)
            pose = validated_transform(placed.world_from_object) @ cached.asset.object_from_track
            result.append((track_id, pose))
        return tuple(result)

    def render_object_coverage(
        self, world_from_rig: np.ndarray, timestamp_us: int, camera: str,
        placed: PlacedObject, dynamic_objects=(),
    ) -> np.ndarray:
        """Return visible object coverage from pure-white and pure-black scene copies.

        Copies have the scene asset's grid and use the ordinary camera image quality.
        The caller lists actors and all other scene objects, excluding this object.
        """
        import numpy as np

        if self.asset_directory is None:
            raise ValueError("Object coverage requires NuRecRenderer.asset_directory")
        images = []
        for color in ("white", "black"):
            gaussian, track_id = self._coverage_copy(placed.asset, color)
            pose = placed.world_from_object @ gaussian.object_from_track
            images.append(self._render_image(world_from_rig, timestamp_us, camera,
                          (*dynamic_objects, (track_id, pose)), self.image_quality))
        return np.clip((images[0].astype(float) - images[1]).mean(axis=2) / 255, 0., 1.)

    def _write_asset(self, asset: GaussianAsset, filename: str) -> None:
        import os
        import tempfile

        if self.asset_directory is None:
            raise ValueError("NuRec assets require asset_directory")
        path = Path(self.asset_directory) / filename
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
                stream.write(asset.ply)
            os.replace(stream.name, path)
            os.chmod(path, 0o644)

    def _insert_track(self, track_id: str) -> None:
        """Insert only the known track that NRE reported missing."""
        from .asset_editing import EDIT_ASSETS_METHOD, asset_editing_messages

        if self._channel is None:
            raise RuntimeError("Call load_scene before rendering")
        messages = asset_editing_messages()
        request = messages["EditAssetsRequest"](scene_id=self._loaded_scene_id)
        server_directory = (Path(self.server_asset_directory) if self.server_asset_directory is not None
                            else Path(self.asset_directory).resolve())
        record = self._known_tracks[track_id]
        asset, filename = record.asset, record.filename
        self._write_asset(asset, filename)
        track = request.insert.add(id=track_id, semantic_class="protruding_object",
                                   asset_id=str(server_directory / filename))
        track.object_size.size_x, track.object_size.size_y, track.object_size.size_z = asset.object_size
        for timestamp_us in self._trajectory_bounds_us:
            entry = track.trajectory.poses.add(timestamp_us=timestamp_us)
            pose = _protobuf_pose(self._parked_pose())
            entry.pose.vec.x, entry.pose.vec.y, entry.pose.vec.z = pose.vec.x, pose.vec.y, pose.vec.z
            entry.pose.quat.w, entry.pose.quat.x = pose.quat.w, pose.quat.x
            entry.pose.quat.y, entry.pose.quat.z = pose.quat.y, pose.quat.z
        edit = self._channel.unary_unary(
            EDIT_ASSETS_METHOD, request_serializer=type(request).SerializeToString,
            response_deserializer=messages["EditAssetsResponse"].FromString)
        response = edit(request)
        if not response.success:
            raise RuntimeError(f"NRE edit_assets failed: {response.message}")

    def _render_request(self, call, request):
        import re

        import grpc

        inserted = set()
        while True:
            try:
                return call(request)
            except grpc.RpcError as error:
                if error.code() != grpc.StatusCode.INVALID_ARGUMENT:
                    raise
                match = re.search(r"""(['"])([^'"]+)\1\s+is not in list""", str(error.details()))
                if match is None or match[2] not in self._known_tracks:
                    raise
                track_id = match[2]
                if track_id in inserted:
                    raise RuntimeError(
                        f"NRE still cannot find track {track_id!r} after insertion") from error
                inserted.add(track_id)
                self._insert_track(track_id)

    def _render_image(self, world_from_rig, timestamp_us: int, camera: str, dynamic_objects,
                      image_quality: float):
        if self.origin_bounds is not None:
            first, last = self.origin_bounds
            if not first - 1e-6 <= timestamp_us / 1e6 <= last + 1e-6:
                raise ValueError("NuRec render timestamp exceeds recorded support")

        import cv2
        import numpy as np
        from alpasim_grpc.v0 import common_pb2, sensorsim_pb2

        if self._stub is None:
            raise RuntimeError("Call load_scene before rendering")
        if camera not in self._camera_specs:
            raise ValueError(f"Camera {camera!r} was not loaded")
        spec = self._camera_specs[camera]

        rig = _protobuf_pose(world_from_rig)
        from .simulator import _compose_pose

        cam = _compose_pose(rig, self._camera_poses[camera], common_pb2)
        objects = []
        for track_id, world_from_actor in dynamic_objects:
            actor_pose = _protobuf_pose(world_from_actor)
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
            image_quality=image_quality,
            insert_ego_mask=False,
        )
        ret = self._render_request(self._stub.render_rgb, req)
        img = cv2.imdecode(np.frombuffer(ret.image_bytes, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"NRE returned an undecodable image for camera {camera!r}")
        model = self._camera_models[camera]
        if img.shape[:2] != (model.height, model.width):
            raise ValueError(
                f"NRE returned image dimensions that are not native for camera {camera!r}: "
                f"expected {(model.height, model.width)}, got {img.shape[:2]}"
            )
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # HWC uint8 RGB for the AVStack


__all__ = [
    'Renderer',
    'NuRecRenderer',
]
