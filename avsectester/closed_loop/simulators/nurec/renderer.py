"""NuRec RGB rendering and native camera/pose conversion at the RPC boundary."""

from __future__ import annotations

import copy
import dataclasses
import math
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, TYPE_CHECKING

from avsectester.geometry.poses import validated_transform
from avsectester.validation import require_nonnegative_number, require_positive_number

if TYPE_CHECKING:
    from collections.abc import Sequence

    from avsectester.geometry.cameras import Camera, CameraCalibration, FThetaCamera
    from avsectester.rendering import PlacedObject
    from .probe_assets import ProbeAsset
    import numpy as np


_PROBE_COLOURS = (("white", (1., 1., 1.)), ("black", (0., 0., 0.)))


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
    @property
    def calibration(self) -> dict[str, CameraCalibration]:
        """Camera calibrations supplied by a loaded renderer, empty by default."""
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
    """Real NuRec renderer: one stateless ``SensorsimService.render_rgb`` per frame to an nre server.

    The heavy neural rendering lives in the ``nre-ga`` server (like the CARLA server), so this stays a
    thin, stateless client — no runtime, no controller/physics/traffic microservices, no callback
    inversion. Lazily imports the gRPC stubs. Connect it to a running renderer and a loaded NuRec
    scene. ``service`` exposes the native RPC client after loading. ``render_at`` supports exact
    recorded poses for initial-sequence inspection without advancing a driving policy.

    Cameras render at their native resolution; a different image size returned by NRE raises
    ``ValueError``.

    ``render_scene_visibility`` inserts probe copies of objects into the loaded scene. It writes
    their Gaussian PLY files to ``probe_directory``, which the NRE server must read at
    ``server_probe_directory`` (the same path by default). ``track_padding_m`` is the server's
    track padding, which sets the inserted probes' scale.
    """

    endpoint: str = "127.0.0.1:50051"
    scene_id: str | None = None
    cameras: list[str] = field(default_factory=lambda: ["camera_front_wide_120fov"])
    image_quality: float = 95.0
    probe_directory: str | Path | None = None
    server_probe_directory: str | None = None
    track_padding_m: tuple[float, float, float] = (1.0, 1.0, 0.25)

    def copy(self) -> NuRecRenderer:
        """The same configuration without a connection, loaded scene or run origin."""
        return dataclasses.replace(self)

    def __post_init__(self) -> None:
        padding = tuple(float(value) for value in self.track_padding_m)
        if len(padding) != 3 or not all(math.isfinite(value) and value >= 0 for value in padding):
            raise ValueError("track_padding_m must contain three finite nonnegative values")
        self.track_padding_m = padding
        self._probes: dict[tuple[int, str], tuple[Any, ProbeAsset, str]] = {}
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
        self._trajectory_bounds_us = (int(poses[0].timestamp_us), max(int(poses[-1].timestamp_us),
                                                                      int(poses[0].timestamp_us) + 1))
        self._t0 = (
            poses[0].timestamp_us if self.origin_timestamp_us is None else self.origin_timestamp_us
        )
        for camera in self.cameras:
            self._camera_models[camera] = ftheta_camera_from_spec(self._camera_specs[camera])

    @property
    def calibration(self) -> dict[str, CameraCalibration]:
        """Loaded camera lenses and optical-camera-to-rig poses, empty before loading."""
        from avsectester.geometry.cameras import CameraCalibration

        if self._stub is None:
            return {}
        return {name: CameraCalibration(
                    self.camera_model(name), validated_transform(pose_from_proto(self._camera_poses[name])))
                for name in self.cameras}

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

    def render_scene_visibility(
        self, world_from_rig: np.ndarray, timestamp_us: int, camera: str,
        objects: Sequence[PlacedObject], dynamic_objects=(), *, occlusion_tolerance_m: float = 0.,
    ) -> np.ndarray:
        """Pixels where the reconstructed scene does not hide ``objects`` in an NRE render.

        Each object has two opaque Gaussian probes, white and black, under track IDs derived from
        their content, so clients rendering the same object share them. Probes are inserted into
        the loaded scene when the server reports them missing: on first use, and after the server
        drops the scene's edits because it reloaded the scene or an edit of it failed.
        Both probes are rendered with ``dynamic_objects`` at this capture, as maximum-quality JPEG:
        server-side PNG encoding is about fifteen times slower, and the resulting visibility
        differs only in single edge pixels. A pixel is visible when the white render exceeds the
        black one by at least half the color range.
        Rejects probe pixels outside the objects' projected boxes, which indicate an asset scale
        or axis convention that differs from the measured NRE behavior.
        ``occlusion_tolerance_m`` moves each probe that distance toward the camera, along the ray
        to its bounding-box center and at most halfway there, so scene content less than about
        that distance in front of an object does not hide it. Composed objects keep their poses;
        the moved probe projects slightly larger, which the composition plan clips to the object.
        """
        import grpc
        import numpy as np

        from avsectester.geometry.projection import project_box

        if self.probe_directory is None:
            raise ValueError("Rendered-probe occlusion requires NuRecRenderer.probe_directory, readable by "
                             "the NRE server; or use object_occlusion='geometry'")
        require_nonnegative_number(occlusion_tolerance_m, "occlusion_tolerance_m")
        cam_from_world = self.cam_from_world(world_from_rig, camera)
        camera_centre = np.linalg.inv(cam_from_world)[:3, 3]
        tracks: dict[str, list] = {colour: [] for colour, _ in _PROBE_COLOURS}
        boxes = []
        for placed in objects:
            world_from_probe = placed.world_from_object.copy()
            centre = world_from_probe @ np.array([0., 0., placed.asset.extent[2] / 2, 1.])
            ray = centre[:3] - camera_centre
            distance = float(np.linalg.norm(ray))
            if occlusion_tolerance_m and distance > 0:
                world_from_probe[:3, 3] -= min(occlusion_tolerance_m, distance / 2) * ray / distance
            for colour, rgb in _PROBE_COLOURS:
                probe, track_id = self._probe(placed.asset, colour, rgb)
                world_from_track = world_from_probe @ probe.object_from_track
                tracks[colour].append((track_id, world_from_track))
            boxes.append((world_from_track, placed.asset.extent))

        def render(colour: str) -> np.ndarray:
            try:
                return self._render_image(world_from_rig, timestamp_us, camera,
                                          (*dynamic_objects, *tracks[colour]), 100.0)
            except grpc.RpcError as error:
                # NRE rejects a request naming a track that its loaded scene does not hold.
                if error.code() != grpc.StatusCode.INVALID_ARGUMENT or "is not in list" not in str(error.details()):
                    raise
            self._insert_probes(objects)
            return self._render_image(world_from_rig, timestamp_us, camera,
                                      (*dynamic_objects, *tracks[colour]), 100.0)

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

    def _probe(self, asset: Any, colour: str, rgb: tuple[float, float, float]) -> tuple[ProbeAsset, str]:
        """A probe of ``asset`` in one color and its content-derived track ID, cached per asset."""
        import hashlib

        from .probe_assets import probe_asset

        key = (id(asset), colour)
        if key not in self._probes:
            probe = probe_asset(asset, rgb, track_padding_m=self.track_padding_m)
            digest = hashlib.sha256(probe.ply + repr(probe.object_size).encode()).hexdigest()[:24]
            # The cached asset reference keeps its id from being reused while the entry exists.
            self._probes[key] = (asset, probe, f"avsectester_probe_{digest}")
        _, probe, track_id = self._probes[key]
        return probe, track_id

    def _insert_probes(self, objects: Sequence[PlacedObject]) -> None:
        """Write the probe PLY files of ``objects`` and insert their tracks into the loaded scene."""
        import os
        import tempfile
        from pathlib import Path

        from .asset_editing import EDIT_ASSETS_METHOD, asset_editing_messages

        assert self.probe_directory is not None
        if self._channel is None:
            raise RuntimeError("Call load_scene before rendering")
        messages = asset_editing_messages()
        request = messages["EditAssetsRequest"](scene_id=self._loaded_scene_id)
        server_directory = Path(self.server_probe_directory or self.probe_directory)
        for placed in objects:
            for colour, rgb in _PROBE_COLOURS:
                probe, track_id = self._probe(placed.asset, colour, rgb)
                path = Path(self.probe_directory) / f"{track_id}.ply"
                if not path.exists():
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
                        stream.write(probe.ply)
                    os.replace(stream.name, path)
                    os.chmod(path, 0o644)
                track = request.insert.add(id=track_id, semantic_class="protruding_object",
                                           asset_id=str(server_directory / path.name))
                track.object_size.size_x, track.object_size.size_y, track.object_size.size_z = probe.object_size
                # NRE requires two trajectory poses; render requests listing the probe set its pose.
                # A request listing no dynamic objects renders every track along its trajectory,
                # so the trajectory lies far below the scene.
                for timestamp_us in self._trajectory_bounds_us:
                    pose = track.trajectory.poses.add(timestamp_us=timestamp_us)
                    pose.pose.vec.z, pose.pose.quat.w = -10_000.0, 1.0
        edit = self._channel.unary_unary(
            EDIT_ASSETS_METHOD, request_serializer=type(request).SerializeToString,
            response_deserializer=messages["EditAssetsResponse"].FromString)
        response = edit(request)
        if not response.success:
            raise RuntimeError(f"NRE edit_assets failed: {response.message}")

    def _render_image(self, world_from_rig, timestamp_us: int, camera: str, dynamic_objects,
                      image_quality: float):
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
            image_quality=image_quality,
            insert_ego_mask=False,
        )
        ret = self._stub.render_rgb(req)
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
