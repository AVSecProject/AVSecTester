"""Neural-reconstruction backend with in-process ego dynamics and pluggable rendering.

``NuRecBackend`` owns the control loop. ``StubRenderer`` supports offline use, while
``NuRecRenderer`` calls a separate SensorsimService for RGB images and exposes its native RPC
client for metadata access. Host-bound insertions use the shared geometric insertion renderer.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from itertools import pairwise
from typing import Any, ClassVar, Protocol, TYPE_CHECKING

from avsectester.backend import validate_command_modes
from avsectester.capture import CaptureBackend
from avsectester.plane import Control, Observation, WorldSnapshot
from avsectester.rendering.compositor import InsertionRenderer

if TYPE_CHECKING:
    from avsectester.runtime import RenderRequest


@dataclass
class EgoPose:
    """Ego world state we own and integrate (2-D kinematic dynamics)."""

    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0  # rad
    speed: float = 0.0  # m/s
    t: float = 0.0  # s


class EgoDynamics(Protocol):
    """A custom ego-motion implementation, with an optional ``command_modes`` declaration."""

    def step(self, pose: EgoPose, control: Control, dt: float) -> EgoPose:
        """Advance the physical ego pose by ``dt`` seconds using the final command."""
        ...


@dataclass
class KinematicBicycle:
    """A simple, transparent ego dynamics model — the shared 'physical control' for NuRec worlds.

    Maps a :class:`~avsectester.plane.Control` (throttle/steer/brake) to motion. Pure and in-process,
    so a step is trivially inspectable and deterministic.
    """

    wheelbase: float = 2.8
    max_accel: float = 3.0  # m/s^2 at full throttle
    max_brake: float = 8.0  # m/s^2 at full brake
    max_steer: float = 0.6  # rad at full steer
    command_modes: ClassVar[frozenset[str]] = frozenset({"actuator"})

    def step(self, pose: EgoPose, control: Control, dt: float) -> EgoPose:
        validate_command_modes(control, self.command_modes)
        accel = control.throttle * self.max_accel - control.brake * self.max_brake
        speed = max(0.0, pose.speed + accel * dt)
        steer = control.steer * self.max_steer
        yaw = pose.yaw + (speed / self.wheelbase) * math.tan(steer) * dt
        return EgoPose(
            x=pose.x + speed * math.cos(yaw) * dt,
            y=pose.y + speed * math.sin(yaw) * dt,
            yaw=yaw,
            speed=speed,
            t=pose.t + dt,
        )


@dataclass
class TrajectoryFollower:
    """Advance the ego along a planned trajectory (for trajectory-output stacks like Alpamayo).

    ``Control.trajectory`` is ``[((x,y,z),(w,x,y,z),t_us), ...]`` in the **rig frame** (x forward,
    y left) at absolute microsecond timestamps. Each step interpolates the rig-frame position at
    ``pose.t + dt`` and places the ego there in world coordinates — an open-loop follower (the ego
    realizes exactly what the policy planned). With no trajectory it coasts to a stop.
    """

    decel: float = 4.0  # m/s^2 when coasting with no plan
    command_modes: ClassVar[frozenset[str]] = frozenset({"trajectory"})

    def step(self, pose: EgoPose, control: Control, dt: float) -> EgoPose:
        validate_command_modes(control, self.command_modes)
        traj = control.trajectory or []
        if not traj:
            speed = max(0.0, pose.speed - self.decel * dt)
            return EgoPose(pose.x, pose.y, pose.yaw, speed, pose.t + dt)
        dx, dy = self._interp_rig_xy(traj, (pose.t + dt) * 1e6, pose.t * 1e6)
        c, s = math.cos(pose.yaw), math.sin(pose.yaw)
        dist = math.hypot(dx, dy)
        yaw = pose.yaw + math.atan2(dy, dx) if dist > 1e-3 else pose.yaw
        return EgoPose(
            x=pose.x + dx * c - dy * s,
            y=pose.y + dx * s + dy * c,
            yaw=yaw,
            speed=dist / dt,
            t=pose.t + dt,
        )

    @staticmethod
    def _interp_rig_xy(traj: list, target_us: float, origin_us: float) -> tuple[float, float]:
        pts = [(t_us, xyz[0], xyz[1]) for xyz, _quat, t_us in traj]
        if target_us <= pts[0][0]:
            # Rig-frame displacements start at the current ego position. A physics
            # substep before the first waypoint must cover only part of that segment.
            interval = pts[0][0] - origin_us
            fraction = (
                min(1.0, max(0.0, (target_us - origin_us) / interval))
                if interval > 0 else 1.0
            )
            return fraction * pts[0][1], fraction * pts[0][2]
        for (t0, x0, y0), (t1, x1, y1) in pairwise(pts):
            if target_us <= t1:
                a = (target_us - t0) / (t1 - t0) if t1 > t0 else 0.0
                return x0 + a * (x1 - x0), y0 + a * (y1 - y0)
        return pts[-1][1], pts[-1][2]


class Renderer(ABC):
    """Renders sensor observations from the reconstructed scene at a given ego pose."""

    cameras: ClassVar[list[str]] = ["camera_front"]
    calibration: ClassVar[dict] = {}

    def reset_origin(self, source=None):
        """Replace runtime origin without modifying configured renderer defaults."""
        self._origin_transform = deepcopy(getattr(self, "start_transform", None))
        self._origin_timestamp_us = getattr(self, "start_timestamp_us", None)
        self._origin_bounds = getattr(self, "source_time_bounds", None)
        if source is not None:
            self._origin_transform = deepcopy(source.payload["scene"].ego.pose)
            self._origin_timestamp_us = round(source.start.timestamp * 1e6)
            self._origin_bounds = (source.payload["first_timestamp"], source.payload["terminal_timestamp"])

    @property
    def origin_transform(self):
        return deepcopy(getattr(self, "_origin_transform", getattr(self, "start_transform", None)))

    @property
    def origin_timestamp_us(self):
        return getattr(self, "_origin_timestamp_us", getattr(self, "start_timestamp_us", None))

    @property
    def origin_bounds(self):
        return getattr(self, "_origin_bounds", getattr(self, "source_time_bounds", None))

    def load_scene(self, scene: Any) -> None:
        """Load a reconstruction once (no-op for the stub)."""

    @abstractmethod
    def render(self, pose: EgoPose, camera: str) -> Any:
        """Render one camera at ``pose`` — stateless given the loaded scene."""

    def close(self) -> None:
        pass


class StubRenderer(Renderer):
    """A placeholder renderer — no scene, no server: returns a black HWC uint8 frame. Lets the whole
    loop run and be debugged, and gives image-consuming stacks (e.g. Alpamayo) a valid tensor. Swap
    in :class:`NuRecRenderer` for real imagery without touching the backend or the AV stack.
    """

    def __init__(
        self, cameras: list[str] | None = None, height: int = 480, width: int = 640
    ) -> None:
        self.cameras = list(cameras) if cameras else ["camera_front"]
        self.height, self.width = height, width

    def render(self, pose: EgoPose, camera: str):
        import numpy as np

        return np.zeros((self.height, self.width, 3), dtype=np.uint8)


class NuRecBackend(CaptureBackend):
    """In-process neural-reconstruction world: own the ego dynamics + loop, render each frame.

    Mirrors ``CarlaBackend``'s shape (reset spawns/loads, step actuates+advances+senses) but keeps
    everything in one process behind a pluggable :class:`Renderer`.
    """

    def __init__(
        self,
        config: dict,
        renderer: Renderer | None = None,
        dynamics: EgoDynamics | None = None,
    ) -> None:
        super().__init__()
        self.config = deepcopy(config)
        self.simulator_version = str(self.config.get("simulator_version", "1"))
        self.renderer = renderer or StubRenderer()
        self.dynamics = dynamics or KinematicBicycle(**self.config.get("dynamics", {}))
        self.dt = float(self.config.get("dt", 0.05))
        if not math.isfinite(self.dt) or self.dt <= 0:
            raise ValueError("NuRec decision time step must be positive and finite")
        self.pose = EgoPose()
        self.frame = 0
        self._render_requests: dict[str, RenderRequest] = {}
        self._insertion_data: dict[tuple, Any] = {}
        self._source_end: float | None = None

    @property
    def supported_stages(self) -> frozenset[str]:
        return frozenset({
            "world.setup", "world.step.pre", "world.step.post", "render.pre", "render.post",
        })

    @property
    def command_modes(self) -> frozenset[str] | None:
        """Delegate command capabilities to the selected ego dynamics implementation."""
        return getattr(self.dynamics, "command_modes", None)

    @property
    def native(self) -> dict[str, Any]:
        """Native resources for explicit world changes or custom renderer operations."""
        return {
            "renderer": self.renderer,
            "dynamics": self.dynamics,
            "render_requests": deepcopy(self._render_requests),
        }

    def render_request(self, camera: str) -> RenderRequest:
        """Return the effective request that produced this camera's current observation.

        Both native camera names and configured output aliases are accepted. The returned
        request is independent of the stored geometry, including its ego pose.
        """
        aliases = self.config.get("camera_aliases", {})
        native_camera = next((key for key, alias in aliases.items() if alias == camera), camera)
        if native_camera not in self._render_requests:
            raise RuntimeError(f"No current render request for camera {camera!r}")
        return deepcopy(self._render_requests[native_camera])

    def ground_truth(self) -> WorldSnapshot:
        """Return the physical ego state, independent of any model-visible estimate."""
        return WorldSnapshot(self.pose.t, self.frame, deepcopy(self.pose), self.pose.speed)

    source_kinds = ("nurec",)
    can_render = True

    @property
    def decision_time_step(self):
        return self.dt

    @property
    def identity(self):
        return {**super().identity, "config": deepcopy(self.config),
                "renderer": type(self.renderer).__name__,
                "endpoint": getattr(self.renderer, "endpoint", None)}

    def _initialize_world(self, source, initialization, environment):
        if initialization is not None:
            raise ValueError("CARLA replay initialization does not apply to NuRec")
        if environment:
            raise ValueError(f"Unsupported NuRec environment overrides: {sorted(environment)}")
        self._render_requests.clear()
        self._insertion_data.clear()
        self._source_end = None
        config = deepcopy(self.config)
        if source is not None:
            scene = source.payload["scene"]
            import numpy as np

            pose = np.asarray(scene.ego.pose, dtype=float)
            config["scene"] = scene.source.get("scene_id")
            config["ego0"] = {
                "x": float(pose[0, 3]), "y": float(pose[1, 3]),
                "yaw": math.atan2(pose[1, 0], pose[0, 0]), "speed": scene.ego.speed,
            }
            self._source_end = source.payload["terminal_timestamp"] - source.start.timestamp
            self.initialization_report = {
                "kind": "recorded", "source": asdict(source.start),
                "time_step": source.payload.get("time_step"),
                "interpolation": source.payload.get("interpolation"),
                "terminal_timestamp": source.payload["terminal_timestamp"], "pre_roll_steps": 0,
            }
        reset_origin = getattr(self.renderer, "reset_origin", None)
        if callable(reset_origin):
            reset_origin(source)
        elif source is not None:
            raise TypeError("Source initialization requires renderer.reset_origin(SourceData)")
        self.renderer.load_scene(config.get("scene"))
        self.pose = EgoPose(**config.get("ego0", {}))
        self.frame = 0

    def _advance_native(self, control):
        if self._source_end is not None and round((self.pose.t + self.dt) * 1e6) > round(self._source_end * 1e6):
            raise ValueError("NuRec step exceeds terminal source support")
        self._render_requests.clear()
        self.pose = self.dynamics.step(self.pose, control, self.dt)
        self.frame += 1

    def render(self, observation, request):
        from avsectester.datasets.nurec import NuRecDataset
        from avsectester.perturbations.image import read_camera_rgb, write_camera_rgb
        from avsectester.services import RenderResult

        scene = request.scene if request.scene is not None else (
            self._source.payload["scene"] if self._source is not None else None)
        if scene is None:
            raise ValueError("NuRec insertion rendering requires source scene geometry")
        key = (scene.source["usdz_path"], scene.source["sensor"], request.sensor_id,
               scene.source.get("time_step", 0.1))
        if key not in self._insertion_data:
            self._insertion_data[key] = NuRecDataset(
                [key[0]], sensor=key[1], camera=key[2], time_step=key[3])
        data = self._insertion_data[key]
        adapter = self._insertion_renderer(scene, data, request.insertions,
                                          request.roles, request.compositor)
        sample = read_camera_rgb(observation, request.sensor_id)
        if sample is None:
            raise ValueError(f"Missing NuRec camera {request.sensor_id!r}")
        key, image = sample
        result = write_camera_rgb(observation, key, adapter(observation, image))
        return RenderResult(request.frame, result, {}, deepcopy(adapter.evidence),
                            deepcopy(adapter.resolved))

    def _insertion_renderer(self, scene, data, insertions, bindings=None, compositor=None):
        """Configure a host-bound insertion callable without connecting to the renderer.

        ``bindings`` accepts a selected match's role-to-track-ID tuples. Actor poses are sampled at
        the effective render timestamp and camera pose. Model-visible time and localization do
        not move the geometry. An explicit render-viewpoint change updates camera geometry while
        the victim's physical pose still comes from backend ground truth.
        """
        backend = self
        import numpy as np
        from avsectester.insertion import ActorPose
        from avsectester.rendering.types import InsertionGeometry
        from avsectester.rendering.visibility import CuboidVisibilityEstimator

        renderer = backend.renderer
        objects = {obj.track_id: obj for obj in scene.objects}
        selected = {
            role: tuple(objects[track_id] for track_id in ids)
            for role, ids in (bindings or {}).items()
        }
        from avsectester.scenarios.context import FilterContext
        from avsectester.rendering.cameras import FThetaCamera as Lens

        aliases = FilterContext(scene, bindings=selected).actor_aliases
        local_center = np.eye(4)
        local_center[:3, 3] = scene.ego.center
        path = scene.source["usdz_path"]

        def geometry(observation):
            pose = backend.ground_truth().vehicle_state
            request = backend.render_request(data.sensor)
            adapter.camera = renderer.camera_model(request.camera)
            actors = data.actor_poses(path, renderer.timestamp_us(request.pose))
            for alias, track_id in aliases.items():
                if track_id in actors:
                    if alias in actors and alias != track_id:
                        raise ValueError(f"Role {alias!r} conflicts with an actor track ID")
                    actors[alias] = actors[track_id]
            victim = ActorPose(renderer.rig_transform(pose) @ local_center, scene.ego.extent)
            return InsertionGeometry(
                actors, victim, renderer.cam_from_world(request.pose, request.camera),
            )

        # Recorded intrinsics let the adapter be registered before reset. Each geometry
        # call then selects the loaded camera that produced the actual render request.
        recorded = scene.cameras[data.camera].model
        camera = Lens(
            cx=float(recorded.principal_point[0]), cy=float(recorded.principal_point[1]),
            angle_to_pixeldist=tuple(recorded.angle_to_pixeldist),
            width=recorded.width, height=recorded.height, max_angle=recorded.max_angle,
            linear_cde=tuple(recorded.linear_cde),
        )
        adapter = NuRecInsertions(
            insertions,
            camera,
            geometry,
            compositor,
            visibility_estimator=CuboidVisibilityEstimator(),
            camera_name=data.camera,
        )
        return adapter

    def _observe(self) -> Observation:
        from avsectester.runtime import RenderRequest

        aliases = self.config.get("camera_aliases", {})
        sensor_data = {}
        calibration = {aliases.get(key, key): value for key, value in self.renderer.calibration.items()}
        for camera in self.renderer.cameras:
            # A render request may use a false viewpoint without moving the physical ego.
            request = self._emit(
                "render.pre", RenderRequest(deepcopy(self.pose), camera), sensor=camera,
            )
            self._render_requests[camera] = deepcopy(request)
            if request.camera != camera:
                output_key = aliases.get(camera, camera)
                calibration.pop(output_key, None)
                if request.camera in self.renderer.calibration:
                    calibration[output_key] = self.renderer.calibration[request.camera]
            image = self.renderer.render(request.pose, request.camera)
            sensor_data[aliases.get(camera, camera)] = self._emit(
                "render.post", image, sensor=camera,
            )
        return self.copy_observation(Observation(
            t=self.pose.t,
            frame=self.frame,
            sensor_data=sensor_data,
            calibration=calibration,
            vehicle_state=self.pose,
            ego_speed=self.pose.speed,
        ))

    def close(self) -> None:
        self._capture = None
        self._world_initialized = False
        self.renderer.close()

    # -- checkpointing: the whole world state is a small picklable dict --------------------------

    def checkpoint(self) -> dict:
        return {"pose": asdict(self.pose), "frame": self.frame}

    def restore(self, ckpt: dict) -> None:
        self.pose = EgoPose(**ckpt["pose"])
        self.frame = ckpt["frame"]
        self._render_requests.clear()
        self._world_initialized = False
        self._capture = None
        self._capture = self._observe()
        self._world_initialized = True


def _compose_pose(a, b, pb):
    """SE(3) compose two protobuf poses: ``a @ b``. AlpaSim builds the camera's world pose as
    ``world_rig @ rig_to_camera`` (rig_to_camera = the camera's pose in the rig)."""
    aw, ax, ay, az = a.quat.w, a.quat.x, a.quat.y, a.quat.z
    bw, bx, by, bz = b.quat.w, b.quat.x, b.quat.y, b.quat.z
    q = pb.Quat(  # quaternion product a*b
        w=aw * bw - ax * bx - ay * by - az * bz,
        x=aw * bx + ax * bw + ay * bz - az * by,
        y=aw * by - ax * bz + ay * bw + az * bx,
        z=aw * bz + ax * by - ay * bx + az * bw,
    )
    tx, ty, tz = (
        b.vec.x,
        b.vec.y,
        b.vec.z,
    )  # rotate b's translation by a's rotation: v + 2w(q×v)+2(q×(q×v))
    cx, cy, cz = ay * tz - az * ty, az * tx - ax * tz, ax * ty - ay * tx
    ccx, ccy, ccz = ay * cz - az * cy, az * cx - ax * cz, ax * cy - ay * cx
    vec = pb.Vec3(
        x=a.vec.x + tx + 2 * (aw * cx + ccx),
        y=a.vec.y + ty + 2 * (aw * cy + ccy),
        z=a.vec.z + tz + 2 * (aw * cz + ccz),
    )
    return pb.Pose(vec=vec, quat=q)


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

    start_timestamp_us: int | None = None
    start_transform: Any = None
    source_time_bounds: tuple[float, float] | None = None

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

    def start_pose(self):
        """Scene-frame (x, y, yaw) the backend should spawn the ego at. None until load_scene."""
        import math

        if self.origin_transform is not None:
            transform = self.origin_transform
            return {
                "x": float(transform[0][3]),
                "y": float(transform[1][3]),
                "yaw": math.atan2(transform[1][0], transform[0][0]),
            }
        if self._start_pose is None:
            return None
        p, q = self._start_pose.vec, self._start_pose.quat
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        return {"x": p.x, "y": p.y, "yaw": yaw}

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

    def camera_model(self, camera: str | None = None):
        """The rendered camera's lens model (:class:`~avsectester.rendering.cameras.FThetaCamera`)
        for world-anchored insertion, available after :meth:`load_scene`."""
        from avsectester.rendering.cameras import FThetaCamera

        return FThetaCamera.from_nurec(self._camera_specs.get(camera, self._spec))

    def rig_transform(self, pose: EgoPose):
        """Preserve the selected frame's height/tilt while evolving planar position and heading."""
        import numpy as np
        from avsectester.rendering.cameras import planar_rig_pose

        if self.origin_transform is None:
            z = self._start_pose.vec.z if self._start_pose is not None else 0.0
            return planar_rig_pose(pose.x, pose.y, pose.yaw, z)
        anchor = np.asarray(self.origin_transform, dtype=float)
        anchor_yaw = math.atan2(anchor[1, 0], anchor[0, 0])
        current = planar_rig_pose(pose.x, pose.y, pose.yaw, anchor[2, 3])
        initial_flat = planar_rig_pose(0, 0, anchor_yaw)
        current[:3, :3] = current[:3, :3] @ initial_flat[:3, :3].T @ anchor[:3, :3]
        return current

    def cam_from_world(self, pose: EgoPose, camera: str | None = None):
        """The same camera pose used by the render request, including selected-frame tilt."""
        import numpy as np
        from avsectester.rendering.cameras import pose_from_proto

        extrinsic = self._camera_poses.get(camera, self._rig_to_camera)
        return np.linalg.inv(self.rig_transform(pose) @ pose_from_proto(extrinsic))

    def camera_transform(self, world_from_rig, camera: str | None = None):
        """World-to-optical-camera transform for an explicit recorded rig pose."""
        import numpy as np
        from avsectester.rendering.cameras import pose_from_proto

        extrinsic = self._camera_poses.get(camera, self._rig_to_camera)
        return np.linalg.inv(np.asarray(world_from_rig) @ pose_from_proto(extrinsic))

    def render(self, pose: EgoPose, camera: str):
        """Render one RGB frame at the ego pose via a single stateless render_rgb call."""
        return self.render_at(self.rig_transform(pose), self.timestamp_us(pose), camera)

    def render_at(self, world_from_rig, timestamp_us: int, camera: str):
        """Render a recorded or user-provided SE(3) pose at an absolute scene timestamp."""
        if self.origin_bounds is not None:
            first, last = self.origin_bounds
            if not first - 1e-6 <= timestamp_us / 1e6 <= last + 1e-6:
                raise ValueError("NuRec render timestamp exceeds source support")

        import cv2
        import numpy as np
        from alpasim_grpc.v0 import common_pb2, sensorsim_pb2

        from scipy.spatial.transform import Rotation

        if self._stub is None:
            raise RuntimeError("Call load_scene before rendering")
        if camera not in self._camera_specs:
            raise ValueError(f"Camera {camera!r} was not loaded")
        spec = self._camera_specs[camera]
        transform = np.asarray(world_from_rig, dtype=float)
        qx, qy, qz, qw = Rotation.from_matrix(transform[:3, :3]).as_quat()
        rig = common_pb2.Pose(
            vec=common_pb2.Vec3(x=transform[0, 3], y=transform[1, 3], z=transform[2, 3]),
            quat=common_pb2.Quat(w=qw, x=qx, y=qy, z=qz),
        )
        cam = _compose_pose(rig, self._camera_poses[camera], common_pb2)
        frame_us = int(timestamp_us)
        req = sensorsim_pb2.RGBRenderRequest(
            scene_id=self._loaded_scene_id,
            resolution_h=spec.resolution_h,
            resolution_w=spec.resolution_w,
            camera_intrinsics=spec,
            frame_start_us=frame_us,
            frame_end_us=frame_us + 1,  # render at an instant: a non-empty [t, t+1) interval
            sensor_pose=sensorsim_pb2.PosePair(start_pose=cam, end_pose=cam),
            image_format=sensorsim_pb2.ImageFormat.JPEG,
            image_quality=self.image_quality,
            insert_ego_mask=False,
        )
        ret = self._stub.render_rgb(req)
        img = cv2.imdecode(np.frombuffer(ret.image_bytes, np.uint8), cv2.IMREAD_COLOR)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # HWC uint8 RGB for the AV stack


class NuRecInsertions(InsertionRenderer):
    """Insertion renderer with optional known-cuboid visibility for NuRec observations.

    ``geometry(observation)`` returns InsertionGeometry with actor/victim poses and the
    world-to-camera matrix. Use with ``frame_perturbation`` to change the model input.
    """

    def __init__(
        self,
        insertions,
        camera,
        geometry,
        compositor=None,
        visibility_estimator=None,
        camera_name="front",
    ):
        def evidence(observation, resolved, state):
            return {
                item.id: visibility_estimator.estimate(
                    item,
                    self.camera,
                    state.cam_from_world,
                    occluders=state.actors,
                    other_insertions=tuple(other for other in resolved if other.id != item.id),
                    camera_name=camera_name,
                )
                for item in resolved
            }

        super().__init__(
            insertions,
            camera,
            geometry,
            compositor,
            evidence_provider=evidence if visibility_estimator is not None else None,
        )
