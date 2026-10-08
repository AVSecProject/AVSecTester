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
from typing import Any, ClassVar

from avsectester.backend import WorldBackend
from avsectester.plane import Control, Observation
from avsectester.simulators.patch_insertion import InsertionRenderer


@dataclass
class EgoPose:
    """Ego world state we own and integrate (2-D kinematic dynamics)."""

    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0  # rad
    speed: float = 0.0  # m/s
    t: float = 0.0  # s


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

    def step(self, pose: EgoPose, control: Control, dt: float) -> EgoPose:
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

    def step(self, pose: EgoPose, control: Control, dt: float) -> EgoPose:
        traj = control.trajectory or []
        if not traj:
            speed = max(0.0, pose.speed - self.decel * dt)
            return EgoPose(pose.x, pose.y, pose.yaw, speed, pose.t + dt)
        dx, dy = self._interp_rig_xy(traj, (pose.t + dt) * 1e6)
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
    def _interp_rig_xy(traj: list, target_us: float) -> tuple[float, float]:
        pts = [(t_us, xyz[0], xyz[1]) for xyz, _quat, t_us in traj]
        if target_us <= pts[0][0]:
            return pts[0][1], pts[0][2]
        for (t0, x0, y0), (t1, x1, y1) in pairwise(pts):
            if target_us <= t1:
                a = (target_us - t0) / (t1 - t0) if t1 > t0 else 0.0
                return x0 + a * (x1 - x0), y0 + a * (y1 - y0)
        return pts[-1][1], pts[-1][2]


class Renderer(ABC):
    """Renders sensor observations from the reconstructed scene at a given ego pose."""

    cameras: ClassVar[list[str]] = ["camera_front"]
    calibration: ClassVar[dict] = {}

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


class NuRecBackend(WorldBackend):
    """In-process neural-reconstruction world: own the ego dynamics + loop, render each frame.

    Mirrors ``CarlaBackend``'s shape (reset spawns/loads, step actuates+advances+senses) but keeps
    everything in one process behind a pluggable :class:`Renderer`.
    """

    def __init__(
        self,
        config: dict,
        renderer: Renderer | None = None,
        dynamics: KinematicBicycle | None = None,
    ) -> None:
        self.config = deepcopy(config)
        self.renderer = renderer or StubRenderer()
        self.dynamics = dynamics or KinematicBicycle(**self.config.get("dynamics", {}))
        self.dt = float(self.config.get("dt", 0.05))
        self.pose = EgoPose()
        self.frame = 0

    def reset(self) -> Observation:
        self.renderer.load_scene(self.config.get("scene"))
        self.pose = EgoPose(**self.config.get("ego0", {}))
        self.frame = 0
        return self._observe()

    def step(self, control: Control) -> Observation:
        self.pose = self.dynamics.step(self.pose, control, self.dt)  # shared physics, in-process
        self.frame += 1
        return self._observe()

    def _observe(self) -> Observation:
        aliases = self.config.get("camera_aliases", {})
        sensor_data = {
            aliases.get(cam, cam): self.renderer.render(self.pose, cam)
            for cam in self.renderer.cameras
        }
        return Observation(
            t=self.pose.t,
            frame=self.frame,
            sensor_data=sensor_data,
            calibration={aliases.get(k, k): v for k, v in self.renderer.calibration.items()},
            vehicle_state=self.pose,
            ego_speed=self.pose.speed,
        )

    def close(self) -> None:
        self.renderer.close()

    # -- checkpointing: the whole world state is a small picklable dict --------------------------

    def checkpoint(self) -> dict:
        return {"pose": asdict(self.pose), "frame": self.frame}

    def restore(self, ckpt: dict) -> None:
        self.pose = EgoPose(**ckpt["pose"])
        self.frame = ckpt["frame"]


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
            self.scene_id = want
        elif want and len(matches) == 1:
            self.scene_id = matches[0]
        elif want or not available:
            raise ValueError(f"Scene {want!r} did not resolve to exactly one renderer scene")
        else:
            self.scene_id = available[0]
        cams = self._stub.get_available_cameras(
            sensorsim_pb2.AvailableCamerasRequest(scene_id=self.scene_id)
        )
        native_cameras = {c.logical_id: c for c in cams.available_cameras}
        for name in self.cameras:
            if name not in native_cameras:
                raise ValueError(f"Camera {name!r} is not available in {self.scene_id!r}")
        self._camera_specs = {name: native_cameras[name].intrinsics for name in self.cameras}
        self._camera_poses = {name: native_cameras[name].rig_to_camera for name in self.cameras}
        self._spec = self._camera_specs[self.cameras[0]]
        self._rig_to_camera = self._camera_poses[self.cameras[0]]
        poses = (
            self._stub.get_available_trajectories(
                sensorsim_pb2.AvailableTrajectoriesRequest(scene_id=self.scene_id)
            )
            .available_trajectories[0]
            .trajectory.poses
        )
        self._start_pose = poses[0].pose  # scene world-frame origin for the ego
        self._t0 = (
            poses[0].timestamp_us if self.start_timestamp_us is None else self.start_timestamp_us
        )

    def start_pose(self):
        """Scene-frame (x, y, yaw) the backend should spawn the ego at. None until load_scene."""
        import math

        if self.start_transform is not None:
            transform = self.start_transform
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
        """The rendered camera's lens model (:class:`~avsectester.simulators.camera_models.FThetaCamera`)
        for world-anchored insertion, available after :meth:`load_scene`."""
        from avsectester.simulators.camera_models import FThetaCamera

        return FThetaCamera.from_nurec(self._camera_specs.get(camera, self._spec))

    def rig_transform(self, pose: EgoPose):
        """Preserve the selected frame's height/tilt while evolving planar position and heading."""
        import numpy as np
        from avsectester.simulators.camera_models import planar_rig_pose

        if self.start_transform is None:
            z = self._start_pose.vec.z if self._start_pose is not None else 0.0
            return planar_rig_pose(pose.x, pose.y, pose.yaw, z)
        anchor = np.asarray(self.start_transform, dtype=float)
        anchor_yaw = math.atan2(anchor[1, 0], anchor[0, 0])
        current = planar_rig_pose(pose.x, pose.y, pose.yaw, anchor[2, 3])
        initial_flat = planar_rig_pose(0, 0, anchor_yaw)
        current[:3, :3] = current[:3, :3] @ initial_flat[:3, :3].T @ anchor[:3, :3]
        return current

    def cam_from_world(self, pose: EgoPose, camera: str | None = None):
        """The same camera pose used by the render request, including selected-frame tilt."""
        import numpy as np
        from avsectester.simulators.camera_models import pose_from_proto

        extrinsic = self._camera_poses.get(camera, self._rig_to_camera)
        return np.linalg.inv(self.rig_transform(pose) @ pose_from_proto(extrinsic))

    def camera_transform(self, world_from_rig, camera: str | None = None):
        """World-to-optical-camera transform for an explicit recorded rig pose."""
        import numpy as np
        from avsectester.simulators.camera_models import pose_from_proto

        extrinsic = self._camera_poses.get(camera, self._rig_to_camera)
        return np.linalg.inv(np.asarray(world_from_rig) @ pose_from_proto(extrinsic))

    def render(self, pose: EgoPose, camera: str):
        """Render one RGB frame at the ego pose via a single stateless render_rgb call."""
        return self.render_at(self.rig_transform(pose), self.timestamp_us(pose), camera)

    def render_at(self, world_from_rig, timestamp_us: int, camera: str):
        """Render a recorded or user-provided SE(3) pose at an absolute scene timestamp."""

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
            scene_id=self.scene_id,
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

    ``geometry(observation)`` provides stable actor world poses, the victim pose and the
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
            actors, _victim, camera_transform = state
            return {
                item.id: visibility_estimator.estimate(
                    item,
                    camera,
                    camera_transform,
                    occluders=actors,
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
