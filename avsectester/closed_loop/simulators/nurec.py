"""Neural-reconstruction simulator with in-process ego dynamics and pluggable rendering.

``NuRecSimulator`` owns the control loop. ``StubRenderer`` supports offline use, while
``NuRecRenderer`` calls a separate SensorsimService for RGB images and exposes its native RPC
client for metadata access. Host-bound insertions use the shared geometric insertion renderer.
"""

from __future__ import annotations

import copy
import dataclasses
import math
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from itertools import pairwise
from typing import Any, ClassVar, Protocol, TYPE_CHECKING

from avsectester.closed_loop.datatypes import validate_command_modes
from avsectester.closed_loop.simulator import Simulator, SimulatorRuntime
from avsectester.closed_loop.datatypes import Control, Observation, WorldSnapshot

if TYPE_CHECKING:
    from avsectester.closed_loop.hooks import CameraRenderRequest
    from avsectester.closed_loop.simulator import SensorView


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

    Maps a :class:`~avsectester.closed_loop.datatypes.Control` (throttle/steer/brake) to motion. Pure and in-process,
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

    def copy(self) -> Renderer:
        """A renderer with this configuration and no connection, loaded scene or run origin.

        Every simulator runtime renders through its own copy. Override this when a renderer
        holds resources that a shallow copy would share.
        """
        return copy.copy(self)

    def reset_origin(self, start=None):
        """Replace runtime origin without modifying configured renderer defaults."""
        self._origin_transform = deepcopy(getattr(self, "start_transform", None))
        self._origin_timestamp_us = getattr(self, "start_timestamp_us", None)
        self._origin_bounds = getattr(self, "recorded_time_bounds", None)
        if start is not None:
            self._origin_transform = deepcopy(start.payload["scene"].ego.pose)
            self._origin_timestamp_us = round(start.frame.timestamp * 1e6)
            self._origin_bounds = (start.payload["first_timestamp"], start.payload["terminal_timestamp"])

    @property
    def origin_transform(self):
        return deepcopy(getattr(self, "_origin_transform", getattr(self, "start_transform", None)))

    @property
    def origin_timestamp_us(self):
        return getattr(self, "_origin_timestamp_us", getattr(self, "start_timestamp_us", None))

    @property
    def origin_bounds(self):
        return getattr(self, "_origin_bounds", getattr(self, "recorded_time_bounds", None))

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
    in :class:`NuRecRenderer` for real imagery without touching the simulator or the AVStack.
    """

    def __init__(
        self, cameras: list[str] | None = None, height: int = 480, width: int = 640
    ) -> None:
        self.cameras = list(cameras) if cameras else ["camera_front"]
        self.height, self.width = height, width

    def render(self, pose: EgoPose, camera: str):
        import numpy as np

        return np.zeros((self.height, self.width, 3), dtype=np.uint8)


class NuRecSimulator(Simulator):
    """A NuRec neural reconstruction with in-process ego dynamics and a pluggable renderer.

    ``renderer`` is a configuration template that is never connected itself: every runtime
    renders through its own copy (``Renderer.copy``), with its own connection, loaded scene and
    origin. ``dynamics`` is also a template: every runtime integrates with its own copy, so a
    handler that changes it, for example ``max_brake``, changes only that runtime.
    """

    dataset_kinds = ("NuRecDataset",)
    can_render = True

    def __init__(
        self,
        config: dict,
        renderer: Renderer | None = None,
        dynamics: EgoDynamics | None = None,
    ) -> None:
        self.config = deepcopy(config)
        self.simulator_version = str(self.config.get("simulator_version", "1"))
        self.renderer = renderer or StubRenderer()
        self.dynamics = dynamics or KinematicBicycle(**self.config.get("dynamics", {}))
        self.dt = float(self.config.get("dt", 0.05))
        if not math.isfinite(self.dt) or self.dt <= 0:
            raise ValueError("NuRec decision time step must be positive and finite")

    def create_runtime(self, seed: int | None = None) -> NuRecSimulatorRuntime:
        """A new live world; NuRec rendering and ego dynamics are deterministic, so ``seed`` is unused."""
        return NuRecSimulatorRuntime(self)

    @property
    def supported_stages(self) -> frozenset[str]:
        return frozenset({
            "simulator.setup", "simulator.step.pre", "simulator.step.post", "render.pre", "render.post",
        })

    @property
    def command_modes(self) -> frozenset[str] | None:
        """Delegate command capabilities to the selected ego dynamics implementation."""
        return getattr(self.dynamics, "command_modes", None)

    @property
    def decision_time_step(self):
        return self.dt

    @property
    def identity(self):
        return {**super().identity, "config": deepcopy(self.config),
                "renderer": type(self.renderer).__name__,
                "endpoint": getattr(self.renderer, "endpoint", None)}


class NuRecSimulatorRuntime(SimulatorRuntime):
    """One live NuRec world: the ego pose and dynamics loop, and its own renderer connection."""

    def __init__(self, simulator: NuRecSimulator) -> None:
        super().__init__(simulator)
        self.config = deepcopy(simulator.config)
        # Duck-typed renderers without ``copy`` are copied shallowly, like Renderer.copy.
        copy_renderer = getattr(simulator.renderer, "copy", None)
        self.renderer = copy_renderer() if callable(copy_renderer) else copy.copy(simulator.renderer)
        self.dynamics = deepcopy(simulator.dynamics)
        self.dt = simulator.dt
        self.pose = EgoPose()
        self.frame = 0
        self._render_requests: dict[str, CameraRenderRequest] = {}
        # Recorded actor tracks, by archive; recorded files are immutable.
        self._track_datasets: dict[tuple, Any] = {}
        self._actor_states: tuple[int, dict] | None = None
        self._recorded_end: float | None = None

    @property
    def native(self) -> dict[str, Any]:
        """Native resources for explicit world changes or custom renderer operations."""
        return {
            "renderer": self.renderer,
            "dynamics": self.dynamics,
            "render_requests": deepcopy(self._render_requests),
        }

    def render_request(self, camera: str) -> CameraRenderRequest:
        """Return the effective request that produced this camera's current observation.

        Both native camera names and configured output aliases are accepted. The returned
        request is independent of the stored geometry, including its ego pose.
        """
        aliases = self.config.get("camera_aliases", {})
        native_camera = next((key for key, alias in aliases.items() if alias == camera), camera)
        if native_camera not in self._render_requests:
            raise RuntimeError(f"No current render request for camera {camera!r}")
        return deepcopy(self._render_requests[native_camera])

    def _state(self) -> WorldSnapshot:
        """Return the physical ego state, independent of any model-visible estimate.

        Other actors follow their recorded tracks, so their states come from the start's
        recording at the current physical time; without a start there are none.
        """
        return WorldSnapshot(self.pose.t, self.frame, deepcopy(self.pose), self.pose.speed,
                             deepcopy(self._recorded_actors()))

    def _tracks(self):
        from avsectester.datasets.nurec import NuRecDataset

        scene = self._start.payload["scene"]
        key = (scene.source["usdz_path"], scene.source["camera_id"], scene.source.get("time_step", 0.1))
        if key not in self._track_datasets:
            self._track_datasets[key] = NuRecDataset([key[0]], camera_id=key[1], time_step=key[2])
        return scene, self._track_datasets[key]

    def _recorded_actors(self) -> dict:
        if self._start is None or "usdz_path" not in self._start.payload["scene"].source:
            return {}
        if self._actor_states is None or self._actor_states[0] != self.frame:
            scene, dataset = self._tracks()
            timestamp = round((self._start.frame.timestamp + self.pose.t) * 1e6)
            self._actor_states = (self.frame, dataset.actor_states(scene.source["usdz_path"], timestamp))
        return self._actor_states[1]

    def _initialize_world(self, start, initialization, environment):
        if initialization is not None:
            raise ValueError("CARLA replay initialization does not apply to NuRec")
        if environment:
            raise ValueError(f"Unsupported NuRec environment overrides: {sorted(environment)}")
        self._render_requests.clear()
        self._actor_states = None
        self._recorded_end = None
        config = deepcopy(self.config)
        if start is not None:
            scene = start.payload["scene"]
            import numpy as np

            pose = np.asarray(scene.ego.pose, dtype=float)
            config["scene"] = scene.source.get("scene_id")
            config["ego0"] = {
                "x": float(pose[0, 3]), "y": float(pose[1, 3]),
                "yaw": math.atan2(pose[1, 0], pose[0, 0]), "speed": scene.ego.speed,
            }
            self._recorded_end = start.payload["terminal_timestamp"] - start.frame.timestamp
            self.initialization_report = {
                "kind": "recorded", "start_frame": asdict(start.frame),
                "time_step": start.payload.get("time_step"),
                "interpolation": start.payload.get("interpolation"),
                "terminal_timestamp": start.payload["terminal_timestamp"], "pre_roll_steps": 0,
            }
        reset_origin = getattr(self.renderer, "reset_origin", None)
        if callable(reset_origin):
            reset_origin(start)
        elif start is not None:
            raise TypeError("Source initialization requires renderer.reset_origin(ScenarioStart)")
        self.renderer.load_scene(config.get("scene"))
        self.pose = EgoPose(**config.get("ego0", {}))
        self.frame = 0

    def _advance_native(self, control):
        if self._recorded_end is not None and round((self.pose.t + self.dt) * 1e6) > round(self._recorded_end * 1e6):
            raise ValueError("NuRec step exceeds terminal recorded support")
        self._render_requests.clear()
        self.pose = self.dynamics.step(self.pose, control, self.dt)
        self.frame += 1

    def sensor_view(self, sensor: str) -> SensorView:
        """View of the current capture; recorded actors come from the recorded clip."""
        if self._start is None:
            raise ValueError("NuRec sensor views require a scenario start with recorded actors")
        scene, dataset = self._tracks()
        return nurec_sensor_view(self, scene, dataset, sensor)

    def _observe(self) -> Observation:
        from avsectester.closed_loop.hooks import CameraRenderRequest

        aliases = self.config.get("camera_aliases", {})
        sensor_data = {}
        calibration = {aliases.get(key, key): value for key, value in self.renderer.calibration.items()}
        for camera in self.renderer.cameras:
            # A render request may use a false viewpoint without moving the physical ego.
            request = self._emit(
                "render.pre", CameraRenderRequest(deepcopy(self.pose), camera), sensor=camera,
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
    recorded_time_bounds: tuple[float, float] | None = None

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

    def start_pose(self):
        """Scene-frame (x, y, yaw) the simulator should spawn the ego at. None until load_scene."""
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
        """The rendered camera's lens model (:class:`~avsectester.geometry.cameras.FThetaCamera`)
        for world-anchored insertion, available after :meth:`load_scene`."""
        from avsectester.geometry.cameras import FThetaCamera

        return FThetaCamera.from_nurec(self._camera_specs.get(camera, self._spec))

    def rig_transform(self, pose: EgoPose):
        """Preserve the selected frame's height/tilt while evolving planar position and heading."""
        import numpy as np
        from avsectester.geometry.cameras import planar_rig_pose

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
        from avsectester.geometry.cameras import pose_from_proto

        extrinsic = self._camera_poses.get(camera, self._rig_to_camera)
        return np.linalg.inv(self.rig_transform(pose) @ pose_from_proto(extrinsic))

    def camera_transform(self, world_from_rig, camera: str | None = None):
        """World-to-optical-camera transform for an explicit recorded rig pose."""
        import numpy as np
        from avsectester.geometry.cameras import pose_from_proto

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


def nurec_sensor_view(simulator_runtime, scene, dataset, sensor=None, data=None):
    """Build the sensor view of one NuRec camera capture.

    Actors are sampled at the effective render timestamp and the camera at the effective render
    pose, so an explicit render-viewpoint change moves both. The ego cuboid comes from the true
    physical pose; model-visible time and localization never move the geometry. ``dataset``
    provides ``actor_poses(path, timestamp_us)`` for the recorded clip of ``scene``. ``data``
    replaces the captured image, for callers that render into an already modified frame.
    """
    import numpy as np

    from avsectester.geometry.poses import ActorPose
    from avsectester.closed_loop.datatypes import read_camera_rgb
    from avsectester.closed_loop.simulator import SensorView

    renderer = simulator_runtime.renderer
    aliases = getattr(simulator_runtime, "config", {}).get("camera_aliases", {})
    native_camera = sensor if sensor is not None else dataset.camera_id
    native_camera = next((key for key, alias in aliases.items() if alias == native_camera),
                         native_camera)
    output_key = aliases.get(native_camera, native_camera)
    request = simulator_runtime.render_request(native_camera)
    truth = simulator_runtime.state()
    timestamp = renderer.timestamp_us(request.pose)
    local_center = np.eye(4)
    local_center[:3, 3] = scene.ego.center
    ego = ActorPose(renderer.rig_transform(truth.vehicle_state) @ local_center, scene.ego.extent)
    if data is None:
        captured = read_camera_rgb(simulator_runtime.observe(), output_key)
        if captured is None:
            raise ValueError(f"Missing NuRec camera {output_key!r}")
        data = captured[1]
    return SensorView(
        output_key, data, renderer.camera_model(request.camera),
        renderer.cam_from_world(request.pose, request.camera),
        dataset.actor_poses(scene.source["usdz_path"], timestamp), ego, truth.t, truth.frame,
        native={"render_request": request, "timestamp_us": timestamp},
    )
