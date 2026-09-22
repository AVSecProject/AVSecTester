"""In-process neural-reconstruction world backend (NuRec / OmniDreams) — we own the loop.

Unlike AlpaSim's gRPC runtime (which owns the loop and calls the driver back), this
:class:`~avsectester.backend.WorldBackend` owns the closed loop *in process*, exactly like
``CarlaBackend``: ``step()`` integrates the ego dynamics and renders the camera from the
reconstructed scene. The only thing that may cross a boundary is the **stateless** per-frame render
(``render(pose) -> image``) — the loop, ego state, and dynamics stay in one process. That makes it
easy to set breakpoints in ``step()``, inspect the frame + state, and checkpoint the world as a plain
picklable dict (``checkpoint``/``restore``).

The renderer is pluggable: :class:`StubRenderer` (default) needs nothing and returns a deterministic
placeholder, so the whole loop runs and is debugged with no server; :class:`NuRecRenderer` wraps a
real NuRec ``SensorsimService`` (one stateless ``render_rgb`` call per frame) and is wired later.
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


@dataclass
class EgoPose:
    """Ego world state we own and integrate (2-D kinematic; enough for a closed-loop drive)."""

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


def ftheta_project(pts_cam: Any, principal_point: tuple, angle_to_pixeldist_poly: list):
    """Project camera-frame points through an NVIDIA f-theta (fisheye) camera to pixels.

    f-theta maps the ray's angle from the optical axis (+z) to a radial pixel distance via a forward
    polynomial: ``r(theta) = sum(a[i] * theta**i)``; the pixel is that radius from the principal point
    along the ray's image-plane direction. Returns ``(px (N,2), z)`` — ``z`` (camera depth) is returned
    so callers can cull points behind the camera. Validated against a live NuRec render.
    """
    import numpy as np

    p = np.asarray(pts_cam, dtype=np.float64)
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    rho = np.hypot(x, y)
    theta = np.arctan2(rho, z)  # angle from the +z optical axis
    r = sum(c * theta**i for i, c in enumerate(angle_to_pixeldist_poly))
    ux = np.where(rho > 1e-9, x / rho, 0.0)
    uy = np.where(rho > 1e-9, y / rho, 0.0)
    px = np.stack([principal_point[0] + r * ux, principal_point[1] + r * uy], axis=1)
    return px, z


def ftheta_rays(width: int, height: int, principal_point: tuple, pixeldist_to_angle_poly: list):
    """Per-pixel **unit ray directions** (camera frame) for an f-theta camera — the inverse of
    :func:`ftheta_project`. Uses the backward polynomial ``theta(r) = sum(b[i]·r^i)`` (pixel radius ->
    angle from the +z axis); the ray is ``(sinθ·cosφ, sinθ·sinφ, cosθ)`` at azimuth φ. ``(H,W,3)``."""
    import numpy as np

    uu, vv = np.meshgrid(np.arange(width), np.arange(height))
    du, dv = uu - principal_point[0], vv - principal_point[1]
    r = np.hypot(du, dv)
    theta = sum(b * r**i for i, b in enumerate(pixeldist_to_angle_poly))
    phi = np.arctan2(dv, du)
    st = np.sin(theta)
    return np.stack([st * np.cos(phi), st * np.sin(phi), np.cos(theta)], axis=-1)


def _quat_to_R(w: float, x: float, y: float, z: float):
    import numpy as np

    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _ego_world_to_cam(renderer: NuRecRenderer, pose):
    """(world_to_cam rotation Rᵀ, camera-centre translation) for the ego ``pose`` — shared by the NuRec
    projectors. Composes the ego rig pose with ``rig_to_camera`` (camera->world), then inverts."""
    import numpy as np

    r2c = renderer._rig_to_camera
    r2c_R = _quat_to_R(r2c.quat.w, r2c.quat.x, r2c.quat.y, r2c.quat.z)
    r2c_t = np.array([r2c.vec.x, r2c.vec.y, r2c.vec.z])
    z0 = renderer._start_pose.vec.z if renderer._start_pose is not None else 0.0
    c, s = math.cos(pose.yaw), math.sin(pose.yaw)
    rig_R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    rig_t = np.array([pose.x, pose.y, z0])
    world_cam_R = rig_R @ r2c_R          # camera -> world
    world_cam_t = rig_t + rig_R @ r2c_t
    return world_cam_R.T, world_cam_t     # X_cam = R.T @ (X_world - t)


def nurec_road_decal(renderer: NuRecRenderer, observation: Observation, ahead: float = 4.5,
                     lateral: float = 0.0, size_u: float = 1.6, size_v: float = 0.9,
                     yaw: float = 0.0):
    """Build ``(xyz, decal)`` for a **per-pixel road decal** ahead of the ego (for ``decal_project``).

    NuRec exposes no depth sensor, but the road plane is known exactly (``z=0`` in the rig frame, camera
    ~1.6 m above it). So every pixel's f-theta ray is intersected with that plane to get its true 3-D
    ground point — the patch then conforms to the road per pixel with correct fisheye curvature and
    foreshortening (a homography can't, and is wrong for a fisheye). ``size_u`` runs along the lane,
    ``size_v`` across it. Returns ``(xyz (H,W,3), DecalFrame)`` or None before the scene is loaded.
    """
    import numpy as np

    from avsectester.attacks.patch_composite import DecalFrame

    spec = getattr(renderer, "_spec", None)
    if spec is None or getattr(renderer, "_rig_to_camera", None) is None:
        return None
    fp = spec.ftheta_param
    pose = observation.vehicle_state
    Rt, cam_t = _ego_world_to_cam(renderer, pose)

    # decal frame on the road (world): centre ahead (+ lateral) of the ego, axes along/across the lane.
    fwd = np.array([math.cos(pose.yaw), math.sin(pose.yaw), 0.0])
    left = np.array([-math.sin(pose.yaw), math.cos(pose.yaw), 0.0])
    u_w = math.cos(yaw) * fwd + math.sin(yaw) * left  # in-plane "along" direction, optionally turned
    o_w = np.array([pose.x, pose.y, 0.0]) + ahead * fwd + lateral * left
    up_w = np.array([0.0, 0.0, 1.0])

    def to_cam(pt):
        return Rt @ (pt - cam_t)

    o_c = to_cam(o_w)
    u_c = Rt @ u_w  # directions: rotation only
    v_c = Rt @ left if yaw == 0 else Rt @ (-math.sin(yaw) * fwd + math.cos(yaw) * left)
    n_c = Rt @ up_w
    decal = DecalFrame(origin=o_c, u_axis=u_c, v_axis=v_c, normal=n_c,
                       size_u=size_u, size_v=size_v, thickness=0.05)

    rays = ftheta_rays(spec.resolution_w, spec.resolution_h,
                       (fp.principal_point_x, fp.principal_point_y), list(fp.pixeldist_to_angle_poly))
    denom = rays @ n_c
    with np.errstate(divide="ignore", invalid="ignore"):
        tdist = (o_c @ n_c) / denom          # ray-plane intersection distance
    tdist = np.where(denom < -1e-6, tdist, -1.0)  # only rays pointing at the ground (denom<0) hit it
    xyz = tdist[..., None] * rays              # camera-frame ground points (z<=0 rows -> behind, culled)
    return xyz, decal


def nurec_panel_quad(renderer: NuRecRenderer, ahead: float = 10.0, half_w: float = 0.65,
                     z_lo: float = 0.7, z_hi: float = 1.7, yaw: float = 0.5):
    """Return ``quad_of(obs) -> (4,2) px | None`` for a fixed world panel ahead of the ego.

    The NuRec analog of :func:`avsectester.simulators.carla.lead_rear_quad`: since the render API
    exposes no actor 3-D boxes, the patch target is a **virtual panel** planted in the reconstructed
    scene ``ahead`` metres in front of the ego start (a billboard / stand-in lead surface), a modest
    ``half_w`` × (``z_hi`` − ``z_lo``) poster turned ``yaw`` radians about the vertical so it is seen
    obliquely — the projection is then a foreshortened trapezoid that reads as a real angled surface,
    not a fronto-parallel sticker. Each frame it composes the ego rig pose with the camera extrinsic,
    transforms the panel corners world->camera, and projects through the f-theta model. Reads the
    renderer's spec lazily (populated at reset).
    """
    import numpy as np

    from avsectester.attacks.patch_composite import order_quad

    def quad_of(observation: Observation) -> Any:
        r2c = getattr(renderer, "_rig_to_camera", None)
        spec = getattr(renderer, "_spec", None)
        if r2c is None or spec is None:
            return None
        fp = spec.ftheta_param
        pp = (fp.principal_point_x, fp.principal_point_y)
        poly = list(fp.angle_to_pixeldist_poly)
        rt, cam_t = _ego_world_to_cam(renderer, observation.vehicle_state)
        # world panel ahead of the start pose (rig +x fwd, +y left, +z up), turned by `yaw` about the
        # vertical so the plane is seen obliquely -> a foreshortened (trapezoidal) warp that reads as a
        # real angled surface rather than a fronto-parallel sticker. u = in-plane horizontal direction.
        u = np.array([math.sin(yaw), math.cos(yaw), 0.0])
        panel = np.array([[ahead, 0.0, z_hi] - half_w * u, [ahead, 0.0, z_hi] + half_w * u,
                          [ahead, 0.0, z_lo] + half_w * u, [ahead, 0.0, z_lo] - half_w * u])
        pc = (panel - cam_t) @ rt.T                 # world->camera: Rᵀ(P - t) per corner
        px, z = ftheta_project(pc, pp, poly)
        if np.any(z <= 0.1):                        # any corner behind the camera -> skip
            return None
        return order_quad(px)

    return quad_of


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
        sensor_data = {cam: self.renderer.render(self.pose, cam) for cam in self.renderer.cameras}
        return Observation(
            t=self.pose.t,
            frame=self.frame,
            sensor_data=sensor_data,
            calibration=self.renderer.calibration,
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
    ``world_rig @ rig_to_camera`` (rig_to_camera = the camera's pose in the rig); we replicate it."""
    aw, ax, ay, az = a.quat.w, a.quat.x, a.quat.y, a.quat.z
    bw, bx, by, bz = b.quat.w, b.quat.x, b.quat.y, b.quat.z
    q = pb.Quat(  # quaternion product a*b
        w=aw * bw - ax * bx - ay * by - az * bz,
        x=aw * bx + ax * bw + ay * bz - az * by,
        y=aw * by - ax * bz + ay * bw + az * bx,
        z=aw * bz + ax * by - ay * bx + az * bw,
    )
    tx, ty, tz = b.vec.x, b.vec.y, b.vec.z  # rotate b's translation by a's rotation: v + 2w(q×v)+2(q×(q×v))
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
    inversion. Lazily imports the gRPC stubs; wire it against a running renderer + a loaded NuRec
    scene. Not exercised yet on this box (needs the nre-ga server + a scene reconstruction).
    """

    endpoint: str = "127.0.0.1:50051"
    scene_id: str | None = None
    cameras: list[str] = field(default_factory=lambda: ["camera_front_wide_120fov"])
    calibration: dict = field(default_factory=dict)
    image_quality: float = 95.0

    def __post_init__(self) -> None:
        self._stub = None
        self._spec = None  # CameraSpec for our camera (intrinsics), queried from the scene
        self._rig_to_camera = None  # camera pose in the rig (extrinsic), composed onto the ego pose
        self._start_pose = None  # scene-frame ego start pose (from the recorded trajectory)
        self._t0 = 0  # scene-clip start timestamp (us); render times map from our sim clock onto it

    def load_scene(self, scene: Any) -> None:
        """Connect to the nre-ga renderer and read the scene id, camera spec, and start pose/time."""
        import grpc
        from alpasim_grpc.v0 import common_pb2, sensorsim_pb2, sensorsim_pb2_grpc

        self._stub = sensorsim_pb2_grpc.SensorsimServiceStub(grpc.insecure_channel(self.endpoint))
        available = list(self._stub.get_available_scenes(common_pb2.Empty()).scene_ids)
        want = scene or self.scene_id
        self.scene_id = next((s for s in available if want and want in s), available[0])
        cams = self._stub.get_available_cameras(
            sensorsim_pb2.AvailableCamerasRequest(scene_id=self.scene_id)
        )
        cam = next(c for c in cams.available_cameras if c.logical_id == self.cameras[0])
        self._spec = cam.intrinsics
        self._rig_to_camera = cam.rig_to_camera  # extrinsic: camera pose in the rig
        poses = self._stub.get_available_trajectories(
            sensorsim_pb2.AvailableTrajectoriesRequest(scene_id=self.scene_id)
        ).available_trajectories[0].trajectory.poses
        self._start_pose = poses[0].pose  # scene world-frame origin for the ego
        self._t0 = poses[0].timestamp_us

    def start_pose(self):
        """Scene-frame (x, y, yaw) the backend should spawn the ego at; None until load_scene."""
        import math

        if self._start_pose is None:
            return None
        p, q = self._start_pose.vec, self._start_pose.quat
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        return {"x": p.x, "y": p.y, "yaw": yaw}

    def render(self, pose: EgoPose, camera: str):
        """Render one RGB frame at the ego pose via a single stateless render_rgb call."""
        # ego rig pose in the scene world frame -> common.Pose (2-D; z from the recorded start)
        import math

        import cv2
        import numpy as np
        from alpasim_grpc.v0 import common_pb2, sensorsim_pb2

        z = self._start_pose.vec.z if self._start_pose is not None else 0.0
        rig = common_pb2.Pose(
            vec=common_pb2.Vec3(x=pose.x, y=pose.y, z=z),
            quat=common_pb2.Quat(w=math.cos(pose.yaw / 2), x=0.0, y=0.0, z=math.sin(pose.yaw / 2)),
        )
        cam = _compose_pose(rig, self._rig_to_camera, common_pb2)  # world_cam = world_rig @ rig_to_camera
        frame_us = self._t0 + int(pose.t * 1e6)  # advance the dynamic scene with our sim clock
        req = sensorsim_pb2.RGBRenderRequest(
            scene_id=self.scene_id,
            resolution_h=self._spec.resolution_h,
            resolution_w=self._spec.resolution_w,
            camera_intrinsics=self._spec,
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
