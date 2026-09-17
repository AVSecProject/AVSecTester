"""AlpaSim as a :class:`~avsectester.backend.WorldBackend` — a neural-reconstruction world model.

AlpaSim (NVlabs/alpasim) is a closed-loop simulator whose runtime services (NuRec/OmniDreams
sensor rendering + controller + physics + traffic) are a gRPC **client** of the AV policy: each frame
the runtime streams camera frames and egomotion to an ``EgodriverService`` and calls
``drive() -> Trajectory``. That is exactly our sim<->stack contract at AlpaSim's wire boundary, so we
bridge it by **implementing ``EgodriverService`` as a relay** and inverting AlpaSim's push-callbacks
into our pull-based ``step()``:

    AlpaSim runtime ── submit_image_observation ─┐
                    ── submit_egomotion_observation ─┤ (stash latest)
                    ── drive(DriveRequest) ──────────┘→ assemble Observation → obs_queue
                                                       ← DriveResponse(Trajectory) ← act_queue ← Control

``reset()`` starts the relay gRPC server and the AlpaSim runtime (pointed at our endpoint) and
returns the first Observation (the runtime's first ``drive``); ``step(control)`` hands the control's
trajectory back as the ``DriveResponse`` and returns the next ``drive``'s Observation.

Mapping (grounded in ``alpasim_grpc/v0/{egodriver,common}.proto``):
  * ``RolloutCameraImage.camera_image`` (``image_bytes`` per ``logical_id``) -> ``Observation.sensor_data``
  * ``RolloutEgoTrajectory`` (poses) + ``DynamicState.linear_velocity`` -> ``vehicle_state`` / ``ego_speed``
  * scene camera calibration (intrinsics + rig extrinsics) -> ``Observation.calibration``
  * our ``Control.trajectory`` -> ``common.Trajectory`` (repeated ``PoseAtTime``) in ``DriveResponse``

The **relay + interface mapping are verified over real gRPC** against AlpaSim's own compiled protos,
using a fake runtime in place of the NuRec renderer (``scripts/alpasim_roundtrip.py``): a camera frame
+ egomotion in, the stack's trajectory out. What is not yet exercised here is the live neural runtime
(NuRec rendering + controller/physics + a real Alpamayo policy) -- ``_launch_runtime`` marks that
boundary. gRPC / ``alpasim_grpc`` imports are lazy, so this module loads without AlpaSim installed.
AlpaSim's policies output trajectories, so the matching AV box is a trajectory-output stack (e.g. an
Alpamayo ``AVStack``), not the lidar-modular one.
"""

from __future__ import annotations

import math
import queue
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from avsectester.backend import WorldBackend
from avsectester.plane import Control, Observation

_TERMINATE = object()  # sentinel put on the action queue to end the session


@dataclass
class EgoState:
    """The ego's self-knowledge extracted from AlpaSim egomotion — a light, avstack-free state."""

    position: tuple[float, float, float]
    orientation: tuple[float, float, float, float]  # quaternion (w, x, y, z)
    velocity: tuple[float, float, float]
    t_us: int


# --- pure mapping helpers (no gRPC / no alpasim import; unit-testable) ---------------------------


def ego_from_egomotion(rollout_ego_trajectory: Any) -> EgoState | None:
    """Extract the latest :class:`EgoState` from a ``RolloutEgoTrajectory`` (proto-like object).

    Uses the last pose of ``trajectory.poses`` and the last ``dynamic_states`` velocity. Returns
    None if no pose is present. Written against attribute access so it tests with simple fakes.
    """
    poses = list(getattr(getattr(rollout_ego_trajectory, "trajectory", None), "poses", []) or [])
    if not poses:
        return None
    last = poses[-1]
    vec, quat = last.pose.vec, last.pose.quat
    states = list(getattr(rollout_ego_trajectory, "dynamic_states", []) or [])
    vel = states[-1].linear_velocity if states else None
    velocity = (vel.x, vel.y, vel.z) if vel is not None else (0.0, 0.0, 0.0)
    return EgoState(
        position=(vec.x, vec.y, vec.z),
        orientation=(quat.w, quat.x, quat.y, quat.z),
        velocity=velocity,
        t_us=int(getattr(last, "timestamp_us", 0)),
    )


def speed_of(ego: EgoState | None) -> float:
    if ego is None:
        return 0.0
    vx, vy, vz = ego.velocity
    return math.sqrt(vx * vx + vy * vy + vz * vz)


def control_waypoints(control: Control) -> list[tuple[tuple[float, float, float], tuple, int]]:
    """Normalize a :class:`Control`'s trajectory into ``[((x,y,z), (w,x,y,z), t_us), ...]``.

    AlpaSim consumes a trajectory (the policy's intended path in the rig frame), so an AlpaSim-facing
    AVStack must set ``Control.trajectory`` to a sequence of waypoints; each may be
    ``((x,y,z), (w,x,y,z), t_us)`` or ``((x,y,z), t_us)`` (identity rotation assumed). Actuator-only
    controls (throttle/steer/brake with no trajectory) are rejected -- they don't fit AlpaSim.
    """
    traj = control.trajectory
    if not traj:
        raise ValueError(
            "AlpaSim expects a trajectory: set Control.trajectory (a trajectory-output AV stack, "
            "e.g. Alpamayo). Actuator-only controls (throttle/steer/brake) are not applicable."
        )
    waypoints = []
    for point in traj:
        if len(point) == 3:
            xyz, quat, t_us = point
        else:
            xyz, t_us = point
            quat = (1.0, 0.0, 0.0, 0.0)
        waypoints.append((tuple(map(float, xyz)), tuple(map(float, quat)), int(t_us)))
    return waypoints


class AlpaSimBackend(WorldBackend):
    """Drive AlpaSim through the interface by relaying its ``EgodriverService`` into ``run``'s loop.

    ``config`` selects the scene, cameras/renderer, and runtime endpoint (see
    ``configs/alpasim_scenario.yaml``). ``endpoint`` is where this relay serves the EgodriverService
    that the AlpaSim runtime connects to.
    """

    def __init__(
        self, config: dict, endpoint: str = "0.0.0.0:50100", launch_runtime: bool = True
    ) -> None:
        self.config = deepcopy(config)
        self.endpoint = endpoint
        # When False, an AlpaSim runtime is started/managed elsewhere and connects to `endpoint`
        # (e.g. wired as the driver-0 service, or a round-trip test) -- we only serve the relay.
        self._do_launch = launch_runtime
        self._obs_q: queue.Queue = queue.Queue(maxsize=1)
        self._act_q: queue.Queue = queue.Queue(maxsize=1)
        self._server = None
        self._runtime = None
        self._latest_images: dict[str, dict] = {}
        self._latest_ego: Any = None
        self._calibration: dict = {}

    # -- EgodriverService relay methods (invoked by the AlpaSim runtime on gRPC threads) ----------

    def start_session(self, request, context):
        self._calibration = self._load_calibration()
        return self._empty()

    def submit_image_observation(self, request, context):
        image = request.camera_image
        self._latest_images[image.logical_id] = {
            "image_bytes": image.image_bytes,
            "frame_start_us": int(image.frame_start_us),
            "frame_end_us": int(image.frame_end_us),
        }
        return self._empty()

    def submit_egomotion_observation(self, request, context):
        self._latest_ego = request
        return self._empty()

    def submit_route(self, request, context):
        return self._empty()

    def drive(self, request, context):
        """The world asks the driver to act: assemble an Observation, wait for our Control."""
        self._obs_q.put(self._assemble_observation(request))
        control = self._act_q.get()
        if control is _TERMINATE:
            return self._drive_response(terminate=True)
        return self._drive_response(trajectory=control_waypoints(control))

    def close_session(self, request, context):
        return self._empty()

    def _assemble_observation(self, drive_request) -> Observation:
        ego = ego_from_egomotion(self._latest_ego)
        return Observation(
            t=int(getattr(drive_request, "time_now_us", 0)) / 1e6,
            frame=int(getattr(drive_request, "time_query_us", 0)),
            sensor_data=dict(self._latest_images),  # {logical_id: {image_bytes, frame_*_us}}
            calibration=self._calibration,
            vehicle_state=ego,
            ego_speed=speed_of(ego),
        )

    # -- WorldBackend interface -------------------------------------------------------------------

    def reset(self) -> Observation:
        self._start_server()
        if self._do_launch:
            self._launch_runtime()  # the live AlpaSim runtime, pointed at self.endpoint
        return self._obs_q.get()  # blocks until the runtime's first drive()

    def step(self, control: Control) -> Observation:
        self._act_q.put(control)
        return self._obs_q.get()

    def close(self) -> None:
        try:
            self._act_q.put_nowait(_TERMINATE)
        except queue.Full:
            pass
        if self._runtime is not None:
            self._stop_runtime()
        if self._server is not None:
            self._server.stop(grace=1.0)
            self._server = None

    # -- live-integration boundary (needs the alpasim / alpasim_grpc packages + a NuRec scene) ----

    def _start_server(self) -> None:
        """Serve this relay as an EgodriverService the AlpaSim runtime connects to."""
        from concurrent import futures

        import grpc  # lazy: only needed for a live run
        from alpasim_grpc.v0 import egodriver_pb2_grpc  # generated from egodriver.proto

        servicer = _RelayServicer(self, egodriver_pb2_grpc.EgodriverServiceServicer)
        self._server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
        egodriver_pb2_grpc.add_EgodriverServiceServicer_to_server(servicer, self._server)
        self._server.add_insecure_port(self.endpoint)
        self._server.start()

    def _launch_runtime(self) -> None:
        """Start the AlpaSim runtime (sensorsim/controller/physics/traffic) for this scene, pointed
        at ``self.endpoint``. This is the live boundary — it requires the AlpaSim install, a NuRec
        reconstruction for the scene, and the gated NVIDIA models. Wire it to AlpaSim's runtime
        launcher (docs/TUTORIAL.md: docker-compose / run-on-slurm) with an egodriver endpoint of
        ``self.endpoint``; not runnable on a box without AlpaSim."""
        raise NotImplementedError(
            "Launch the AlpaSim runtime for this scene pointed at endpoint "
            f"{self.endpoint!r} (see AlpaSim docs/TUTORIAL.md). Requires the alpasim install, a "
            "NuRec scene, and NVIDIA driver models — not available on this host."
        )

    def _stop_runtime(self) -> None:
        pass

    def _load_calibration(self) -> dict:
        """Camera intrinsics + rig extrinsics for the scene (AlpaSim camera_catalog / usdz
        calibration). Filled at session start on a live run; empty here."""
        return {}

    # -- proto builders (lazy; only reached on a live run) ----------------------------------------

    def _empty(self):
        from alpasim_grpc.v0 import common_pb2

        return common_pb2.Empty()

    def _drive_response(self, trajectory=None, terminate: bool = False):
        from alpasim_grpc.v0 import common_pb2, egodriver_pb2

        traj = common_pb2.Trajectory()
        for (x, y, z), (qw, qx, qy, qz), t_us in trajectory or []:
            traj.poses.add(
                pose=common_pb2.Pose(
                    vec=common_pb2.Vec3(x=x, y=y, z=z),
                    quat=common_pb2.Quat(w=qw, x=qx, y=qy, z=qz),
                ),
                timestamp_us=t_us,
            )
        return egodriver_pb2.DriveResponse(trajectory=traj, terminate_session=terminate)


class _RelayServicer:
    """Adapts :class:`AlpaSimBackend`'s relay methods to a generated EgodriverServiceServicer.

    Built lazily (the base class comes from the generated gRPC stubs) so the module imports without
    ``alpasim_grpc``. Delegates each RPC to the backend.
    """

    def __new__(cls, backend: AlpaSimBackend, servicer_base):
        namespace = {
            name: (lambda self, req, ctx, _m=getattr(backend, name): _m(req, ctx))
            for name in (
                "start_session",
                "close_session",
                "submit_image_observation",
                "submit_egomotion_observation",
                "submit_route",
                "drive",
            )
        }
        return type("EgoDriverRelay", (servicer_base,), namespace)()
