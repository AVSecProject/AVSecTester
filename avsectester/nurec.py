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
    """A deterministic placeholder renderer — no scene, no server. Lets the loop run and be debugged.

    Returns a small dict standing in for a rendered frame; swap in :class:`NuRecRenderer` for real
    imagery without touching the backend or the AV stack.
    """

    def render(self, pose: EgoPose, camera: str) -> dict:
        return {"camera": camera, "ego_xy": (round(pose.x, 3), round(pose.y, 3)), "t": pose.t}


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
    cameras: list[str] = field(default_factory=lambda: ["camera_front_wide"])
    calibration: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._stub = None

    def load_scene(self, scene: Any) -> None:
        import grpc
        from alpasim_grpc.v0 import sensorsim_pb2_grpc

        self._stub = sensorsim_pb2_grpc.SensorsimServiceStub(grpc.insecure_channel(self.endpoint))
        self.scene_id = scene or self.scene_id
        # (load/select the scene + read camera calibration from the service here)

    def render(self, pose: EgoPose, camera: str) -> Any:
        raise NotImplementedError(
            "Wire SensorsimService.render_rgb(RGBRenderRequest(scene, camera, pose)) here against a "
            "running nre-ga renderer with a loaded NuRec scene (not available on this box yet)."
        )
