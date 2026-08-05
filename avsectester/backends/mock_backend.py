"""Simulator-free baseline backend (no CARLA required).

Synthesizes a straight-road world in avstack types and runs the SAME loop as the
CARLA backend — real avstack perception (``Passthrough3DObjectDetector``), tracking
(``BasicBoxTracker3D``), a longitudinal controller, the forward-collision brake reflex,
and the ``add_perception_hook`` attack/defense seam — so the whole engine (attacks,
metrics, escalation DAG, reports) runs end-to-end without a simulator.

Only the *world* is mocked: the ego is a 1-D kinematic point driving +x, with a couple
of static background vehicles beside the lane. Everything else is the production stack.
Heavy avstack imports stay lazy so the package remains importable core-only.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

from ..config import BACKENDS
from ..core.interfaces import Backend

_PHANTOM_LIKE_EXTENT = [1.6, 1.8, 4.0]  # h, w, l


@BACKENDS.register_module()
class MockBackend(Backend):
    """Deterministic, simulator-free closed-loop backend."""

    def __init__(
        self,
        frames: int = 140,
        target_speed: float = 6.0,
        dt: float = 0.05,
        n_background: int = 2,
        accel: float = 8.0,
        decel: float = 8.0,
        brake_distance: float = 8.0,
        brake_corridor: float = 2.5,
    ) -> None:
        self.frames = frames
        self.target_speed = target_speed
        self.dt = dt
        self.n_background = n_background
        self.accel = accel
        self.decel = decel
        self.brake_distance = brake_distance
        self.brake_corridor = brake_corridor

        self._perception_hooks: list[Callable] = []
        self._perception = None
        self._tracker = None
        self._ctx = None
        self._frame = 0
        self._x = 0.0
        self._v = 0.0

    def profile(self):
        """A ground-truth passthrough stack with a tracker; exposes both perception seams."""
        from ..core.capability import Capability, StackProfile

        return StackProfile.of(
            seams=["perception_input", "perception_out"],
            capabilities=[Capability.GT_PERCEPTION, Capability.TRACKER],
        )

    def attach(self, plugin: Callable, seam: str = "perception_out") -> None:
        """Attach at ``perception_input`` (object-level pre-loop) or ``perception_out``
        (detector post-hook via the avstack hook adapter)."""
        if seam == "perception_input":
            self._perception_hooks.append(plugin)
        else:
            from ..hooks import attach as attach_hook

            attach_hook(self._perception, plugin, seam, self._ctx)

    # -- object construction ---------------------------------------------------
    @staticmethod
    def _object(obj_type: str, oid: int, xyz, t: float, yaw: float = 0.0):
        import numpy as np
        from avstack.environment.objects import ObjectState
        from avstack.geometry import (
            Acceleration,
            AngularVelocity,
            Attitude,
            Box3D,
            GlobalOrigin3D,
            Position,
            Velocity,
        )
        from avstack.geometry import transformations as tforms

        pos = Position(np.asarray(xyz, dtype=float), GlobalOrigin3D)
        att = Attitude(tforms.transform_orientation([0, 0, yaw], "euler", "quat"), GlobalOrigin3D)
        box = Box3D(pos, att, _PHANTOM_LIKE_EXTENT, where_is_t="bottom")
        obj = ObjectState(obj_type, ID=oid)
        obj.set(
            t,
            pos,
            box,
            Velocity(np.zeros(3), GlobalOrigin3D),
            Acceleration(np.zeros(3), GlobalOrigin3D),
            att,
            AngularVelocity(np.quaternion(1), GlobalOrigin3D),
        )
        return obj

    def _ego_state(self, t: float):
        import numpy as np
        from avstack.environment.objects import ObjectState
        from avstack.geometry import (
            Acceleration,
            AngularVelocity,
            Attitude,
            Box3D,
            GlobalOrigin3D,
            Position,
            Velocity,
        )

        pos = Position(np.array([self._x, 0.0, 0.0]), GlobalOrigin3D)
        att = Attitude(np.quaternion(1), GlobalOrigin3D)  # heading +x
        box = Box3D(pos, att, _PHANTOM_LIKE_EXTENT, where_is_t="bottom")
        ego = ObjectState("car", ID=0)
        ego.set(
            t,
            pos,
            box,
            Velocity(np.array([self._v, 0.0, 0.0]), GlobalOrigin3D),
            Acceleration(np.zeros(3), GlobalOrigin3D),
            att,
            AngularVelocity(np.quaternion(1), GlobalOrigin3D),
        )
        return ego

    def _background(self, t: float) -> list:
        # static vehicles beside the lane (out of the brake corridor)
        out = []
        for i in range(self.n_background):
            side = 6.0 if i % 2 == 0 else -6.0
            out.append(self._object("car", 100 + i, [20.0 + 10.0 * i, side, 0.0], t))
        return out

    # -- lifecycle -------------------------------------------------------------
    def build(self, spec: Any = None) -> None:
        from avstack.modules.perception.object3d import Passthrough3DObjectDetector
        from avstack.modules.tracking.tracker3d import BasicBoxTracker3D

        if spec is not None:
            ic = getattr(getattr(spec, "scenario", None), "initial_conditions", {}) or {}
            self.frames = ic.get("frames", self.frames)
            self.target_speed = ic.get("target_speed", self.target_speed)

        from ..hooks import RunContext

        self._perception = Passthrough3DObjectDetector()
        self._tracker = BasicBoxTracker3D()
        self._ctx = RunContext(run_id="mock")
        self._frame = 0
        self._x = 0.0
        self._v = 0.0

    def step(self) -> dict[str, Any]:
        from avstack.datastructs import DataContainer
        from avstack.geometry import GlobalOrigin3D

        from .common import forward_hazard

        frame = self._frame
        t = frame * self.dt
        ego_state = self._ego_state(t)
        gt = self._background(t)
        self._ctx.tick(frame, t, ego_state=ego_state, ground_truth=gt)

        data = DataContainer(frame, t, gt, "mock")
        for hook in self._perception_hooks:  # perception_input seam (object level)
            data = hook(data, ego_state=ego_state)
        n_input = len(data)

        # perception_out attacks/defenses fire as post-hooks inside this call
        detections = self._perception(data, frame=frame)
        self._tracker(detections, platform=GlobalOrigin3D)
        confirmed = self._tracker.tracks_confirmed

        hazard = forward_hazard(ego_state, confirmed, self.brake_distance, self.brake_corridor)
        if hazard is not None:
            throttle, brake = 0.0, 1.0
        else:
            throttle = max(0.0, min(1.0, self.target_speed - self._v))
            brake = 0.0

        # integrate simple longitudinal kinematics
        a = throttle * self.accel - brake * self.decel
        self._v = max(0.0, self._v + a * self.dt)
        self._x += self._v * self.dt
        self._frame += 1

        return {
            "frame": frame,
            "t": t,
            "n_input": n_input,
            "n_detections": len(detections),
            "n_tracks": len(confirmed),
            "ego_speed": float(self._v),
            "throttle": throttle,
            "brake": brake,
            "steer": 0.0,
            "hazard_dist": hazard,
            "braking": hazard is not None,
        }

    def run(self) -> Iterator[dict[str, Any]]:
        for _ in range(self.frames):
            yield self.step()

    def close(self) -> None:
        self._perception = None
        self._tracker = None
