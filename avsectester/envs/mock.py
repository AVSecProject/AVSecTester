"""Simulator-free mock environment + system (no CARLA required).

``MockEnv`` is the world: a 1-D kinematic ego driving +x with a couple of static background
vehicles beside the lane; ``step(control)`` integrates ``(throttle, brake)`` and yields the
next frame. ``MockSystem`` is the AV stack: a ground-truth passthrough detector + tracker +
forward-collision brake reflex, with attacks/defenses fired at its perception seams.

Together they exercise the whole framework (attacks, defenses, metrics) without a simulator.
Heavy avstack imports stay lazy so the package is importable core-only.
"""

from __future__ import annotations

from typing import Any

from ..config import ENVIRONMENTS, SYSTEMS
from ..core.environment import Environment
from ..core.frame import Frame
from ..core.seam import Seam
from ..core.system import Outcome, System
from .common import forward_hazard, make_object


@ENVIRONMENTS.register_module()
class MockEnv(Environment):
    def __init__(self, frames: int = 140, dt: float = 0.05, n_background: int = 2,
                 accel: float = 8.0, decel: float = 8.0) -> None:
        self.frames = frames
        self.dt = dt
        self.n_background = n_background
        self.accel = accel
        self.decel = decel
        self._i = 0
        self._x = 0.0
        self._v = 0.0

    def _frame(self) -> Frame:
        t = self._i * self.dt
        ego = make_object("car", 0, [self._x, 0.0, 0.0], t, velocity=(self._v, 0.0, 0.0))
        # static vehicles beside the lane (out of the brake corridor)
        gt = [
            make_object("car", 100 + i, [20.0 + 10.0 * i, 6.0 if i % 2 == 0 else -6.0, 0.0], t)
            for i in range(self.n_background)
        ]
        return Frame(index=self._i, timestamp=t, ego=ego, ground_truth=gt)

    def reset(self) -> Frame:
        self._i, self._x, self._v = 0, 0.0, 0.0
        return self._frame()

    def step(self, control: Any = None) -> tuple[Frame, bool]:
        throttle, brake = control or (0.0, 0.0)
        a = throttle * self.accel - brake * self.decel
        self._v = max(0.0, self._v + a * self.dt)
        self._x += self._v * self.dt
        self._i += 1
        return self._frame(), self._i >= self.frames


@SYSTEMS.register_module()
class MockSystem(System):
    seams = (Seam.PERCEPTION_INPUT, Seam.PERCEPTION_OUT)

    def __init__(self, target_speed: float = 6.0, brake_distance: float = 8.0,
                 brake_corridor: float = 2.5) -> None:
        super().__init__()
        self.target_speed = target_speed
        self.brake_distance = brake_distance
        self.brake_corridor = brake_corridor
        from avstack.modules.perception.object3d import Passthrough3DObjectDetector
        from avstack.modules.tracking.tracker3d import BasicBoxTracker3D

        self._perception = Passthrough3DObjectDetector()
        self._tracker = BasicBoxTracker3D()

    def process(self, frame: Frame) -> Outcome:
        from avstack.datastructs import DataContainer
        from avstack.geometry import GlobalOrigin3D

        data = DataContainer(frame.index, frame.timestamp, list(frame.ground_truth or []), "mock")
        data = self.fire(Seam.PERCEPTION_INPUT, data, frame)   # object-level attacks/defenses
        n_input = len(data)

        detections = self._perception(data, frame=frame.index)
        detections = self.fire(Seam.PERCEPTION_OUT, detections, frame)  # detection-level
        self._tracker(detections, platform=GlobalOrigin3D)
        confirmed = self._tracker.tracks_confirmed

        hazard = forward_hazard(frame.ego, confirmed, self.brake_distance, self.brake_corridor)
        v = float(frame.ego.velocity.norm())
        if hazard is not None:
            throttle, brake = 0.0, 1.0
        else:
            throttle, brake = max(0.0, min(1.0, self.target_speed - v)), 0.0

        record = {
            "frame": frame.index, "t": frame.timestamp,
            "n_input": n_input, "n_detections": len(detections), "n_tracks": len(confirmed),
            "ego_speed": v, "throttle": throttle, "brake": brake, "steer": 0.0,
            "hazard_dist": hazard, "braking": hazard is not None,
        }
        return Outcome(control=(throttle, brake), record=record)
