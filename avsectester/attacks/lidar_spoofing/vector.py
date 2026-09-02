"""The LiDAR-spoofing vector: the shared mechanism for adding/removing obstacles.

Both object-spoofing (inject a phantom -> false positive) and object-removal (delete a real
obstacle -> false negative) attacks compose this vector. It owns:

- the **bindings** — a LiDAR-spoofing attack is most faithfully delivered at ``raw_lidar``
  (spray/erase points, requires a real cloud + a neural detector), with an object-level
  proxy at ``perception_input`` on a ground-truth passthrough stack;
- the **geometry** — anchoring a world-fixed phantom from an ego-relative offset, building
  the phantom ``ObjectState``;
- the **feasibility constraints** — a point budget and field-of-view checks;
- the per-seam **primitives** — object-level add/remove now; raw-point add/remove are the
  optimization-track counterparts (declared, not yet implemented).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..vector import AttackVector

# phantom vehicle box [h, w, l] in metres
PHANTOM_EXTENT = [1.6, 1.8, 4.0]


class LidarSpoofingVector(AttackVector):
    # Object-level injection at the detector input, on a ground-truth passthrough stack. The
    # raw-point realization (spraying into the LiDAR cloud) is the optimization track and is
    # not wired yet, so ``raw_lidar`` is intentionally not declared here.
    seams = ("perception_input",)

    def __init__(self, n_points_budget: int = 1000) -> None:
        self.n_points_budget = n_points_budget

    # -- feasibility constraints ----------------------------------------------
    def check_budget(self, n_points: int) -> None:
        if n_points > self.n_points_budget:
            raise ValueError(
                f"n_points={n_points} exceeds the declared LiDAR budget "
                f"(<= {self.n_points_budget})"
            )

    # -- geometry (pure; per-run caching lives on the attack) -----------------
    @staticmethod
    def anchor_world(ego_state: Any, offset_xyz: list[float]) -> Any:
        """World position of an obstacle placed ``offset_xyz`` = [fwd, left, up] from the ego.

        Fixing the phantom in the world (rather than tracking the ego) makes it a stationary
        obstacle the ego then approaches — the realistic case.
        """
        import numpy as np
        from avstack.geometry import transformations as tforms

        rot = tforms.get_rot_yaw_matrix(ego_state.attitude.yaw, "+z")
        return ego_state.position.x + rot @ np.asarray(offset_xyz, dtype=float)

    @staticmethod
    def make_phantom_object(
        timestamp: float, world_xyz: Any, ego_state: Any,
        obj_type: str = "car", score: float = 1.0, oid: int = 90001,
        extent: list[float] | None = None,
    ) -> Any:
        """Build a stationary phantom ``ObjectState`` at ``world_xyz`` (global frame)."""
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

        pos = Position(np.asarray(world_xyz, dtype=float), GlobalOrigin3D)
        att = Attitude(ego_state.attitude.q, GlobalOrigin3D)
        box = Box3D(pos, att, extent or PHANTOM_EXTENT, where_is_t="bottom")
        obj = ObjectState(obj_type, ID=oid, score=score)
        obj.set(
            timestamp, pos, box,
            Velocity(np.zeros(3), GlobalOrigin3D),
            Acceleration(np.zeros(3), GlobalOrigin3D),
            att,
            AngularVelocity(np.quaternion(1), GlobalOrigin3D),
        )
        return obj

    @staticmethod
    def forward_offset(obj: Any, ego_state: Any) -> tuple[float, float]:
        """(forward, lateral) metres of ``obj`` in the ego's heading frame."""
        import numpy as np
        from avstack.geometry import transformations as tforms

        rel = np.asarray(obj.position.x, dtype=float) - np.asarray(ego_state.position.x, dtype=float)
        rot = tforms.get_rot_yaw_matrix(ego_state.attitude.yaw, "+z")
        local = rot.T @ rel  # world -> ego heading frame
        return float(local[0]), float(local[1])

    # -- object-level primitives (perception_input seam) ----------------------
    @staticmethod
    def add_object(container: Any, obj: Any) -> Any:
        """Append ``obj`` to a detector-input ``DataContainer`` (in place)."""
        container.append(obj)
        return container

    @staticmethod
    def remove_objects(container: Any, predicate: Callable[[Any], bool]) -> Any:
        """Return a new container with objects matching ``predicate`` dropped."""
        from avstack.datastructs import DataContainer

        kept = [o for o in container if not predicate(o)]
        return DataContainer(
            container.frame, container.timestamp, kept,
            getattr(container, "source_identifier", "atk"),
        )

    def select_forward_target(
        self, container: Any, ego_state: Any, corridor: float = 3.0, max_range: float = 40.0,
    ) -> Any:
        """The nearest object ahead of the ego within ``corridor`` lateral / ``max_range`` fwd."""
        best, best_fwd = None, float("inf")
        for o in container:
            fwd, lat = self.forward_offset(o, ego_state)
            if 0.0 < fwd < min(best_fwd, max_range) and abs(lat) <= corridor:
                best, best_fwd = o, fwd
        return best

    # -- raw-point primitives (raw_lidar seam) — optimization track -----------
    def add_points(self, cloud: Any, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(
            "raw-LiDAR point injection is the optimization track (Phase D): naive point "
            "spraying does not fool the detector; use a WhiteboxLidarDetector + PGD."
        )

    def remove_points(self, cloud: Any, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(
            "raw-LiDAR point removal (occlusion/erasure) is the optimization track (Phase D)."
        )
