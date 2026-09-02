"""Shared environment/system helpers (avstack geometry; imported lazily)."""

from __future__ import annotations

from typing import Any

_EXTENT = [1.6, 1.8, 4.0]  # h, w, l


def forward_hazard(ego_state: Any, tracks: Any, brake_distance: float, brake_corridor: float):
    """Nearest confirmed track's forward distance (m) inside the ego corridor, else None."""
    import numpy as np
    from avstack.geometry import transformations as tforms

    rot = tforms.get_rot_yaw_matrix(ego_state.attitude.yaw, "+z")
    fwd, left = rot[:, 0], rot[:, 1]
    ego_pos = ego_state.position.x
    nearest = None
    for tk in tracks:
        d = np.asarray(tk.position.x) - ego_pos
        dx, dy = float(np.dot(d, fwd)), float(np.dot(d, left))
        if 0.0 < dx < brake_distance and abs(dy) < brake_corridor:
            nearest = dx if nearest is None else min(nearest, dx)
    return nearest


def make_object(obj_type: str, oid: int, xyz, t: float, velocity=(0.0, 0.0, 0.0)):
    """Build an avstack ObjectState at ``xyz`` (global frame)."""
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

    pos = Position(np.asarray(xyz, dtype=float), GlobalOrigin3D)
    att = Attitude(np.quaternion(1), GlobalOrigin3D)  # heading +x
    box = Box3D(pos, att, _EXTENT, where_is_t="bottom")
    obj = ObjectState(obj_type, ID=oid)
    obj.set(
        t, pos, box,
        Velocity(np.asarray(velocity, dtype=float), GlobalOrigin3D),
        Acceleration(np.zeros(3), GlobalOrigin3D),
        att,
        AngularVelocity(np.quaternion(1), GlobalOrigin3D),
    )
    return obj
