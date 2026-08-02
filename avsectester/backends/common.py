"""Shared backend helpers (frame-agnostic; used by CarlaBackend and MockBackend)."""

from __future__ import annotations

from typing import Any


def forward_hazard(
    ego_state: Any, tracks: Any, brake_distance: float, brake_corridor: float
) -> float | None:
    """Nearest confirmed track's forward distance (m) inside the ego corridor, else None.

    ``tracks`` is any iterable of objects exposing ``.position.x`` (global xyz). The ego
    forward/left axes come from its yaw, so this works identically for CARLA and the mock.
    """
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
