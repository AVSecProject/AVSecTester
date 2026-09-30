"""Fake traffic lights — a camera-facing *natural physical object* attack (image family).

The "truck with multiple traffic lights on it" example of docs/PROJECT.md: real signal heads mounted
where the ego's camera reads them as live traffic lights, e.g. carried on the rear of the lead
vehicle, or on a roadside board. Like :mod:`avsectester.attacks.sign_spoof`, the attack owns only the
**payload** (the signal-head image) and the **target** (where); how it renders is shared simulation in
:mod:`avsectester.simulators.patch_insertion`.

The payload is drawn procedurally (numpy/PIL, no external asset): one or more standard 3-lens signal
heads in black housings, on a board, with a chosen lens lit (red / yellow / green). Reuse it as:

  * a board carried on the lead vehicle's rear — image-anchored via
    :func:`avsectester.attacks.sign_spoof.vehicle_sign_quad` (pass this module's ``aspect``);
  * a roadside rig — world-anchored via :class:`avsectester.attacks.sign_spoof.RoadsideSign`.

Only numpy/PIL here; no simulator imports.
"""

from __future__ import annotations

import numpy as np

from avsectester.attacks.sign_spoof import RoadsideSign

# (housing, lens-off, red, yellow, green) BGR-agnostic RGB; drawn on transparent background
_HOUSING = (28, 28, 30)
_OFF = (45, 45, 48)
_LENS = {"red": (220, 40, 40), "yellow": (240, 200, 40), "green": (40, 200, 90)}


def _disc(img: np.ndarray, cx: int, cy: int, r: int, color) -> None:
    ys, xs = np.ogrid[:img.shape[0], :img.shape[1]]
    m = (xs - cx) ** 2 + (ys - cy) ** 2 <= r * r
    img[m, :3] = color
    img[m, 3] = 255


def signal_head_rgba(lit: str = "red", height: int = 300) -> np.ndarray:
    """One vertical 3-lens signal head (red/yellow/green top-to-bottom) with ``lit`` lens on; RGBA."""
    w = max(2, height // 3)
    img = np.zeros((height, w, 4), np.uint8)
    pad = max(2, w // 12)
    img[pad:-pad, pad:-pad, :3] = _HOUSING  # rounded-ish black housing
    img[pad:-pad, pad:-pad, 3] = 255
    r = (w - 4 * pad) // 2
    for i, name in enumerate(("red", "yellow", "green")):
        cy = int(height * (i + 0.5) / 3)
        base = _LENS[name] if name == lit else _OFF
        _disc(img, w // 2, cy, r, base)
        if name == lit:  # a brighter core so it reads as emitting
            _disc(img, w // 2, cy, max(1, r - r // 3), tuple(min(255, int(c * 1.25)) for c in base))
    return img


def traffic_lights_rgba(n: int = 3, lit: str = "red", height: int = 300, gap_frac: float = 0.25,
                        board: tuple | None = (35, 37, 40)) -> np.ndarray:
    """``n`` signal heads in a row, all showing ``lit``, mounted on an optional ``board``; RGBA.

    ``n`` heads is the "multiple traffic lights" payload; ``gap_frac`` is the spacing between heads as a
    fraction of a head's width. ``board=None`` gives just the heads on a transparent background.
    """
    head = signal_head_rgba(lit, height)
    hh, hw = head.shape[:2]
    gap = int(hw * gap_frac)
    marg = gap
    W = n * hw + (n - 1) * gap + 2 * marg
    H = hh + 2 * marg
    out = np.zeros((H, W, 4), np.uint8)
    if board is not None:
        out[..., :3] = board
        out[..., 3] = 255
    for i in range(n):
        x0 = marg + i * (hw + gap)
        head_a = head[..., 3:4] / 255.0
        region = out[marg:marg + hh, x0:x0 + hw, :3].astype(np.float32)
        out[marg:marg + hh, x0:x0 + hw, :3] = (head[..., :3] * head_a + region * (1 - head_a)).astype(np.uint8)
        out[marg:marg + hh, x0:x0 + hw, 3] = np.maximum(out[marg:marg + hh, x0:x0 + hw, 3], head[..., 3])
    return out


def aspect_of(face: np.ndarray) -> float:
    """height / width of a payload, for the vehicle-rear quad."""
    return face.shape[0] / face.shape[1]


def roadside_rig(face: np.ndarray, x: float, y: float, width: float = 2.4, mount_height: float = 2.2,
                 yaw: float = 0.0, ground_z: float = 0.0) -> RoadsideSign:
    """The traffic-light board on a roadside post (bottom edge ``mount_height`` up — overhead-ish)."""
    return RoadsideSign(x=x, y=y, face=face, width=width, mount_height=mount_height, ground_z=ground_z,
                        yaw=yaw, post=True, post_width=0.12)
