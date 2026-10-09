"""Traffic-sign payloads for host-bound or world-fixed insertions.

``SignAsset`` defines the face and posts in local coordinates. Use ``Insertion``
with ``WorldPlacement`` for a roadside sign, or ``PlaneAsset`` with
``AttachedPlacement`` for a sign face installed on a vehicle.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from avsectester.insertion import PlaneSurface

ASSETS = Path(__file__).resolve().parents[2] / "assets" / "signs"
STOP_SIGN = ASSETS / "mutcd_r1-1_stop.png"  # public-domain MUTCD R1-1 face (see assets/signs/SOURCES.md)


def sign_rgba(path: str | Path = STOP_SIGN, size: int | None = None) -> np.ndarray:
    """Load a sign face as ``(H, W, 4)`` uint8 RGBA (transparent outside the sign's outline)."""
    from PIL import Image

    im = Image.open(path).convert("RGBA")
    if size:
        im = im.resize((size, size), Image.LANCZOS)
    return np.asarray(im).copy()


def post_rgba(height: int = 256, width: int = 16, color: tuple = (150, 152, 155)) -> np.ndarray:
    """A galvanized sign post texture: flat grey, shaded across its width like a round/U-channel post."""
    shade = 0.65 + 0.35 * np.sin(np.linspace(0.15, math.pi - 0.15, width))  # darker at the edges
    img = np.zeros((height, width, 4), np.uint8)
    img[..., :3] = (np.asarray(color, np.float32)[None, None, :] * shade[None, :, None]).astype(np.uint8)
    img[..., 3] = 255
    return img


@dataclass
class SignAsset:
    """A local planar sign and optional posts, ready for an ``Insertion``.

    The origin is on the ground beneath the face. Local +x is the face normal,
    +y is the viewer's right and +z is up. ``mount_height`` locates the bottom
    edge of the face. World position and orientation belong to the insertion.
    """

    face: np.ndarray = field(default_factory=sign_rgba, repr=False)
    width: float = 0.9
    mount_height: float = 1.5
    post: bool = True
    post_width: float = 0.07
    n_posts: int = 1

    def planes(self) -> Sequence[PlaneSurface]:
        if not np.isfinite([self.width, self.mount_height, self.post_width]).all():
            raise ValueError("Sign dimensions must be finite")
        if self.width <= 0 or self.mount_height < 0 or self.post_width <= 0:
            raise ValueError("Sign width and post width must be positive, mount height nonnegative")
        if not isinstance(self.n_posts, int) or self.n_posts < 1:
            raise ValueError("n_posts must be a positive integer")
        height = self.width * self.face.shape[0] / self.face.shape[1]

        def rectangle(x, y, width, bottom, top, texture):
            return PlaneSurface(np.array([
                [x, y - width / 2, top], [x, y + width / 2, top],
                [x, y + width / 2, bottom], [x, y - width / 2, bottom],
            ]), texture)

        surfaces = []
        if self.post:
            offsets = [0.0] if self.n_posts == 1 else np.linspace(-0.35, 0.35, self.n_posts) * self.width
            for offset in offsets:
                surfaces.append(rectangle(-0.03, offset, self.post_width, 0,
                                          self.mount_height + height / 2, post_rgba()))
        surfaces.append(rectangle(0, 0, self.width, self.mount_height,
                                  self.mount_height + height, self.face))
        return surfaces
