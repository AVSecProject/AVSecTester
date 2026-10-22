"""Reusable sign faces and support posts for world-object assets."""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .images import load_rgba_image
from .objects import ObjectAsset, PlaneSurface


STOP_SIGN_FACE = Path(__file__).resolve().parents[1] / "assets" / "signs" / "mutcd_r1-1_stop.png"


def post_rgba(height_pixels: int = 256, width_pixels: int = 16,
              color: tuple = (150, 152, 155)) -> np.ndarray:
    """A galvanized sign post texture: flat grey, shaded across its width like a round/U-channel post."""
    shade = 0.65 + 0.35 * np.sin(np.linspace(0.15, math.pi - 0.15, width_pixels))  # darker at the edges
    img = np.zeros((height_pixels, width_pixels, 4), np.uint8)
    img[..., :3] = (np.asarray(color, np.float32)[None, None, :] * shade[None, :, None]).astype(np.uint8)
    img[..., 3] = 255
    return img


@dataclass
class SignAsset:
    """A local planar sign and optional posts in a bottom-centred object frame.

    The origin is on the ground beneath the face. Local +X is the front face normal,
    +Y points left and +Z points up. Lengths are in metres: ``width_m`` is the face width,
    ``mount_height_m`` locates the bottom edge of the face and ``post_width_m`` is each post's
    width. World position and orientation belong to a PlacedObject.
    """

    face: np.ndarray | str | Path = field(default=STOP_SIGN_FACE, repr=False)
    width_m: float = 0.9
    mount_height_m: float = 1.5
    post: bool = True
    post_width_m: float = 0.07
    post_count: int = 1

    def __post_init__(self):
        self.face = load_rgba_image(self.face)
        if not isinstance(self.post, bool):
            raise TypeError("post must be boolean")

        if not np.isfinite([self.width_m, self.mount_height_m, self.post_width_m]).all():
            raise ValueError("Sign dimensions must be finite")
        if self.width_m <= 0 or self.mount_height_m < 0 or self.post_width_m <= 0:
            raise ValueError("Sign width_m and post_width_m must be positive, mount_height_m nonnegative")
        if isinstance(self.post_count, bool) or not isinstance(self.post_count, int) or self.post_count < 1:
            raise ValueError("post_count must be a positive integer")

    def object_asset(self, category: str) -> ObjectAsset:
        """Return these local surfaces as a bottom-centred world-object asset."""
        return ObjectAsset.from_surfaces(self.planes(), category)

    def planes(self) -> Sequence[PlaneSurface]:
        height = self.width_m * self.face.shape[0] / self.face.shape[1]

        def rectangle(x, y, width, bottom, top, texture):
            return PlaneSurface(np.array([
                [x, y - width / 2, top], [x, y + width / 2, top],
                [x, y + width / 2, bottom], [x, y - width / 2, bottom],
            ]), texture)

        surfaces = []
        if self.post:
            offsets = [0.0] if self.post_count == 1 else np.linspace(-0.35, 0.35, self.post_count) * self.width_m
            for offset in offsets:
                surfaces.append(rectangle(-0.03, offset, self.post_width_m, 0,
                                          self.mount_height_m + height / 2, post_rgba()))
        surfaces.append(rectangle(0, 0, self.width_m, self.mount_height_m,
                                  self.mount_height_m + height, self.face))
        return surfaces


def stop_sign_asset(**options) -> ObjectAsset:
    """Build the supplied stop-sign face with optional sign dimensions and support posts."""
    return SignAsset(face=STOP_SIGN_FACE, **options).object_asset("stop_sign")
