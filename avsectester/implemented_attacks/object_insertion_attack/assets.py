"""Sign, person-poster and traffic-light assets owned by ObjectInsertionAttack."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from collections.abc import Sequence

import numpy as np

from avsectester.validation import require_nonnegative_integer, require_positive_integer
from avsectester.rendering.objects import ObjectAsset, PlaneSurface
from avsectester.rendering.textures import STOP_SIGN_FACE, load_rgba_image


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
    post_width_m: float = 0.07
    post_count: int = 1

    def __post_init__(self):
        self.face = load_rgba_image(self.face)
        if not np.isfinite([self.width_m, self.mount_height_m, self.post_width_m]).all():
            raise ValueError("Sign dimensions must be finite")
        if self.width_m <= 0 or self.mount_height_m < 0 or self.post_width_m <= 0:
            raise ValueError("Sign width_m and post_width_m must be positive, mount_height_m nonnegative")
        require_nonnegative_integer(self.post_count, "post_count")

    def object_asset(self, category: str) -> ObjectAsset:
        """Return this sign's panel and optional supports as a bottom-centred object asset."""
        return ObjectAsset.from_panel(
            self.face, category=category, width_m=self.width_m,
            mount_height_m=self.mount_height_m,
            post_count=self.post_count, post_width_m=self.post_width_m,
        )

    def planes(self) -> Sequence[PlaneSurface]:
        return self.object_asset("sign").surfaces


def poster_rgba(person: np.ndarray, aspect: float = 1.5, fill: float = 0.86,
                paper: tuple = (228, 226, 218), border: tuple = (40, 40, 42), border_frac: float = 0.025) -> np.ndarray:
    """Print ``person`` on an opaque poster (height/width = ``aspect``) with a thin frame.

    The figure is scaled to ``fill`` of the poster height, centred horizontally, feet near the bottom.
    """
    from PIL import Image

    ph, pw = person.shape[:2]
    H = 900
    W = round(H / aspect)
    poster = np.zeros((H, W, 4), np.uint8)
    poster[..., :3] = paper
    poster[..., 3] = 255
    s = min(fill * H / ph, 0.9 * W / pw)
    fig = np.asarray(Image.fromarray(person).resize((max(1, int(pw * s)), max(1, int(ph * s))), Image.LANCZOS))
    fh, fw = fig.shape[:2]
    y0, x0 = H - fh - int(0.05 * H), (W - fw) // 2
    a = fig[..., 3:4].astype(np.float32) / 255.0
    region = poster[y0:y0 + fh, x0:x0 + fw, :3].astype(np.float32)
    poster[y0:y0 + fh, x0:x0 + fw, :3] = (fig[..., :3] * a + region * (1 - a)).astype(np.uint8)
    b = max(2, int(border_frac * W))
    poster[:b, :, :3] = poster[-b:, :, :3] = border
    poster[:, :b, :3] = poster[:, -b:, :3] = border
    return poster

def standee(person: np.ndarray, height_m: float = 1.75) -> SignAsset:
    """A local life-size cut-out standing on the ground, without posts."""
    return SignAsset(face=person, width_m=height_m * person.shape[1] / person.shape[0],
                     mount_height_m=0.0, post_count=0)

def billboard(person: np.ndarray, width_m: float = 1.4, mount_height_m: float = 0.6) -> SignAsset:
    """A printed poster on two legs. Placement is supplied separately by ``Insertion``."""
    return SignAsset(face=poster_rgba(person), width_m=width_m, mount_height_m=mount_height_m,
                     post_width_m=0.09, post_count=2)

_HOUSING = (28, 28, 30)

_OFF = (45, 45, 48)

_LENS = {"red": (220, 40, 40), "yellow": (240, 200, 40), "green": (40, 200, 90)}

def _disc(img: np.ndarray, cx: int, cy: int, r: int, color) -> None:
    ys, xs = np.ogrid[:img.shape[0], :img.shape[1]]
    m = (xs - cx) ** 2 + (ys - cy) ** 2 <= r * r
    img[m, :3] = color
    img[m, 3] = 255

def signal_head_rgba(lit: str = "red", texture_height_pixels: int = 300) -> np.ndarray:
    """One vertical 3-lens signal head (red/yellow/green top-to-bottom) with ``lit`` lens on; RGBA."""
    if lit not in _LENS:
        raise ValueError(f"Unknown traffic-light state {lit!r}")
    if isinstance(texture_height_pixels, bool) or not isinstance(texture_height_pixels, int) or texture_height_pixels < 24:
        raise ValueError("traffic-light texture_height_pixels must be an integer of at least 24 pixels")
    w = max(2, texture_height_pixels // 3)
    img = np.zeros((texture_height_pixels, w, 4), np.uint8)
    pad = max(2, w // 12)
    img[pad:-pad, pad:-pad, :3] = _HOUSING  # rounded-ish black housing
    img[pad:-pad, pad:-pad, 3] = 255
    r = (w - 4 * pad) // 2
    for i, name in enumerate(("red", "yellow", "green")):
        cy = int(texture_height_pixels * (i + 0.5) / 3)
        base = _LENS[name] if name == lit else _OFF
        _disc(img, w // 2, cy, r, base)
        if name == lit:  # a brighter core so it reads as emitting
            _disc(img, w // 2, cy, max(1, r - r // 3), tuple(min(255, int(c * 1.25)) for c in base))
    return img

def traffic_lights_rgba(head_count: int = 3, lit: str = "red", texture_height_pixels: int = 300, gap_fraction: float = 0.25,
                        board: tuple | None = (35, 37, 40)) -> np.ndarray:
    """``head_count`` signal heads in a row, all showing ``lit``, mounted on an optional ``board``; RGBA.

    ``head_count`` heads is the "multiple traffic lights" payload; ``gap_fraction`` is the spacing between heads as a
    fraction of a head's width. ``board=None`` gives just the heads on a transparent background.
    """
    require_positive_integer(head_count, "head_count")
    if not math.isfinite(gap_fraction) or gap_fraction < 0:
        raise ValueError("gap_fraction must be finite and nonnegative")
    head = signal_head_rgba(lit, texture_height_pixels)
    hh, hw = head.shape[:2]
    gap = int(hw * gap_fraction)
    marg = gap
    W = head_count * hw + (head_count - 1) * gap + 2 * marg
    H = hh + 2 * marg
    out = np.zeros((H, W, 4), np.uint8)
    if board is not None:
        out[..., :3] = board
        out[..., 3] = 255
    for i in range(head_count):
        x0 = marg + i * (hw + gap)
        head_a = head[..., 3:4] / 255.0
        region = out[marg:marg + hh, x0:x0 + hw, :3].astype(np.float32)
        out[marg:marg + hh, x0:x0 + hw, :3] = (head[..., :3] * head_a + region * (1 - head_a)).astype(np.uint8)
        out[marg:marg + hh, x0:x0 + hw, 3] = np.maximum(out[marg:marg + hh, x0:x0 + hw, 3], head[..., 3])
    return out

def roadside_rig(face: np.ndarray | str | Path, width_m: float = 2.4, mount_height_m: float = 2.2) -> SignAsset:
    """Local traffic-light board and post, with its bottom edge at ``mount_height_m``."""
    return SignAsset(face=face, width_m=width_m, mount_height_m=mount_height_m, post_width_m=0.12)

@dataclass
class PersonAsset:
    """A cut-out standee or a printed person poster, with its origin at ground level.

    Lengths are in metres: a standee uses ``height_m``; a billboard uses ``width_m`` and
    ``mount_height_m``."""

    image: np.ndarray | str | Path
    mode: str = "billboard"
    height_m: float = 1.75
    width_m: float = 1.4
    mount_height_m: float = 0.6

    def __post_init__(self):
        self.image = load_rgba_image(self.image)
        if self.mode not in {"billboard", "standee"}:
            raise ValueError("PersonAsset mode must be billboard or standee")
        if (not np.isfinite([self.height_m, self.width_m, self.mount_height_m]).all()
                or min(self.height_m, self.width_m) <= 0 or self.mount_height_m < 0):
            raise ValueError("Person height_m and width_m must be positive, mount_height_m nonnegative")

    def planes(self):
        asset = (billboard(self.image, self.width_m, self.mount_height_m) if self.mode == "billboard"
                 else standee(self.image, self.height_m))
        return asset.planes()


@dataclass
class TrafficLightAsset:
    """A traffic-light board and post, whose origin is the ground beneath the board.

    ``texture_height_pixels`` sets each head's raster resolution. ``width_m`` and
    ``mount_height_m`` set the board width and bottom height in world metres."""

    head_count: int = 3
    lit: str = "red"
    texture_height_pixels: int = 300
    gap_fraction: float = 0.25
    board: tuple | None = (35, 37, 40)
    width_m: float = 2.4
    mount_height_m: float = 2.2

    def __post_init__(self):
        if self.lit not in _LENS:
            raise ValueError(f"Unknown traffic-light state {self.lit!r}")
        require_positive_integer(self.head_count, "head_count")
        if isinstance(self.texture_height_pixels, bool) or not isinstance(self.texture_height_pixels, int) or self.texture_height_pixels < 24:
            raise ValueError("traffic-light texture_height_pixels must be an integer of at least 24 pixels")
        if not math.isfinite(self.gap_fraction) or self.gap_fraction < 0:
            raise ValueError("gap_fraction must be finite and nonnegative")
        if (not np.isfinite([self.width_m, self.mount_height_m]).all()
                or self.width_m <= 0 or self.mount_height_m < 0):
            raise ValueError("Traffic-light width_m must be positive and mount_height_m nonnegative")

    def planes(self):
        face = traffic_lights_rgba(self.head_count, self.lit, self.texture_height_pixels, self.gap_fraction, self.board)
        return roadside_rig(face, self.width_m, self.mount_height_m).planes()
