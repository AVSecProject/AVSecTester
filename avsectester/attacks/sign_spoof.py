"""Fake traffic sign — a camera-facing *natural physical object* attack (image family).

A benign-looking but misplaced traffic sign (here a STOP sign) put where the ego's camera sees it,
e.g. planted on the roadside of a road with no intersection, or carried on the back of the lead
vehicle ("a stop sign at the rear of a car" in docs/PROJECT.md). No optimization: the payload is the
standard sign face itself.

As with :mod:`avsectester.attacks.physical_patch`, the attack owns only the **payload** (which sign)
and the **target** (where); how it renders into the frame is simulation, shared in
:mod:`avsectester.simulators.patch_insertion`:

  * :class:`RoadsideSign` — world-anchored: a sign face + post at a fixed scene position, rendered by
    ray casting through the real camera model (exact under NuRec's f-theta fisheye), so it grows and
    shifts with correct perspective as the ego approaches. Use :func:`roadside_sign_insert`.
  * :func:`vehicle_sign_quad` — image-anchored: a square quad on the lead vehicle's rear, found from a
    2-D detector box (planar warp, as ``nurec_patch_demo`` does for the patch).

Both plug into the run loop as a ``perturb(Observation)`` via
:func:`avsectester.simulators.patch_insertion.frame_perturbation`, so the AV stack perceives the sign.
Pixel helpers are numpy/PIL only.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

ASSETS = Path(__file__).resolve().parents[1] / "assets" / "signs"
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


def _rect(center: np.ndarray, right: np.ndarray, w: float, z_top: float, z_bottom: float) -> np.ndarray:
    """World rectangle (TL,TR,BR,BL as seen from its front) centred on ``center`` (x, y), vertical."""
    left, rgt = center - right * (w / 2), center + right * (w / 2)
    return np.array([[*left, z_top], [*rgt, z_top], [*rgt, z_bottom], [*left, z_bottom]], dtype=np.float64)


@dataclass
class RoadsideSign:
    """A sign planted at a fixed scene-world position (x forward, y left, z up — the NuRec scene frame).

    ``yaw`` turns the face about the vertical: 0 faces an ego driving along +x; positive angles it
    toward the road centre-line from a right-side (negative ``y``) placement. Real signs: bottom edge
    >= 1.5 m above the ground (rural), face 0.75 m (30") on ordinary roads, 0.9 m (36") on multilane.
    """

    x: float
    y: float
    face: np.ndarray = field(default_factory=sign_rgba, repr=False)
    width: float = 0.9
    mount_height: float = 1.5  # bottom edge of the face above the ground
    ground_z: float = 0.0
    yaw: float = 0.0
    post: bool = True
    post_width: float = 0.07
    n_posts: int = 1  # 2: a board on two legs at +-35% of its width

    def planes(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """The textured world rectangles to render, back to front: ``[(corners 4x3, rgba), ...]``."""
        normal = np.array([-math.cos(self.yaw), -math.sin(self.yaw)])  # face points back at the ego
        right = np.array([math.sin(self.yaw), -math.cos(self.yaw)])  # viewer's right along the face
        c = np.array([self.x, self.y])
        h = self.width * self.face.shape[0] / self.face.shape[1]
        bottom = self.ground_z + self.mount_height
        planes = []
        if self.post:  # from the ground to the face's centre, just behind the face
            offsets = [0.0] if self.n_posts == 1 else np.linspace(-0.35, 0.35, self.n_posts) * self.width
            for off in offsets:
                planes.append((_rect(c - normal * 0.03 + right * off, right, self.post_width, bottom + h / 2,
                                     self.ground_z), post_rgba()))
        planes.append((_rect(c, right, self.width, bottom + h, bottom), self.face))
        return planes


def roadside_sign_insert(sign: RoadsideSign, compositor: Any, camera: Any,
                         cam_from_world_of: Callable[[Any], np.ndarray],
                         soften: float = 0.0) -> Callable[[Any, np.ndarray], np.ndarray]:
    """``insert(observation, rgb) -> rgb`` that renders ``sign`` into the frame at the observation's ego
    pose. ``camera`` is a lens model and ``cam_from_world_of(observation)`` the 4x4 world->camera pose
    (NuRec: ``renderer.camera_model()`` / ``lambda o: renderer.cam_from_world(o.vehicle_state)``)."""
    planes = sign.planes()

    def _insert(observation: Any, rgb: np.ndarray) -> np.ndarray:
        return compositor.apply_planes(rgb, camera, cam_from_world_of(observation), planes, soften=soften)

    return _insert


def vehicle_sign_quad(detect: Callable[[Any], Any], width_frac: float = 0.4, v_center: float = 0.45,
                      central: float = 0.25, aspect: float = 1.0) -> Callable[[Any], Any]:
    """``quad_of(observation)``: a quad (height/width ``aspect``, square by default) on the lead vehicle's
    rear from a 2-D detector box, for :meth:`PatchCompositor.apply` — a STOP sign (or a poster) carried
    on the back of the car ahead. The lead is the
    nearest detection in the ego lane, so the sign stays on one vehicle as others overtake; a missed
    detection holds the previous quad for a few frames instead of dropping the sign."""
    from avsectester.simulators.patch_insertion import detector_quad, hold_quad

    return hold_quad(detector_quad(detect, width_frac=width_frac, v_center=v_center, central=central,
                                   aspect=aspect, pick="lane"))


def quad_insert(quad_of: Callable[[Any], Any], compositor: Any,
                face: np.ndarray) -> Callable[[Any, np.ndarray], np.ndarray]:
    """``insert(observation, rgb) -> rgb`` that warps ``face`` onto ``quad_of(observation)`` (if any)."""

    def _insert(observation: Any, rgb: np.ndarray) -> np.ndarray:
        quad = quad_of(observation)
        return rgb if quad is None else compositor.apply(rgb, quad, face)

    return _insert
