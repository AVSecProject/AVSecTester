"""Inserted pedestrian image — a camera-facing *natural physical object* attack (image family).

A picture of a pedestrian where no pedestrian is: the "human being poster / billboard on the road"
example of docs/PROJECT.md (after *Phantom of the ADAS*, Nassi et al., CCS 2020). Two props, both
world-anchored like :class:`avsectester.attacks.object_insertion.sign_spoof.RoadsideSign` (they reuse it — a sign is
any textured board at a fixed scene position):

  * :func:`standee` — a life-size cardboard cut-out of a person standing on the ground, e.g. on the
    shoulder next to the ego lane;
  * :func:`billboard` — the person printed on a poster board on two legs by the roadside.

:func:`poster_rgba` also makes the face for a poster carried on the lead vehicle's rear (an ad on a
truck), placed image-anchored with :func:`avsectester.attacks.object_insertion.sign_spoof.vehicle_sign_quad`.

The payload is a real person cut out of a camera image (``scripts/extract_person_cutouts.py`` builds
them from nuScenes with SAM). Those cut-outs are CC BY-NC-SA derived data, so none ship with the repo:
point :func:`load_cutout` at your own asset directory.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from avsectester.attacks.object_insertion.sign_spoof import RoadsideSign


def load_cutout(path: str | Path) -> np.ndarray:
    """A person cut-out as ``(H, W, 4)`` uint8 RGBA (transparent background)."""
    from PIL import Image

    return np.asarray(Image.open(path).convert("RGBA")).copy()


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


def standee(person: np.ndarray, x: float, y: float, height: float = 1.75, yaw: float = 0.0,
            ground_z: float = 0.0) -> RoadsideSign:
    """A life-size cut-out (``height`` m tall) standing on the ground at scene position (x, y)."""
    return RoadsideSign(x=x, y=y, face=person, width=height * person.shape[1] / person.shape[0],
                        mount_height=0.0, ground_z=ground_z, yaw=yaw, post=False)


def billboard(person: np.ndarray, x: float, y: float, width: float = 1.4, mount_height: float = 0.6,
              yaw: float = 0.0, ground_z: float = 0.0) -> RoadsideSign:
    """The person printed on a ``width``-m poster board on two legs, bottom edge ``mount_height`` up."""
    return RoadsideSign(x=x, y=y, face=poster_rgba(person), width=width, mount_height=mount_height,
                        ground_z=ground_z, yaw=yaw, post=True, post_width=0.09, n_posts=2)
