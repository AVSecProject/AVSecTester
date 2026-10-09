"""Person cut-outs and printed boards for the common insertion pipeline.

Cut-outs can be loaded from user assets or extracted by ``extract_person_cutouts.py``.
They contain no simulator-specific placement or vehicle selection logic.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from avsectester.attacks.object_insertion.sign_spoof import SignAsset


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


def standee(person: np.ndarray, height: float = 1.75) -> SignAsset:
    """A local life-size cut-out standing on the ground, without posts."""
    return SignAsset(face=person, width=height * person.shape[1] / person.shape[0],
                     mount_height=0.0, post=False)


def billboard(person: np.ndarray, width: float = 1.4, mount_height: float = 0.6) -> SignAsset:
    """A printed poster on two legs. Placement is supplied separately by ``Insertion``."""
    return SignAsset(face=poster_rgba(person), width=width, mount_height=mount_height,
                     post=True, post_width=0.09, n_posts=2)
