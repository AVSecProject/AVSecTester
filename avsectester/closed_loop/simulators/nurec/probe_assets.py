"""Single-colour Gaussian-splat copies of framework objects for insertion into NRE scenes.

NRE inserts a 3D Gaussian PLY as a dynamic-object track. Its public documentation does not state
how PLY coordinates map to the track, so the mapping below was measured on nre-ga 26.04: the track
frame (origin at the object's bounding-box centre) holds ``TRACK_FROM_PLY @ p * scale``, where
``scale = max(object_size - track_padding_m)``. Probe objects inserted with it land on their
projected pixels within 0.1 px.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from avsectester.rendering import ObjectAsset

TRACK_FROM_PLY = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
_SH_ZERO = 0.28209479177387814  # zeroth-order spherical-harmonic basis value
_OPAQUE_LOGIT = 8.0
_THICKNESS_M = 0.001
_PLY_PROPERTIES = ("x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
                   "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3")


@dataclass(frozen=True)
class ProbeAsset:
    """An opaque, single-colour Gaussian PLY of an asset's visible texels, and its NRE track box.

    The track origin is the asset's bounding-box centre, ``object_from_track`` places it in the
    asset frame, and ``object_size`` is the box size plus NRE's track padding.
    """

    ply: bytes
    object_from_track: np.ndarray
    object_size: tuple[float, float, float]


def _quaternions_wxyz(rotation: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    x, y, z, w = Rotation.from_matrix(rotation).as_quat()
    return np.array([w, x, y, z])


def probe_asset(asset: ObjectAsset, rgb: tuple[float, float, float], *,
                track_padding_m: tuple[float, float, float], spacing_m: float = 0.01) -> ProbeAsset:
    """Cover each surface's opaque texels (alpha at least one half) with flat ``rgb`` Gaussians.

    Gaussians lie on a grid of at most ``spacing_m`` and at most one per texel. ``rgb`` values are
    in [0, 1].
    """
    extent = np.asarray(asset.extent, dtype=float)
    scale = float(extent.max())
    if scale <= 0:
        raise ValueError("A probe asset needs a nonzero extent")
    centre = np.array([0., 0., extent[2] / 2])
    colour = (np.asarray(rgb, dtype=float) - 0.5) / _SH_ZERO
    rows = []
    for surface in asset.surfaces:
        top_left, top_right, _, bottom_left = surface.corners
        across, down = top_right - top_left, bottom_left - top_left
        height, width = surface.texture.shape[:2]
        count_across = int(np.clip(round(np.linalg.norm(across) / spacing_m), 1, width))
        count_down = int(np.clip(round(np.linalg.norm(down) / spacing_m), 1, height))
        s, t = np.meshgrid((np.arange(count_across) + .5) / count_across,
                           (np.arange(count_down) + .5) / count_down)
        opaque = surface.texture[(t * height).astype(int), (s * width).astype(int), 3] >= 128
        s, t = s[opaque], t[opaque]
        if not s.size:
            continue
        points = top_left + s[:, None] * across + t[:, None] * down
        normal = np.cross(across, down)
        axes = np.column_stack((normal / np.linalg.norm(normal), across / np.linalg.norm(across),
                                down / np.linalg.norm(down)))
        if np.linalg.det(axes) < 0:
            axes[:, 2] *= -1
        sigma = np.array([_THICKNESS_M, np.linalg.norm(across) / count_across * .6,
                          np.linalg.norm(down) / count_down * .6])
        count = len(points)
        rows.append(np.column_stack((
            (points - centre) @ TRACK_FROM_PLY / scale,  # TRACK_FROM_PLY.T applied to row vectors
            np.tile(colour, (count, 1)), np.full(count, _OPAQUE_LOGIT),
            np.tile(np.log(sigma / scale), (count, 1)),
            np.tile(_quaternions_wxyz(TRACK_FROM_PLY.T @ axes), (count, 1)),
        )))
    if not rows:
        raise ValueError("A probe asset needs at least one opaque texel")
    vertices = np.concatenate(rows).astype("<f4")
    header = (f"ply\nformat binary_little_endian 1.0\nelement vertex {len(vertices)}\n"
              + "".join(f"property float {name}\n" for name in _PLY_PROPERTIES) + "end_header\n")
    object_from_track = np.eye(4)
    object_from_track[:3, 3] = centre
    padding = np.asarray(track_padding_m, dtype=float)
    return ProbeAsset(header.encode() + vertices.tobytes(), object_from_track,
                      tuple(float(value) for value in extent + padding))
