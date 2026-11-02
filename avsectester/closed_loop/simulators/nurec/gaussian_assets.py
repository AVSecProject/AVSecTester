"""Gaussian-splat assets for NRE object rendering and visibility probes.

The track frame is centered on the object box. NRE holds TRACK_FROM_PLY @ p * scale,
where scale is max(object_size - track_padding_m).
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import numpy as np

from avsectester.rendering import ObjectAsset

TRACK_FROM_PLY = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
_SH_ZERO = 0.28209479177387814
_OPAQUE_LOGIT = 8.0
_THICKNESS_M = 0.001
_PLY_PROPERTIES = ("x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
                   "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3")


@dataclass(frozen=True)
class GaussianAsset:
    """Gaussian PLY bytes, object-from-track transform and padded NRE object size."""

    ply: bytes
    object_from_track: np.ndarray
    object_size: tuple[float, float, float]


def _quaternions_wxyz(rotation: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    x, y, z, w = Rotation.from_matrix(rotation).as_quat()
    return np.array([w, x, y, z])


def gaussian_asset(
    asset: ObjectAsset, colors: Sequence[np.ndarray | tuple[float, float, float]], *,
    spacing_m: float, lateral_scale: float, track_padding_m: tuple[float, float, float],
    _probe_grid: bool = False,
) -> GaussianAsset:
    """Sample opaque face centers and alpha-weighted area colors into flat Gaussians."""
    import cv2

    if len(colors) != len(asset.surfaces):
        raise ValueError("Provide one color source per asset surface")
    if not np.isfinite([spacing_m, lateral_scale]).all() or min(spacing_m, lateral_scale) <= 0:
        raise ValueError("Gaussian spacing and lateral scale must be positive")
    extent = np.asarray(asset.extent, dtype=float)
    scale = float(extent.max())
    if scale <= 0:
        raise ValueError("A Gaussian asset needs a nonzero extent")
    center = np.array([0., 0., extent[2] / 2])
    rows = []
    for surface, source in zip(asset.surfaces, colors):
        top_left, top_right, _, bottom_left = surface.corners
        across, down = top_right - top_left, bottom_left - top_left
        height, width = surface.texture.shape[:2]
        # Probe grids round to the nearest count. Scene grids round up,
        # capped at one sample per texel.
        count = round if _probe_grid else np.ceil
        count_across = int(np.clip(count(np.linalg.norm(across) / spacing_m), 1, width))
        count_down = int(np.clip(count(np.linalg.norm(down) / spacing_m), 1, height))
        s, t = np.meshgrid((np.arange(count_across) + .5) / count_across,
                           (np.arange(count_down) + .5) / count_down)
        opaque = surface.texture[(t * height).astype(int), (s * width).astype(int), 3] >= 128
        s, t = s[opaque], t[opaque]
        if not s.size:
            continue
        if isinstance(source, tuple):
            surface_colors = np.tile((np.asarray(source, dtype=float) - .5) / _SH_ZERO, (len(s), 1))
        else:
            rgba = np.asarray(source)
            if rgba.shape != surface.texture.shape:
                raise ValueError("Face colors must have the surface texture's RGBA shape")
            alpha = rgba[..., 3].astype(float) / 255
            weighted = np.concatenate((rgba[..., :3].astype(float) * alpha[..., None],
                                       alpha[..., None]), axis=2)
            averaged = cv2.resize(weighted, (count_across, count_down), interpolation=cv2.INTER_AREA)
            rgb = np.divide(averaged[..., :3], averaged[..., 3:],
                            out=np.zeros_like(averaged[..., :3]), where=averaged[..., 3:] > 0)
            surface_colors = (rgb[opaque] / 255 - .5) / _SH_ZERO
        points = top_left + s[:, None] * across + t[:, None] * down
        normal = np.cross(across, down)
        axes = np.column_stack((normal / np.linalg.norm(normal), across / np.linalg.norm(across),
                                down / np.linalg.norm(down)))
        if np.linalg.det(axes) < 0:
            axes[:, 2] *= -1
        sigma = np.array([_THICKNESS_M, np.linalg.norm(across) / count_across * lateral_scale,
                          np.linalg.norm(down) / count_down * lateral_scale])
        size = len(points)
        rows.append(np.column_stack((
            (points - center) @ TRACK_FROM_PLY / scale,
            surface_colors, np.full(size, _OPAQUE_LOGIT),
            np.tile(np.log(sigma / scale), (size, 1)),
            np.tile(_quaternions_wxyz(TRACK_FROM_PLY.T @ axes), (size, 1)),
        )))
    if not rows:
        raise ValueError("A Gaussian asset needs at least one opaque texel")
    vertices = np.concatenate(rows).astype("<f4")
    header = (f"ply\nformat binary_little_endian 1.0\nelement vertex {len(vertices)}\n"
              + "".join(f"property float {name}\n" for name in _PLY_PROPERTIES) + "end_header\n")
    object_from_track = np.eye(4)
    object_from_track[:3, 3] = center
    padding = np.asarray(track_padding_m, dtype=float)
    return GaussianAsset(header.encode() + vertices.tobytes(), object_from_track,
                         tuple(float(value) for value in extent + padding))


def probe_asset(asset: ObjectAsset, rgb: tuple[float, float, float], *,
                track_padding_m: tuple[float, float, float], spacing_m: float = .01) -> GaussianAsset:
    """Single-color probes on the nearest-count grid at 0.6 times its spacing."""
    return gaussian_asset(asset, (rgb,) * len(asset.surfaces), spacing_m=spacing_m,
                          lateral_scale=.6, track_padding_m=track_padding_m, _probe_grid=True)
