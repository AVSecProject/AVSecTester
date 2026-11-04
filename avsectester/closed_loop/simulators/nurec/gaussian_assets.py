"""Gaussian-splat assets for NRE object rendering and visibility probes.

The track frame is centered on the object box. NRE holds TRACK_FROM_PLY @ p * scale,
where scale is max(object_size - track_padding_m).
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from avsectester.rendering import ObjectAsset

if TYPE_CHECKING:
    from avsectester.rendering.meshes import MeshAsset

TRACK_FROM_PLY = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
_SH_ZERO = 0.28209479177387814
_OPAQUE_LOGIT = 8.0
_THICKNESS_M = 0.001
SCENE_GRID_SPACING_M = 0.005
SCENE_LATERAL_SCALE = 2.0
PROBE_GRID_SPACING_M = 0.01
PROBE_LATERAL_SCALE = 0.9
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
        rows.append(_gaussian_rows(points, surface_colors, sigma, axes, extent))
    if not rows:
        raise ValueError("A Gaussian asset needs at least one opaque texel")
    return _pack_asset(np.concatenate(rows), extent, track_padding_m)


def probe_asset(asset: ObjectAsset, rgb: tuple[float, float, float], *,
                track_padding_m: tuple[float, float, float],
                spacing_m: float = PROBE_GRID_SPACING_M) -> GaussianAsset:
    """Single-color probes with lateral sigma equal to nine tenths of the grid-cell size."""
    return gaussian_asset(asset, (rgb,) * len(asset.surfaces), spacing_m=spacing_m,
                          lateral_scale=PROBE_LATERAL_SCALE,
                          track_padding_m=track_padding_m, _probe_grid=True)


def _gaussian_rows(points, colors, sigma, axes, extent):
    scale = float(np.max(extent))
    center = np.array([0., 0., extent[2] / 2])
    size = len(points)
    if axes.ndim == 2:
        quaternions = np.tile(_quaternions_wxyz(TRACK_FROM_PLY.T @ axes), (size, 1))
    else:
        from scipy.spatial.transform import Rotation

        rotations = np.einsum("ij,njk->nik", TRACK_FROM_PLY.T, axes)
        quaternions = Rotation.from_matrix(rotations).as_quat()[:, [3, 0, 1, 2]]
    scales = np.log(sigma / scale)
    if scales.ndim == 1:
        scales = np.tile(scales, (size, 1))
    return np.column_stack((
        (points - center) @ TRACK_FROM_PLY / scale,
        colors, np.full(size, _OPAQUE_LOGIT), scales, quaternions,
    ))


def _pack_asset(rows, extent, track_padding_m):
    vertices = rows.astype("<f4")
    header = (f"ply\nformat binary_little_endian 1.0\nelement vertex {len(vertices)}\n"
              + "".join(f"property float {name}\n" for name in _PLY_PROPERTIES) + "end_header\n")
    object_from_track = np.eye(4)
    object_from_track[:3, 3] = [0., 0., extent[2] / 2]
    padding = np.asarray(track_padding_m, dtype=float)
    return GaussianAsset(header.encode() + vertices.tobytes(), object_from_track,
                         tuple(float(value) for value in extent + padding))


def _triangle_samples(vertices, faces, spacing_m):
    """Sample an equally spaced square grid in each triangle's longest-edge frame."""
    triangle = vertices[faces]
    edges = np.stack((triangle[:, 1] - triangle[:, 0],
                      triangle[:, 2] - triangle[:, 1],
                      triangle[:, 0] - triangle[:, 2]), axis=1)
    lengths = np.linalg.norm(edges, axis=2)
    normal = np.cross(edges[:, 0], -edges[:, 2])
    area_twice = np.linalg.norm(normal, axis=1)
    valid = area_twice > 0
    triangle, edges, lengths = triangle[valid], edges[valid], lengths[valid]
    normal, area_twice = normal[valid], area_twice[valid]
    if not len(triangle):
        return None
    longest_edge = lengths.argmax(axis=1)
    longest = lengths[np.arange(len(lengths)), longest_edge]
    order = (longest_edge[:, None] + np.arange(3)) % 3
    ordered = np.take_along_axis(triangle, order[..., None], axis=1)
    origin = ordered[:, 0]
    across = (ordered[:, 1] - origin) / longest[:, None]
    normal = normal / area_twice[:, None]
    down = np.cross(normal, across)
    height = area_twice / longest
    third_x = np.einsum("ij,ij->i", ordered[:, 2] - origin, across)
    # The longest edge's bounding rectangle has twice the triangle's area. Grid centers
    # therefore require about two candidates per retained sample, even for slender faces.
    count_across = np.floor(longest / spacing_m + .5).astype(np.int64)
    count_down = np.floor(height / spacing_m + .5).astype(np.int64)
    counts = count_across * count_down
    owner = np.repeat(np.arange(len(triangle)), counts)
    local = np.arange(len(owner)) - np.repeat(np.cumsum(counts) - counts, counts)
    x = (local % count_across[owner] + .5) * spacing_m
    y = (local // count_across[owner] + .5) * spacing_m
    third = y / height[owner]
    second = (x - third_x[owner] * third) / longest[owner]
    ordered_weights = np.column_stack((1 - second - third, second, third))
    inside = np.all(ordered_weights >= 0, axis=1)
    owner, ordered_weights = owner[inside], ordered_weights[inside]
    weights = np.empty_like(ordered_weights)
    np.put_along_axis(weights, order[owner], ordered_weights, axis=1)
    # Sub-grid triangles may contain no grid center. Keep their centroid so a positive-area
    # face is not lost solely because it is smaller or narrower than the sampling interval.
    missing = np.flatnonzero(np.bincount(owner, minlength=len(triangle)) == 0)
    owner = np.concatenate((owner, missing))
    weights = np.concatenate((weights, np.full((len(missing), 3), 1 / 3)))
    points = np.einsum("ni,nij->nj", weights, triangle[owner])
    axes = np.stack((normal, across, down), axis=2)
    return points, weights, owner, valid, axes, area_twice / 2


def _uv_pixels(uv, width, height):
    x = np.mod(uv[:, 0] * (width - 1), width)
    y = np.mod((1 - uv[:, 1]) * (height - 1), height)
    return x, y


def _bilinear(image, uv):
    height, width = image.shape[:2]
    x, y = _uv_pixels(uv, width, height)
    x0, y0 = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    x1, y1 = (x0 + 1) % width, (y0 + 1) % height
    tx, ty = (x - x0)[:, None], (y - y0)[:, None]
    return ((1 - ty) * ((1 - tx) * image[y0, x0] + tx * image[y0, x1])
            + ty * ((1 - tx) * image[y1, x0] + tx * image[y1, x1]))


def _texture_samples(texture, uv, triangle_uv, triangle_area, owner, spacing_m):
    import cv2

    height, width = texture.shape[:2]
    x, y = _uv_pixels(uv, width, height)
    full_alpha = texture[np.round(y).astype(np.int64) % height,
                         np.round(x).astype(np.int64) % width, 3]
    alpha = texture[..., 3:].astype(float) / 255
    image = np.concatenate((texture[..., :3].astype(float) * alpha, alpha), axis=2)
    pyramid = [image]
    while min(image.shape[:2]) > 1 or max(image.shape[:2]) > 1:
        image = cv2.resize(image, (max(1, (image.shape[1] + 1) // 2),
                                   max(1, (image.shape[0] + 1) // 2)),
                           interpolation=cv2.INTER_AREA)
        pyramid.append(image)
    across = triangle_uv[:, 1] - triangle_uv[:, 0]
    down = triangle_uv[:, 2] - triangle_uv[:, 0]
    texels = np.abs(across[:, 0] * down[:, 1] - across[:, 1] * down[:, 0])
    texels *= .5 * (width - 1) * (height - 1)
    footprint = texels / triangle_area * spacing_m**2
    levels = np.clip(np.floor(.5 * np.log2(np.maximum(1, footprint))).astype(np.int64),
                     0, len(pyramid) - 1)[owner]
    sampled = np.empty((len(uv), 4))
    for level, image in enumerate(pyramid):
        selected = levels == level
        if np.any(selected):
            sampled[selected] = _bilinear(image, uv[selected])
    rgb = np.divide(sampled[:, :3], sampled[:, 3:],
                    out=np.zeros_like(sampled[:, :3]), where=sampled[:, 3:] > 0)
    return rgb, full_alpha


def mesh_gaussian_asset(
    asset: MeshAsset, color: str | tuple[float, float, float], *,
    spacing_m: float, lateral_scale: float, track_padding_m: tuple[float, float, float],
) -> GaussianAsset:
    """Sample mesh triangles into flat Gaussians with shared textured and coverage geometry.

    Every nondegenerate triangle supplies a square grid in its longest-edge plane frame.
    Triangles without a grid center supply their centroid. Lateral sigma is the requested
    spacing times ``lateral_scale``, independent of triangle dimensions.
    Full-resolution alpha selects samples. Texture colors use bilinear samples of an
    alpha-premultiplied mip pyramid. Coverage copies change only the zero-order SH colors.
    """
    if not np.isfinite([spacing_m, lateral_scale]).all() or min(spacing_m, lateral_scale) <= 0:
        raise ValueError("Gaussian spacing and lateral scale must be positive")
    if isinstance(color, str) and color != "texture":
        raise ValueError("Mesh Gaussian color must be 'texture' or an RGB tuple")
    extent = np.asarray(asset.extent, dtype=float)
    if extent.max() <= 0:
        raise ValueError("A Gaussian asset needs a nonzero extent")
    rows = []
    for part in asset.parts:
        samples = _triangle_samples(part.vertices, part.faces, spacing_m)
        if samples is None:
            continue
        points, weights, owner, valid, axes, triangle_area = samples
        faces = part.faces[valid]
        if part.texture is not None:
            triangle_uv = part.uv[faces]
            uv = np.einsum("ni,nij->nj", weights, triangle_uv[owner])
            rgb, alpha = _texture_samples(part.texture, uv, triangle_uv, triangle_area, owner, spacing_m)
        elif part.vertex_colors is not None:
            rgba = np.einsum("ni,nij->nj", weights, part.vertex_colors[faces[owner]])
            rgb, alpha = rgba[:, :3], rgba[:, 3]
        else:
            rgb = np.tile(part.color[:3], (len(points), 1)).astype(float)
            alpha = np.full(len(points), part.color[3])
        opaque = alpha >= 128
        if not np.any(opaque):
            continue
        if isinstance(color, str):
            colors = (rgb[opaque] / 255 - .5) / _SH_ZERO
        else:
            colors = np.tile((np.asarray(color, dtype=float) - .5) / _SH_ZERO,
                             (int(opaque.sum()), 1))
        sigma = np.array([_THICKNESS_M, spacing_m * lateral_scale, spacing_m * lateral_scale])
        rows.append(_gaussian_rows(points[opaque], colors, sigma, axes[owner[opaque]], extent))
    if not rows:
        raise ValueError("A Gaussian asset needs at least one opaque triangle sample")
    return _pack_asset(np.concatenate(rows), extent, track_padding_m)
