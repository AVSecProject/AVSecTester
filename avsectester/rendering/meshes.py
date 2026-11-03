"""Triangle mesh assets with explicit file coordinates, scale and basic colors."""
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def _array(value: Any, dtype: Any, name: str, shape: tuple[int | None, ...]) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True)
    if result.ndim != len(shape) or any(
            wanted is not None and actual != wanted for actual, wanted in zip(result.shape, shape)):
        raise ValueError(f"{name} must have shape {shape}")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain finite values")
    result.setflags(write=False)
    return result


def _colors(value: Any, name: str, shape: tuple[int | None, ...]) -> np.ndarray:
    data = np.asarray(value)
    if data.dtype != np.uint8:
        raise ValueError(f"{name} must be uint8 RGBA")
    return _array(data, np.uint8, name, shape)


@dataclass(frozen=True)
class MeshPart:
    """One triangle mesh with UV texture, vertex RGBA or a uniform RGBA color.

    Vertices use object coordinates in meters. UV coordinates follow trimesh: u increases
    rightward and v increases upward. Alpha values of at least 128 are opaque.
    """

    vertices: np.ndarray
    faces: np.ndarray
    uv: np.ndarray | None = None
    texture: np.ndarray | None = None
    vertex_colors: np.ndarray | None = None
    color: np.ndarray | None = None

    def __post_init__(self) -> None:
        vertices = _array(self.vertices, np.float64, "vertices", (None, 3))
        faces_input = np.asarray(self.faces)
        if not np.issubdtype(faces_input.dtype, np.integer):
            raise ValueError("faces must contain integer indices")
        faces = _array(faces_input, np.int64, "faces", (None, 3))
        if not len(vertices) or not len(faces):
            raise ValueError("MeshPart requires vertices and triangle faces")
        if np.any(faces < 0) or np.any(faces >= len(vertices)):
            raise ValueError("faces contain invalid vertex indices")
        textured = self.uv is not None or self.texture is not None
        if sum((textured, self.vertex_colors is not None, self.color is not None)) != 1:
            raise ValueError("MeshPart requires exactly one color source")
        if textured:
            if self.uv is None or self.texture is None:
                raise ValueError("A textured MeshPart requires both uv and texture")
            uv = _array(self.uv, np.float64, "uv", (len(vertices), 2))
            texture = _colors(self.texture, "texture", (None, None, 4))
            if not texture.shape[0] or not texture.shape[1]:
                raise ValueError("texture must be nonempty")
            object.__setattr__(self, "uv", uv)
            object.__setattr__(self, "texture", texture)
        elif self.vertex_colors is not None:
            object.__setattr__(self, "vertex_colors",
                               _colors(self.vertex_colors, "vertex_colors", (len(vertices), 4)))
        else:
            object.__setattr__(self, "color", _colors(self.color, "color", (4,)))
        object.__setattr__(self, "vertices", vertices)
        object.__setattr__(self, "faces", faces)


@dataclass(frozen=True)
class MeshAsset:
    """Colored mesh parts in x-forward, y-left, z-up coordinates with a bottom-center origin.

    All vertices must fit inside the box described by the full length, width and height in
    ``extent``. Mesh assets have no nominal texture.
    """

    parts: tuple[MeshPart, ...]
    extent: tuple[float, float, float]
    category: str

    def __post_init__(self) -> None:
        parts = tuple(self.parts)
        if not parts or any(not isinstance(part, MeshPart) for part in parts):
            raise ValueError("MeshAsset requires at least one MeshPart")
        if not isinstance(self.category, str) or not self.category.strip():
            raise ValueError("MeshAsset.category must be a nonempty string")
        extent = np.asarray(self.extent, dtype=float)
        if extent.shape != (3,) or not np.isfinite(extent).all() or np.any(extent < 0):
            raise ValueError("MeshAsset.extent must contain three finite nonnegative dimensions")
        lower = np.array([-extent[0] / 2, -extent[1] / 2, 0.])
        upper = np.array([extent[0] / 2, extent[1] / 2, extent[2]])
        for part in parts:
            if np.any(part.vertices < lower - 1e-6) or np.any(part.vertices > upper + 1e-6):
                raise ValueError("Mesh vertices must lie inside the bottom-centered extent")
        object.__setattr__(self, "parts", parts)
        object.__setattr__(self, "extent", tuple(float(value) for value in extent))


def _axis(name: str) -> np.ndarray:
    if name not in ("+x", "-x", "+y", "-y", "+z", "-z"):
        raise ValueError("Axes must be one of +x, -x, +y, -y, +z, -z")
    result = np.zeros(3)
    result["xyz".index(name[1])] = 1 if name[0] == "+" else -1
    return result


def _rgba(value: Any) -> np.ndarray:
    color = np.asarray(value)
    if color.shape not in ((3,), (4,)) or not np.isfinite(color).all():
        raise ValueError("color must contain three or four finite integer channels")
    if np.any(color < 0) or np.any(color > 255) or np.any(color != np.round(color)):
        raise ValueError("color channels must be integers in [0, 255]")
    if len(color) == 3:
        color = np.append(color, 255)
    return color.astype(np.uint8)


def _material_colors(material: Any) -> tuple[np.ndarray | None, np.ndarray | None]:
    image = getattr(material, "baseColorTexture", None)
    factor = getattr(material, "baseColorFactor", None)
    pbr = hasattr(material, "baseColorTexture")
    if not pbr:
        image = getattr(material, "image", None)
        factor = getattr(material, "diffuse", None)
    if image is not None:
        rgba = np.asarray(image.convert("RGBA")).copy()
        if pbr and factor is not None:
            rgba = np.round(rgba.astype(float) * np.asarray(factor) / 255).astype(np.uint8)
    elif factor is not None:
        rgba = _rgba(factor)
    else:
        return None, None
    if pbr:
        mode = getattr(material, "alphaMode", None) or "OPAQUE"
        if mode == "OPAQUE":
            rgba[..., 3] = 255
        elif mode == "MASK":
            cutoff = getattr(material, "alphaCutoff", None)
            cutoff = .5 if cutoff is None else float(cutoff)
            rgba[..., 3] = (rgba[..., 3].astype(float) / 255 >= cutoff) * 255
    return (rgba, None) if image is not None else (None, rgba)


def _mesh_part(mesh: Any, fallback: np.ndarray | None) -> MeshPart:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    visual = mesh.visual
    texture = uniform = uv = colors = None
    if getattr(visual, "kind", None) == "texture":
        texture, uniform = _material_colors(visual.material)
        if texture is not None:
            uv = visual.uv
            if uv is None:
                raise ValueError("A textured mesh requires UV coordinates")
    elif getattr(visual, "kind", None) == "vertex":
        colors = np.asarray(visual.vertex_colors, dtype=np.uint8)
    elif getattr(visual, "kind", None) == "face":
        colors = np.repeat(np.asarray(visual.face_colors, dtype=np.uint8), 3, axis=0)
        vertices = vertices[faces.reshape(-1)]
        faces = np.arange(len(vertices), dtype=np.int64).reshape(-1, 3)
    has_color = any(value is not None for value in (texture, uniform, colors))
    if fallback is not None and has_color:
        raise ValueError("color conflicts with colors provided by the mesh file")
    if not has_color:
        if fallback is None:
            raise ValueError("Mesh file has no colors. Provide color explicitly")
        uniform = fallback
    return MeshPart(vertices, faces, uv, texture, colors, uniform)


def load_mesh_asset(
    path: str | Path, *, category: str, scale: float = 1., forward_axis: str = "+x",
    up_axis: str = "+z", origin: str = "bottom_center", color: Sequence[int] | None = None,
) -> MeshAsset:
    """Load OBJ, glTF/GLB, mesh PLY or STL as colored parts in framework object coordinates.

    ``scale`` converts file units to meters. Explicit forward and up axes define a right-handed
    mapping to +X and +Z. ``bottom_center`` centers the horizontal bounds and moves the bottom
    to zero. ``file`` preserves the converted origin and requires nonnegative z. ``color``
    supplies RGB or RGBA channels only when the file has no color. External material resources
    resolve relative to the mesh file. Only base colors are used.
    """
    path = Path(path)
    suffixes = (".obj", ".gltf", ".glb", ".ply", ".stl")
    if path.suffix.lower() not in suffixes:
        raise ValueError("Supported mesh formats: OBJ, glTF/GLB, mesh PLY, STL")
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be positive and finite")
    forward, up = _axis(forward_axis), _axis(up_axis)
    if np.dot(forward, up) != 0:
        raise ValueError("forward_axis and up_axis must be orthogonal")
    if origin not in ("bottom_center", "file"):
        raise ValueError("origin must be 'bottom_center' or 'file'")
    fallback = None if color is None else _rgba(color)
    try:
        import trimesh
    except ImportError as error:
        raise ImportError("Mesh loading requires trimesh. Install the rendering extra") from error
    loaded = trimesh.load(path, process=False)
    meshes = loaded.dump(concatenate=False) if isinstance(loaded, trimesh.Scene) else (loaded,)
    parts = [_mesh_part(mesh, fallback) for mesh in meshes
             if isinstance(mesh, trimesh.Trimesh) and len(mesh.faces)]
    if not parts:
        raise ValueError("Mesh asset has no triangle faces. Gaussian PLY assets are not supported")
    rotation = np.stack((forward, np.cross(up, forward), up))
    converted = [part.vertices @ rotation.T * scale for part in parts]
    all_vertices = np.concatenate(converted)
    if origin == "bottom_center":
        lower, upper = all_vertices.min(axis=0), all_vertices.max(axis=0)
        offset = np.array([(lower[0] + upper[0]) / 2, (lower[1] + upper[1]) / 2, lower[2]])
        converted = [vertices - offset for vertices in converted]
        extent = tuple(upper - lower)
    else:
        if np.any(all_vertices[:, 2] < 0):
            raise ValueError("origin='file' requires z >= 0. Use origin='bottom_center' instead")
        extent = (2 * np.abs(all_vertices[:, 0]).max(),
                  2 * np.abs(all_vertices[:, 1]).max(), all_vertices[:, 2].max())
    normalized = tuple(
        MeshPart(vertices, part.faces, part.uv, part.texture, part.vertex_colors, part.color)
        for part, vertices in zip(parts, converted))
    return MeshAsset(normalized, extent, category)


__all__ = ["MeshPart", "MeshAsset", "load_mesh_asset"]
