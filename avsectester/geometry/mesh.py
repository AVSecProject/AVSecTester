"""CPU triangle-mesh ray queries requiring Embree through the optional nurec extra."""

from __future__ import annotations

from functools import lru_cache
from collections.abc import Sequence
from typing import Any

import numpy as np

from .cameras import Camera, camera_ray_mask
from .poses import validated_transform


@lru_cache(maxsize=1)
def _dependencies() -> tuple[Any, Any, Any]:
    try:
        import trimesh
        from embreex.rtcore_scene import EmbreeScene
        from embreex.mesh_construction import TriangleMesh
    except (ImportError, OSError) as error:
        raise ImportError(
            "Mesh depth requires trimesh and embreex; install avsectester[nurec]"
        ) from error
    return trimesh, EmbreeScene, TriangleMesh


class MeshRaycaster:
    """An immutable triangle mesh accelerated by Embree.

    ``vertices`` are finite world coordinates with shape ``(N, 3)``; ``faces`` are
    integer vertex indices with shape ``(M, 3)``. Construction copies both arrays.
    Queries return Euclidean distances along each ray, with ``inf`` for no hit.
    Embree is required for mesh queries. Its optional installation dependencies load only
    when this class is used.
    """

    @staticmethod
    def check_dependencies() -> None:
        """Fail early when the optional CPU mesh-query dependencies are unavailable."""
        _dependencies()

    def __init__(self, vertices: np.ndarray, faces: np.ndarray) -> None:
        trimesh, self._scene_type, self._geometry_type = _dependencies()
        vertices = np.array(vertices, dtype=np.float64, copy=True)
        faces = np.asarray(faces)
        if vertices.ndim != 2 or vertices.shape[1:] != (3,) or not np.isfinite(vertices).all():
            raise ValueError("Mesh vertices must be finite coordinates with shape (N, 3)")
        if faces.ndim != 2 or faces.shape[1:] != (3,) or faces.dtype.kind not in "iu":
            raise ValueError("Mesh faces must be integer indices with shape (M, 3)")
        faces = np.array(faces, dtype=np.int64, copy=True)
        if faces.size and (faces.min() < 0 or faces.max() >= len(vertices)):
            raise ValueError("Mesh faces contain an invalid vertex index")
        # Do not clean or expand the scene's millions of triangles. Keeping the
        # original indexed mesh also avoids a second, much larger triangle array.
        # Translate before Embree converts ray origins to float32; otherwise a
        # scene's large world offset would unnecessarily reduce local precision.
        self._origin = (vertices.min(axis=0) / 2 + vertices.max(axis=0) / 2
                        if len(vertices) else np.zeros(3))
        vertices -= self._origin
        self._vertices, self._faces = vertices, faces
        self._vertices.setflags(write=False)
        self._faces.setflags(write=False)
        self._native_origin = vertices.min(axis=0) if len(vertices) else np.zeros(3)
        self._native_scale = 1.
        if len(faces):
            # Match trimesh's referenced-vertex bounding-box scale once. Queries
            # use only these immutable arrays and the native scene, never its hash cache.
            referenced = np.zeros(len(vertices), dtype=bool)
            referenced[faces] = True
            used = vertices[referenced]
            extent = used.max(axis=0) - used.min(axis=0)
            scale = float((extent ** 2).sum() ** .5)
            self._native_scale = 100. / (scale if scale >= trimesh.constants.tol.zero else 1.)
        self._scene: Any = None

    def _native_scene(self) -> Any:
        if self._scene is None:
            vertices = ((self._vertices - self._native_origin) * self._native_scale).astype(np.float32)
            scene = self._scene_type()
            self._geometry_type(scene=scene, vertices=vertices,
                                indices=self._faces.astype(np.int32))
            self._scene = scene
        return self._scene

    def distances(self, origins: np.ndarray, directions: np.ndarray) -> np.ndarray:
        """First-hit distances for paired ``(N, 3)`` origins and directions.

        Directions are normalized internally. Zero/nonfinite directions and
        nonfinite origins are rejected. Work is batched to bound temporary memory.
        """
        origins = np.asarray(origins, dtype=np.float64)
        directions = np.asarray(directions, dtype=np.float64)
        if origins.ndim != 2 or origins.shape[1:] != (3,) or origins.shape != directions.shape:
            raise ValueError("Ray origins and directions must have matching shape (N, 3)")
        lengths = np.linalg.norm(directions, axis=1)
        if not np.isfinite(origins).all() or not np.isfinite(lengths).all() or (lengths <= 0).any():
            raise ValueError("Ray origins must be finite and directions finite and nonzero")
        distances = np.full(len(origins), np.inf, dtype=np.float64)
        if not len(self._faces):
            return distances
        origins = origins - self._origin
        directions = directions / lengths[:, None]
        for start in range(0, len(origins), 65536):
            stop = start + 65536
            ray_origins, rays = origins[start:stop], directions[start:stop]
            # Preserve the established float32 conversion and normalization order
            # before Embree traversal, including trimesh's local-origin scaling.
            native_origins = ray_origins.astype(np.float32).astype(np.float64)
            native_origins = ((native_origins - self._native_origin) * self._native_scale).astype(np.float32)
            native_rays = rays.astype(np.float32)
            inverse_lengths = np.sqrt(np.dot(native_rays * native_rays, [1., 1., 1.]))
            inverse_lengths **= -1
            native_rays = (native_rays * inverse_lengths[:, None]).astype(np.float32)
            triangle_ids = self._native_scene().run(native_origins, native_rays)
            hits = np.flatnonzero(triangle_ids >= 0)
            if not len(hits):
                continue
            # Embree selects the first triangle using float32. Recompute that
            # triangle's intersection in float64 to retain metric depth precision.
            triangles = self._vertices[self._faces[triangle_ids[hits]]]
            normals = np.cross(triangles[:, 1] - triangles[:, 0],
                               triangles[:, 2] - triangles[:, 0])
            denominator = np.einsum("ij,ij->i", normals, rays[hits])
            numerator = np.einsum("ij,ij->i", normals, triangles[:, 0] - ray_origins[hits])
            values = np.divide(numerator, denominator, out=np.full(len(hits), np.inf),
                               where=np.abs(denominator) > 1e-15)
            values[values < 0] = np.inf
            distances[start + hits] = values
        return distances


def camera_depth(camera: Camera, cam_from_world: np.ndarray,
                 raycasters: Sequence[MeshRaycaster]) -> np.ndarray:
    """Optical-axis z depth at pixel centres, taking the nearest of all supplied meshes.

    Rays use the camera's own lens model and are transformed into world coordinates.
    Pixels outside the valid forward lens domain, and rays without a hit, are positive inf.
    The returned float64 array has shape ``(camera.height, camera.width)``.
    """
    raycasters = tuple(raycasters)
    if any(not isinstance(raycaster, MeshRaycaster) for raycaster in raycasters):
        raise TypeError("raycasters must contain MeshRaycaster instances")
    world_from_camera = np.linalg.inv(validated_transform(cam_from_world))
    depth = np.full(camera.height * camera.width, np.inf, dtype=np.float64)
    for start in range(0, len(depth), 65536):
        indices = np.arange(start, min(start + 65536, len(depth)))
        pixels = np.column_stack((indices % camera.width + .5, indices // camera.width + .5))
        rays = np.asarray(camera.unproject(pixels), dtype=np.float64)
        lengths = np.linalg.norm(rays, axis=1)
        valid = camera_ray_mask(camera, rays) & np.isfinite(lengths) & (lengths > 0)
        if not valid.any():
            continue
        optical_rays = rays[valid] / lengths[valid, None]
        world_rays = optical_rays @ world_from_camera[:3, :3].T
        origins = np.broadcast_to(world_from_camera[:3, 3], world_rays.shape)
        distances = np.full(len(world_rays), np.inf, dtype=np.float64)
        for raycaster in raycasters:
            distances = np.minimum(distances, raycaster.distances(origins, world_rays))
        depth[indices[valid]] = distances * optical_rays[:, 2]
    return depth.reshape(camera.height, camera.width)
