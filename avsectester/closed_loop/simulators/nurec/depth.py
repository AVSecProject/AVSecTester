"""Cached NuRec scene meshes and per-capture dynamic actor depth."""

from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

from avsectester.data_types import ActorState

if TYPE_CHECKING:
    from avsectester.geometry.cameras import Camera
    from avsectester.geometry.mesh import MeshRaycaster
    import numpy as np


@lru_cache(maxsize=2)
def _cached_scene_raycaster(path: str, modified_ns: int, size: int) -> MeshRaycaster:
    from avsectester.datasets.nurec import read_scene_mesh
    from avsectester.geometry.mesh import MeshRaycaster

    return MeshRaycaster(*read_scene_mesh(path))


def _scene_raycaster(path: str) -> MeshRaycaster:
    """Share two immutable scene BVHs by resolved archive path, mtime_ns and size."""
    source = Path(path).resolve()
    stat = source.stat()
    return _cached_scene_raycaster(str(source), stat.st_mtime_ns, stat.st_size)


@lru_cache(maxsize=4)
def _cached_ground_raycaster(path: str, modified_ns: int, size: int,
                             mesh_name: str) -> MeshRaycaster:
    import io
    import zipfile
    import trimesh
    from avsectester.geometry.mesh import MeshRaycaster

    with zipfile.ZipFile(path) as archive:
        if mesh_name not in archive.namelist():
            raise ValueError(f"NuRec ground contact archive {path!r} is missing: {mesh_name}")
        mesh = trimesh.load(io.BytesIO(archive.read(mesh_name)), file_type="ply", process=False)
    return MeshRaycaster(mesh.vertices, mesh.faces)


def _ground_raycaster(path: str, mesh_name: str) -> MeshRaycaster:
    source = Path(path).resolve()
    stat = source.stat()
    return _cached_ground_raycaster(str(source), stat.st_mtime_ns, stat.st_size, mesh_name)


def _actor_depth(camera: Camera, cam_from_world: np.ndarray,
                 actors: Mapping[str, ActorState], depth: np.ndarray) -> np.ndarray:
    """Intersect each captured actor only in its padded projected image rectangle."""
    import numpy as np
    from avsectester.geometry.boxes import box_ray_distances
    from avsectester.geometry.cameras import camera_ray_mask
    from avsectester.geometry.projection import project_box

    world_from_camera = np.linalg.inv(cam_from_world)
    minimum_world_ray_length = np.linalg.svd(world_from_camera[:3, :3], compute_uv=False).min()
    for actor_id, state in actors.items():
        if state.extent is None:
            raise ValueError(f"NuRec mesh depth requires bounding-box extent for actor {actor_id!r}")
        if min(state.extent) <= 0:
            continue
        world_from_box = state.box_transform()
        bounds = project_box(camera, cam_from_world, world_from_box, state.extent)
        if bounds is None:
            continue
        lower = np.maximum(np.floor(bounds[:2]).astype(int) - 2, (0, 0))
        upper = np.minimum(np.ceil(bounds[2:]).astype(int) + 2, (camera.width, camera.height))
        x0, y0 = lower
        x1, y1 = upper
        region = depth[y0:y1, x0:x1]
        camera_from_box = cam_from_world @ world_from_box
        nearest_z = (camera_from_box[2, 3]
                     - np.abs(camera_from_box[2, :3]) @ (np.asarray(state.extent) / 2))
        # A scene hit in front of the entire box cannot be replaced by this actor.
        # The singular-value factor also preserves the existing Euclidean-distance
        # convention for recorded rotations with finite calibration precision.
        nearest_depth = np.nextafter(max(0., nearest_z) * minimum_world_ray_length, -np.inf)
        indices = np.flatnonzero((region > nearest_depth).ravel())
        if not len(indices):
            continue
        pixels = np.column_stack((indices % (x1 - x0) + x0 + .5,
                                  indices // (x1 - x0) + y0 + .5))
        rays = np.asarray(camera.unproject(pixels), dtype=np.float64)
        lengths = np.linalg.norm(rays, axis=1)
        valid = camera_ray_mask(camera, rays) & np.isfinite(lengths) & (lengths > 0)
        if valid.any():
            unit_rays = rays[valid] / lengths[valid, None]
            distances = box_ray_distances(world_from_box, state.extent,
                                          unit_rays, world_from_camera)
            # Recorded calibration rotations have finite precision. Mesh queries
            # normalize the transformed world rays, so preserve that Euclidean
            # distance convention when merging analytic cuboid intersections.
            world_lengths = np.linalg.norm(unit_rays @ world_from_camera[:3, :3].T, axis=1)
            actor_depth = distances * world_lengths * unit_rays[:, 2]
            rows = indices[valid] // (x1 - x0) + y0
            columns = indices[valid] % (x1 - x0) + x0
            depth[rows, columns] = np.minimum(depth[rows, columns], actor_depth)
    return depth
