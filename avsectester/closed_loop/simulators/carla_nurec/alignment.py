"""Align a reconstructed NuRec scene with the right-handed CARLA world."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike

from avsectester.geometry.geodesy import enu_from_ecef
from avsectester.geometry.poses import make_pose, validated_transform


@dataclass(frozen=True, eq=False)
class SceneAlignment:
    """Convert between NuRec scene coordinates and right-handed CARLA world coordinates."""

    scene_to_carla_world: np.ndarray
    """CARLA-world-from-scene transform in meters and right-handed axes."""
    _carla_world_to_scene: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        forward = validated_transform(self.scene_to_carla_world)
        rotation = forward[:3, :3].T
        inverse = make_pose(rotation, -rotation @ forward[:3, 3])
        inverse.setflags(write=False)
        object.__setattr__(self, "scene_to_carla_world", forward)
        object.__setattr__(self, "_carla_world_to_scene", inverse)

    @classmethod
    def from_scene(
        cls, xodr_origin: tuple[float, float, float], ecef_from_scene: ArrayLike,
    ) -> SceneAlignment:
        """Build the alignment from an OpenDRIVE origin and a scene-to-ECEF transform."""
        return cls(enu_from_ecef(*xodr_origin) @ validated_transform(ecef_from_scene))

    def to_carla(self, scene_from_pose: ArrayLike) -> np.ndarray:
        """Transform a scene-world pose into the right-handed CARLA world."""
        return self.scene_to_carla_world @ np.asarray(scene_from_pose, dtype=float)

    def from_carla(self, carla_world_from_pose: ArrayLike) -> np.ndarray:
        """Transform a right-handed CARLA world pose into the scene world."""
        return self._carla_world_to_scene @ np.asarray(carla_world_from_pose, dtype=float)

    def vector_to_carla(self, scene_vectors: ArrayLike) -> np.ndarray:
        """Rotate scene-world vectors into the right-handed CARLA world without translating."""
        return np.asarray(scene_vectors, dtype=float) @ self.scene_to_carla_world[:3, :3].T

    def vector_from_carla(self, carla_world_vectors: ArrayLike) -> np.ndarray:
        """Rotate right-handed CARLA world vectors into the scene world without translating."""
        return np.asarray(carla_world_vectors, dtype=float) @ self._carla_world_to_scene[:3, :3].T


__all__ = ['SceneAlignment']
