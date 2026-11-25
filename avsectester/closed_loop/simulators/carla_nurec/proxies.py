"""Represent recorded NuRec road users with collision-enabled CARLA actors."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from avsectester.data_types import ActorState
from avsectester.geometry.boxes import oriented_boxes_distance

from ..carla.conversions import carla_transform
from ..carla.vehicles import BlueprintGeometry, best_fit_blueprint
from ..carla.world import CarlaWorldSession
from .alignment import SceneAlignment


class RecordedTrackProxies:
    """Maintain kinematic CARLA obstacles for recorded vehicle, rider and person tracks."""

    def __init__(
        self, session: CarlaWorldSession, alignment: SceneAlignment,
        vehicle_geometries: Sequence[BlueprintGeometry], walker_geometries: Sequence[BlueprintGeometry],
    ) -> None:
        self.session = session
        self.alignment = alignment
        self._geometries = {"vehicle": vehicle_geometries, "rider": vehicle_geometries,
                            "person": walker_geometries}
        # Each live proxy keeps its blueprint geometry and the actor pose queued for the next tick.
        self._actors: dict[str, tuple[Any, BlueprintGeometry, np.ndarray] | None] = {}
        self._missing: set[str] = set()

    @property
    def missing(self) -> frozenset[str]:
        """Recorded tracks for which a proxy could not be created during this session."""
        return frozenset(self._missing)

    @property
    def count(self) -> int:
        """Number of live proxies."""
        return sum(proxy is not None for proxy in self._actors.values())

    def sync(self, actors: Mapping[str, ActorState]) -> None:
        """Queue moves, remove disappeared tracks and spawn newly visible proxies."""
        active = {identity: state for identity, state in actors.items()
                  if state.category in self._geometries}
        for identity in sorted(self._actors.keys() - active.keys()):
            proxy = self._actors.pop(identity)
            if proxy is not None:
                self.session.destroy(proxy[0])
        for identity, state in sorted(active.items()):
            proxy = self._actors.get(identity)
            if identity in self._actors and proxy is None:
                continue
            pose = self.alignment.to_carla(state.box_transform())
            if proxy is not None:
                actor, geometry, _ = proxy
                pose[:3, 3] -= pose[:3, :3] @ geometry.box_location
                self.session.place(actor, pose)
                self._actors[identity] = actor, geometry, pose
                continue
            assert state.category is not None
            geometries = self._geometries[state.category]
            self._actors[identity] = None
            if not geometries or state.extent is None:
                self._missing.add(identity)
                continue
            geometry = best_fit_blueprint(geometries, state.extent)
            pose[:3, 3] -= pose[:3, :3] @ geometry.box_location
            blueprint = self.session.blueprint(geometry.id)
            staging = pose.copy()
            staging_z = self.session.level_top_z() + 10.0
            for attempt in range(6):
                staging[2, 3] = staging_z + 6.0 * attempt
                actor = self.session.spawn(blueprint, carla_transform(staging))
                if actor is not None:
                    self.session.place_kinematic(actor, pose)
                    self._actors[identity] = actor, geometry, pose
                    break
            else:
                self._missing.add(identity)

    def touching(self, world_from_box: ArrayLike, extent: ArrayLike) -> list[str]:
        """Return sorted track IDs whose proxy box overlaps or touches the given box from above.

        Boxes use right-handed CARLA world coordinates and full extents. Proxy poses are the
        poses queued by the latest ``sync``, which the next world tick applies.
        """
        return [identity for identity, proxy in sorted(self._actors.items()) if proxy is not None
                and oriented_boxes_distance(world_from_box, extent, proxy[1].box_transform(proxy[2]),
                                            proxy[1].dimensions) == 0.]

    def destroy_all(self) -> None:
        """Queue destruction of every live proxy."""
        self.sync({})


__all__ = ["RecordedTrackProxies"]
