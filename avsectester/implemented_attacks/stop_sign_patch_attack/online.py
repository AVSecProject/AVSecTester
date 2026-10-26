"""Deploy one replica stop-sign face during online setup."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from avsectester.attack import AttackRuntime, OnlineContext
from avsectester.attack.insertion import InsertObject
from avsectester.rendering import Harmonizer, ObjectAsset


REPLICA_OBJECT_ID = "stop_sign_replica"


class StopSignPatchAttackRuntime(AttackRuntime):
    """Insert one replica face; the simulator owns subsequent capture and cleanup."""

    def __init__(self, asset: ObjectAsset, world_from_object: np.ndarray,
                 harmonizer: Harmonizer | None = None) -> None:
        self.asset = asset
        self.harmonizer = harmonizer
        self.world_from_object = world_from_object.copy()

    def setup(self, context: OnlineContext) -> Sequence[InsertObject]:
        return (InsertObject(REPLICA_OBJECT_ID, self.asset, self.world_from_object,
                             harmonizer=self.harmonizer),)
