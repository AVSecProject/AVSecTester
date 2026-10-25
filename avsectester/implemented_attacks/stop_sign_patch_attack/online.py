"""Deploy one transparent patch panel during online setup."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from avsectester.attack import AttackRuntime, OnlineContext
from avsectester.attack.insertion import InsertObject
from avsectester.rendering import ObjectAsset


class StopSignPatchAttackRuntime(AttackRuntime):
    """Insert one patch; the simulator owns subsequent capture and cleanup."""

    def __init__(self, asset: ObjectAsset, world_from_object: np.ndarray) -> None:
        self.asset = asset
        self.world_from_object = world_from_object.copy()

    def setup(self, context: OnlineContext) -> Sequence[InsertObject]:
        return (InsertObject("stop_sign_patch", self.asset, self.world_from_object),)
