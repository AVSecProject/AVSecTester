"""Deploy the selected vehicle mesh before the first capture."""
import numpy as np

from avsectester.attack import AttackRuntime, OnlineContext
from avsectester.attack.insertion import InsertObject
from avsectester.geometry.poses import validated_transform
from avsectester.rendering import Harmonizer, MeshAsset


class VehicleInsertionAttackRuntime(AttackRuntime):
    def __init__(self, asset: MeshAsset, world_from_object: np.ndarray,
                 harmonizer: Harmonizer | None) -> None:
        self.asset = asset
        self.world_from_object = validated_transform(world_from_object)
        self.harmonizer = harmonizer

    def setup(self, context: OnlineContext) -> tuple[InsertObject, ...]:
        return (InsertObject("inserted_vehicle", self.asset, self.world_from_object,
                             harmonizer=self.harmonizer),)
