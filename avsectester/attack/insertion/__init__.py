"""Public assets, calibrated compositing, visibility and actions for user attacks."""
from .assets import (
    Insertion, PlaneAsset, PlanarAsset,
    ResolvedInsertion, resolve_insertion,
)
from .compositor import PatchCompositor, render_resolved
from .action import InsertObject, InsertionAttackAction
from .types import Visibility, VisibilityEvidence
from .visibility import (
    CuboidVisibilityEstimator, DepthVisibilityEstimator, visibility_from_depth, visibility_from_masks,
)


__all__ = [
    'Insertion',
    'PlaneAsset',
    'PlanarAsset',
    'ResolvedInsertion',
    'resolve_insertion',
    'PatchCompositor',
    'render_resolved',
    'InsertObject',
    'InsertionAttackAction',
    'Visibility',
    'VisibilityEvidence',
    'CuboidVisibilityEstimator',
    'DepthVisibilityEstimator',
    'visibility_from_depth',
    'visibility_from_masks',
]
