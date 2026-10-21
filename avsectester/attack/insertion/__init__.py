"""Public assets, calibrated compositing, visibility and actions for user attacks."""
from .assets import (
    CheckerboardTexture, ImageTexture, Insertion, PlaneAsset, PlaneSurface, PlanarAsset,
    ResolvedInsertion, load_rgba_image, resolve_insertion,
)
from .compositor import PatchCompositor, render_resolved, warp_patch
from .action import InsertionAttackAction
from .harmonizers import ClassicHarmonizer, Harmonizer, PCTNetHarmonizer
from .types import Visibility, VisibilityEvidence
from .visibility import (
    CuboidVisibilityEstimator, DepthVisibilityEstimator, visibility_from_depth, visibility_from_masks,
)

__all__ = [
    "CheckerboardTexture", "ImageTexture", "Insertion", "PlaneAsset", "PlaneSurface", "PlanarAsset",
    "ResolvedInsertion", "load_rgba_image", "resolve_insertion", "PatchCompositor", "render_resolved",
    "warp_patch", "InsertionAttackAction", "ClassicHarmonizer", "Harmonizer", "PCTNetHarmonizer",
    "Visibility", "VisibilityEvidence", "CuboidVisibilityEstimator", "DepthVisibilityEstimator",
    "visibility_from_depth", "visibility_from_masks",
]
