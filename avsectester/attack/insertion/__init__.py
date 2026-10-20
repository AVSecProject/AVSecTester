"""Public assets, calibrated compositing, visibility and effects for user attacks."""
from .assets import (
    CheckerboardTexture, ImageTexture, Insertion, PlaneAsset, PlaneSurface, PlanarAsset,
    ResolvedInsertion, load_rgba_image, resolve_insertion,
)
from .compositor import PatchCompositor, render_resolved, warp_patch
from .effect import InsertionAttackEffect
from .harmonizers import ClassicHarmonizer, Harmonizer, PCTNetHarmonizer
from .types import Visibility, VisibilityEvidence
from .visibility import (
    CuboidVisibilityEstimator, DepthVisibilityEstimator, visibility_from_depth, visibility_from_masks,
)

__all__ = [
    "CheckerboardTexture", "ImageTexture", "Insertion", "PlaneAsset", "PlaneSurface", "PlanarAsset",
    "ResolvedInsertion", "load_rgba_image", "resolve_insertion", "PatchCompositor", "render_resolved",
    "warp_patch", "InsertionAttackEffect", "ClassicHarmonizer", "Harmonizer", "PCTNetHarmonizer",
    "Visibility", "VisibilityEvidence", "CuboidVisibilityEstimator", "DepthVisibilityEstimator",
    "visibility_from_depth", "visibility_from_masks",
]
