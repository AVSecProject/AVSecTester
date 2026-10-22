"""Simulator-independent world objects, planar rendering and appearance compositing."""

from .objects import ObjectAsset, PlacedObject, PlaneSurface
from .harmonizers import ClassicHarmonizer, Harmonizer, PCTNetHarmonizer
from .surfaces import render_surfaces
from .compositing import CompositeResult, ObjectVisibility, blend_rgba, composite_objects
from .images import load_rgba_image
from .signs import SignAsset, STOP_SIGN_FACE, post_rgba, stop_sign_asset

__all__ = [
    "ObjectAsset", "PlacedObject", "PlaneSurface", "ClassicHarmonizer", "Harmonizer",
    "PCTNetHarmonizer", "render_surfaces", "CompositeResult", "ObjectVisibility", "blend_rgba", "composite_objects",
    "load_rgba_image", "SignAsset", "STOP_SIGN_FACE", "post_rgba", "stop_sign_asset",
]
