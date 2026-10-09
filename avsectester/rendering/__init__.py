"""Shared camera, surface sampling and visibility interfaces for insertion rendering."""

from .cameras import Camera
from .types import (
    EvidenceProvider,
    GeometryProvider,
    GeometryVisibilityEstimator,
    InsertionGeometry,
    Visibility,
    VisibilityEvidence,
)

__all__ = [
    "Camera",
    "EvidenceProvider",
    "GeometryProvider",
    "GeometryVisibilityEstimator",
    "InsertionGeometry",
    "Visibility",
    "VisibilityEvidence",
]
