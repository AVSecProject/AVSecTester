"""Data shared by visibility providers and insertion renderers."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Visibility:
    """Measured fraction or a coarse dataset representative. None camera means source-wide scope."""

    fraction: float
    source: str
    camera: str | None = None
    label: str | None = None

    def __post_init__(self):
        if not math.isfinite(self.fraction) or not 0 <= self.fraction <= 1:
            raise ValueError("visibility fraction must be in [0, 1]")


@dataclass(frozen=True)
class VisibilityEvidence:
    """One insertion's visibility and image-sized sampling results for one camera/instant.

    ``reference_pixels`` includes the full off-image alpha silhouette. Masks and
    ``target_depth`` are cropped to the image. Depth is optical-axis Z in metres.
    ``sampled_rgba`` optionally carries the exact surface samples used by the estimator,
    so compositing can reuse them. Custom providers may omit it.
    """

    visibility: Visibility | None
    reference_mask: np.ndarray
    visible_mask: np.ndarray
    target_depth: np.ndarray
    reference_pixels: int
    reason: str = ""
    sampled_rgba: np.ndarray | None = None
