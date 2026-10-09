"""Data shared by visibility providers and insertion renderers."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import NamedTuple, Protocol

import numpy as np

from avsectester.insertion import ActorPose, ResolvedInsertion
from avsectester.plane import Observation
from .cameras import Camera


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


class InsertionGeometry(NamedTuple):
    """One observation's world geometry, with distances in metres.

    Actor keys are stable IDs or bound role aliases. Actor/victim poses map local axes
    into world axes. ``cam_from_world`` maps world points into optical camera axes
    (x right, y down, z forward). Tuple unpacking remains supported.
    """

    actors: Mapping[str, ActorPose]
    victim: ActorPose
    cam_from_world: np.ndarray


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


class GeometryProvider(Protocol):
    def __call__(self, observation: Observation) -> InsertionGeometry: ...


class EvidenceProvider(Protocol):
    def __call__(
        self,
        observation: Observation,
        resolved: tuple[ResolvedInsertion, ...],
        geometry: InsertionGeometry,
    ) -> Mapping[str, VisibilityEvidence]: ...


class GeometryVisibilityEstimator(Protocol):
    """Visibility from known actor geometry and other insertions, without scene depth."""

    def estimate(
        self,
        subject: ResolvedInsertion,
        camera: Camera,
        cam_from_world: np.ndarray,
        *,
        occluders: Mapping[str, ActorPose],
        camera_name: str,
        other_insertions: Sequence[ResolvedInsertion] = (),
    ) -> VisibilityEvidence: ...
