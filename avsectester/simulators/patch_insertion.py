"""Shared geometry-based insertion rendering and observation adapters.

``InsertionRenderer`` resolves explicit placements against current actor poses.
``render_resolved`` uses the same sampling and visibility evidence as selection.
The image-space warp primitive is also available for user-defined image operations.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from avsectester.plane import Observation
from avsectester.insertion import Insertion, ResolvedInsertion
from avsectester.rendering.cameras import Camera
from avsectester.rendering.types import InsertionGeometry, GeometryProvider, EvidenceProvider, VisibilityEvidence
from avsectester.simulators.viz import View, camera_view

from avsectester.rendering.harmonizers import Harmonizer, ClassicHarmonizer


def warp_patch(frame_rgb: np.ndarray, dst_quad: np.ndarray, patch_rgba: np.ndarray):
    """Perspective-warp ``patch_rgba`` onto the image quad ``dst_quad`` (4x2, TL,TR,BR,BL); composite.

    Returns ``(composite_rgb uint8, mask uint8 HxW)`` — the patch alpha-blended over the frame.
    """
    import cv2

    h, w = frame_rgb.shape[:2]
    ph, pw = patch_rgba.shape[:2]
    src = np.array([[0, 0], [pw, 0], [pw, ph], [0, ph]], dtype=np.float32)
    homography = cv2.getPerspectiveTransform(src, np.asarray(dst_quad, dtype=np.float32))
    warped = cv2.warpPerspective(patch_rgba, homography, (w, h), flags=cv2.INTER_LINEAR)
    alpha = warped[:, :, 3:4].astype(np.float32) / 255.0
    comp = frame_rgb.astype(np.float32) * (1 - alpha) + warped[:, :, :3].astype(np.float32) * alpha
    mask = (warped[:, :, 3] > 127).astype(np.uint8) * 255
    return comp.astype(np.uint8), mask


def render_plane(
    frame_rgb: np.ndarray,
    camera: Any,
    cam_from_world: np.ndarray,
    corners_world: np.ndarray,
    texture_rgba: np.ndarray,
    soften: float = 0.0,
    margin: int = 2,
):
    """Render a textured planar rectangle anchored in the **world** into the frame; composite it.

    The world-anchored counterpart of :func:`warp_patch`: instead of an image quad, the target is a
    3-D rectangle ``corners_world`` (4x3, TL,TR,BR,BL as seen from the front) and a real ``camera``
    (:mod:`avsectester.rendering.cameras` — pinhole or f-theta fisheye). Every pixel in the
    rectangle's image footprint is cast as a ray and intersected with the plane, so the result is exact
    for a distorting lens (a 4-corner homography is not). The texture is pre-filtered to its on-screen
    size (no aliasing on distant objects); ``soften`` (px sigma) optionally blurs it to match a soft
    render. Returns ``(composite_rgb uint8, mask uint8 HxW)`` like :func:`warp_patch`, or the frame and
    an empty mask when the plane is behind the camera / out of view.
    """
    import cv2
    from avsectester.insertion import PlaneSurface
    from avsectester.rendering.geometry import prepare_surface, sample_image

    if frame_rgb.shape[:2] != (camera.height, camera.width):
        raise ValueError("Frame dimensions do not match camera calibration")
    sampler = prepare_surface(PlaneSurface(corners_world, texture_rgba), camera, cam_from_world)
    sampled, _ = sample_image(sampler, camera, margin=margin)
    sampled = sampled.astype(np.float32)
    if soften > 0:
        bounds = sampler.bounds or (0, 0, camera.width, camera.height)
        padding = margin - 2 if sampler.bounds is not None else 0
        x0, y0 = max(0, bounds[0] - padding), max(0, bounds[1] - padding)
        x1, y1 = min(camera.width, bounds[2] + padding), min(camera.height, bounds[3] + padding)
        if x1 > x0 and y1 > y0:
            sampled[y0:y1, x0:x1] = cv2.GaussianBlur(sampled[y0:y1, x0:x1], (0, 0), soften)
    alpha = sampled[:, :, 3:4].astype(np.float32) / 255
    composite = frame_rgb.astype(np.float32) * (1 - alpha) + sampled[:, :, :3] * alpha
    mask = (sampled[:, :, 3] > 127).astype(np.uint8) * 255
    return composite.astype(np.uint8), mask


class PatchCompositor:
    """Appearance settings for geometry-based insertion compositing.

    ``soften`` is the Gaussian sigma in pixels for resolved insertion rendering.
    It does not change geometry or expand visibility masks.
    """

    def __init__(self, harmonizer: Harmonizer | None = None, *, soften: float = 0.0) -> None:
        if not np.isfinite(soften) or soften < 0:
            raise ValueError("soften must be finite and nonnegative")
        self.soften = float(soften)
        self.harmonizer = harmonizer or ClassicHarmonizer()

def render_resolved(
    frame: np.ndarray,
    camera: Camera,
    cam_from_world: np.ndarray,
    resolved: ResolvedInsertion,
    evidence: VisibilityEvidence | None = None,
    *,
    compositor: PatchCompositor | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Composite a resolved insertion with per-pixel surface depth and visibility.

    ``evidence`` is the estimator result for this same pose and camera. Its visible
    mask prevents the insertion from painting over foreground geometry. Without
    evidence, only the insertion's own surfaces occlude one another. An optional
    compositor supplies harmonization, applied within the visible region only.

    Returns ``(RGB image, uint8 mask)``. Unknown visibility is an explicit error,
    because treating missing evidence as visible would silently change the scene.
    """
    from avsectester.rendering.geometry import insertion_image

    shape = frame.shape[:2]
    if shape != (camera.height, camera.width):
        raise ValueError("Frame dimensions do not match camera calibration")
    if evidence is not None:
        if evidence.visibility is None:
            raise ValueError(f"Insertion visibility is unknown: {evidence.reason}")
        if evidence.visible_mask.shape != shape:
            raise ValueError("Visibility mask dimensions do not match the image")
    if evidence is not None and getattr(evidence, "sampled_rgba", None) is not None:
        rgba, depth = evidence.sampled_rgba, evidence.target_depth
    else:
        rgba, depth = insertion_image(resolved, camera, cam_from_world)
    active = np.isfinite(depth)
    if evidence is not None:
        active &= evidence.visible_mask
    if compositor is not None and compositor.soften > 0:
        import cv2

        # Appearance filtering never expands the geometric visibility mask.
        rgba = cv2.GaussianBlur(rgba.astype(np.float32), (0, 0), compositor.soften)
    output = frame.copy()
    alpha = rgba[active, 3:4].astype(np.float32) / 255
    output[active] = (frame[active] * (1 - alpha) + rgba[active, :3] * alpha).astype(np.uint8)
    union = active.astype(np.uint8) * 255
    if compositor is not None and active.any():
        harmonized = compositor.harmonizer(output, union, frame)
        output[active] = harmonized[active]
    return output, union


def frame_perturbation(
    insert: Callable[[Observation, Any], Any], camera: str | None = None, base: View | None = None
) -> Callable[[Observation], Observation]:
    """Return ``perturb(obs) -> obs`` that rewrites one camera's frame with ``insert(obs, rgb) -> rgb``.

    The sensor-plane attack seam for backends whose camera payload is a raw RGB ndarray (NuRec): the
    AV stack itself perceives the inserted object. ``carla.insertion_perturbation`` is the
    CARLA ``ImageData`` analogue. A visualization callback alone does not change model input.
    """
    from dataclasses import replace

    def _perturb(observation: Observation) -> Observation:
        data = observation.sensor_data
        if not data:
            return observation
        key = camera if camera in data else next(iter(data))
        rgb = camera_view(observation, key) if base is None else base(observation)
        if rgb is None:
            return observation
        new_data = dict(data)
        new_data[key] = insert(observation, rgb)
        return replace(observation, sensor_data=new_data)

    return _perturb


class InsertionRenderer:
    """Resolve bound objects and render them through the same geometry used by selection.

    ``geometry(observation)`` returns :class:`InsertionGeometry`. An optional
    ``evidence_provider(observation, resolved, geometry)`` returns visibility evidence by insertion
    ID. No filters run here. Use ``frame_perturbation`` or a backend adapter to update model inputs.
    """

    def __init__(
        self,
        insertions: Sequence[Insertion],
        camera: Camera,
        geometry: GeometryProvider,
        compositor: PatchCompositor | None = None,
        evidence_provider: EvidenceProvider | None = None,
    ):
        self.insertions = tuple(insertions)
        if len({item.id for item in self.insertions}) != len(self.insertions):
            raise ValueError("Insertion IDs must be unique")
        self.camera = camera
        self.geometry = geometry
        self.compositor = compositor
        self.evidence_provider = evidence_provider
        self.resolved = ()
        self.evidence = {}

    def __call__(self, observation: Observation, rgb: np.ndarray) -> np.ndarray:
        from avsectester.insertion import resolve_insertion

        state = self.geometry(observation)
        if not isinstance(state, InsertionGeometry):
            raise TypeError("geometry must return InsertionGeometry")
        self.resolved = tuple(
            resolve_insertion(item, state.actors, state.victim) for item in self.insertions
        )
        self.evidence = (
            self.evidence_provider(observation, self.resolved, state)
            if self.evidence_provider is not None
            else {}
        )
        result = rgb
        for item in self.resolved:
            result, _ = render_resolved(
                result,
                self.camera,
                state.cam_from_world,
                item,
                self.evidence[item.id] if self.evidence_provider else None,
                compositor=self.compositor,
            )
        return result
