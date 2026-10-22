"""Shared geometry-based insertion rendering and compositing.

``render_resolved`` uses the same sampling and visibility evidence as selection.
The image-space warp primitive is also available for user-defined image operations.
"""

from __future__ import annotations

import numpy as np

from .assets import ResolvedInsertion
from avsectester.geometry.cameras import Camera
from avsectester.attack.insertion.types import VisibilityEvidence

from avsectester.rendering import Harmonizer, blend_rgba, render_surfaces


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
    comp = blend_rgba(frame_rgb, warped, warped[:, :, 3] > 0)
    mask = (warped[:, :, 3] > 127).astype(np.uint8) * 255
    return comp.astype(np.uint8), mask


class PatchCompositor:
    """Appearance settings for geometry-based insertion compositing.

    ``soften`` is the Gaussian sigma in pixels for resolved insertion rendering.
    It does not change geometry or expand visibility masks.
    ``harmonizer=None`` disables harmonization. Geometry, alpha compositing and
    softening apply independently of the selected harmonizer.
    """

    def __init__(
        self, harmonizer: Harmonizer | None = None, *, soften: float = 0.0,
    ) -> None:
        if not np.isfinite(soften) or soften < 0:
            raise ValueError("soften must be finite and nonnegative")
        if harmonizer is not None and not callable(harmonizer):
            raise TypeError("harmonizer must implement Harmonizer")
        self.soften = float(soften)
        self.harmonizer = harmonizer


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

    Returns ``(RGB image, uint8 mask)``. Unknown visibility is an explicit error.
    Online insertion actions handle this evidence as a recorded no-op before compositing.
    """
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
        rgba, depth = render_surfaces(resolved.planes(), camera, cam_from_world)
    active = np.isfinite(depth)
    if evidence is not None:
        active &= evidence.visible_mask
    output = blend_rgba(frame, rgba, active,
                        harmonizer=None if compositor is None else compositor.harmonizer,
                        soften=0.0 if compositor is None else compositor.soften)
    union = active.astype(np.uint8) * 255
    return output, union
