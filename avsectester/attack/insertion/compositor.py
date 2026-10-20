"""Shared geometry-based insertion rendering and compositing.

``render_resolved`` uses the same sampling and visibility evidence as selection.
The image-space warp primitive is also available for user-defined image operations.
"""

from __future__ import annotations

import numpy as np

from .assets import ResolvedInsertion
from avsectester.geometry.cameras import Camera
from avsectester.attack.insertion.types import VisibilityEvidence

from avsectester.attack.insertion.harmonizers import Harmonizer, ClassicHarmonizer


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


class PatchCompositor:
    """Appearance settings for geometry-based insertion compositing.

    ``soften`` is the Gaussian sigma in pixels for resolved insertion rendering.
    It does not change geometry or expand visibility masks.
    ``harmonize`` enables the selected harmonizer. When disabled, it is not called
    or loaded, and the same geometry, alpha compositing and softening still apply.
    """

    def __init__(
        self, harmonizer: Harmonizer | None = None, *, soften: float = 0.0,
        harmonize: bool = True,
    ) -> None:
        if not isinstance(harmonize, bool):
            raise TypeError("harmonize must be a bool")
        if not np.isfinite(soften) or soften < 0:
            raise ValueError("soften must be finite and nonnegative")
        if harmonizer is not None and not callable(harmonizer):
            raise TypeError("harmonizer must implement Harmonizer")
        self.soften = float(soften)
        self.harmonize = harmonize
        self.harmonizer = harmonizer if harmonizer is not None else ClassicHarmonizer()


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
    Online insertion effects handle this evidence as a recorded no-op before compositing.
    """
    from avsectester.attack.insertion.geometry import insertion_image

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
    if compositor is not None and compositor.harmonize and active.any():
        harmonized = compositor.harmonizer(output, union, frame)
        output[active] = harmonized[active]
    return output, union
