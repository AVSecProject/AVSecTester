"""Pixel visibility including occlusion and image truncation.

The denominator is the complete target silhouette from the same viewpoint without external
occluders. Render a larger reference canvas at the original focal length when necessary.
``image_bounds`` identifies the actual camera image on that canvas. Pixels outside it are invisible.
A bounding box or a silhouette already clipped by the image is not a complete reference.
"""

import numpy as np
from avsectester.scenarios.scene import Visibility


def _image_mask(shape, image_bounds):
    if image_bounds is None:
        return np.ones(shape, dtype=bool)
    if len(image_bounds) != 4 or any(not isinstance(v, (int, np.integer)) for v in image_bounds):
        raise ValueError("Image bounds must be integer (left, top, right, bottom)")
    left, top, right, bottom = image_bounds
    height, width = shape
    if not (0 <= left < right <= width and 0 <= top < bottom <= height):
        raise ValueError("Image bounds must lie inside the reference canvas")
    mask = np.zeros(shape, dtype=bool)
    mask[top:bottom, left:right] = True
    return mask


def visibility_from_masks(
    reference, visible, *, camera: str, image_bounds=None
) -> Visibility | None:
    """Divide visible in-image pixels by the FULL unoccluded reference silhouette.

    Both masks use the reference canvas. ``image_bounds=None`` means that the camera covers the
    whole canvas and the reference is known to fit. An empty reference is unknown, not zero.
    A nonempty reference entirely outside image_bounds has zero visibility.
    """
    reference, visible = np.asarray(reference, dtype=bool), np.asarray(visible, dtype=bool)
    if reference.ndim != 2 or visible.shape != reference.shape:
        raise ValueError("Visibility masks must be aligned two-dimensional arrays")
    in_image = _image_mask(reference.shape, image_bounds)
    denominator = int(reference.sum())
    if denominator == 0:
        return None
    return Visibility(
        float(np.count_nonzero(reference & visible & in_image) / denominator),
        source="rendered_masks",
        camera=camera,
    )


def visibility_from_depth(
    footprint,
    target_depth,
    scene_depth,
    *,
    camera: str,
    tolerance_m: float = 0.0002,
    image_bounds=None,
) -> Visibility | None:
    """Compare target and scene depth in the same convention and pixel projection.

    footprint is the FULL projected silhouette, including portions outside the camera image.
    target_depth may come from an isolated reference render or known surface geometry. The
    target and camera poses must match the scene measurement. Depth outside image_bounds is
    not needed and may be NaN. Depth inside the target's in-image footprint must be valid.
    """
    footprint = np.asarray(footprint, dtype=bool)
    target_depth, scene_depth = np.asarray(target_depth), np.asarray(scene_depth)
    if (
        footprint.ndim != 2
        or target_depth.shape != footprint.shape
        or scene_depth.shape != footprint.shape
    ):
        raise ValueError("Footprint and depth maps must be aligned two-dimensional arrays")
    if not np.isfinite(tolerance_m) or tolerance_m < 0:
        raise ValueError("Depth tolerance must be finite and nonnegative")
    required = footprint & _image_mask(footprint.shape, image_bounds)
    if not np.all(
        np.isfinite(target_depth[required])
        & (target_depth[required] > 0)
        & np.isfinite(scene_depth[required])
        & (scene_depth[required] > 0)
    ):
        raise ValueError("Depth evidence is missing or invalid inside the target footprint")
    result = visibility_from_masks(
        footprint,
        target_depth <= scene_depth + tolerance_m,
        camera=camera,
        image_bounds=image_bounds,
    )
    return None if result is None else Visibility(result.fraction, "depth_comparison", camera)
