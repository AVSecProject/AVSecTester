"""Patch placement geometry — pure numpy (no torch), so it imports and tests without a GPU stack."""

from __future__ import annotations

import numpy as np


def homography_4pt(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """3x3 homography mapping the 4 ``src`` points to the 4 ``dst`` points (DLT)."""
    a = []
    for (x, y), (u, v) in zip(src, dst):
        a.append([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])
        a.append([0, 0, 0, x, y, 1, -v * x, -v * y, -v])
    _, _, vt = np.linalg.svd(np.asarray(a, dtype=np.float64))
    h = vt[-1].reshape(3, 3)
    return h / h[2, 2]


def rear_panel_homography(box, patch_hw, wfrac=0.7, hfrac=0.5, ycenter=0.55) -> np.ndarray:
    """Homography from patch pixels -> a centered rectangle on a target box's rear panel.

    ``box`` = xyxy in image pixels; ``patch_hw`` = (Hp, Wp). The patch covers ``wfrac``x``hfrac`` of
    the box, centered horizontally and placed at vertical fraction ``ycenter`` (the rear panel).
    """
    x0, y0, x1, y1 = box
    hp, wp = patch_hw
    bw, bh = x1 - x0, y1 - y0
    cx, cy = (x0 + x1) / 2.0, y0 + bh * ycenter
    pw, ph = bw * wfrac, bh * hfrac
    dst = np.array([[cx - pw / 2, cy - ph / 2], [cx + pw / 2, cy - ph / 2],
                    [cx + pw / 2, cy + ph / 2], [cx - pw / 2, cy + ph / 2]], dtype=np.float64)
    src = np.array([[0, 0], [wp, 0], [wp, hp], [0, hp]], dtype=np.float64)
    return homography_4pt(src, dst)
