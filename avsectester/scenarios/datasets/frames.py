"""Concrete :class:`~avsectester.scenarios.source.FrameSource`s — raw, unlabeled pixel sources.

A frame source is the *pixel* side of a trace: it yields :class:`Frame`s (image + calibration) with no
ground truth. Wrap one in a labeler (e.g. :class:`~...datasets.detector.DetectorLabeler`) to get a
labeled :class:`~avsectester.scenarios.source.Dataset` the requirement predicate can run on.

``ImageFolder`` is the simplest source: a fixed list of image paths under one shared intrinsic — a bag of
real frames (a nuScenes CAM_FRONT dump, a NuRec render sequence written to disk, an ad-hoc folder).
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from avsectester.scenarios.source import Frame, FrameSource


class ImageFolder(FrameSource):
    """A :class:`FrameSource` over a fixed list of image paths sharing one intrinsic ``k``.

    ``camera`` is the logical name the requirement targets (default ``front``). Image dimensions are read
    lazily from each file, so the paths need not exist until iterated (cv2 imported lazily)."""

    def __init__(self, image_paths: list[str], k: np.ndarray, camera: str = "front") -> None:
        self.image_paths = list(image_paths)
        self.k = np.asarray(k, dtype=np.float64)
        self.camera = camera

    def frames(self) -> Iterator[Frame]:
        import cv2

        for path in self.image_paths:
            bgr = cv2.imread(path)
            if bgr is None:
                continue
            h, w = bgr.shape[:2]
            yield Frame(image_path=path, k=self.k, width=w, height=h, camera=self.camera)
