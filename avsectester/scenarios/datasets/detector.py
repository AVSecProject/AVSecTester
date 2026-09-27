"""A **labeling strategy**: derive :class:`SceneGT` from a 2-D detector run on a frame source.

This is *not* a dataset of its own — contrast :class:`~...datasets.nuscenes.NuScenesDataset`, which reads
real 3-D labels. ``DetectorLabeler`` **composes over** any :class:`~avsectester.scenarios.source.FrameSource`
(a folder of images, NuRec renders) and turns it into a labeled
:class:`~avsectester.scenarios.source.Dataset` by *synthesising* the ground truth with a detector — for
datasets that ship no accessible 3-D annotations (nuScenes GT is permission-locked here; NuRec/Alpamayo
ships none).

``box2d`` comes straight from the detector; ``distance`` is monocular-estimated from the box height + a
nominal vehicle height and the focal length. ``yaw`` (viewpoint) and ``visibility`` are **not observable
from a 2-D box**, so they are set to rear-facing / fully-visible — documented approximations a 3-D
detector or real labels would replace. For a *detection-removal* attack this detector-derived state is
also the attack-relevant signal: you can only remove a detection that exists.

``detect`` is injected (``rgb -> [(xyxy, score, label)]``, vehicles only), so this stays dataset- and
model-agnostic (a COCO detector for nuScenes/NuRec, etc.). Imports cv2 lazily.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

import numpy as np

from avsectester.backend import WorldBackend
from avsectester.scenarios.datasets.frames import ImageFolder
from avsectester.scenarios.datasets.nuscenes import RecordedFrameBackend
from avsectester.scenarios.scene import CameraCalib, EgoState, ObjectGT, SceneGT
from avsectester.scenarios.source import Dataset, FrameSource


class DetectorLabeler(Dataset):
    """Wrap a :class:`FrameSource`; run ``detect`` on each frame and emit detector-derived ``SceneGT``.

    ``frames`` is the pixel source to label. ``detect`` maps ``rgb -> [(xyxy, score, label)]``.
    ``vehicle_height`` is the assumed metric car height used for the monocular distance estimate. The
    result is an ordinary :class:`Dataset`, so it drops straight into a ``DatasetFilter``."""

    def __init__(self, frames: FrameSource, detect: Callable[[Any], Any],
                 vehicle_height: float = 1.55) -> None:
        self.frames = frames
        self.detect = detect
        self.vehicle_height = vehicle_height

    @classmethod
    def over_images(cls, image_paths: list[str], detect: Callable[[Any], Any], k: np.ndarray,
                    camera: str = "front", vehicle_height: float = 1.55) -> DetectorLabeler:
        """Convenience: label a bare list of image paths (wraps them in an :class:`ImageFolder`)."""
        return cls(ImageFolder(image_paths, k, camera=camera), detect, vehicle_height=vehicle_height)

    def scenes(self) -> Iterator[SceneGT]:
        import cv2

        for frame in self.frames.frames():
            bgr = cv2.imread(frame.image_path)
            if bgr is None:
                continue
            rgb = bgr[:, :, ::-1]
            k = np.asarray(frame.k, dtype=np.float64)
            f, cx = float(k[0, 0]), float(k[0, 2])
            objects = []
            for i, (box, _score, _label) in enumerate(self.detect(rgb)):
                x1, y1, x2, y2 = (float(v) for v in box)
                bh = max(y2 - y1, 1.0)
                distance = f * self.vehicle_height / bh               # monocular depth from box height
                bcx = (x1 + x2) / 2.0
                lateral = -(bcx - cx) / f * distance                  # +left, from the box centre offset
                objects.append(ObjectGT(
                    track_id=f"{i}", category="vehicle",
                    center=(distance, lateral, self.vehicle_height / 2),
                    yaw=0.0, box2d={frame.camera: (x1, y1, x2, y2)}, visibility=1.0))
            calib = CameraCalib(name=frame.camera, width=frame.width, height=frame.height, model=k)
            yield SceneGT(frame=0, t=0.0, ego=EgoState(speed=0.0),
                          cameras={frame.camera: calib}, objects=objects,
                          source={"dataset": "detector", "image_path": frame.image_path})

    def make_backend(self, scene: SceneGT) -> WorldBackend:
        return RecordedFrameBackend(scene.source["image_path"])
