"""``SceneGT`` from a 2-D detector on real images — for datasets with no accessible 3-D labels.

nuScenes GT here is permission-locked, and NuRec/Alpamayo ships no annotations, so ground truth is not
always available. This adapter builds ``SceneGT`` from a detector run on the real frames: ``box2d`` comes
straight from the detector, and distance is monocular-estimated from the box height + a nominal vehicle
height and the camera focal length. ``yaw`` (viewpoint) and ``visibility`` are **not observable from a
2-D box**, so they are set to rear-facing / fully-visible — documented approximations that a 3-D detector
or real labels would replace. For a *detection-removal* attack this detector-derived state is also the
attack-relevant signal: you can only remove a detection that exists.

``detect`` is injected (``rgb -> [(xyxy, score, label)]``, vehicles only), so this stays dataset- and
model-agnostic (a COCO detector for nuScenes/NuRec, etc.). Imports cv2 lazily.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

import numpy as np

from avsectester.backend import WorldBackend
from avsectester.scenarios.datasets.nuscenes import RecordedFrameBackend
from avsectester.scenarios.scene import CameraCalib, EgoState, ObjectGT, SceneGT
from avsectester.scenarios.source import Dataset


class DetectorDataset(Dataset):
    """Iterate ``image_paths``, run ``detect`` on each, and emit ``SceneGT`` with detector-derived state.

    ``k`` is the camera intrinsic (for the monocular distance estimate); ``vehicle_height`` the assumed
    metric height of a car. ``camera`` is the logical name the requirement targets (default ``front``)."""

    def __init__(self, image_paths: list[str], detect: Callable[[Any], Any], k: np.ndarray,
                 camera: str = "front", vehicle_height: float = 1.55) -> None:
        self.image_paths = list(image_paths)
        self.detect = detect
        self.k = np.asarray(k, dtype=np.float64)
        self.camera = camera
        self.vehicle_height = vehicle_height

    def scenes(self) -> Iterator[SceneGT]:
        import cv2

        f, cx = float(self.k[0, 0]), float(self.k[0, 2])
        for path in self.image_paths:
            bgr = cv2.imread(path)
            if bgr is None:
                continue
            rgb = bgr[:, :, ::-1]
            h, w = rgb.shape[:2]
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
                    yaw=0.0, box2d={self.camera: (x1, y1, x2, y2)}, visibility=1.0))
            calib = CameraCalib(name=self.camera, width=w, height=h, model=self.k)
            yield SceneGT(frame=0, t=0.0, ego=EgoState(speed=0.0),
                          cameras={self.camera: calib}, objects=objects,
                          source={"dataset": "detector", "image_path": path})

    def make_backend(self, scene: SceneGT) -> WorldBackend:
        return RecordedFrameBackend(scene.source["image_path"])
