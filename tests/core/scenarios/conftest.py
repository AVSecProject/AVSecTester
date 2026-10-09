"""Small initial-scene and insertion factories shared by selection tests."""

import numpy as np
from avsectester.rendering.types import Visibility

import pytest

from avsectester.insertion import AttachedPlacement, Insertion, Orientation, PlaneAsset
from avsectester.scenarios import CameraCalib, EgoState, ObjectGT, SceneGT


@pytest.fixture
def scene():
    def make(frame=0, distances=(8, 12), *, ids=("a", "b")):
        optical_to_ego = np.eye(4)
        optical_to_ego[:3, :3] = [[0, 0, 1], [-1, 0, 0], [0, -1, 0]]
        optical_to_ego[2, 3] = 1.5
        camera = CameraCalib("front", 160, 120, (80, 80, 60), optical_to_ego)
        return SceneGT(
            frame,
            frame * 0.1,
            EgoState(pose=np.eye(4)),
            {"front": camera},
            [
                ObjectGT(track, "vehicle", (x, 0, 0.75), extent=(4, 2, 1.5), visibility=Visibility(0.0, source="provided"))
                for track, x in zip(ids, distances)
            ],
        )

    return make


@pytest.fixture
def patch():
    def make(identifier="patch", host="host"):
        return Insertion(
            identifier,
            PlaneAsset(np.full((4, 4, 4), 255, np.uint8), 0.6, 0.4),
            AttachedPlacement(host, "rear_center", (-0.05, 0, 0.2)),
            Orientation("follow_host", (0, 0, 180)),
        )

    return make
