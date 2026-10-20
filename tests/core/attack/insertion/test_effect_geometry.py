"""Portable attacks complete public preparation, execution and offline evaluation."""
from dataclasses import replace
import numpy as np
import pytest

from avsectester.attack.insertion import PlaneAsset
from avsectester.attack.placement import WorldPlacement
from avsectester.closed_loop.simulator import SensorView
from avsectester.datasets import ActorState
from avsectester.geometry.cameras import FThetaCamera, PinholeCamera
from avsectester.geometry.poses import ActorPose

CAM_FROM_WORLD = np.array([[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1.]])
CAMERA = PinholeCamera(np.array([[50, 0, 50], [0, 50, 40], [0, 0, 1.]]), 100, 80)


def host_state():
    pose = np.eye(4)
    pose[0, 3] = 10
    return ActorState(pose, (0., 0., 0.), "car", (4., 2., 2.))


@pytest.mark.parametrize("lens", [CAMERA, FThetaCamera(50, 40, (0, 50), 100, 80,
                                                       (0, 1 / 50), max_angle=1.5)])
def test_sensor_effect_uses_depth_when_available_and_cuboids_otherwise(lens):
    from avsectester.attack.insertion import Insertion
    from avsectester.attack.insertion.effect import InsertionAttackEffect
    from avsectester.attack.placement import Orientation
    effect = InsertionAttackEffect("front", Insertion("plane",
        PlaneAsset(np.full((8, 8, 4), 255, np.uint8), 2., 1.),
        WorldPlacement((6., 0., 0.)), Orientation("face_ego")))
    view = SensorView("front", np.zeros((80, 100, 3), np.uint8), lens, CAM_FROM_WORLD,
                      {}, ActorPose(np.eye(4)), 0., 0)
    assert effect.apply(view.data, view).any()
    blocked = replace(view, depth=np.full((80, 100), 2.))
    assert not effect.apply(blocked.data, blocked).any()
    cuboid = replace(view, actors={"occluder": ActorPose(host_state().pose, (4., 2., 2.))})
    assert effect.evidence(cuboid).visibility.source == "cuboid_estimate"
