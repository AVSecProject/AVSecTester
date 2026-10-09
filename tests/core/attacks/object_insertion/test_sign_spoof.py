"""Sign assets retain physical dimensions under explicit world placement."""

import math

import numpy as np
import pytest

from avsectester.attacks.object_insertion.sign_spoof import SignAsset
from avsectester.insertion import ActorPose, Insertion, Orientation, WorldPlacement, resolve_insertion


@pytest.mark.parametrize("yaw", [0, 20, -35])
def test_sign_world_geometry_preserves_face_and_post_positions(yaw):
    insertion = Insertion("sign", SignAsset(), WorldPlacement((20, -5, 0)),
                          Orientation("fixed_world", (0, 0, 180 + yaw)))
    post, face = resolve_insertion(insertion, {}, ActorPose(np.eye(4))).planes()
    angle = math.radians(yaw)
    right = np.array([math.sin(angle), -math.cos(angle), 0])
    center = np.array([20, -5, 1.95])
    expected = np.array([center - right * .45 + [0, 0, .45],
                         center + right * .45 + [0, 0, .45],
                         center + right * .45 - [0, 0, .45],
                         center - right * .45 - [0, 0, .45]])
    np.testing.assert_allclose(face.corners, expected, atol=1e-12)
    assert face.texture[0, 0, 3] == 0
    np.testing.assert_allclose(post.corners[:2].mean(axis=0),
                               [20 + .03 * math.cos(angle), -5 + .03 * math.sin(angle), 1.95])
    assert post.corners[3, 2] == pytest.approx(0)
