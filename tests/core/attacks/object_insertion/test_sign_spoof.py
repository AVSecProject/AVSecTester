"""Roadside sign payload and placement geometry."""

import math

import numpy as np


def test_roadside_sign_planes_geometry():
    from avsectester.attacks.object_insertion.sign_spoof import RoadsideSign

    s = RoadsideSign(x=20.0, y=-5.0, width=0.9, mount_height=1.5)
    (post, _), (face, tex) = s.planes()
    assert tex.shape[2] == 4 and tex[0, 0, 3] == 0  # RGBA, transparent outside the octagon
    assert np.allclose(face[:, 0], 20.0)  # yaw 0: the face is a plane of constant x
    assert face[0, 1] > face[1, 1]  # TL is on the viewer's left (+y) when facing the ego
    assert np.isclose(face[2, 2], 1.5) and np.isclose(face[0, 2] - face[3, 2], 0.9)
    assert np.isclose(post[3, 2], 0.0) and post[0, 0] > 20.0  # post: from the ground, behind the face
    turned = RoadsideSign(x=20.0, y=-5.0, yaw=math.radians(20)).planes()[-1][0]
    assert not np.allclose(turned[:, 0], 20.0)  # yaw rotates the face out of the x-plane
