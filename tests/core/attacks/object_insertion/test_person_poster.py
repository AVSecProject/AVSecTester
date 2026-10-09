"""Pedestrian standees, billboards and printed poster payloads."""

import numpy as np
import pytest


def _person(h=120, w=40):
    """A synthetic cut-out: an opaque person-shaped column on a transparent background."""
    img = np.zeros((h, w, 4), np.uint8)
    img[:, 10:30, :3] = (60, 70, 90)
    img[:, 10:30, 3] = 255
    return img


def test_person_standee_and_billboard_geometry():
    from avsectester.attacks.object_insertion.person_poster import billboard, standee

    (surface,) = standee(_person(), height=1.75).planes()
    face = surface.corners  # no post
    assert np.isclose(face[0, 2] - face[3, 2], 1.75) and np.isclose(face[3, 2], 0.0)  # on the ground
    assert np.isclose(np.linalg.norm(face[1] - face[0]), 1.75 * 40 / 120)  # keeps the cut-out's aspect
    planes = billboard(_person(), width=1.4, mount_height=0.6).planes()
    assert len(planes) == 3  # two legs + the board
    legs = [p.corners for p in planes[:2]]
    assert not np.allclose(legs[0], legs[1]) and all(np.isclose(leg[3, 2], 0.0) for leg in legs)
    board = planes[-1].corners
    assert np.isclose(board[3, 2], 0.6) and np.isclose(np.linalg.norm(board[1] - board[0]), 1.4)


def test_poster_prints_person_on_opaque_paper():
    from avsectester.attacks.object_insertion.person_poster import poster_rgba

    p = poster_rgba(_person(), aspect=1.5)
    assert p.shape[0] / p.shape[1] == pytest.approx(1.5, rel=0.01)
    assert (p[..., 3] == 255).all()  # opaque board
    mid = p[p.shape[0] // 2, p.shape[1] // 2, :3]
    assert tuple(mid) == (60, 70, 90)  # the figure is printed in the centre
    assert tuple(p[p.shape[0] // 2, p.shape[1] // 8, :3]) == (228, 226, 218)  # paper around it
