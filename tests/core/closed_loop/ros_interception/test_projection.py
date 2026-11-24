"""Pose corrections and timestamp joins at the ROS boundary."""

import math

import numpy as np

from avsectester.closed_loop.ros_interception.projection import Pose, PoseDelta, ProjectionQueue


def test_pose_delta_moves_poses_rigidly():
    start = Pose((1., 2., 3.), (0., 0., 0., 1.))
    switch = math.sqrt(0.5)
    end = Pose((10., 20., 30.), (0., 0., switch, switch))
    delta = PoseDelta.between(start, end)
    assert delta is not None
    assert delta.apply(start) == end
    moved = delta.apply(Pose((2., 2., 3.), start.rotation))
    np.testing.assert_allclose(moved.position, (10., 21., 30.))
    np.testing.assert_allclose(moved.rotation, end.rotation)
    assert PoseDelta.between(start, Pose(start.position, (0., 0., 0., -1.))) is None


def test_transforms_wait_for_their_pose_and_fall_back_only_when_it_cannot_arrive():
    delta = PoseDelta.between(Pose((0., 0., 0.), (0., 0., 0., 1.)), Pose((1., 0., 0.), (0., 0., 0., 1.)))
    waiting = ProjectionQueue[str](timeout_s=0.5)
    # Transforms keep their order and wait for their own poses beyond one pose period.
    waiting.enqueue("first", [2], 1.)
    waiting.enqueue("second", [3], 1.01)
    assert waiting.drain(1.2) == []
    waiting.pose(2, None)
    first, = waiting.drain(1.2)
    assert first.message == "first" and first.pairs == ((None, True),)
    waiting.pose(3, delta)
    second, = waiting.drain(1.2)
    assert second.message == "second" and second.pairs == ((delta, True),)

    # A later pose shows that a transform's own pose will not arrive.
    waiting.enqueue("unmatched", [4], 1.3)
    waiting.pose(5, delta)
    fallback, = waiting.drain(1.3)
    assert fallback.message == "unmatched" and fallback.pairs == ((delta, False),)

    # The timeout only releases a transform when the pose topic has stopped.
    waiting.enqueue("stopped", [6], 1.4)
    assert waiting.drain(1.89) == []
    stopped, = waiting.drain(1.91)
    assert stopped.pairs == ((delta, False),)
