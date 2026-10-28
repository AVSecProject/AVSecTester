"""Static bearing tracking with biased monocular ranges."""

import math

import numpy as np
import pytest

from avsectester.closed_loop.avstacks.implemented.modules.tracker import StopSignTracker
from avsectester.closed_loop.simulators.ego_dynamics import PlanarEgoState


def test_update():
    tracker = StopSignTracker()
    rng = np.random.default_rng(0)
    sign = np.array([40., 5.])
    errors = []
    for x in np.arange(0., 35.1, .5):
        delta = sign - np.array([x, 0.])
        distance = np.linalg.norm(delta) * .7
        bearing = math.atan2(delta[1], delta[0]) + rng.normal(0., .007)
        measured = np.array([distance * math.cos(bearing), distance * math.sin(bearing), 0.])
        tracker.update(PlanarEgoState(x=x), np.zeros(3), [measured])
        assert len(tracker.tracks) == 1
        track = tracker.tracks[0]
        assert track.track_id == 0
        np.testing.assert_allclose(track.covariance, track.covariance.T)
        assert np.linalg.eigvalsh(track.covariance).min() >= 0.
        if x >= 25.:
            errors.append(np.linalg.norm(track.position - sign))
    assert max(errors) < 1.
    # Snapshots are detached from subsequent caller modifications.
    tracker.tracks[0].position[:] = 0.
    assert np.linalg.norm(tracker.tracks[0].position - sign) < 1.
    with pytest.raises(ValueError, match="camera_position_rig"):
        tracker.update(PlanarEgoState(), np.zeros(2), [])


def test_association():
    tracker = StopSignTracker()
    # The camera is translated in the rig and the rig is rotated in the world.
    state = PlanarEgoState(x=3., y=-2., yaw=math.pi / 2)
    camera = np.array([1., .5, 1.2])
    signs = [np.array([40., -7.5, 0.]), np.array([40., 7.5, 0.])]
    tracker.update(state, camera, signs)
    assert [track.track_id for track in tracker.tracks] == [0, 1]
    np.testing.assert_allclose(tracker.tracks[0].position, [10.5, 38.])
    np.testing.assert_allclose(tracker.tracks[1].position, [-4.5, 38.])
    tracker.update(state, camera, signs)
    assert len(tracker.tracks) == 2


def test_reset():
    tracker = StopSignTracker()
    tracker.update(PlanarEgoState(), np.zeros(3),
                   [np.array([40., 5., 0.]), np.array([40., 20., 0.])])
    assert [track.track_id for track in tracker.tracks] == [0, 1]
    tracker.reset()
    assert tracker.tracks == ()
    tracker.update(PlanarEgoState(), np.zeros(3), [np.array([10., 2., 0.])])
    assert tracker.tracks[0].track_id == 0


def test_stop_sign_tracker():
    assert StopSignTracker().tracks == ()
    with pytest.raises(ValueError, match="positive"):
        StopSignTracker(range_sigma_fraction=0.)
    with pytest.raises(ValueError, match="positive"):
        StopSignTracker(bearing_sigma_rad=-.1)
