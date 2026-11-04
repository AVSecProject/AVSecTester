"""Bearing sign tracking and scored Kalman vehicle tracking."""

import math

import numpy as np
import pytest

from avsectester.closed_loop.avstacks.implemented.modules.tracker import StopSignTracker, VehicleTracker
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


def test_vehicle_tracker():
    rng = np.random.default_rng(29)
    tracker = VehicleTracker()
    state = PlanarEgoState(x=3., y=-2., yaw=math.pi / 2)
    face = np.array([[10., 2., 0.], [10., 1., 0.], [10., 0., 0.]])
    tracker.update(state, 0., [face], [.9])
    assert tracker.tracks == ()
    tracker.update(state, .1, [face], [.9])
    initial = tracker.tracks
    assert initial[0].track_id == 0 and initial[0].hits == 2
    np.testing.assert_allclose(initial[0].position, [2., 8.])
    np.testing.assert_array_equal(initial[0].velocity, [0., 0.])
    assert initial[0].half_width_m == pytest.approx(1.)
    # A stationary target is observed with fixed-seed position noise from a moving ego.
    target = np.array([25., 2.])
    speeds = []
    for frame in range(1, 201):
        ego = PlanarEgoState(x=frame * .02)
        center = target - np.array([ego.x, ego.y]) + rng.normal(0., .25, 2)
        points = np.array([[center[0], center[1] + 1., 0.],
                           [center[0], center[1], 0.], [center[0], center[1] - 1., 0.]])
        tracker.update(ego, frame * .1 + .1, [points], [.9])
        speeds.extend(np.linalg.norm(track.velocity) for track in tracker.tracks)
    assert np.median(speeds[-100:]) < .3
    np.testing.assert_allclose(tracker.tracks[-1].position, target, atol=.3)

    tracker.reset()
    assert tracker.tracks == ()
    for frame in range(151):
        t = frame * .1
        center = np.array([10. + 2. * t, 3.]) + rng.normal(0., .03, 2)
        points = np.array([[center[0], center[1] + .8, 0.],
                           [center[0], center[1], 0.], [center[0], center[1] - .8, 0.]])
        tracker.update(PlanarEgoState(), t, [points], [.9])
    assert tracker.tracks[0].track_id == 0
    np.testing.assert_allclose(tracker.tracks[0].velocity, [2., 0.], atol=.15)
    assert tracker.tracks[0].half_width_m == pytest.approx(.8)
    snapshot = tracker.tracks[0]
    tracker.tracks[0].position[:] = 0.
    tracker.tracks[0].velocity[:] = 99.
    assert tracker.tracks[0].position[0] > 30.
    assert tracker.tracks[0].velocity[0] < 3.
    # Equal-time updates do not advance a moving prediction.
    previous = tracker.tracks[0].position.copy()
    tracker.update(PlanarEgoState(), 15., [points], [.9])
    np.testing.assert_allclose(tracker.tracks[0].position, previous, atol=.03)
    tracker.update(PlanarEgoState(), 15.1, [], [])
    assert tracker.tracks == ()
    points[:, 0] += .4
    tracker.update(PlanarEgoState(), 15.2, [points], [.9])
    assert tracker.tracks[0].position[0] > snapshot.position[0]
    assert tracker.tracks[0].track_id == snapshot.track_id
    assert tracker.tracks[0].hits == 153
    np.testing.assert_allclose(snapshot.position, [40., 3.], atol=.15)
    with pytest.raises(ValueError, match="nondecreasing"):
        tracker.update(PlanarEgoState(), 15.1, [], [])
    with pytest.raises(ValueError, match=r"finite \(3, 3\)"):
        tracker.update(PlanarEgoState(), 15.4, [np.zeros(3)], [.9])
    with pytest.raises(ValueError, match="scores"):
        tracker.update(PlanarEgoState(), 15.4, [points], [])
    with pytest.raises(ValueError, match="scores"):
        tracker.update(PlanarEgoState(), 15.4, [points], [np.nan])
    with pytest.raises(ValueError, match="positive"):
        VehicleTracker(association_gate_chi2=0.)

    # Prediction preserves identities as the targets cross and measurement order changes.
    tracker.reset()
    for frame in range(61):
        t = frame * .1
        centers = [np.array([20. + 2. * t, -3. + t]), np.array([32. - 2. * t, 3. - t])]
        measured = [np.array([[center[0], center[1] + .5, 0.],
                              [center[0], center[1], 0.], [center[0], center[1] - .5, 0.]])
                    for center in centers[::-1]]
        if frame % 2:
            measured.reverse()
        tracker.update(PlanarEgoState(), t, measured, [.9, .9])
    tracks = {track.track_id: track for track in tracker.tracks}
    np.testing.assert_allclose(tracks[0].position, centers[1], atol=.05)
    np.testing.assert_allclose(tracks[1].position, centers[0], atol=.05)
    np.testing.assert_allclose(tracks[0].velocity, [-2., -1.], atol=.05)
    np.testing.assert_allclose(tracks[1].velocity, [2., 1.], atol=.05)
    far = np.array([[100., 1., 0.], [100., 0., 0.], [100., -1., 0.]])
    tracker.update(PlanarEgoState(), 6.1, [far], [.9])
    assert tracker.tracks == ()
    tracker.update(PlanarEgoState(), 6.2, [far], [.9])
    assert {track.track_id for track in tracker.tracks} == {2}
    assert tracker.tracks[-1].hits == 2


def test_vehicle_score_association():
    tracker = VehicleTracker()
    assert tracker.max_missed_updates == 10
    state = PlanarEgoState()
    face = np.array([[10., 1., 0.], [10., 0., 0.], [10., -1., 0.]])
    far = face + np.array([20., 0., 0.])
    tracker.update(state, 0., [face, far], [.5, .49])
    assert tracker.tracks == ()
    tracker.update(state, .1, [face, far], [.1, .4])
    assert len(tracker.tracks) == 1
    assert tracker.tracks[0].track_id == 0 and tracker.tracks[0].hits == 2
    # A high-score association wins before a closer low-score measurement is considered.
    high_face = face + np.array([.2, 0., 0.])
    high_face[:, 1] *= 2
    tracker.update(state, .2, [high_face, face], [.5, .49])
    assert tracker.tracks[0].half_width_m == pytest.approx(2.)
    assert tracker.tracks[0].hits == 3
    tracker.update(state, .3, [], [])
    assert tracker.tracks == ()
    tracker.update(state, .4, [face], [.49])
    assert tracker.tracks == ()
    tracker.update(state, .5, [face], [.5])
    assert tracker.tracks[0].track_id == 0 and tracker.tracks[0].hits == 4
    for index in range(9):
        tracker.update(state, .6 + index * .1, [], [])
        assert tracker.tracks == ()
    tracker.update(state, 1.5, [face], [.5])
    assert tracker.tracks[0].track_id == 0 and tracker.tracks[0].hits == 5
    for index in range(10):
        tracker.update(state, 1.6 + index * .1, [], [])
        assert tracker.tracks == ()
    tracker.update(state, 2.6, [face], [.5])
    assert tracker.tracks == ()
    tracker.update(state, 2.7, [face], [.5])
    assert tracker.tracks[0].track_id == 1 and tracker.tracks[0].hits == 2

    equal_thresholds = VehicleTracker(high_score_threshold=.1, low_score_threshold=.1, min_hits=1)
    equal_thresholds.update(state, 0., [face], [.1])
    assert len(equal_thresholds.tracks) == 1
    with pytest.raises(ValueError, match="threshold"):
        VehicleTracker(high_score_threshold=.2, low_score_threshold=.3)
    with pytest.raises(ValueError, match="threshold"):
        VehicleTracker(high_score_threshold=1.1)
