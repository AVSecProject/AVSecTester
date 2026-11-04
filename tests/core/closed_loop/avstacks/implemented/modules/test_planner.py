"""Tracked-sign stops, cruise and trajectory-controller interoperability."""

import numpy as np
import pytest

from avsectester.closed_loop.avstacks.implemented.modules.planner import RuleBasedPlanner
from avsectester.closed_loop.avstacks.implemented.modules.tracker import SignTrack, VehicleTrack
from avsectester.closed_loop.simulators.ego_dynamics import AlpaSimController, PlanarEgoState
from avsectester.data_types import Observation


def observation(state, route=None):
    return Observation(t=state.t, frame=round(state.t * 10), vehicle_state=state,
                       ego_speed=state.speed, route=route)


@pytest.mark.parametrize("stop_before_sign_m", [5.5, 10.])
def test_rule_based_planner(stop_before_sign_m):
    planner = RuleBasedPlanner(stop_before_sign_m=stop_before_sign_m, stop_duration_s=2.)
    state = PlanarEgoState(speed=13.)
    planner.reset(observation(state))
    records = []
    for frame in range(260):
        tracks = [SignTrack(0, np.array([50., 0.]), np.eye(2))]
        command = planner.step(observation(state), tracks)
        log = planner.component_log
        records.append((state, command, log))
        speed = log["planned_speed_mps"]
        state = PlanarEgoState(x=state.x + speed * .1, speed=speed, t=(frame + 1) * .1)
    stopped = [(current, command, log) for current, command, log in records if log["state"] == "STOPPED"]
    stopped_at, stopped_position = stopped[0][0].t, stopped[0][0].x
    resumed = [(current, command, log) for current, command, log in records
               if current.t > stopped_at and log["state"] != "STOPPED"]
    resumed_at = resumed[0][0].t
    assert records[0][2]["braking"] is False
    assert records[0][2]["acceleration_mps2"] == pytest.approx(0.)
    assert records[0][1].trajectory[-1][0][0] < 13. * planner.trajectory_horizon_s
    assert all(log["acceleration_mps2"] <= 0. for _, _, log in records if log["braking"])
    for _, command, _ in stopped:
        np.testing.assert_allclose([point[0] for point in command.trajectory], 0.)
    assert all(current.t - stopped_at < 2. + 1e-9 for current, _, _ in stopped)
    assert all(log["state"] == "CRUISE" for _, _, log in resumed)
    target = 50. - planner.stop_before_sign_m
    assert stopped_position is not None and target - .5 <= stopped_position <= target + .2
    assert stopped_at <= 2 * target / 13. + 1.
    assert resumed_at is not None and resumed_at - stopped_at >= 2.
    assert state.speed > 5. and state.x > 50.
    planner.reset(observation(PlanarEgoState(speed=13.)))
    planner.step(observation(PlanarEgoState(speed=13.)), [SignTrack(0, np.array([50., 0.]), np.eye(2))])
    assert planner.component_log["state"] == "STOPPING"
    assert planner.component_log["target_track_id"] == 0


def test_braking_latch():
    planner = RuleBasedPlanner()
    initial = PlanarEgoState(speed=13.)
    planner.reset(observation(initial))
    command = planner.step(observation(initial), [SignTrack(0, np.array([50., 0.]), np.eye(2))])
    assert planner.component_log["braking"] is False
    assert planner.component_log["acceleration_mps2"] == pytest.approx(0.)
    assert command.trajectory[-1][0][0] < 65.
    # Crossing the trigger in the forecast did not latch the current decision.
    planner.step(observation(PlanarEgoState(x=1.3, speed=13., t=.1)),
                 [SignTrack(0, np.array([50., 0.]), np.eye(2))])
    assert planner.component_log["braking"] is False
    planner.step(observation(PlanarEgoState(x=25., speed=13., t=.2)),
                 [SignTrack(0, np.array([50., 0.]), np.eye(2))])
    assert planner.component_log["braking"] is True
    # A lower required deceleration stays latched, without a comfortable-rate floor.
    planner.step(observation(PlanarEgoState(x=26., speed=5., t=.3)),
                 [SignTrack(0, np.array([50., 0.]), np.eye(2))])
    assert planner.component_log["braking"] is True
    assert planner.component_log["acceleration_mps2"] == pytest.approx(-25. / 37.)
    # A nearer, distinct target clears the old target's latch.
    planner.step(observation(PlanarEgoState(x=26., speed=5., t=.4)),
                 [SignTrack(1, np.array([38., 0.]), np.eye(2))])
    assert planner.component_log["braking"] is False
    planner.step(observation(PlanarEgoState(x=80., speed=5., t=.5)), ())
    assert planner.component_log["state"] == "CRUISE"
    assert planner.component_log["braking"] is False


def test_late_detection():
    planner = RuleBasedPlanner()
    state = PlanarEgoState(speed=13.)
    planner.reset(observation(state))
    stopped = None
    records = []
    for index in range(100):
        planner.step(observation(state), [SignTrack(0, np.array([15.5, 0.]), np.eye(2))])
        log = planner.component_log
        records.append(log)
        if log["state"] == "STOPPED":
            stopped = state
            break
        speed = log["planned_speed_mps"]
        state = PlanarEgoState(x=state.x + speed * .1, speed=speed, t=(index + 1) * .1)
    assert records[0]["braking"] is True
    assert records[0]["acceleration_mps2"] == -planner.max_deceleration_mps2
    assert all(log["braking"] is True and log["acceleration_mps2"] <= 0.
               for log in records if log["state"] != "STOPPED")
    assert stopped is not None
    assert stopped.speed < planner.stop_speed_mps
    assert stopped.x > 15.5 - planner.stop_before_sign_m


def test_cruise():
    planner = RuleBasedPlanner()
    state = PlanarEgoState(speed=13., t=2.)
    planner.reset(observation(state))
    command = planner.step(observation(state), [SignTrack(0, np.array([-5., 0.]), np.eye(2)),
                                                 SignTrack(1, np.array([100., 0.]), np.eye(2))])
    assert planner.component_log["state"] == "CRUISE"
    assert planner.component_log["acceleration_mps2"] == pytest.approx(0.)
    assert command.trajectory[0] == ((0., 0., 0.), (1., 0., 0., 0.), 2_000_000)
    assert command.trajectory[-1][0] == pytest.approx((65., 0., 0.))
    times = [point[2] for point in command.trajectory]
    assert len(times) == 51 and all(b > a for a, b in zip(times, times[1:]))
    # Arc-length sampling follows the supplied route, then continues its final tangent.
    route = np.array([[0., 0., 0.], [30., 0., 0.], [30., 20., 0.]])
    planner.reset(observation(state, route))
    command = planner.step(observation(state, route), ())
    assert command.trajectory[-1][0] == pytest.approx((30., 35., 0.))
    assert command.trajectory[-1][1] == pytest.approx((np.sqrt(.5), 0., 0., np.sqrt(.5)))
    changed = RuleBasedPlanner(trajectory_step_s=.2, trajectory_horizon_s=1.)
    changed.reset(observation(state))
    assert [p[2] for p in changed.step(observation(state), ()).trajectory] == [
        2_000_000, 2_200_000, 2_400_000, 2_600_000, 2_800_000, 3_000_000,
    ]
    with pytest.raises(ValueError, match="trajectory_horizon_s"):
        RuleBasedPlanner(trajectory_horizon_s=.1, trajectory_step_s=.2)
    with pytest.raises(ValueError, match="microseconds"):
        RuleBasedPlanner(trajectory_step_s=.0000001)
    # Sub-microsecond rounding at a partial final step must not duplicate a timestamp.
    short = RuleBasedPlanner(trajectory_horizon_s=.10000000001)
    short.reset(observation(state))
    assert [p[2] for p in short.step(observation(state), ()).trajectory] == [2_000_000, 2_100_000]


def test_sign_relevance():
    planner = RuleBasedPlanner()
    state = PlanarEgoState(speed=13.)
    planner.reset(observation(state))
    planner.step(observation(state), [SignTrack(0, np.array([50., 11.]), np.eye(2))])
    assert planner.component_log["state"] == "CRUISE"
    planner.step(observation(state), [SignTrack(1, np.array([40., 10.]), np.eye(2))])
    assert planner.component_log["state"] == "STOPPING"
    assert planner.component_log["distance_to_stop_m"] == pytest.approx(34.5)


def test_alpasim_trajectory():
    pytest.importorskip("alpasim_controller")
    dynamics = AlpaSimController()
    planner = RuleBasedPlanner()
    state = PlanarEgoState(x=10., y=-2., yaw=.7, speed=13., t=.2)
    planner.reset(observation(state))
    dynamics.reset(state)
    for _ in range(3):
        command = planner.step(observation(state), ())
        state = dynamics.step(state, command, .1)
    assert state.speed == pytest.approx(13., abs=.1)
    assert state.t == pytest.approx(.5)
    assert np.isfinite([state.x, state.y, state.yaw]).all()
    stationary = PlanarEgoState(t=1.)
    planner.reset(observation(stationary))
    command = planner.step(observation(stationary), [SignTrack(0, np.array([planner.stop_before_sign_m, 0.]), np.eye(2))])
    assert planner.component_log["state"] == "STOPPED"
    dynamics.reset(stationary)
    advanced = dynamics.step(stationary, command, .1)
    assert advanced.speed == pytest.approx(0., abs=1e-8)
    assert [advanced.x, advanced.y] == pytest.approx([0., 0.], abs=1e-8)


def test_vehicle_following():
    planner = RuleBasedPlanner()
    state = PlanarEgoState(speed=8.)
    planner.reset(observation(state))
    vehicles = (
        VehicleTrack(1, np.array([45., 0.]), np.zeros(2), 1.),
        VehicleTrack(0, np.array([25., 0.]), np.zeros(2), 1.),
    )
    gaps, accelerations = [], []
    for index in range(300):
        planner.step(observation(state), (), vehicles)
        log = planner.component_log
        gap = 25. - state.x - planner.ego_front_offset_m
        assert log["lead_track_id"] == 0
        assert log["lead_gap_m"] == pytest.approx(gap)
        assert log["lead_speed_mps"] == 0.
        assert log["lead_lateral_interval_m"] == [-1., 1.]
        gaps.append(gap)
        accelerations.append(log["acceleration_mps2"])
        speed = log["planned_speed_mps"]
        state = PlanarEgoState(x=state.x + (state.speed + speed) * .05,
                              speed=speed, t=(index + 1) * .1)
    assert accelerations[0] < 0.
    assert state.speed < .1
    assert min(gaps) > 0.
    assert gaps[-1] == pytest.approx(planner.min_gap_m, abs=.1)

    # A moving leader at the IDM equilibrium gap preserves speed through the horizon.
    planner = RuleBasedPlanner(cruise_speed_mps=10.)
    state = PlanarEgoState(speed=5.)
    gap = (2. + 5. * 1.5) / np.sqrt(1 - (5. / 10.) ** 4)
    moving = VehicleTrack(2, np.array([gap + 3.7, 0.]), np.array([5., 0.]), 1.)
    planner.reset(observation(state))
    command = planner.step(observation(state), (), (moving,))
    assert planner.component_log["acceleration_mps2"] == pytest.approx(0., abs=1e-12)
    assert planner.component_log["lead_speed_mps"] == 5.
    assert command.trajectory[-1][0] == pytest.approx((25., 0., 0.), abs=1e-10)

    # A roadside vehicle's near edge stays outside the ego corridor on both sides.
    ignored = (
        VehicleTrack(3, np.array([10., 2.]), np.zeros(2), .5),
        VehicleTrack(4, np.array([-1., 0.]), np.zeros(2), 1.),
        VehicleTrack(5, np.array([61., 0.]), np.zeros(2), 1.),
        VehicleTrack(9, np.array([10., -2.]), np.zeros(2), .5),
    )
    planner.reset(observation(state))
    clean = planner.step(observation(state), ())
    planner.reset(observation(state))
    assert planner.step(observation(state), (), ignored).trajectory == clean.trajectory
    assert planner.component_log["lead_track_id"] is None
    assert planner.component_log["lead_gap_m"] is None
    assert planner.component_log["lead_speed_mps"] is None
    assert planner.component_log["lead_lateral_interval_m"] is None

    # Partial overlap is relevant even when the contact center is outside the corridor.
    for side in (-1., 1.):
        overlap = VehicleTrack(10, np.array([10., side * 2.]), np.zeros(2), .8)
        planner.reset(observation(state))
        planner.step(observation(state), (), (overlap,))
        assert planner.component_log["lead_track_id"] == 10
        assert planner.component_log["lead_lateral_interval_m"] == pytest.approx(
            [side * 2. - .8, side * 2. + .8])

    # A curved rig-frame route determines relevance and the world-velocity projection.
    route = np.array([[0., 0., 0.], [10., 0., 0.], [10., 40., 0.]])
    turned = observation(PlanarEgoState(x=10., y=20., yaw=np.pi / 2, speed=3.), route)
    around_turn = VehicleTrack(6, np.array([0., 30.]), np.array([-4., 0.]), .9)
    planner.reset(turned)
    planner.step(turned, (), (around_turn,))
    assert planner.component_log["lead_track_id"] == 6
    assert planner.component_log["lead_gap_m"] == pytest.approx(20. - 3.7)
    assert planner.component_log["lead_speed_mps"] == pytest.approx(4.)
    assert planner.component_log["lead_lateral_interval_m"] == pytest.approx([-.9, .9])
    reverse = VehicleTrack(6, around_turn.position, np.array([4., 0.]), .9)
    planner.step(turned, (), (reverse,))
    assert planner.component_log["lead_speed_mps"] == 0.

    # Sign braking and following combine by taking the more conservative acceleration.
    state = PlanarEgoState(speed=8.)
    signs = (SignTrack(0, np.array([10., 0.]), np.eye(2)),)
    sign_only, lead_only, combined = RuleBasedPlanner(), RuleBasedPlanner(), RuleBasedPlanner()
    for driver in (sign_only, lead_only, combined):
        driver.reset(observation(state))
    sign_only.step(observation(state), signs)
    lead_only.step(observation(state), (), vehicles)
    combined.step(observation(state), signs, vehicles)
    assert combined.component_log["acceleration_mps2"] == min(
        sign_only.component_log["acceleration_mps2"], lead_only.component_log["acceleration_mps2"])
    too_close = VehicleTrack(7, np.array([2., 0.]), np.zeros(2), 1.)
    combined.reset(observation(state))
    combined.step(observation(state), (), (too_close,))
    assert combined.component_log["acceleration_mps2"] == -combined.max_deceleration_mps2

    stationary = PlanarEgoState()
    combined.reset(observation(stationary))
    stopped = combined.step(observation(stationary),
        (SignTrack(0, np.array([5.5, 0.]), np.eye(2)),), vehicles)
    assert combined.component_log["state"] == "STOPPED"
    np.testing.assert_array_equal([point[0] for point in stopped.trajectory], 0.)
    with pytest.raises(ValueError, match="min_gap_m"):
        RuleBasedPlanner(min_gap_m=-1.)
    with pytest.raises(ValueError, match="vehicle_lookahead_m"):
        RuleBasedPlanner(vehicle_lookahead_m=0.)
    with pytest.raises(ValueError, match="ego_half_width_m"):
        RuleBasedPlanner(ego_half_width_m=0.)
    with pytest.raises(ValueError, match="lateral_margin_m"):
        RuleBasedPlanner(lateral_margin_m=-.1)
    with pytest.raises(ValueError, match="position and velocity"):
        planner.step(turned, (), (VehicleTrack(8, np.zeros(2), np.array([np.nan, 0.]), 1.),))
    with pytest.raises(ValueError, match="half_width_m"):
        planner.step(turned, (), (VehicleTrack(8, np.array([5., 0.]), np.zeros(2), -.1),))
