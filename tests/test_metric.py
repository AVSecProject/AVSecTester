"""The impact metric must not call an attack a success unless the clean run actually drove.

Covers the three outcomes: a real induced stop (success), a degenerate run where the clean ego
never moved (inconclusive, not success), and braking induced without a full stop.
"""

import pytest
from avsectester.metric import impact
from avsectester.plane import FrameRecord, Trace


def _trace(speeds, brakes):
    recs = [
        FrameRecord(
            frame=i,
            t=i * 0.05,
            n_detections=5,
            speed=s,
            throttle=0.0 if b > 0 else 0.75,
            brake=b,
            steer=0.0,
        )
        for i, (s, b) in enumerate(zip(speeds, brakes))
    ]
    return Trace(records=recs)


def test_real_induced_stop_is_success():
    clean = _trace([0, 1, 2, 3, 4, 5], [0, 0, 0, 0, 0, 0])  # cruises to 5 m/s, never brakes
    attacked = _trace([0, 1, 2, 1, 0, 0], [0, 0, 1, 1, 1, 1])  # braked to a stop
    r = impact(clean, attacked)
    assert r.clean_drove and r.induced_stop and r.attack_succeeded
    assert "forced an unsafe stop" in r.verdict


def test_clean_never_drove_is_inconclusive_not_success():
    # the exact case that motivated this: both runs ~stationary, attacked has brake frames
    clean = _trace([0.1, 0.15, 0.15, 0.15], [0, 0, 0, 0])
    attacked = _trace([0.1, 0.15, 0.15, 0.15], [0, 1, 1, 1])
    r = impact(clean, attacked)
    assert not r.clean_drove
    assert not r.attack_succeeded  # induced_braking alone must NOT be a success
    assert r.verdict.startswith("INCONCLUSIVE")


def test_braking_without_stop_is_not_success():
    clean = _trace([0, 2, 4, 5, 5, 5], [0, 0, 0, 0, 0, 0])  # cruising
    attacked = _trace([0, 2, 4, 3, 2, 2], [0, 0, 1, 1, 1, 1])  # slowed but ends at 2 m/s (> stop)
    r = impact(clean, attacked)
    assert r.clean_drove and r.induced_braking and not r.induced_stop
    assert not r.attack_succeeded
    assert "INDUCED BRAKING" in r.verdict


@pytest.mark.parametrize(
    "speeds,brakes",
    [
        ([0, 2, 5], [0, 0, 0]),
        ([5, 2, 0], [0, 1, 1]),
    ],
)
def test_identical_drives_have_no_attack_impact(make_trace, speeds, brakes):
    result = impact(make_trace(speeds, brakes), make_trace(speeds, brakes))
    assert result.clean_drove
    assert not result.induced_braking
    assert not result.induced_stop
    assert not result.attack_succeeded
    assert result.verdict == "no impact (attack did not change the drive)"


def test_both_runs_stop_naturally_is_not_attack_success(make_trace):
    result = impact(make_trace([5, 2, 0], [0, 1, 1]), make_trace([5, 0, 0], [1, 1, 1]))
    assert result.clean_drove and result.induced_braking
    assert not result.induced_stop
    assert not result.attack_succeeded


@pytest.mark.parametrize(
    "peak,clean_final,attacked_final,expected",
    [
        (0.999, 0.999, 0.0, False),  # clean driving threshold is inclusive
        (1.0, 1.0, 0.0, True),
        (5.0, 5.0, 0.5, True),  # stopping threshold is inclusive
        (5.0, 5.0, 0.501, False),
        (5.0, 0.5, 0.0, False),  # baseline also stopped
        (5.0, 0.501, 0.0, True),
    ],
)
def test_verdict_speed_boundaries(make_trace, peak, clean_final, attacked_final, expected):
    result = impact(
        make_trace([0, peak, clean_final]),
        make_trace([0, 0, attacked_final], [0, 1, 1]),
    )
    assert result.attack_succeeded is expected


def test_custom_speed_thresholds_are_used(make_trace):
    clean, attacked = make_trace([0, 2, 2]), make_trace([0, 1, 0.75], [0, 1, 1])
    assert impact(clean, attacked, stop_speed=0.8, baseline_speed=2.0).attack_succeeded
    assert not impact(clean, attacked, stop_speed=0.7, baseline_speed=2.0).attack_succeeded
    assert not impact(clean, attacked, stop_speed=0.8, baseline_speed=2.1).attack_succeeded


def test_impact_reports_measurements_from_each_run(make_trace):
    result = impact(make_trace([0, 6, 4]), make_trace([0, 3, 0.2], [0, 1, 1]))
    assert result.clean_peak_speed == 6
    assert result.clean_final_speed == 4
    assert result.attacked_final_speed == 0.2
    assert result.clean_brake_frames == 0
    assert result.attacked_brake_frames == 2
    assert result.clean_drove and result.induced_braking and result.induced_stop
    assert "ATTACK SUCCEEDED" in str(result)


def test_empty_runs_are_inconclusive(make_trace):
    result = impact(make_trace([]), make_trace([]))
    assert not result.attack_succeeded
    assert result.verdict.startswith("INCONCLUSIVE")


@pytest.mark.parametrize("attacked_speeds", [[], [0.0]])
@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="impact() accepts missing/truncated attacked traces as successful stops",
)
def test_incomplete_attacked_run_is_inconclusive(make_trace, attacked_speeds):
    # A stopped first frame or no frames cannot establish the outcome of the requested drive.
    result = impact(make_trace([0, 2, 5]), make_trace(attacked_speeds))
    assert not result.attack_succeeded
    assert result.verdict.startswith("INCONCLUSIVE")
