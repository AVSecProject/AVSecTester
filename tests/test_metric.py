"""The impact metric must not call an attack a success unless the clean run actually drove.

Covers the three outcomes: a real induced stop (success), a degenerate run where the clean ego
never moved (inconclusive, not success), and braking induced without a full stop.
"""

from avsectester.metric import impact
from avsectester.scenario import FrameRecord, Trace


def _trace(speeds, brakes):
    recs = [
        FrameRecord(
            frame=i, t=i * 0.05, n_detections=5, speed=s,
            throttle=0.0 if b > 0 else 0.75, brake=b, steer=0.0,
        )
        for i, (s, b) in enumerate(zip(speeds, brakes))
    ]
    return Trace(records=recs)


def test_real_induced_stop_is_success():
    clean = _trace([0, 1, 2, 3, 4, 5], [0, 0, 0, 0, 0, 0])       # cruises to 5 m/s, never brakes
    attacked = _trace([0, 1, 2, 1, 0, 0], [0, 0, 1, 1, 1, 1])    # braked to a stop
    r = impact(clean, attacked)
    assert r.clean_drove and r.induced_stop and r.attack_succeeded
    assert "forced an unsafe stop" in r.verdict


def test_clean_never_drove_is_inconclusive_not_success():
    # the exact case that motivated this: both runs ~stationary, attacked has brake frames
    clean = _trace([0.1, 0.15, 0.15, 0.15], [0, 0, 0, 0])
    attacked = _trace([0.1, 0.15, 0.15, 0.15], [0, 1, 1, 1])
    r = impact(clean, attacked)
    assert not r.clean_drove
    assert not r.attack_succeeded          # induced_braking alone must NOT be a success
    assert r.verdict.startswith("INCONCLUSIVE")


def test_braking_without_stop_is_not_success():
    clean = _trace([0, 2, 4, 5, 5, 5], [0, 0, 0, 0, 0, 0])       # cruising
    attacked = _trace([0, 2, 4, 3, 2, 2], [0, 0, 1, 1, 1, 1])    # slowed but ends at 2 m/s (> stop)
    r = impact(clean, attacked)
    assert r.clean_drove and r.induced_braking and not r.induced_stop
    assert not r.attack_succeeded
    assert "INDUCED BRAKING" in r.verdict
