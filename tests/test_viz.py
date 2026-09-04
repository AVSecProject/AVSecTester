"""The driving-impact plot renders from two traces (needs matplotlib; skipped otherwise)."""

import pytest
from avsectester.scenario import FrameRecord, Trace


def _trace(speeds, brakes):
    recs = [
        FrameRecord(frame=i, t=i * 0.05, n_detections=5, speed=s,
                    throttle=0.0 if b > 0 else 0.75, brake=b, steer=0.0)
        for i, (s, b) in enumerate(zip(speeds, brakes))
    ]
    return Trace(records=recs)


def test_plot_impact_writes_png(tmp_path):
    pytest.importorskip("matplotlib")
    from avsectester.viz import plot_impact

    clean = _trace([0, 1, 2, 3, 4, 5], [0, 0, 0, 0, 0, 0])
    attacked = _trace([0, 1, 2, 1, 0, 0], [0, 0, 1, 1, 1, 1])
    out = tmp_path / "impact.png"
    path = plot_impact(clean, attacked, str(out))
    assert out.exists() and out.stat().st_size > 0
    assert path == str(out)
