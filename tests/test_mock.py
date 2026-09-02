"""End-to-end on the mock env/system: attacks impact driving; a defense mitigates."""

from __future__ import annotations

import pytest

pytest.importorskip("avstack")

from avsectester.attacks import ObjectSpoofingAttack, PhantomDetectionAttack
from avsectester.core import run, run_experiment
from avsectester.defenses import ScoreGateDefense
from avsectester.envs import MockEnv, MockSystem
from avsectester.metrics import ImpactMetric


def test_clean_run_cruises_without_braking():
    tr = run(MockEnv(frames=80), MockSystem(target_speed=6.0))
    assert tr.count("braking") == 0
    assert tr.series("ego_speed")[-1] > 3.0


def test_phantom_detection_impacts_driving():
    res = run_experiment(
        lambda: MockEnv(frames=100), MockSystem, ImpactMetric(),
        attack=PhantomDetectionAttack(target_xyz=[6.0, 0.0, -1.5]),
    )
    assert res.metrics["impacted"] is True
    assert res.metrics["brake_frames_clean"] == 0 and res.metrics["stopped"]


def test_score_gate_mitigates_object_spoof():
    res = run_experiment(
        lambda: MockEnv(frames=100), MockSystem, ImpactMetric(),
        attack=ObjectSpoofingAttack(target_xyz=[6.0, 0.0, 0.0], score=0.3),
        defense=ScoreGateDefense(threshold=0.5),
    )
    assert res.metrics["impacted"] is True
    assert res.defended_metrics["impacted"] is False   # defense removed the low-score phantom
