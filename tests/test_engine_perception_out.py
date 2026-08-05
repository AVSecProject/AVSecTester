"""Engine end-to-end with a detection-level attack at the perception_out seam (no CARLA).

Complements test_engine.py (which drives the perception_input seam): here the phantom is
injected as a detector post-hook, verifying the seam-aware attach routing and that a
detection-level attack escalates to an unsafe stop on the CARLA-free MockBackend.
"""
from __future__ import annotations

import pytest

pytest.importorskip("avstack")

from avsectester.core.engine import ExperimentRunner
from avsectester.core.experiment import ExperimentSpec


def _spec() -> ExperimentSpec:
    return ExperimentSpec.model_validate(
        {
            "name": "phantom-detection-mock",
            "system": {"name": "mock-av"},
            "scenario": {
                "backend": {"type": "MockBackend", "frames": 140, "target_speed": 6.0},
                "initial_conditions": {"frames": 140, "target_speed": 6.0},
            },
            "attack": {
                "spec": {"type": "PhantomDetectionAttack", "target_xyz": [10.0, 0.0, 0.0], "score": 0.9},
                "threat_model": {
                    "goal": "phantom brake", "target": "detector output",
                    "success_criteria": "unsafe stop",
                },
            },
        }
    )


def test_detection_level_attack_escalates_on_mock():
    result = ExperimentRunner(_spec()).run()
    # clean cruises; attacked brakes to a stop for the phantom detection
    assert result.metrics["escalated"] is True
    assert result.metrics["min_speed_clean"] > 3.0
    assert result.metrics["final_speed_attacked"] < 1.0
    # the phantom shows up as an extra detection -> confirmed track
    assert result.attacked_trace.series("tracking", "count")[-1] >= 1
