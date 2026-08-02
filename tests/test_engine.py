"""End-to-end engine test on the CARLA-free MockBackend.

Exercises the whole framework: spec -> runner -> MockBackend -> attack/defense hooks
-> escalation metric + DAG -> report. Needs avstack (perception/tracking); no CARLA.
"""
from __future__ import annotations

import pytest

pytest.importorskip("avstack")

# ensure plugins self-register
import avsectester.attacks
import avsectester.backends.mock_backend
import avsectester.defenses
import avsectester.metrics  # noqa: F401
from avsectester.core.engine import ExperimentRunner
from avsectester.core.experiment import ExperimentSpec
from avsectester.reports import render_report

_SPEC = {
    "name": "mock_escalation_test",
    "system": {"name": "mock_av"},
    "scenario": {
        "backend": {"type": "MockBackend", "frames": 120, "target_speed": 6.0},
        "initial_conditions": {"frames": 120},
    },
    "attack": {
        "spec": {"type": "LidarSpoofAttack", "target_xyz": [12.0, 0.0, 0.0], "score": 0.3},
        "threat_model": {
            "goal": "phantom brake",
            "knowledge": "blackbox",
            "access": ["sensor"],
            "target": "ego perception",
            "success_criteria": "ego stops",
        },
    },
    "defense": {"spec": {"type": "ScoreGateDefense", "threshold": 0.5}},
}


def test_engine_escalation_and_mitigation():
    spec = ExperimentSpec.model_validate(_SPEC)
    result = ExperimentRunner(spec).run()
    m = result.metrics

    # attack escalates end to end
    assert m["activated"] is True
    assert m["reached_consequence"] is True
    assert m["stopped"] is True
    assert m["escalated"] is True
    assert m["stages_reached"] == [
        "attack_surface", "perception", "tracking", "control", "consequence"
    ]
    assert m["propagation_depth"] == 5
    assert m["brake_frames_clean"] == 0
    assert m["brake_frames_attacked"] > 0
    # clean ego was cruising (post-warmup), attacked was forced to a stop
    assert m["min_speed_clean"] > 3.0
    assert m["min_speed_attacked"] < 0.5

    # DAG populated + root cause at the attack surface
    assert result.dag.graph.number_of_nodes() == 5
    assert result.dag.root_cause().stage.value == "attack_surface"
    assert len(result.dag.consequence_paths()) >= 1

    # defense mitigates
    assert result.mitigated is True
    assert result.defended_metrics["escalated"] is False

    # report renders
    report = render_report(result)
    assert "ESCALATED" in report and "mitigated" in report


def test_engine_clean_only_no_attack():
    spec_dict = {k: v for k, v in _SPEC.items() if k not in ("attack", "defense")}
    spec = ExperimentSpec.model_validate(spec_dict)
    result = ExperimentRunner(spec).run()
    assert result.metrics["escalated"] is False
    assert result.metrics["activated"] is False
