"""Generic evaluation retains ordered logs, missing evidence and metric failures."""

import numpy as np
import pytest

from avsectester.evaluation import (
    DrivingMetrics, Metric, RunGroup, StopImpact, TrajectoryDeviation,
    UnavailableMetricValue, evaluate_groups,
)
from avsectester.runlog import RunLog


class UserMetric(Metric):
    def evaluate(self, group):
        return {name: {"seed": np.int64(group.seed),
                       "confidence": UnavailableMetricValue("No confidence measurement"),
                       "detected": UnavailableMetricValue("No detection measurement", "boolean")}
                for name in group.runs}


class BrokenMetric(Metric):
    def evaluate(self, group):
        raise RuntimeError("unavailable scoring service")


class InvalidMetric(Metric):
    def evaluate(self, group):
        return {"clean": {"probe": 1.}, "unknown": {"probe": 2.}}


class InterruptedMetric(Metric):
    def evaluate(self, group):
        raise KeyboardInterrupt


def test_evaluate_groups(segment, make_log):
    groups = (
        RunGroup(0, segment, 1, {
            "clean": make_log([3., 3.]), "attacked": make_log([1., 1.]),
            "rain": make_log([2., 2.]),
        }, treatment_attacks={"clean": None, "attacked": "first", "rain": "second"}, part="test"),
        RunGroup(0, segment, 0, {
            "clean": make_log([3., 3.]), "attacked": make_log([1., 1.]),
            "rain": RunLog(status="failed"),
        }, errors={"rain": "sensor disconnected"},
                 treatment_attacks={"clean": None, "attacked": "first", "rain": "second"}, part="test"),
    )
    report = evaluate_groups(groups, (
        DrivingMetrics(), TrajectoryDeviation(), StopImpact(), UserMetric(), BrokenMetric(), InvalidMetric(),
    ), notes=("Recorded evidence only.",))
    assert [group["seed"] for group in report.groups] == [1, 0]
    first, last = report.groups
    assert first["part"] == "test"
    assert first["treatment_attacks"] == {"clean": None, "attacked": "first", "rain": "second"}
    assert list(first["treatments"]) == ["clean", "attacked", "rain"]
    assert [(pair["treatment_a"], pair["treatment_b"]) for pair in first["comparisons"]] == [
        ("clean", "attacked"), ("clean", "rain"), ("attacked", "rain"),
    ]
    pair = first["comparisons"][0]
    assert pair["deltas"]["DrivingMetrics.final_speed"] == -2.
    assert "TrajectoryDeviation.final_deviation" in pair["metrics"]
    assert {"StopImpact.a_to_b.induced_stop", "StopImpact.b_to_a.induced_stop"} <= pair["metrics"].keys()
    assert pair["deltas"]["UserMetric.confidence"] is None
    assert pair["reasons"]["delta.UserMetric.confidence"] == "One run measurement is unavailable"
    clean = first["treatments"]["clean"]
    assert clean["metrics"]["UserMetric.seed"] == 1
    assert isinstance(clean["metrics"]["UserMetric.seed"], int)
    assert clean["metrics"]["UserMetric.confidence"] is None
    assert clean["reasons"]["UserMetric.confidence"] == "No confidence measurement"
    assert clean["metric_types"]["UserMetric.detected"] == "boolean"
    assert "InvalidMetric.probe" not in clean["metrics"]
    assert last["treatments"]["rain"]["status"] == "failed"
    assert last["treatments"]["rain"]["error"] == "sensor disconnected"
    assert last["comparisons"][1]["metrics"] is None
    assert "rain=failed" in last["comparisons"][1]["reason"]
    assert report.metric_errors[0]["error"] == "RuntimeError: unavailable scoring service"
    assert report.metric_errors[1]["error"] == "ValueError: Metric returned unknown or incomplete treatment 'unknown'"
    assert report.notes == ["Recorded evidence only."]
    assert [row["treatment"] for row in report.treatment_summaries] == ["clean", "attacked", "rain"]
    assert {metric["type"].split(":")[-1] for metric in report.metrics} >= {"DrivingMetrics", "UserMetric"}
    with pytest.raises(KeyboardInterrupt):
        evaluate_groups(groups[:1], (InterruptedMetric(),))
