"""Generic evaluation retains ordered logs, missing evidence and metric failures."""

import numpy as np
import pytest

from avsectester.evaluation import (
    DrivingMetrics, Metric, PairMetric, RunGroup, StopImpact, TrajectoryDeviation,
    UnavailableMetricValue, evaluate_groups,
)
from avsectester.runlog import RunLog


class UserMetric(Metric):
    """User measurements from saved evidence.

    This second paragraph is excluded from the default summary.
    """

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


class BrokenDescriptionMetric(UserMetric):
    def describe(self):
        raise RuntimeError("description unavailable")


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
    assert pair["deltas"]["DrivingMetrics.final_speed"] == -2.0
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
    assert "repeat" not in first and "repeat" not in report.metric_errors[0]
    assert report.repeat_pairs == [] and "repeat_pairs" not in report.as_dict()
    assert all("mean_differences" not in pair for pair in report.pair_summaries)
    assert [row["treatment"] for row in report.treatment_summaries] == ["clean", "attacked", "rain"]
    assert {metric["type"].split(":")[-1] for metric in report.metrics} >= {"DrivingMetrics", "UserMetric"}
    specifications = {item["label"]: item for item in report.as_dict()["metrics"]}
    driving = specifications["DrivingMetrics"]
    assert driving["summary"] and driving["fields"]["final_speed"]["description"]
    assert driving["fields"]["final_speed"]["unit"] == "m/s"
    assert specifications["StopImpact"]["kind"] == "pair"
    assert "a_to_b.reference_drove" in specifications["StopImpact"]["fields"]
    assert specifications["UserMetric"]["summary"] == "User measurements from saved evidence."
    assert specifications["UserMetric"]["fields"] == {}
    described = evaluate_groups(groups[:1], (BrokenDescriptionMetric(),))
    assert described.metric_errors == [{
        "metric": "BrokenDescriptionMetric", "error": "RuntimeError: description unavailable",
        "phase": "metric description",
    }]
    assert described.groups[0]["treatments"]["clean"]["metrics"]["BrokenDescriptionMetric.seed"] == 1
    assert described.metrics[0]["label"] == "BrokenDescriptionMetric"
    with pytest.raises(KeyboardInterrupt):
        evaluate_groups(groups[:1], (InterruptedMetric(),))


def test_repeated_groups(segment, make_log):
    class BrokenRepeatMetric(PairMetric):
        def evaluate(self, pair):
            if pair.treatment_a == pair.treatment_b:
                raise RuntimeError("Missing repeat evidence")
            return {}

    groups = (
        RunGroup(0, segment, 0, {"clean": make_log([1.]), "attacked": make_log([0.], shift=1.)}),
        RunGroup(0, segment, 0, {"clean": make_log([9.], shift=5.),
                                 "attacked": make_log([8.], shift=4.)}, repeat=2),
        RunGroup(0, segment, 0, {"clean": make_log([3.], shift=2.),
                                 "attacked": RunLog(status="failed")}, repeat=1),
        RunGroup(0, segment, 1, {"clean": make_log([0.])}),
        RunGroup(1, segment, 0, {"clean": make_log([0.])}),
        RunGroup(0, segment, 0, {"clean": make_log([0.])}, part="validation"),
    )
    report = evaluate_groups(groups, (DrivingMetrics(), TrajectoryDeviation(),
                                      BrokenRepeatMetric(), UserMetric()), repeats=3)
    assert [group["repeat"] for group in report.groups] == [0, 2, 1, 0, 0, 0]
    assert report.as_dict()["repeats"] == 3
    assert report.as_dict()["repeat_pairs"] == report.repeat_pairs
    assert [(row["treatment"], row["repeat_a"], row["repeat_b"]) for row in report.repeat_pairs] == [
        ("clean", 0, 1), ("clean", 0, 2), ("clean", 1, 2), ("attacked", 0, 2),
    ]
    assert all(row["seed"] == row["segment_index"] == 0 and row["part"] is None
               for row in report.repeat_pairs)
    deviations = [pair["metrics"]["TrajectoryDeviation.final_deviation"] for pair in report.repeat_pairs]
    assert deviations == [2., 5., 3., 3.]
    assert len(report.metric_errors) == 4
    first_error = report.metric_errors[0]
    assert first_error["repeat_a"] == 0 and first_error["repeat_b"] == 1
    assert first_error["treatment"] == "clean"
    comparison, clean_noise, attacked_noise, validation_noise = report.pair_summaries
    assert comparison["deltas"]["DrivingMetrics.final_speed"]["mean"] == -1.
    assert comparison["mean_differences"]["DrivingMetrics.final_speed"] == pytest.approx(1.4)
    assert comparison["mean_differences"]["DrivingMetrics.stopped"] == pytest.approx(.1)
    assert comparison["mean_differences"]["UserMetric.confidence"] is None
    assert "No completed measurement" in comparison["mean_difference_reasons"]["UserMetric.confidence"]
    assert clean_noise["treatment_a"] == clean_noise["treatment_b"] == "clean"
    assert attacked_noise["treatment_a"] == attacked_noise["treatment_b"] == "attacked"
    assert clean_noise["deltas"] == {} and "mean_differences" not in clean_noise
    assert clean_noise["groups"] == clean_noise["total_groups"] == 3
    score = clean_noise["metrics"]["TrajectoryDeviation.final_deviation"]
    assert (score["count"], score["min"], score["median"], score["max"]) == (3, 2., 3., 5.)
    assert attacked_noise["groups"] == attacked_noise["total_groups"] == 1
    assert validation_noise == {
        "part": "validation", "treatment_a": "clean", "treatment_b": "clean",
        "groups": 0, "total_groups": 0, "deltas": {}, "metrics": {},
    }

    failed_logs = {"clean": RunLog(status="failed"), "attacked": RunLog(status="failed")}
    failed = evaluate_groups([RunGroup(0, segment, 0, failed_logs, repeat=repeat)
                              for repeat in range(2)], (DrivingMetrics(), TrajectoryDeviation()), repeats=2)
    assert failed.as_dict()["repeat_pairs"] == []
    pair = failed.pair_summaries[0]
    assert pair["mean_differences"]["DrivingMetrics.final_speed"] is None
    assert pair["mean_difference_reasons"]["DrivingMetrics.final_speed"] == (
        "No completed measurement for clean or attacked"
    )
    for invalid in (groups[:2], (groups[0], groups[0])):
        with pytest.raises(ValueError, match="RunGroup.repeat must be below repeats and unique"):
            evaluate_groups(invalid, (), repeats=2)


def test_metric_labels(segment, make_log):
    first_type = type("SharedMetric", (UserMetric,), {"__module__": "user_metrics.first"})
    second_type = type("SharedMetric", (UserMetric,), {"__module__": "user_metrics.second"})
    group = RunGroup(0, segment, 0, {"clean": make_log([2.])})
    report = evaluate_groups((group,), (first_type(), second_type()))
    assert [item["label"] for item in report.metrics] == [
        "user_metrics.first:SharedMetric", "user_metrics.second:SharedMetric",
    ]
    measurements = report.groups[0]["treatments"]["clean"]["metrics"]
    assert measurements["user_metrics.first:SharedMetric.seed"] == 0
    assert measurements["user_metrics.second:SharedMetric.seed"] == 0
