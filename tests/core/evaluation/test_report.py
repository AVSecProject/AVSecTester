"""Reports retain run statuses, pairwise direction, statistics and evaluation failures."""

import csv
import json
from types import SimpleNamespace

import numpy as np
import pytest

from avsectester.evaluation import (
    DetectionMetrics, EvaluationConfiguration, Metric, PairMetric, StopImpact,
    TrajectoryDeviation, UnavailableMetricValue, evaluate,
)
from avsectester.experiment.models import ExperimentResult, TreatmentResult
from avsectester.runlog import RunLog


def configuration(output=None, metrics=None, treatments=("clean", "attack")):
    return SimpleNamespace(output=output, evaluation=EvaluationConfiguration(metrics=metrics),
                           treatments=tuple(SimpleNamespace(name=name) for name in treatments))


@pytest.fixture
def result_factory(group_factory, log_factory):
    def result(clean=None, attack=None, *, extra=None, segment_index=0, seed=0):
        segment = group_factory({}).segment
        logs = {"clean": clean if clean is not None else log_factory([2, 3]),
                "attack": attack if attack is not None else log_factory([2, .5], [0, 1])}
        logs.update(extra or {})
        runs = tuple(TreatmentResult("patch", segment_index, segment, name, seed, run,
                                     "failed on purpose" if run.status != "completed" else None)
                     for name, run in logs.items())
        return ExperimentResult(runs)
    return result


def test_report_saves_runs_pairs_statistics_and_plots(tmp_path, result_factory):
    report = evaluate(result_factory(), configuration(tmp_path))
    data = json.loads((tmp_path / "evaluation/report.json").read_text())
    pair = data["groups"][0]["comparisons"][0]
    assert pair["treatment_a"] == "clean" and pair["treatment_b"] == "attack"
    assert pair["deltas"]["DrivingMetrics.final_speed"] == -2.5
    assert pair["deltas"]["DrivingMetrics.stopped"] == 1
    assert pair["metrics"]["TrajectoryDeviation.final_deviation"] == 0
    assert "TrajectoryDeviation.final_deviation" not in pair["deltas"]
    summary = data["treatment_summaries"][1]
    assert summary["metrics"]["DrivingMetrics.final_speed"] == {
        "kind": "number", "count": 1, "mean": .5, "std": 0, "std_ddof": 0}
    assert summary["metrics"]["DrivingMetrics.stopped"]["true_fraction"] == 1
    text = report.format_summary()
    assert "(ratio)" in text and "(mean)" in text and "B - A" in text
    assert len(list((tmp_path / "evaluation/plots").glob("*.png"))) == 1
    with (tmp_path / "evaluation/runs.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2 and rows[0]["status"] == "completed"
    assert rows[1]["DrivingMetrics.final_speed"] == "0.5"
    obsolete = tmp_path / "evaluation/plots/obsolete.png"
    obsolete.touch()
    evaluate(result_factory(), configuration(tmp_path))
    assert not obsolete.exists()


def test_all_pairs_follow_configuration_order_and_pair_metrics_run_once(result_factory, log_factory):
    calls = []
    class CountPairs(PairMetric):
        def evaluate(self, pair):
            calls.append((pair.treatment_a, pair.treatment_b))
            return {"value": 1}
    data = result_factory(extra={"rain": log_factory([1, 1], shift=2)})
    settings = configuration(metrics=(CountPairs(), TrajectoryDeviation()), treatments=("rain", "attack", "clean"))
    report = evaluate(data, settings)
    assert calls == [("rain", "attack"), ("rain", "clean"), ("attack", "clean")]
    assert [row["treatment"] for row in report.treatment_summaries] == ["rain", "attack", "clean"]
    assert len(report.groups[0]["comparisons"]) == 3
    assert report.groups[0]["comparisons"][0]["metrics"]["TrajectoryDeviation.final_deviation"] == 2


def test_failed_member_makes_pair_unavailable_with_reason(result_factory):
    report = evaluate(result_factory(clean=RunLog(status="failed")), configuration())
    assert report.groups[0]["treatments"]["clean"]["metrics"] == {}
    pair = report.groups[0]["comparisons"][0]
    assert pair["metrics"] is None and pair["deltas"]["DrivingMetrics.final_speed"] is None
    assert "clean=failed" in pair["reason"]
    assert report.treatment_summaries[0]["groups"] == 0
    assert report.treatment_summaries[0]["total_groups"] == report.treatment_summaries[0]["failed_runs"] == 1


def test_default_set_is_experiment_wide_and_all_failed_reason_is_accurate(result_factory, log_factory):
    first = result_factory()
    defaults = evaluate(first, configuration())
    assert [item["type"].split(":")[-1] for item in defaults.metrics] == [
        "DrivingMetrics", "TrajectoryDeviation", "DetectionMetrics"]
    second = result_factory(clean=log_factory([1], detections=None),
                            attack=log_factory([1], detections=None), seed=1)
    report = evaluate(ExperimentResult(first.runs + second.runs), configuration())
    assert [item["type"].split(":")[-1] for item in report.metrics] == [
        "DrivingMetrics", "TrajectoryDeviation"]
    assert "not all provide detections" in report.notes[0]
    assert report.notes[0] in report.format_summary()
    failed = result_factory(clean=RunLog(status="failed"), attack=RunLog(status="failed"))
    report = evaluate(failed, configuration())
    assert "no completed run" in report.notes[0]
    assert "driving-system" not in report.notes[0]


def test_metric_exceptions_preserve_status_reports_and_later_metrics(tmp_path, result_factory):
    class Broken(Metric):
        def evaluate(self, group):
            raise RuntimeError("metric exploded")
    class Good(Metric):
        def evaluate(self, group):
            return {name: {"seed": np.int64(99), "status": np.float32(2), "bool": np.bool_(True)}
                    for name in group.runs}
    report = evaluate(result_factory(), configuration(tmp_path, metrics=(Broken(), Good())))
    assert report.metric_errors[0]["error"] == "RuntimeError: metric exploded"
    data = json.loads((tmp_path / "evaluation/report.json").read_text())
    assert data["metric_errors"] and data["groups"][0]["treatments"]["clean"]["status"] == "completed"
    assert data["groups"][0]["treatments"]["clean"]["metrics"]["Good.seed"] == 99
    with (tmp_path / "evaluation/runs.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["status"] == "completed" and rows[0]["seed"] == "0"
    assert rows[0]["Good.seed"] == "99" and rows[0]["Good.status"] == "2.0"


def test_unavailable_reasons_boolean_type_and_invalid_values(result_factory):
    class Custom(Metric):
        def evaluate(self, group):
            return {name: {"flag": UnavailableMetricValue("No custom data", "boolean")} for name in group.runs}
    report = evaluate(result_factory(), configuration(metrics=(Custom(),)))
    assert report.groups[0]["treatments"]["clean"]["reasons"]["Custom.flag"] == "No custom data"
    assert report.treatment_summaries[0]["metrics"]["Custom.flag"] == {
        "kind": "proportion", "count": 0, "true_fraction": None}
    class Bad(Metric):
        def evaluate(self, group):
            return {"clean": {"bad": [1, 2]}}
    report = evaluate(result_factory(), configuration(metrics=(Bad(),)))
    assert "finite scalar" in report.metric_errors[0]["error"]


def test_multiple_groups_seeds_population_std_and_boolean_deltas(result_factory, log_factory):
    first = result_factory()
    second = result_factory(clean=log_factory([4, 5]), attack=log_factory([4, 2]), segment_index=1, seed=4)
    report = evaluate(ExperimentResult(first.runs + second.runs), configuration())
    summary = report.treatment_summaries[0]
    assert summary["groups"] == 2
    assert summary["metrics"]["DrivingMetrics.final_speed"]["mean"] == 4
    assert summary["metrics"]["DrivingMetrics.final_speed"]["std"] == 1
    assert report.pair_summaries[0]["deltas"]["DrivingMetrics.final_speed"]["mean"] == -2.75
    assert report.pair_summaries[0]["deltas"]["DrivingMetrics.stopped"]["mean"] == .5


def test_explicit_detection_failure_is_recorded_not_raised(tmp_path, result_factory, log_factory):
    report = evaluate(result_factory(clean=log_factory([1], detections=None)),
                      configuration(tmp_path, metrics=(DetectionMetrics(),)))
    assert "TestDriver" in report.metric_errors[0]["error"]
    assert (tmp_path / "evaluation/report.json").exists()


def test_metric_options_are_saved_and_protocol_is_validated(result_factory):
    report = evaluate(result_factory(), configuration(metrics=(DetectionMetrics(max_range=None),)))
    assert report.metrics[0]["options"] == {"match_distance": 2.0, "max_range": None, "category_map": {}}
    with pytest.raises(TypeError, match="Metric"):
        EvaluationConfiguration(metrics=(object(),))
    with pytest.raises(ValueError, match="repeat"):
        EvaluationConfiguration(metrics=(DetectionMetrics(), DetectionMetrics()))


def test_plotting_never_changes_global_matplotlib_backend(tmp_path, result_factory, monkeypatch):
    import matplotlib
    monkeypatch.setattr(matplotlib, "use", lambda *args, **kwargs: pytest.fail("global backend was changed"))
    report = evaluate(result_factory(), configuration(tmp_path))
    assert report.metric_errors == []


def test_distinct_user_metric_types_with_same_class_name_do_not_overwrite(result_factory):
    first = type("Counter", (Metric,), {"__module__": "first_metrics", "evaluate":
                 lambda self, group: {name: {"value": 1} for name in group.runs}})
    second = type("Counter", (Metric,), {"__module__": "second_metrics", "evaluate":
                  lambda self, group: {name: {"value": 2} for name in group.runs}})
    report = evaluate(result_factory(), configuration(metrics=(first(), second())))
    assert report.metric_errors == []
    assert report.groups[0]["treatments"]["clean"]["metrics"] == {
        "first_metrics:Counter.value": 1, "second_metrics:Counter.value": 2}


def test_unserializable_user_metric_options_still_save_run_statuses(tmp_path, result_factory):
    class Custom(Metric):
        def __init__(self, option):
            self.option = option
        def evaluate(self, group):
            return {name: {"value": 1} for name in group.runs}
    report = evaluate(result_factory(), configuration(tmp_path, metrics=(Custom(object()),)))
    assert report.metric_errors[0]["phase"] == "parameter description"
    data = json.loads((tmp_path / "evaluation/report.json").read_text())
    assert data["groups"][0]["treatments"]["clean"]["status"] == "completed"


def test_explicit_stop_impact_is_directional_pair_data_without_another_delta(result_factory):
    report = evaluate(result_factory(), configuration(metrics=(StopImpact(),)))
    pair = report.groups[0]["comparisons"][0]
    assert pair["metrics"]["StopImpact.a_to_b.induced_stop"] is True
    assert pair["metrics"]["StopImpact.b_to_a.induced_stop"] is False
    assert pair["deltas"] == {}
    assert report.groups[0]["treatments"]["clean"]["metrics"] == {}


def test_terminal_summary_is_compact_and_zero_differences_have_no_sign(result_factory, log_factory):
    report = evaluate(result_factory(clean=log_factory([2, 3]), attack=log_factory([2, 3])), configuration())
    text = report.format_summary()
    assert "final_speed (mean)" in text and "stopped (ratio)" in text
    assert "peak_speed" not in text and "mean_speed" not in text and "min_distance" not in text
    assert "DrivingMetrics.peak_speed" in report.groups[0]["treatments"]["clean"]["metrics"]
    assert "-0" not in text
    assert "notes" not in report.groups[0]
