"""Report records format and save scalar measurements without experiment objects."""

import csv
from copy import deepcopy
import json

import numpy as np

from avsectester.evaluation import DrivingMetrics, EvaluationReport, UnavailableMetricValue
from avsectester.evaluation.report import format_number, format_statistic


def test_evaluation_report(tmp_path):
    report = EvaluationReport(
        groups=[{
            "part": None, "treatment_attacks": {"clean": None, "attacked": "attack"}, "segment_index": 0, "scenario_id": "scene", "seed": 0,
            "treatments": {
                "clean": {"status": "completed", "error": None, "metrics": {
                    "DrivingMetrics.final_speed": np.float64(3.),
                    "DrivingMetrics.stopped": np.bool_(False), "UserMetric.seed": np.int64(9),
                }},
                "attacked": {"status": "failed", "error": "sensor disconnected", "metrics": {}},
            },
            "comparisons": [],
        }],
        treatment_summaries=[{
            "part": None, "treatment": "clean", "groups": 1, "total_groups": 1,
            "failed_runs": 0, "metrics": {
                "DrivingMetrics.final_speed": {"kind": "number", "mean": 3., "std": 0., "count": 1},
                "DrivingMetrics.stopped": {"kind": "proportion", "true_fraction": 0., "count": 1},
                "UserMetric.seed": {"kind": "number", "mean": 9., "std": 0., "count": 1},
            },
        }, {
            "part": None, "treatment": "attacked", "groups": 0, "total_groups": 1,
            "failed_runs": 1, "metrics": {},
        }],
        pair_summaries=[{
            "part": None, "treatment_a": "clean", "treatment_b": "attacked",
            "deltas": {"DrivingMetrics.final_speed": {"mean": None}},
        }],
        metrics=[
            {"label": "DrivingMetrics", "fields": {
                name: {"description": field.description, "unit": field.unit}
                for name, field in DrivingMetrics().describe().fields.items()}},
            {"label": "UserMetric", "fields": {"seed": {"unit": None}}},
        ],
        notes=["Only recorded measurements are available."],
    )
    text = report.format_summary()
    assert "DrivingMetrics" in text and "final_speed" in text and "stopped" in text
    assert "final_speed (m/s)" in text and "3.0000" in text and "0 of 1 groups" in text and "—" in text
    assert "UserMetric" in text and "seed" in text and "9.0000" in text and "n = 1" in text
    assert report.notes[0] in text
    assert report.save(tmp_path) == tmp_path and report.path == tmp_path
    data = json.loads((tmp_path / "report.json").read_text())
    assert data == report.as_dict() and data["statistics"]["std_ddof"] == 0
    measurements = data["groups"][0]["treatments"]["clean"]["metrics"]
    assert measurements["UserMetric.seed"] == 9 and measurements["DrivingMetrics.stopped"] is False
    with (tmp_path / "runs.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2 and rows[-1]["status"] == "failed"
    assert rows[0]["UserMetric.seed"] == "9" and rows[0]["seed"] == "0"
    assert list(rows[0])[:8] == [
        "part", "segment_index", "scenario_id", "seed", "treatment", "attack_name", "status", "error",
    ]
    assert rows[0]["attack_name"] == "" and rows[1]["attack_name"] == "attack"


def test_summarize():
    report = EvaluationReport(groups=[
        {"part": None, "treatments": {
            "clean": {"status": "completed", "metrics": {"speed": 3., "stopped": False}},
            "attacked": {"status": "completed", "metrics": {"speed": 1., "stopped": True}},
        }, "comparisons": [{
            "treatment_a": "clean", "treatment_b": "attacked", "status": "completed",
            "deltas": {"speed": -2., "stopped": 1.},
            "metrics": {"separation": 4., "missing_flag": None},
            "metric_types": {"missing_flag": "boolean"},
        }]},
        {"part": None, "treatments": {
            "clean": {"status": "completed", "metrics": {"speed": 5., "stopped": False}},
            "attacked": {"status": "completed", "metrics": {"speed": 3., "stopped": False}},
        }, "comparisons": [{
            "treatment_a": "clean", "treatment_b": "attacked", "status": "completed",
            "deltas": {"speed": -2., "stopped": 0.},
            "metrics": {"separation": 2., "missing_flag": None},
            "metric_types": {"missing_flag": "boolean"},
        }]},
        {"part": None, "treatments": {
            "clean": {"status": "completed", "metrics": {"speed": 4., "stopped": False}},
            "attacked": {"status": "failed", "metrics": {}},
        }, "comparisons": [{
            "treatment_a": "clean", "treatment_b": "attacked", "status": "unavailable",
            "deltas": {"speed": None, "stopped": None}, "metrics": None,
        }]},
    ])
    report.summarize()
    assert [row["treatment"] for row in report.treatment_summaries] == ["clean", "attacked"]
    clean, attacked = report.treatment_summaries
    assert clean["metrics"]["speed"]["mean"] == 4.
    assert np.isclose(clean["metrics"]["speed"]["std"], np.sqrt(2 / 3))
    assert clean["metrics"]["speed"]["std_ddof"] == 0
    assert (attacked["groups"], attacked["total_groups"], attacked["failed_runs"]) == (2, 3, 1)
    assert attacked["metrics"]["stopped"] == {"kind": "proportion", "count": 2, "true_fraction": .5}
    pair, = report.pair_summaries
    assert (pair["groups"], pair["total_groups"]) == (2, 3)
    assert pair["deltas"]["stopped"]["kind"] == "number"
    assert pair["deltas"]["stopped"]["mean"] == .5
    assert pair["metrics"]["separation"]["mean"] == 3.
    assert pair["metrics"]["missing_flag"] == {"kind": "proportion", "count": 0, "true_fraction": None}

    for group in report.groups:
        group["part"] = "train"
    validation = deepcopy(report.groups[0])
    validation["part"] = "validation"
    validation["treatments"]["clean"]["metrics"]["speed"] = 100.
    report.groups.append(validation)
    report.summarize()
    summaries = {(row["part"], row["treatment"]): row for row in report.treatment_summaries}
    assert summaries["train", "clean"]["metrics"]["speed"]["mean"] == 4.
    assert summaries["validation", "clean"]["metrics"]["speed"]["mean"] == 100.
    assert [row["part"] for row in report.pair_summaries] == ["train", "validation"]
    assert "Part: train" in report.format_summary() and "Part: validation" in report.format_summary()


def test_record_error():
    report = EvaluationReport()
    report.record_error("UserMetric", RuntimeError("missing evidence"), seed=4, treatment_a="clean")
    assert report.metric_errors == [{
        "metric": "UserMetric", "error": "RuntimeError: missing evidence",
        "seed": 4, "treatment_a": "clean",
    }]


def test_format_number():
    assert format_number(7) == "7"
    assert format_number(1.0) == "1.0000" and format_number(3.0) == "3.0000"
    assert format_number(-0.0) == "0.0000"
    assert format_number(float("inf")) == "—" and format_number(float("nan")) == "—"
    assert format_number(1.23456789) == "1.2346"
    assert format_number(-.000001) == "0.0000"
    assert format_number(True) == "yes" and format_number(False) == "no"
    assert format_number(None) == "—"
    assert format_number(UnavailableMetricValue("Missing evidence")) == "—"
    assert format_statistic({"mean": 1.23456789, "std": .1, "count": 3}) == "1.2346 ± 0.1000\nn = 3"
    assert format_statistic({"mean": 3., "std": 0., "count": 1}) == "3.0000\nn = 1"
    assert format_statistic({"kind": "proportion", "true_fraction": 2 / 3, "count": 3}) == "0.6667\n2 of 3 groups"
