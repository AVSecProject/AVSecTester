"""Offline reports group public run results and retain failures."""

import csv
from dataclasses import replace
import json

import numpy as np

from avsectester.closed_loop.services import DrivingAdapter
from avsectester.evaluation import DrivingMetrics, EvaluationConfiguration, Metric, TrajectoryDeviation, evaluate
from avsectester.evaluation.report import plot_group
from avsectester.experiment import ExperimentConfiguration, ExperimentResult, RunsConfiguration, TreatmentConfiguration
from avsectester.experiment.models import TreatmentResult
from avsectester.runlog import RunLog
from avsectester.maps import ScenarioMap
from tests.fakes import ConstantAVStack, MemoryDataset, MemorySimulator, MinimalAttack


class UserMetric(Metric):
    def evaluate(self, group):
        return {name: {"seed": np.int64(group.seed),
                       "map_boundaries": len(group.scenario_map.road_boundaries)} for name in group.runs}


class BrokenMetric(Metric):
    def evaluate(self, group):
        raise RuntimeError("unavailable scoring service")


def test_evaluate(tmp_path, segment, make_log):
    configuration = ExperimentConfiguration(
        MemoryDataset(), MemorySimulator(), DrivingAdapter(lambda seed: ConstantAVStack()),
        attacks={"attack": MinimalAttack()},
        evaluation=EvaluationConfiguration((DrivingMetrics(), TrajectoryDeviation(), UserMetric(), BrokenMetric())),
        runs=RunsConfiguration(
            treatments=tuple(TreatmentConfiguration(name, None if name == "clean" else "attack")
                             for name in ("clean", "attacked", "rain")),
            output=tmp_path,
        ),
    )
    runs = []
    for index in (0, 1):
        for seed in (0, 1):
            for name, speed in (("clean", 3.), ("attacked", 1.), ("rain", 2.)):
                failed = index == seed == 1 and name == "rain"
                log = (RunLog(status="failed") if failed
                       else make_log([speed + seed, speed + seed], shift=seed + index))
                runs.append(TreatmentResult("attack", index, segment, name, seed, log,
                                            "sensor disconnected" if failed else None))
    result = ExperimentResult(tuple(runs), scenario_maps={"scene": ScenarioMap(
        road_boundaries=(np.array([[0., 1., 0.], [5., 1., 0.]]),),
    )})
    report = evaluate(result, configuration)
    assert len(report.groups) == 4
    assert [row["treatment"] for row in report.treatment_summaries] == ["clean", "attacked", "rain"]
    assert len(report.groups[0]["comparisons"]) == 3
    pair = report.groups[0]["comparisons"][0]
    assert (pair["treatment_a"], pair["treatment_b"]) == ("clean", "attacked")
    assert pair["deltas"]["DrivingMetrics.final_speed"] == -2.
    assert "TrajectoryDeviation.final_deviation" in pair["metrics"]
    last_group = report.groups[-1]
    assert last_group["treatments"]["rain"]["status"] == "failed"
    assert last_group["comparisons"][1]["metrics"] is None
    assert "rain=failed" in last_group["comparisons"][1]["reason"]
    summary = report.treatment_summaries[-1]
    assert (summary["groups"], summary["total_groups"], summary["failed_runs"]) == (3, 4, 1)
    assert np.isclose(summary["metrics"]["DrivingMetrics.final_speed"]["mean"], 7 / 3)
    assert np.isclose(summary["metrics"]["DrivingMetrics.final_speed"]["std"], np.sqrt(2) / 3)
    assert report.metric_errors[0]["error"] == "RuntimeError: unavailable scoring service"
    text = report.format_summary()
    assert "B - A" in text and "(mean)" in text and "(ratio)" in text
    data = json.loads((tmp_path / "evaluation/report.json").read_text())
    assert data == report.as_dict() and data["statistics"]["std_ddof"] == 0
    assert data["groups"][0]["treatments"]["clean"]["metrics"]["UserMetric.seed"] == 0
    assert data["groups"][0]["treatments"]["clean"]["metrics"]["UserMetric.map_boundaries"] == 1
    with (tmp_path / "evaluation/runs.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 12 and rows[-1]["status"] == "failed"
    assert "UserMetric.seed" in rows[0] and rows[0]["seed"] == "0"
    assert len(list((tmp_path / "evaluation/plots").glob("*.png"))) == 4
    defaults = evaluate(result, replace(
        configuration, runs=replace(configuration.runs, output=None), evaluation=EvaluationConfiguration()))
    assert {metric["type"].split(":")[-1] for metric in defaults.metrics} >= {"DrivingMetrics", "TrajectoryDeviation"}


def test_plot_group(tmp_path, make_log):
    path = plot_group({"clean": make_log([3., 2.]),
                       "attacked": make_log([2., 0.], [0., 1.])}, tmp_path / "plot.png",
                      title="Small recorded experiment")
    assert path.read_bytes().startswith(b"\x89PNG")
    empty = plot_group({}, tmp_path / "empty.png")
    assert empty.is_file()
