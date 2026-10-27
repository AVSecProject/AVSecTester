"""Experiments organize saved runs and select evaluation defaults without dataset reads."""

from dataclasses import replace

import numpy as np

from avsectester.closed_loop.avstacks.driving_adapter import DrivingAdapter
from avsectester.evaluation import DrivingMetrics, EvaluationConfiguration, Metric, TrajectoryDeviation
from avsectester.experiment import (
    ExperimentConfiguration, ExperimentResult, RunsConfiguration, SelectionDecision, TreatmentConfiguration,
    TreatmentResult, evaluate,
)
from avsectester.maps import ScenarioMap
from avsectester.runlog import RunLog
from tests.fakes import ConstantAVStack, MemoryDataset, MemorySimulator, MinimalAttack


class MapMetric(Metric):
    def evaluate(self, group):
        return {name: {"boundaries": len(group.scenario_map.road_boundaries)} for name in group.runs}


def test_evaluate(tmp_path, segment, make_log):
    configuration = ExperimentConfiguration(
        MemoryDataset(), MemorySimulator(), DrivingAdapter(lambda seed: ConstantAVStack()),
        attacks={"attack": MinimalAttack(), "other": MinimalAttack()},
        evaluation=EvaluationConfiguration((DrivingMetrics(), TrajectoryDeviation(), MapMetric())),
        runs=RunsConfiguration(
            treatments=(TreatmentConfiguration("clean"), TreatmentConfiguration("attacked", "attack"),
                        TreatmentConfiguration("rain", "other")), output=tmp_path,
        ),
    )
    runs = []
    for index in (0, 1):
        for seed in (0, 1):
            for name in ("rain", "attacked", "clean"):
                log = make_log([2., 2.], detections=None)
                runs.append(TreatmentResult(
                    {"clean": None, "attacked": "attack", "rain": "other"}[name],
                    index, segment, name, seed, log, part="train" if index == 0 else "test"))
    runs[-3] = replace(runs[-3], log=RunLog(status="failed"), error="sensor disconnected")
    selection = SelectionDecision("attack", "scene", "selected")
    result = ExperimentResult(tuple(runs), selections=(selection,), scenario_maps={"scene": ScenarioMap(
        road_boundaries=(np.array([[0., 1., 0.], [5., 1., 0.]]),),
    )})
    report = evaluate(result, configuration)
    assert [(group["segment_index"], group["seed"]) for group in report.groups] == [
        (0, 0), (0, 1), (1, 0), (1, 1),
    ]
    assert all(list(group["treatments"]) == ["clean", "attacked", "rain"] for group in report.groups)
    assert report.groups[-1]["treatments"]["rain"]["error"] == "sensor disconnected"
    assert report.groups[0]["treatments"]["clean"]["metrics"]["MapMetric.boundaries"] == 1
    assert report.selections == [selection.as_dict()]
    assert report.groups[0]["treatment_attacks"] == {"clean": None, "attacked": "attack", "rain": "other"}
    assert [group["part"] for group in report.groups] == ["train", "train", "test", "test"]
    assert (tmp_path / "evaluation/report.json").is_file()
    assert (tmp_path / "evaluation_report.html").is_file()
    assert report.experiment["dataset_type"] == "tests.fakes:MemoryDataset"
    assert report.experiment["scenarios_considered"] == 1
    assert report.experiment["decision_steps"] is None
    assert report.experiment["comparison_groups"] == 4
    assert report.experiment["treatments"][0] == {
        "treatment": "clean", "attack": None, "attack_type": None,
    }
    assert (tmp_path / "evaluation/runs.csv").is_file()
    assert {p.name for p in (tmp_path / "evaluation/plots").glob("*.png")} == {
        f"segment-{index:05d}-seed-{seed}.png" for index in (0, 1) for seed in (0, 1)
    }

    default_configuration = replace(
        configuration, runs=replace(configuration.runs, output=None), evaluation=EvaluationConfiguration())
    defaults = evaluate(result, default_configuration)
    assert [metric["type"].split(":")[-1] for metric in defaults.metrics] == [
        "DrivingMetrics", "TrajectoryDeviation",
    ]
    detected = replace(result, runs=tuple(replace(run, log=replace(
        run.log, steps=[replace(step, detections=[]) for step in run.log.steps])) for run in result.runs))
    assert [metric["type"].split(":")[-1] for metric in evaluate(detected, default_configuration).metrics] == [
        "DrivingMetrics", "TrajectoryDeviation", "DetectionMetrics",
    ]
