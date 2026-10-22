"""Preparation and execution are tested as complete public workflows."""

from dataclasses import replace
import json

import pytest

from avsectester.closed_loop.services import DrivingAdapter
from avsectester.experiment import Experiment, ExperimentConfiguration, RunsConfiguration, TreatmentConfiguration
from avsectester.maps import ScenarioMap
from tests.fakes import ConstantAVStack, MemoryDataset, MemorySimulator, MinimalAttack


def configuration(attack, *, dataset=None, treatments=(), **options):
    return ExperimentConfiguration(
        dataset or MemoryDataset(), MemorySimulator(),
        DrivingAdapter(lambda seed: ConstantAVStack()), {"probe": attack},
        runs=RunsConfiguration(treatments=treatments, **options),
    )


def test_prepare():
    class PreparationAttack(MinimalAttack):
        def run_offline(self, context):
            return {"asset": self.assets, "origins": [
                (segment.scenario_id, segment.start.position, segment.stop.position)
                for segment in context.metadata
            ]}

    config = configuration(
        PreparationAttack(assets="source", fail_scenarios=("broken",)),
        dataset=MemoryDataset(scenario_ids=("first", "broken", "last")),
    )
    prepared = Experiment.prepare(config)
    origins = [(segment.scenario_id, segment.start.position, segment.stop.position)
               for segment in prepared.metadata["probe"]]
    assert origins == [("first", 0, 2), ("first", 2, 4), ("last", 0, 2), ("last", 2, 4)]
    assert prepared.artifacts["probe"] == {"asset": "source", "origins": origins}
    assert [(decision.scenario_id, decision.status) for decision in prepared.selections] == [
        ("first", "selected"), ("broken", "error"), ("last", "selected"),
    ]
    assert "RuntimeError: synthetic selection failed" == prepared.selections[1].error
    with pytest.raises(RuntimeError, match="selection failed"):
        Experiment.prepare(replace(config, runs=replace(config.runs, strict=True)))


def test_execute(tmp_path):
    marker = tmp_path / "stopped.jsonl"
    config = configuration(
        MinimalAttack(stop_marker_path=marker), dataset=MemoryDataset(maps={"scene": ScenarioMap()}),
        seeds=(2, 7), treatments=(
            TreatmentConfiguration("clean"), TreatmentConfiguration("attacked", "probe"),
            TreatmentConfiguration("failed", environment={"world": {"fail": True}}),
        ),
    )
    prepared = Experiment.prepare(config)
    result = Experiment.execute(prepared)
    assert isinstance(result.scenario_maps["scene"], ScenarioMap)
    expected = {(index, seed, treatment) for index in (0, 1) for seed in (2, 7)
                for treatment in ("clean", "attacked", "failed")}
    assert {(run.segment_index, run.seed, run.treatment) for run in result.runs} == expected
    assert {run.treatment for run in result.failures} == {"failed"}
    for run in result.runs:
        if run.treatment == "failed":
            assert run.log.status == "failed" and "RuntimeError" in run.error
            continue
        assert run.log.status == "completed"
        assert run.log.steps[0].physical_pre.frame == run.segment.start.position
        if run.treatment == "attacked":
            progress = [output.value["decision"] for step in run.log.steps
                        for output in step.outputs if output.name == "progress"]
            assert progress == [0, 1]
    assert marker.read_text().strip()
    with pytest.raises(RuntimeError, match="memory simulation failed"):
        Experiment.execute(Experiment.prepare(replace(config, runs=replace(config.runs, strict=True))))

    # A later attack instance is constructed before the first treatment starts. Interruption
    # must propagate and release that pending participant even though its setup never ran.
    interrupted_marker = tmp_path / "pending-stopped.jsonl"
    interrupted = configuration(MinimalAttack(stop_marker_path=interrupted_marker), treatments=(
        TreatmentConfiguration("interrupted", environment={"world": {"interrupt": True}}),
        TreatmentConfiguration("pending", "probe"),
    ))
    with pytest.raises(KeyboardInterrupt):
        Experiment.execute(Experiment.prepare(interrupted))
    stopped = [json.loads(line) for line in interrupted_marker.read_text().splitlines()]
    assert {"scenario": "scene", "start": 0} in stopped
