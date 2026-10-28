"""Saved preparations and recorded results survive the public output-directory boundary."""

from dataclasses import replace
import json

import pytest
import numpy as np

from avsectester.closed_loop.avstacks.driving_adapter import DrivingAdapter
from avsectester.attack import AttackSegment
from avsectester.execution.runner import RecordingPolicy
from avsectester.experiment import (
    ConfigurationMismatchError, Experiment, ExperimentConfiguration, ExperimentResult,
    MetadataConfiguration, PreparedExperiment, RunsConfiguration, TreatmentConfiguration,
    load_configuration, save_metadata,
)
from avsectester.data_types import Unavailable
from avsectester.maps import Lane, ScenarioMap, StopLine, TrafficControl
from tests.fakes import ConstantAVStack, MemoryDataset, MemorySimulator, MinimalAttack


def configuration(*, dataset=None, **options):
    return ExperimentConfiguration(
        dataset or MemoryDataset(), MemorySimulator(), DrivingAdapter(lambda seed: ConstantAVStack()),
        {"probe": MinimalAttack(assets={"value": 3})}, runs=RunsConfiguration(**options),
    )


def measurements(result):
    return [(run.group, run.treatment, run.log.status,
             [(step.step, step.physical_pre.t, step.physical_post.t, step.executed_command)
              for step in run.log.steps]) for run in result.runs]


def test_prepared_experiment_save_and_load(tmp_path):
    config = configuration()
    prepared = Experiment.prepare(config)
    destination = prepared.save(tmp_path / "prepared")
    loaded = PreparedExperiment.load(config, destination)
    saved_preparation = json.loads((destination / "prepared.json").read_text())["preparation"]
    assert loaded.preparation == saved_preparation == prepared.preparation
    assert loaded.artifacts == prepared.artifacts
    assert loaded.metadata == prepared.metadata
    assert [unit.as_dict() for unit in loaded.units] == [unit.as_dict() for unit in prepared.units]
    assert loaded.deployment == prepared.deployment
    assert measurements(Experiment.execute(loaded)) == measurements(Experiment.execute(prepared))
    execution_config = replace(config, runs=replace(
        config.runs, seeds=(0, 5), decision_steps=1, strict=True,
        recording=RecordingPolicy(every_n_steps=2), output=tmp_path / "execution",
        treatments=(TreatmentConfiguration("rain", "probe", {"world": {"weather": "rain"}}),),
    ))
    reused = PreparedExperiment.load(execution_config, destination)
    assert reused.preparation == saved_preparation
    resaved = reused.save(tmp_path / "resaved")
    assert (resaved / "prepared.json").read_bytes() == (destination / "prepared.json").read_bytes()

    specification = {
        "dataset": config.dataset, "simulator": config.simulator, "avstack": config.avstack,
        "attacks": {"probe": {"type": "tests.fakes:MinimalAttack", "options": {
            "decision_steps": 2, "max_segments": 1,
        }}},
    }
    ordered = Experiment.prepare(load_configuration(specification)).save(tmp_path / "ordered")
    specification["attacks"]["probe"]["options"] = {"max_segments": 1, "decision_steps": 2}
    reordered = PreparedExperiment.load(load_configuration(specification), ordered)
    reordered.save(tmp_path / "reordered")
    assert (tmp_path / "reordered/prepared.json").read_bytes() == (ordered / "prepared.json").read_bytes()
    for max_segments in (True, 1.0):
        specification["attacks"]["probe"]["options"]["max_segments"] = max_segments
        with pytest.raises(ConfigurationMismatchError, match="configuration differs"):
            PreparedExperiment.load(load_configuration(specification), ordered)

    invalid = replace(prepared, metadata={"probe": (
        replace(prepared.metadata["probe"][0], user_data={"path": tmp_path}),
    )})
    with pytest.raises(TypeError, match="user_data"):
        invalid.save(tmp_path / "invalid")

    changed = replace(config, attacks={"probe": MinimalAttack(assets={"value": 4})})
    with pytest.raises(ConfigurationMismatchError, match="configuration differs"):
        PreparedExperiment.load(changed, destination)
    different_seed = replace(config, runs=replace(config.runs, seeds=(3,)))
    with pytest.raises(ConfigurationMismatchError, match=r"preparation.seed: old=0, new=3"):
        PreparedExperiment.load(different_seed, destination)
    different_environment = replace(
        config, runs=replace(config.runs, environment={"world": {"weather": "rain"}}),
    )
    with pytest.raises(ConfigurationMismatchError, match=r"offline_environments.probe: old=.*new=.*rain"):
        PreparedExperiment.load(different_environment, destination)

    dataset = MemoryDataset(decision_steps=10)
    frames = dataset.frames("scene")
    source = tmp_path / "late-segment.json"
    save_metadata((AttackSegment("scene", frames[6], frames[8]),), source,
                  dataset_type="tests.fakes:MemoryDataset")
    late = replace(configuration(dataset=dataset), metadata=MetadataConfiguration(load=source))
    saved = Experiment.prepare(late).save(tmp_path / "late")
    with pytest.raises(ValueError, match="Frame is outside"):
        PreparedExperiment.load(replace(late, dataset=MemoryDataset(decision_steps=4)), saved)


def test_experiment_result_load(tmp_path):
    centerline = np.array([[0., 0., 0.], [5., 0., 0.]])
    scenario_map = ScenarioMap(
        lanes={"lane": Lane("lane", centerline, centerline + [0., 1., 0.],
                            centerline - [0., 1., 0.], "driving")},
        traffic_controls={"stop": TrafficControl("stop", "stop_sign", "STOP", np.eye(4),
                                                (.1, 1., 1.), "face_center", ("lane",), "line")},
        stop_lines={"line": StopLine("line", np.array([[2., -1., 0.], [2., 1., 0.]]),
                                     "STOP", "stop_sign", ("lane",), "stop")},
    )
    config = configuration(output=tmp_path / "experiment", dataset=MemoryDataset(
        scenario_ids=("scene", "unmapped"), maps={"scene": scenario_map},
    ))
    prepared = Experiment.prepare(config)
    original = Experiment.execute(prepared)
    loaded = ExperimentResult.load(config.runs.output, config)
    assert measurements(loaded) == measurements(original)
    analysis_config = replace(config, runs=replace(config.runs, environment={"world": {"weather": "rain"}}))
    assert measurements(ExperimentResult.load(config.runs.output, analysis_config)) == measurements(original)
    assert (config.runs.output / "maps/scene.json").is_file()
    assert (config.runs.output / "maps/unmapped.json").is_file()
    np.testing.assert_array_equal(loaded.scenario_maps["scene"].lanes["lane"].centerline, centerline)
    assert loaded.scenario_maps["scene"].traffic_controls["stop"].stop_line_id == "line"
    assert loaded.scenario_maps["scene"].stop_lines["line"].control_id == "stop"
    assert isinstance(loaded.scenario_maps["unmapped"], Unavailable)
    assert loaded.scenario_maps["unmapped"].reason == original.scenario_maps["unmapped"].reason
    clean_only = replace(config, runs=replace(
        config.runs, treatments=(TreatmentConfiguration("clean"),)))
    filtered = ExperimentResult.load(config.runs.output, clean_only)
    assert len(filtered.notices) == 4
    assert all("Skipping saved run for removed treatment" in note for note in filtered.notices)
    assert {run.treatment for run in filtered.runs} == {"clean"}
    failed = original.runs[0]
    assert failed.log.path is not None
    manifest = failed.log.path / "manifest.json"
    manifest.write_text("{incomplete JSON")
    loaded = ExperimentResult.load(config.runs.output, config)
    assert len(loaded.runs) == len(original.runs)
    assert [(run.group, run.treatment) for run in loaded.failures] == [(failed.group, failed.treatment)]
    assert "Run read failed" in loaded.failures[0].error
    assert all(run.log.status == "completed" for run in loaded.runs[1:])

    manifest.unlink()
    incomplete = ExperimentResult.load(config.runs.output, config)
    assert incomplete.failures[0].log.status == "incomplete"
    assert "no saved manifest" in incomplete.failures[0].error
    changed = replace(config, attacks={"probe": MinimalAttack(assets={"value": 4})})
    with pytest.raises(ConfigurationMismatchError, match="configuration differs"):
        ExperimentResult.load(config.runs.output, changed)
    path = config.runs.output / "prepared.json"
    saved = json.loads(path.read_text())
    saved["units"][0]["part"] = "train"
    path.write_text(json.dumps(saved))
    with pytest.raises(ConfigurationMismatchError, match="units or partition"):
        ExperimentResult.load(config.runs.output, config)
