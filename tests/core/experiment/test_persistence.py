"""Saved preparations and recorded results survive the public persistence boundary."""

from dataclasses import replace

import pytest

from avsectester.closed_loop.services import DrivingAdapter
from avsectester.experiment import (
    ConfigurationMismatchError, Experiment, ExperimentConfiguration, ExperimentResult,
    PreparedExperiment,
)
from avsectester.experiment.models import run_directory
from tests.fakes import ConstantAVStack, MemoryDataset, MemorySimulator, MinimalAttack


def configuration(**options):
    return ExperimentConfiguration(
        MemoryDataset(), MemorySimulator(), DrivingAdapter(lambda seed: ConstantAVStack()),
        {"probe": MinimalAttack(assets={"value": 3})}, **options,
    )


def measurements(result):
    return [(run.group, run.treatment, run.log.status,
             [(step.step, step.physical_pre.t, step.physical_post.t, step.executed_command)
              for step in run.log.steps]) for run in result.runs]


def test_save_and_load(tmp_path):
    config = configuration()
    prepared = Experiment.prepare(config)
    destination = prepared.save(tmp_path / "prepared")
    loaded = PreparedExperiment.load(config, destination)
    assert loaded.artifacts == prepared.artifacts
    assert loaded.metadata == prepared.metadata
    assert measurements(Experiment.execute(loaded)) == measurements(Experiment.execute(prepared))


def test_load_rejects_changed_configuration(tmp_path):
    config = configuration()
    Experiment.prepare(config).save(tmp_path)
    changed = replace(config, attacks={"probe": MinimalAttack(assets={"value": 4})})
    with pytest.raises(ConfigurationMismatchError, match="configuration differs"):
        PreparedExperiment.load(changed, tmp_path)


def test_experiment_result_load(tmp_path):
    config = configuration(output=tmp_path / "experiment")
    prepared = Experiment.prepare(config)
    original = Experiment.execute(prepared)
    loaded = ExperimentResult.load(config.output, config)
    assert measurements(loaded) == measurements(original)
    failed = original.runs[0]
    manifest = run_directory(config.output, failed.attack_name, failed.segment_index,
                             failed.seed, failed.treatment) / "manifest.json"
    manifest.write_text("{incomplete JSON")
    loaded = ExperimentResult.load(config.output, config)
    assert len(loaded.runs) == len(original.runs)
    assert [(run.group, run.treatment) for run in loaded.failures] == [(failed.group, failed.treatment)]
    assert "Run read failed" in loaded.failures[0].error
    assert all(run.log.status == "completed" for run in loaded.runs[1:])
