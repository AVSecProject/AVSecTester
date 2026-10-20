"""Opt-in execution of the shipped CARLA experiment through its public workflow."""
from dataclasses import replace
from pathlib import Path

import pytest


@pytest.mark.carla
def test_carla_configuration_prepares_executes_and_evaluates(request, tmp_path):
    from avsectester.experiment import Experiment, ExperimentResult, load_configuration
    from avsectester.evaluation import evaluate

    path = Path(request.config.getoption('--carla-config'))
    config = replace(load_configuration(path), output=tmp_path / 'experiment', decision_steps=3)
    prepared = Experiment.prepare(config)
    assert not [item for item in prepared.selections if item.status == 'error']
    assert any(prepared.metadata.values()), 'The real scenario must select at least one segment'
    result = Experiment.execute(prepared)
    assert result.runs and not result.failures
    for run in result.runs:
        assert len(run.log.steps) == 3
        assert run.log.steps[0].physical_pre.ego is not None
    report = evaluate(result, config)
    restored = ExperimentResult.load(config.output, config)
    assert len(restored.runs) == len(result.runs)
    assert evaluate(restored, config).as_dict() == report.as_dict()
