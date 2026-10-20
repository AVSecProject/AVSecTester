"""Portable attacks complete public preparation, execution and offline evaluation."""

from avsectester.attack.implemented_attacks.speed_feedback_attack.attack import SpeedFeedbackAttack
from avsectester.closed_loop.services import DrivingAdapter
from avsectester.evaluation import evaluate
from avsectester.experiment import Experiment, ExperimentConfiguration, TreatmentConfiguration

def test_attack_prepares_executes_and_evaluates(memory_experiment):
    Data = memory_experiment.Data
    Driver = memory_experiment.Driver
    Sim = memory_experiment.Sim
    attack = SpeedFeedbackAttack(decision_steps=2, max_segments=1, target_speed=5., gain=.5)
    config = ExperimentConfiguration(Data(), Sim(), DrivingAdapter(lambda seed: Driver()),
        attacks={"test": attack}, treatments=(TreatmentConfiguration("clean"),
        TreatmentConfiguration("attack", "test")), strict=True)
    prepared = Experiment.prepare(config)
    assert len(prepared.metadata["test"]) == 1
    result = Experiment.execute(prepared)
    assert not result.failures
    assert len(result.runs) == 2
    assert all(len(run.log.steps) == 2 for run in result.runs)
    report = evaluate(result, config)
    assert len(report.groups) == 1
    clean, attacked = [run.log for run in result.runs]
    assert attacked.steps[1].model_observation.ego_speed > clean.steps[1].model_observation.ego_speed

def test_speed_feedback_state_is_fresh_for_every_runtime():
    attack = SpeedFeedbackAttack(decision_steps=2, target_speed=5., gain=.5)
    first, second = attack.create_runtime(None, None), attack.create_runtime(None, None)
    first.bias = 3.
    assert second.bias == 0.
