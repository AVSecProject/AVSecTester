"""Preparation models define the canonical inputs used by selection and offline work."""

from avsectester.closed_loop.avstacks.driving_adapter import DrivingAdapter
from avsectester.experiment import Experiment, ExperimentConfiguration, ExperimentResult, RunsConfiguration, SelectionDecision
from tests.fakes import ConstantAVStack, MemoryDataset, MemorySimulator, MinimalAttack


def test_preparation_record():
    color = (255, 200, 100)
    attack = MinimalAttack()
    attack.environment = {"world": {"lighting": {"intensity": .5}}}
    configuration = ExperimentConfiguration(
        MemoryDataset(), MemorySimulator(), DrivingAdapter(lambda seed: ConstantAVStack()),
        {"probe": attack}, runs=RunsConfiguration(seeds=(3, 7), environment={"world": {
            "weather": "rain", "lighting": {"street_lights": {"color": color}},
        }}),
    )
    assert Experiment.prepare(configuration).preparation == {
        "seed": 3,
        "timing": {"decision_period_us": None, "sensors": {}, "pose_reporting_interval_us": 0,
                   "planner_delay_us": 0, "force_gt_duration_us": 0,
                   "skip_driver_during_force_gt": False},
        "history": {"duration_us": 0, "samples": 1},
        "split": None,
        "offline_environments": {"probe": {
            "world": {"weather": "rain", "lighting": {
                "street_lights": {"color": list(color)}, "intensity": .5,
            }}, "sensors": [],
        }},
    }
    actual_color = configuration.runs.environment["world"]["lighting"]["street_lights"]["color"]
    assert actual_color == color and isinstance(actual_color, tuple)


def test_selection_errors():
    selected = SelectionDecision("probe", "first", "selected")
    rejected = SelectionDecision("probe", "second", "not_selected", ("No candidate",))
    failed = SelectionDecision("probe", "third", "error", error="Map unavailable")
    result = ExperimentResult((), selections=(selected, rejected, failed))
    assert result.selection_errors == (failed,)
