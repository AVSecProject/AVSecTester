"""CLI workflow, saved preparation and runtime/configuration exit status."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from typer.testing import CliRunner

from avsectester import __version__, cli


@pytest.fixture
def workflow(tmp_path, monkeypatch):
    path = tmp_path / "experiment.yaml"
    path.write_text("dataset: fake\n")
    output = tmp_path / "result"
    configuration = SimpleNamespace(output=output)
    def save_output(directory):
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "prepared.json").write_text("{}")
    selected = SimpleNamespace(save=Mock(side_effect=save_output))
    result = SimpleNamespace(failures=(), selection_errors=(), notices=())
    load = Mock(return_value=configuration)
    validate = Mock()
    def prepare_output(configuration):
        save_output(configuration.output)
        return selected
    prepare = Mock(side_effect=prepare_output)
    execute = Mock(return_value=result)
    report = Mock(return_value=SimpleNamespace(format_summary=lambda: "Report", metric_errors=[]))
    monkeypatch.setattr(cli, "load_configuration", load)
    monkeypatch.setattr(cli, "validate_configuration_environment", validate)
    monkeypatch.setattr(cli.Experiment, "prepare", prepare)
    monkeypatch.setattr(cli.Experiment, "execute", execute)
    monkeypatch.setattr(cli, "_evaluate_result", report)
    return SimpleNamespace(**locals())


def test_run_calls_preparation_execution_and_evaluation(workflow):
    w = workflow
    result = CliRunner().invoke(cli.app, ["run", str(w.path)])
    assert result.exit_code == 0, result.output
    w.load.assert_called_once_with(w.path)
    w.prepare.assert_called_once_with(w.configuration)
    w.selected.save.assert_not_called()
    w.execute.assert_called_once_with(w.selected)
    w.report.assert_called_once_with(w.result, w.configuration)


def test_run_loads_saved_preparation_instead_of_preparing(workflow, monkeypatch):
    w = workflow
    saved = w.tmp_path / "previous"
    saved.mkdir()
    load = Mock(return_value=w.selected)
    monkeypatch.setattr(cli.PreparedExperiment, "load", load)
    result = CliRunner().invoke(cli.app, ["run", str(w.path), "--prepared", str(saved)])
    assert result.exit_code == 0, result.output
    load.assert_called_once_with(w.configuration, saved)
    w.prepare.assert_not_called()
    w.selected.save.assert_called_once_with(w.output)
    w.execute.assert_called_once_with(w.selected)


def test_run_refuses_existing_runs_without_preparing_or_loading(workflow, monkeypatch):
    w = workflow
    (w.output / "runs").mkdir(parents=True)
    marker = w.output / "runs" / "keep"
    marker.write_text("existing")
    load = Mock()
    monkeypatch.setattr(cli.PreparedExperiment, "load", load)
    result = CliRunner().invoke(cli.app, ["run", str(w.path)])
    assert result.exit_code == 2 and "already exists" in result.output
    w.prepare.assert_not_called()
    w.execute.assert_not_called()
    load.assert_not_called()
    assert marker.read_text() == "existing"


@pytest.mark.parametrize("field", ["failures", "selection_errors"])
def test_completed_work_still_gets_a_report_when_other_runs_or_selection_failed(workflow, field):
    w = workflow
    failure = SimpleNamespace(attack_name="example", segment_index=0, seed=0,
                              treatment="attacked", error="RuntimeError: failed", scenario_id="scene",
                              log=SimpleNamespace(status="failed"))
    setattr(w.result, field, (failure,))
    result = CliRunner().invoke(cli.app, ["run", str(w.path)])
    assert result.exit_code == 1
    w.report.assert_called_once_with(w.result, w.configuration)


def test_execution_failure_evaluates_saved_partial_results(workflow, monkeypatch):
    w = workflow
    w.output.mkdir()
    (w.output / "prepared.json").write_text("{}")
    load = Mock(return_value=w.result)
    monkeypatch.setattr(cli.ExperimentResult, "load", load)
    w.execute.side_effect = RuntimeError("driver failed")
    result = CliRunner().invoke(cli.app, ["run", str(w.path)])
    assert result.exit_code == 1 and "driver failed" in result.output
    load.assert_called_once_with(w.output, w.configuration)
    w.report.assert_called_once_with(w.result, w.configuration)


def test_evaluate_reads_saved_runs_without_preparing_or_executing(workflow, monkeypatch):
    w = workflow
    w.output.mkdir()
    (w.output / "prepared.json").write_text("{}")
    load = Mock(return_value=w.result)
    monkeypatch.setattr(cli.ExperimentResult, "load", load)
    result = CliRunner().invoke(cli.app, ["evaluate", str(w.path)])
    assert result.exit_code == 0, result.output
    load.assert_called_once_with(w.output, w.configuration)
    w.prepare.assert_not_called()
    w.execute.assert_not_called()
    w.selected.save.assert_not_called()
    w.report.assert_called_once_with(w.result, w.configuration)


@pytest.mark.parametrize("command", ["run", "evaluate"])
def test_bad_configuration_is_exit_two(workflow, command):
    w = workflow
    w.load.side_effect = ValueError("unknown constructor")
    result = CliRunner().invoke(cli.app, [command, str(w.path)])
    assert result.exit_code == 2 and "unknown constructor" in result.output
    w.prepare.assert_not_called()
    w.execute.assert_not_called()


def test_prepared_mismatch_is_configuration_error(workflow, monkeypatch):
    w = workflow
    saved = w.tmp_path / "previous"
    saved.mkdir()
    monkeypatch.setattr(cli.PreparedExperiment, "load", Mock(side_effect=cli.ConfigurationMismatchError("incompatible")))
    result = CliRunner().invoke(cli.app, ["run", str(w.path), "--prepared", str(saved)])
    assert result.exit_code == 2 and "incompatible" in result.output
    w.execute.assert_not_called()


def test_missing_output_is_configuration_error(workflow):
    w = workflow
    w.configuration.output = None
    result = CliRunner().invoke(cli.app, ["run", str(w.path)])
    assert result.exit_code == 2 and "output" in result.output
    w.prepare.assert_not_called()


@pytest.mark.parametrize("flag", ["--frames", "--gpu", "--mode", "--treatment", "--plot"])
def test_old_flags_are_rejected(workflow, flag):
    result = CliRunner().invoke(cli.app, ["run", str(workflow.path), flag, "1"])
    assert result.exit_code == 2
    workflow.load.assert_not_called()


def test_metric_value_error_is_execution_error_not_configuration_error(workflow):
    w = workflow
    w.report.side_effect = ValueError("DetectionMetrics requires detections")
    result = CliRunner().invoke(cli.app, ["run", str(w.path)])
    assert result.exit_code == 1 and "ValueError" in result.output and "requires detections" in result.output
    w.execute.assert_called_once()


def test_version_is_unchanged():
    result = CliRunner().invoke(cli.app, ["version"])
    assert result.exit_code == 0
    assert result.output.strip() == f"avsectester {__version__}"


def test_failed_prepare_does_not_read_stale_preparation(workflow, monkeypatch):
    w = workflow
    w.output.mkdir()
    (w.output / "prepared.json").write_text("old preparation")
    w.prepare.side_effect = RuntimeError("prepare failed before saving")
    load = Mock()
    monkeypatch.setattr(cli.ExperimentResult, "load", load)
    result = CliRunner().invoke(cli.app, ["run", str(w.path)])
    assert result.exit_code == 1 and "RuntimeError: prepare failed" in result.output
    load.assert_not_called()
    w.report.assert_not_called()


def test_keyboard_interrupt_reports_available_results_and_returns_130(workflow, monkeypatch):
    w = workflow
    load = Mock(return_value=w.result)
    monkeypatch.setattr(cli.ExperimentResult, "load", load)
    w.execute.side_effect = KeyboardInterrupt()
    result = CliRunner().invoke(cli.app, ["run", str(w.path)])
    assert result.exit_code == 130 and "interrupted" in result.output
    load.assert_called_once_with(w.output, w.configuration)
    w.report.assert_called_once_with(w.result, w.configuration)


def test_evaluate_missing_saved_results_has_clear_message(workflow):
    result = CliRunner().invoke(cli.app, ["evaluate", str(workflow.path)])
    assert result.exit_code == 1 and "No saved experiment results found" in result.output


def test_metric_error_is_exit_one_after_writing_report(workflow):
    w = workflow
    w.report.return_value.metric_errors = [{"metric": "UserMetric", "error": "ValueError: invalid"}]
    result = CliRunner().invoke(cli.app, ["run", str(w.path)])
    assert result.exit_code == 1 and "Output directory:" in result.output
    w.report.assert_called_once_with(w.result, w.configuration)


@pytest.fixture
def memory_cli(tmp_path, monkeypatch):
    """Exercise the real public workflow; no mocks replace execution, recording or evaluation."""
    import sys
    from types import ModuleType
    import numpy as np
    import yaml
    from avsectester.attack import Attack, AttackSegment
    from avsectester.closed_loop.avstack import AVStack
    from avsectester.closed_loop.datatypes import Control, Observation, WorldSnapshot
    from avsectester.closed_loop.simulator import Simulator, SimulatorRuntime
    from avsectester.datasets import (
        ActorState, Dataset, FrameData, FrameReference, ScenarioDescriptor, ScenarioStart,
    )
    from avsectester.evaluation.metric import Metric

    module = ModuleType("avsec_cli_memory")
    module.allow_data = True
    module.fail_run = False

    class MemoryDataset(Dataset):
        def scenarios(self):
            if not module.allow_data:
                raise AssertionError("evaluate accessed dataset scenarios")
            return iter((ScenarioDescriptor("scene", "memory"),))

        def frames(self, scenario_id):
            if not module.allow_data:
                raise AssertionError("evaluate accessed dataset frames")
            return tuple(FrameReference(scenario_id, i, str(i), i * .1) for i in range(3))

        def frame_data(self, scenario_id, frame):
            return FrameData(frame, ego=ActorState(np.eye(4), (1, 0, 0), "vehicle", (4, 2, 1)))

        def native(self, scenario_id):
            return {}

        def scenario_start(self, scenario_id, frame):
            return ScenarioStart("memory", scenario_id, frame, {})

    class MemoryRuntime(SimulatorRuntime):
        command_modes = frozenset({"actuator"})

        def _initialize_world(self, start, initialization, world):
            self.position = start.frame.position
            self.speed = .5 if world.get("weather", {}).get("precipitation", 0) else 1.

        def _observe(self):
            return Observation(self.position * .1, self.position,
                               {"front": np.full((4, 4, 3), 100, np.uint8)}, ego_speed=self.speed)

        def _state(self):
            pose = np.eye(4)
            pose[0, 3] = self.position
            return WorldSnapshot(self.position * .1, self.position,
                                 ego_speed=self.speed, ego=ActorState(pose, (self.speed, 0, 0), "vehicle", (4, 2, 1)))

        def _advance_native(self, command):
            self.position += 1

    class MemorySimulator(Simulator):
        dataset_kinds = ("memory",)
        supported_world_settings = frozenset({"weather"})

        def create_runtime(self, seed=None):
            return MemoryRuntime(self)

    class MemoryAVStack(AVStack):
        command_modes = frozenset({"actuator"})
        provides_detections = False

        def __call__(self, observation):
            if module.fail_run:
                raise RuntimeError("memory driving failure")
            image = observation.sensor_data["front"]
            noisy = bool(np.any(image != 100))
            return Control(throttle=0. if noisy else .3, brake=.1 if noisy else 0.)

    class MemoryAttack(Attack):
        def build_metadata(self, scenario):
            return (AttackSegment(scenario.scenario_id, scenario.frames[0], scenario.frames[2]),)

    class FailingMetric(Metric):
        def evaluate(self, group):
            raise ValueError("metric failed")

    for item in (MemoryDataset, MemorySimulator, MemoryAVStack, MemoryAttack, FailingMetric):
        setattr(module, item.__name__, item)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    output = tmp_path / "output"
    specification = {
        "dataset": {"type": "avsec_cli_memory:MemoryDataset"},
        "simulator": {"type": "avsec_cli_memory:MemorySimulator"},
        "avstack": {"type": "avsec_cli_memory:MemoryAVStack"},
        "attacks": {"example": {"type": "avsec_cli_memory:MemoryAttack"}},
        "treatments": [{"name": "clean"}, {"name": "attacked", "attack_name": "example"},
                       {"name": "noisy", "attack_name": "example", "environment": {
                           "world": {"weather": {"precipitation": 70}},
                           "sensors": [{"type": "GaussianNoise", "camera": "front",
                                        "options": {"severity": .3}}]}}],
        "output": str(output),
    }
    config_path = tmp_path / "config.yaml"
    def write_config():
        config_path.write_text(yaml.safe_dump(specification))
    write_config()
    return SimpleNamespace(**locals())


def test_in_memory_cli_run_then_offline_evaluate_writes_all_deliverables(memory_cli):
    import json
    w = memory_cli
    runner = CliRunner()
    result = runner.invoke(cli.app, ["run", str(w.config_path)])
    assert result.exit_code == 0, result.output
    assert (w.output / "prepared.json").is_file()
    assert len(list((w.output / "runs").glob("*/*/*/*/manifest.json"))) == 3
    directory = w.output / "evaluation"
    assert (directory / "report.json").is_file() and (directory / "runs.csv").is_file()
    assert len(list((directory / "plots").glob("*.png"))) == 1
    report = json.loads((directory / "report.json").read_text())
    pairs = {(pair["treatment_a"], pair["treatment_b"]): pair["deltas"]
             for pair in report["pair_summaries"]}
    # The attack has no effects, so only the noisy environment (rain slows, noise brakes) differs.
    expected = {("clean", "attacked"): (0., 0), ("clean", "noisy"): (-.5, 2),
                ("attacked", "noisy"): (-.5, 2)}
    assert set(pairs) == set(expected)
    for pair, (final_speed, brake_steps) in expected.items():
        assert pairs[pair]["DrivingMetrics.final_speed"]["mean"] == final_speed, pair
        assert pairs[pair]["DrivingMetrics.brake_steps"]["mean"] == brake_steps, pair
    original = (directory / "report.json").read_bytes()
    w.module.allow_data = False
    repeated = runner.invoke(cli.app, ["evaluate", str(w.config_path)])
    assert repeated.exit_code == 0, repeated.output
    assert (directory / "report.json").read_bytes() == original


@pytest.mark.parametrize("failure", ["runtime", "metric", "configuration"])
def test_in_memory_cli_failure_exit_codes_and_status_reports(memory_cli, failure):
    import json
    w = memory_cli
    if failure == "runtime":
        w.module.fail_run = True
    elif failure == "metric":
        w.specification["evaluation"] = {"metrics": [{"type": "avsec_cli_memory:FailingMetric"}]}
    else:
        w.specification["attacks"] = {}
    w.write_config()
    result = CliRunner().invoke(cli.app, ["run", str(w.config_path)])
    assert result.exit_code == (2 if failure == "configuration" else 1), result.output
    if failure != "configuration":
        directory = w.output / "evaluation"
        report = json.loads((directory / "report.json").read_text())
        assert (directory / "runs.csv").is_file()
        if failure == "runtime":
            assert "Failed run:" in result.output and "RuntimeError" in result.output
        else:
            assert report["metric_errors"]


@pytest.mark.parametrize("error,code,label", [
    (ValueError("invalid prepared JSON"), 1, "Result load error"),
    (TypeError("invalid record"), 1, "Result load error"),
    (cli.ConfigurationMismatchError("attack names differ"), 2, "Configuration error"),
])
def test_result_read_errors_are_distinct_from_configuration_incompatibility(
    workflow, monkeypatch, error, code, label,
):
    w = workflow
    w.output.mkdir()
    (w.output / "prepared.json").write_text("{}")
    monkeypatch.setattr(cli.ExperimentResult, "load", Mock(side_effect=error))
    result = CliRunner().invoke(cli.app, ["evaluate", str(w.path)])
    assert result.exit_code == code and label in result.output
    w.report.assert_not_called()


@pytest.mark.parametrize("failure", [RuntimeError("strict run failed"), KeyboardInterrupt()])
def test_execution_failure_prints_the_saved_output_directory_once(workflow, monkeypatch, failure):
    monkeypatch.setattr(cli.ExperimentResult, "load", Mock(return_value=workflow.result))
    workflow.execute.side_effect = failure
    result = CliRunner().invoke(cli.app, ["run", str(workflow.path)])
    assert result.exit_code in (1, 130)
    assert result.output.count("Output directory:") == 1


def test_prepare_interrupt_does_not_claim_an_output_directory(workflow):
    workflow.prepare.side_effect = KeyboardInterrupt()
    result = CliRunner().invoke(cli.app, ["run", str(workflow.path)])
    assert result.exit_code == 130
    assert "Output directory:" not in result.output
    assert not workflow.output.exists()


@pytest.mark.parametrize("command", ["run", "evaluate"])
def test_interrupted_evaluation_is_not_restarted_and_reports_possible_partial_files(
    workflow, monkeypatch, command,
):
    w = workflow
    w.output.mkdir()
    (w.output / "prepared.json").write_text("{}")
    load = Mock(return_value=w.result)
    monkeypatch.setattr(cli.ExperimentResult, "load", load)
    w.report.side_effect = KeyboardInterrupt()
    result = CliRunner().invoke(cli.app, [command, str(w.path)])
    assert result.exit_code == 130 and "report files may be incomplete" in result.output
    w.report.assert_called_once()
    assert result.output.count("Output directory:") == 1


def test_removed_treatment_notice_is_plain_cli_output(workflow, monkeypatch):
    w = workflow
    w.output.mkdir()
    (w.output / "prepared.json").write_text("{}")
    w.result.notices = ("Skipping saved run for removed treatment: removed",)
    monkeypatch.setattr(cli.ExperimentResult, "load", Mock(return_value=w.result))
    result = CliRunner().invoke(cli.app, ["evaluate", str(w.path)])
    assert result.exit_code == 0 and w.result.notices[0] in result.output
    assert "RuntimeWarning" not in result.output
