"""Offline result loading validates preparation, identities and incomplete comparisons."""

from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest

from avsectester.attack import Attack, AttackSegment
from avsectester.closed_loop.services import DrivingAdapter
from avsectester.closed_loop.simulator import Simulator
from avsectester.datasets import Dataset, FrameReference, ScenarioDescriptor
from avsectester.experiment import (
    ConfigurationMismatchError, ExperimentConfiguration, ExperimentResult, PreparedExperiment, SelectionDecision,
)
from avsectester.runlog import EventRecord, Recorder, RunLog


class OfflineDataset(Dataset):
    def scenarios(self):
        return iter((ScenarioDescriptor("scene", "memory"),))

    def frames(self, scenario_id):
        return tuple(FrameReference(scenario_id, index, index, index * .1) for index in range(5))

    def frame_data(self, scenario_id, frame):
        raise AssertionError("Evaluation must not request simulator frame data")

    def native(self, scenario_id):
        raise AssertionError("Evaluation must not request native data")

    def scenario_start(self, scenario_id, frame):
        raise AssertionError("Evaluation must not start a scenario")


class OfflineSimulator(Simulator):
    def create_runtime(self, seed=None):
        raise AssertionError("Evaluation must not start a simulator")


class SavedAttack(Attack):
    def __init__(self, scale=1):
        self.scale = scale

    def build_metadata(self, scenario):
        raise AssertionError("Evaluation must not select again")

    def load_artifacts(self, directory):
        raise AssertionError("Evaluation does not need to load artifacts")


def _driver(seed):
    raise AssertionError("Evaluation must not construct a driving model")


@pytest.fixture
def saved(tmp_path):
    configuration = ExperimentConfiguration(
        OfflineDataset(), OfflineSimulator(), DrivingAdapter(_driver),
        attacks={"patch": SavedAttack()}, seeds=(0, 1), output=tmp_path,
    )
    frames = configuration.dataset.frames("scene")
    segments = tuple(AttackSegment("scene", frames[index], frames[index + 2],
                                   user_data={"array": np.arange(2), "asset": Path("assets/patch.png")}) for index in (0, 2))
    prepared = PreparedExperiment(configuration, {"patch": segments}, {"patch": None}, (
        SelectionDecision("patch", "scene", "selected"),
        SelectionDecision("patch", "bad-scene", "error", ("bad input",), "selection failed"),
    ))
    prepared.save(tmp_path)
    return configuration, prepared


def run_path(configuration, index=0, seed=0, treatment="clean"):
    return configuration.output / "runs" / "patch" / f"segment-{index:05d}" / f"seed-{seed}" / treatment


def run_log(prepared, index=0, seed=0, treatment="clean", status="completed"):
    metadata = {"attack_name": "patch", "segment_index": index, "seed": seed,
                "treatment": treatment, "segment": asdict(prepared.metadata["patch"][index])}
    events = [EventRecord("failure", data={"error_type": "RuntimeError", "message": "failed run"})] if status == "failed" else []
    return RunLog(metadata=metadata, status=status, events=events)


def fake_storage(monkeypatch, records):
    for path in records:
        path.mkdir(parents=True)
        (path / "manifest.json").touch()
    monkeypatch.setattr(RunLog, "read", lambda path: records[Path(path)])


def test_load_recovers_multiple_groups_statuses_selections_and_missing_runs(saved, monkeypatch):
    configuration, prepared = saved
    records = {
        run_path(configuration): run_log(prepared),
        run_path(configuration, treatment="patch"): run_log(prepared, treatment="patch", status="failed"),
        run_path(configuration, seed=1): run_log(prepared, seed=1, status="incomplete"),
    }
    fake_storage(monkeypatch, records)
    loaded = ExperimentResult.load(configuration.output, configuration)
    assert len(loaded.runs) == 8 and len(loaded.groups) == 4
    assert len(loaded.failures) == 7
    assert loaded.runs[0].log.status == "completed"
    assert loaded.runs[1].error == "RuntimeError: failed run"
    assert loaded.runs[2].error == "Run ended with status incomplete"
    assert "no saved manifest" in loaded.runs[-1].error
    assert loaded.selections == prepared.selections and len(loaded.selection_errors) == 1
    # Missing outcomes are reported in memory; loading never writes fake logs to disk.
    assert len(list(configuration.output.glob("runs/**/manifest.json"))) == 3


def test_no_runs_cannot_be_mistaken_for_a_completed_experiment(saved):
    configuration, _ = saved
    loaded = ExperimentResult.load(configuration.output, configuration)
    assert len(loaded.runs) == len(loaded.failures) == 8
    assert all(run.log.status == "incomplete" for run in loaded.runs)
    assert not (configuration.output / "runs").exists()


@pytest.mark.parametrize("field,value", [
    ("attack_name", "unknown"), ("segment_index", 5), ("seed", 99),
    ("treatment", "unknown"), ("segment_index", True), ("seed", False),
    ("attack_name", ["invalid"]),
])
def test_unknown_or_invalid_run_identity_marks_only_affected_run_failed(saved, monkeypatch, field, value):
    configuration, prepared = saved
    run = run_log(prepared)
    run.metadata[field] = value
    fake_storage(monkeypatch, {run_path(configuration): run})
    loaded = ExperimentResult.load(configuration.output, configuration)
    assert loaded.runs[0].log.status == "failed"
    assert "Run read failed: ValueError: Run identity" in loaded.runs[0].error


def test_metadata_must_match_the_actual_directory(saved, monkeypatch):
    configuration, prepared = saved
    fake_storage(monkeypatch, {run_path(configuration, seed=1): run_log(prepared)})
    loaded = ExperimentResult.load(configuration.output, configuration)
    assert loaded.runs[2].log.status == "failed"
    assert "directory disagree" in loaded.runs[2].error


def test_segment_metadata_cannot_differ_from_saved_preparation(saved, monkeypatch):
    configuration, prepared = saved
    run = run_log(prepared)
    run.metadata["segment"] = deepcopy(run.metadata["segment"])
    run.metadata["segment"]["roles"] = {"host": ("unrelated",)}
    fake_storage(monkeypatch, {run_path(configuration): run})
    loaded = ExperimentResult.load(configuration.output, configuration)
    assert loaded.runs[0].log.status == "failed"
    assert "segment metadata differs" in loaded.runs[0].error


def test_changed_preparation_configuration_is_rejected_before_reading_runs(saved, monkeypatch):
    configuration, _ = saved
    incompatible = replace(configuration, attacks={"patch": SavedAttack(2)})
    monkeypatch.setattr(RunLog, "read", lambda path: pytest.fail("Must validate preparation first"))
    with pytest.raises(ConfigurationMismatchError, match="configuration differs"):
        ExperimentResult.load(configuration.output, incompatible)


def test_fully_saved_empty_run_logs_round_trip_without_loading_artifacts(saved):
    configuration, prepared = saved
    for index in range(2):
        for seed in configuration.seeds:
            for treatment in ("clean", "patch"):
                path = run_path(configuration, index, seed, treatment)
                metadata = run_log(prepared, index, seed, treatment).metadata
                with Recorder(path, metadata=metadata):
                    pass
    loaded = ExperimentResult.load(configuration.output, configuration)
    assert len(loaded.runs) == 8 and not loaded.failures
    np.testing.assert_array_equal(loaded.runs[0].log.metadata["segment"]["user_data"]["array"], [0, 1])


def test_metadata_comparison_normalizes_numpy_scalars_but_preserves_boolean_types():
    from avsectester.experiment.models import _same_data
    assert _same_data({"position_m": (np.float64(1.5), np.float32(2.0), np.int64(3))},
                      {"position_m": [1.5, 2.0, 3]})
    assert not _same_data(np.bool_(True), 1)
    assert not _same_data(True, np.int64(1))
    assert not _same_data(np.int64(1), 1.0)


def test_offline_result_loading_never_accesses_dataset_scenarios_or_frames(saved, monkeypatch):
    configuration, _ = saved
    def forbidden(*args):
        pytest.fail("Offline results must depend only on saved preparation and logs")
    monkeypatch.setattr(configuration.dataset, "scenarios", forbidden)
    monkeypatch.setattr(configuration.dataset, "checked_frame", forbidden)
    monkeypatch.setattr(configuration.dataset, "frames", forbidden)
    result = ExperimentResult.load(configuration.output, configuration)
    assert len(result.runs) == 8


def test_unreadable_run_does_not_block_other_saved_runs(saved, monkeypatch):
    configuration, prepared = saved
    first, second = run_path(configuration), run_path(configuration, treatment="patch")
    records = {first: run_log(prepared), second: run_log(prepared, treatment="patch")}
    fake_storage(monkeypatch, records)
    def read(path):
        if Path(path) == first:
            raise OSError("broken parquet chunk")
        return records[Path(path)]
    monkeypatch.setattr(RunLog, "read", read)
    result = ExperimentResult.load(configuration.output, configuration)
    assert result.runs[0].log.status == "failed"
    assert "OSError: broken parquet chunk" in result.runs[0].error
    assert result.runs[1].log.status == "completed"


def test_saved_runs_follow_the_single_run_directory_layout(saved, monkeypatch):
    from avsectester.experiment.models import saved_run_directories

    configuration, prepared = saved
    current = run_path(configuration)
    unexpected_seed = run_path(configuration, seed=7)
    stray = configuration.output / "runs" / "patch" / "notes" / "manifest.json"
    fake_storage(monkeypatch, {current: run_log(prepared), unexpected_seed: None})
    stray.parent.mkdir(parents=True)
    stray.touch()
    assert saved_run_directories(configuration.output) == (
        (("patch", 0, 0, "clean"), current), (("patch", 0, 7, "clean"), unexpected_seed))
    result = ExperimentResult.load(configuration.output, configuration)
    assert result.notices == (
        f"Skipping saved run that the saved preparation does not expect: {unexpected_seed}",)


def test_saved_directory_for_removed_treatment_is_noticed_and_skipped(saved, monkeypatch):
    configuration, prepared = saved
    current = run_path(configuration)
    removed = run_path(configuration, treatment="removed")
    fake_storage(monkeypatch, {current: run_log(prepared), removed: None})
    result = ExperimentResult.load(configuration.output, configuration)
    assert len(result.notices) == 1 and "removed treatment" in result.notices[0]
    assert len(result.runs) == 8
    assert result.runs[0].log.status == "completed"


def test_corrupt_preparation_is_a_read_error_not_configuration_mismatch(saved):
    import json
    configuration, _ = saved
    (configuration.output / "prepared.json").write_text("not JSON")
    with pytest.raises(json.JSONDecodeError) as failure:
        ExperimentResult.load(configuration.output, configuration)
    assert not isinstance(failure.value, ConfigurationMismatchError)
