"""Preparation and execution are tested as complete public workflows."""

from dataclasses import replace
import json

import pytest

from avsectester.attack import AttackSegment
from avsectester.closed_loop.avstacks.driving_adapter import DrivingAdapter
from avsectester.closed_loop.simulators import UnsupportedStartError
from avsectester.experiment import (
    ConfigurationMismatchError, Experiment, ExperimentConfiguration, ExperimentResult, PreparedExperiment,
    RunsConfiguration, SplitConfiguration, TreatmentConfiguration, evaluate,
)
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
    restricted = replace(config, avstack=DrivingAdapter(lambda seed: ConstantAVStack(), past_steps=1))
    filtered = Experiment.prepare(restricted)
    assert [segment.start.position for segment in filtered.metadata["probe"]] == [2, 2]
    assert "insufficient recorded history" in filtered.selections[0].reasons[0]
    with pytest.raises(RuntimeError, match="selection failed"):
        Experiment.prepare(replace(config, runs=replace(config.runs, strict=True)))

    # Starts the simulator cannot use are skipped with the simulator's reason.
    class UnusableLastScenario(MemorySimulator):
        def validate_start(self, start):
            super().validate_start(start)
            if start.scenario_id == "last":
                raise UnsupportedStartError("synthetic route failure")

    skipped = Experiment.prepare(replace(config, simulator=UnusableLastScenario()))
    assert {segment.scenario_id for segment in skipped.metadata["probe"]} == {"first"}
    last = next(decision for decision in skipped.selections if decision.scenario_id == "last")
    assert last.status == "not_selected"
    assert any("cannot start: synthetic route failure" in reason for reason in last.reasons)


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


def test_multiple_attacks(tmp_path):
    # Identical selected segments form one comparison group and share its clean run.
    config = ExperimentConfiguration(
        MemoryDataset(), MemorySimulator(), DrivingAdapter(lambda seed: ConstantAVStack()),
        {"green": MinimalAttack(max_segments=1), "blue": MinimalAttack(max_segments=1)},
        runs=RunsConfiguration(treatments=(
            TreatmentConfiguration("clean"), TreatmentConfiguration("green", "green"),
            TreatmentConfiguration("blue", "blue"),
            TreatmentConfiguration("green_rain", "green", environment={"world": {"weather": "rain"}}),
        ), output=tmp_path / "results"),
    )
    result = Experiment.execute(Experiment.prepare(config))
    assert {(run.attack_name, run.treatment) for run in result.runs} == {
        (None, "clean"), ("green", "green"), ("green", "green_rain"), ("blue", "blue"),
    }
    assert all(run.log.status == "completed" for run in result.runs)
    loaded = ExperimentResult.load(tmp_path / "results", config)
    assert {(run.attack_name, run.treatment) for run in loaded.runs} == {
        (run.attack_name, run.treatment) for run in result.runs}
    report = evaluate(loaded, config)
    assert {(row["part"], row["treatment"]) for row in report.treatment_summaries} == {
        (None, "clean"), (None, "green"), (None, "green_rain"), (None, "blue"),
    }
    assert set(result.groups) == {(0, 0)}
    clean_directories = list((tmp_path / "results" / "runs").glob("segment-*/seed-*/clean"))
    assert clean_directories == [tmp_path / "results" / "runs" / "segment-00000" / "seed-0" / "clean"]


def test_split(tmp_path):
    class SplitAttack(MinimalAttack):
        def run_offline(self, context):
            return {"train": context.metadata, "validation": context.validation_metadata}

    config = ExperimentConfiguration(
        MemoryDataset(scenario_ids=("z", "a")), MemorySimulator(),
        DrivingAdapter(lambda seed: ConstantAVStack()),
        {"first": SplitAttack(variants_per_window=2), "second": SplitAttack(variants_per_window=2)},
        split=SplitConfiguration(2, validation_segments=1, seed=0),
        runs=RunsConfiguration(treatments=(
            TreatmentConfiguration("clean"), TreatmentConfiguration("first", "first"),
            TreatmentConfiguration("second", "second"),
        ), output=tmp_path / "split"),
    )
    prepared = Experiment.prepare(config)
    assert len(prepared.units) == 8
    assert [(unit.segment.scenario_id, unit.segment.start.position, unit.segment.stop.position,
             unit.segment.user_data["variant"]) for unit in prepared.units] == [
        (scenario, start, start + 2, variant)
        for scenario in ("z", "a") for start in (0, 2) for variant in (0, 1)
    ]
    assert all(unit.attack_names == ("first", "second") for unit in prepared.units)
    assert [unit.part for unit in prepared.units] == [
        "test", "test", "train", "validation", "train", "test", "test", "test",
    ]
    # Two variants at the same window are independent units, including across parts.
    assert prepared.units[2].part != prepared.units[3].part
    for name in config.attacks:
        assert len(prepared.metadata[name]) == 8
        assert prepared.artifacts[name] == {
            "train": tuple(unit.segment for unit in prepared.units if unit.part == "train"),
            "validation": tuple(unit.segment for unit in prepared.units if unit.part == "validation"),
        }
    repeated = Experiment.prepare(replace(config, runs=replace(config.runs, output=None)))
    assert [unit.part for unit in repeated.units] == [unit.part for unit in prepared.units]
    result = Experiment.execute(prepared)
    assert len(result.runs) == 15
    assert {run.part for run in result.runs} == {"test"}
    assert {run.segment_index for run in result.runs} == {0, 1, 5, 6, 7}
    assert all(run.log.metadata["part"] == "test" for run in result.runs)

    # Execution selection may change without invalidating the prepared partition.
    train_config = replace(config, split=replace(config.split, execute=("train",)),
                           runs=replace(config.runs, output=tmp_path / "train"))
    training = PreparedExperiment.load(train_config, tmp_path / "split")
    assert {run.part for run in Experiment.execute(training).runs} == {"train"}
    with pytest.raises(ConfigurationMismatchError, match="split"):
        PreparedExperiment.load(replace(config, split=replace(config.split, seed=1)), tmp_path / "split")
    with pytest.raises(ValueError, match="8"):
        Experiment.prepare(replace(config, split=SplitConfiguration(7, validation_segments=1),
                                   runs=replace(config.runs, output=None)))


def test_shared_deployment_budget():
    class BudgetAttack(MinimalAttack):
        def run_offline(self, context):
            return context.deployment.ticks

    class SlowDeploymentAttack(BudgetAttack):
        def deployment_ticks(self, segment):
            return 3

    config = ExperimentConfiguration(
        MemoryDataset(), MemorySimulator(), DrivingAdapter(lambda seed: ConstantAVStack()),
        {"fast": BudgetAttack(max_segments=3), "slow": SlowDeploymentAttack(max_segments=3)},
        runs=RunsConfiguration(treatments=(
            TreatmentConfiguration("clean"), TreatmentConfiguration("fast", "fast"),
            TreatmentConfiguration("slow", "slow"),
        )),
    )
    prepared = Experiment.prepare(config)
    assert prepared.artifacts == {"fast": 3, "slow": 3}
    assert {name: [segment.start.position for segment in segments]
            for name, segments in prepared.metadata.items()} == {"fast": [0, 4], "slow": [0, 4]}
    fast_selection = next(row for row in prepared.selections if row.attack_name == "fast")
    assert any("2" in reason and "deployment" in reason for reason in fast_selection.reasons)


    # One attack may declare different budgets in different scenes. Its earlier scene
    # must still be checked against the experiment-wide maximum after all selection.
    class VaryingDeploymentAttack(BudgetAttack):
        def build_metadata(self, scenario):
            start = 1 if scenario.scenario_id == "a" else 3
            return (AttackSegment(scenario.scenario_id, scenario.frames[start], scenario.frames[start + 1]),)

        def deployment_ticks(self, segment):
            return segment.start.position

    varying = Experiment.prepare(configuration(
        VaryingDeploymentAttack(), dataset=MemoryDataset(scenario_ids=("a", "b")),
    ))
    assert varying.deployment.ticks == 3
    assert varying.artifacts == {"probe": 3}
    assert [(segment.scenario_id, segment.start.position) for segment in varying.metadata["probe"]] == [
        ("b", 3),
    ]
    rejected = next(row for row in varying.selections if row.scenario_id == "a")
    assert rejected.status == "not_selected"
    assert any("deployment" in reason and "3" in reason for reason in rejected.reasons)


def test_prepare_resume(tmp_path):
    class CheckpointAttack(MinimalAttack):
        def __init__(self, marker_path, ready_path=None):
            super().__init__(max_segments=1)
            self.marker_path, self.ready_path = marker_path, ready_path

        def build_metadata(self, scenario):
            with self.marker_path.with_suffix(".selection").open("x") as stream:
                stream.write("selected")
            return super().build_metadata(scenario)

        def run_offline(self, context):
            if self.ready_path is not None and not self.ready_path.exists():
                raise RuntimeError("offline dependency unavailable")
            with self.marker_path.open("x") as stream:
                stream.write("prepared")
            return self.marker_path.read_text()

    first_marker, second_marker, ready = (tmp_path / name for name in ("first", "second", "ready"))
    config = ExperimentConfiguration(
        MemoryDataset(), MemorySimulator(), DrivingAdapter(lambda seed: ConstantAVStack()),
        {"first": CheckpointAttack(first_marker), "second": CheckpointAttack(second_marker, ready)},
        runs=RunsConfiguration(output=tmp_path / "prepare", strict=True),
    )
    with pytest.raises(RuntimeError, match="offline dependency unavailable"):
        Experiment.prepare(config)
    output = tmp_path / "prepare"
    assert not (output / "prepared.json").exists()
    assert (output / "preparation-progress.json").is_file()
    assert all((output / "attacks" / name / "metadata.json").is_file() for name in config.attacks)
    first_saved = first_marker.read_bytes()
    with pytest.raises(ConfigurationMismatchError, match="seed"):
        Experiment.prepare(replace(config, runs=replace(config.runs, seeds=(1,))), resume=True)
    ready.touch()
    prepared = Experiment.prepare(config, resume=True)
    assert first_marker.read_bytes() == first_saved
    assert prepared.artifacts == {"first": "prepared", "second": "prepared"}
    assert (output / "prepared.json").is_file()
    # A complete checkpoint also loads directly without repeating either offline phase.
    assert Experiment.prepare(config, resume=True).artifacts == prepared.artifacts
    with pytest.raises(ValueError, match="exist"):
        Experiment.prepare(config)


def test_execute_resume(tmp_path):
    config = configuration(MinimalAttack(max_segments=1), output=tmp_path / "resume", treatments=(
        TreatmentConfiguration("clean"),
        TreatmentConfiguration("failed", environment={"world": {"fail": True}}),
        TreatmentConfiguration("interrupted", environment={"world": {"interrupt": True}}),
        TreatmentConfiguration("pending", "probe"),
    ))
    prepared = Experiment.prepare(config)
    with pytest.raises(KeyboardInterrupt):
        Experiment.execute(prepared)
    base = tmp_path / "resume" / "runs" / "segment-00000" / "seed-0"
    clean_manifest = base / "clean" / "manifest.json"
    saved_clean = clean_manifest.read_bytes()
    retained_file = base / "clean" / "retained.txt"
    retained_file.write_text("preserve completed directory")
    assert json.loads((base / "failed" / "manifest.json").read_text())["status"] == "failed"
    # Recover the fake external world's faults before resuming the same requested run identities.
    config.runs.treatments[1].environment["world"].clear()
    config.runs.treatments[2].environment["world"].clear()
    completed = Experiment.execute(prepared, resume=True)
    assert len(completed.runs) == 4
    assert all(run.log.status == "completed" for run in completed.runs)
    assert clean_manifest.read_bytes() == saved_clean
    assert retained_file.read_text() == "preserve completed directory"
    assert {run.treatment for run in completed.runs} == {"clean", "failed", "interrupted", "pending"}
    all_manifests = {path: path.read_bytes() for path in base.glob("*/manifest.json")}
    assert all_manifests
    Experiment.execute(prepared, resume=True)
    assert {path: path.read_bytes() for path in base.glob("*/manifest.json")} == all_manifests
    assert retained_file.read_text() == "preserve completed directory"


    # A completed manifest with another treatment's attack identity must be replaced.
    pending_manifest = base / "pending" / "manifest.json"
    mismatched = json.loads(pending_manifest.read_text())
    mismatched["metadata"]["attack_name"] = "other"
    pending_manifest.write_text(json.dumps(mismatched))
    replaced_file = base / "pending" / "replaced.txt"
    replaced_file.write_text("discard mismatched run directory")
    repaired = Experiment.execute(prepared, resume=True)
    assert all(run.log.status == "completed" for run in repaired.runs)
    pending = next(run for run in repaired.runs if run.treatment == "pending")
    assert pending.log.metadata["attack_name"] == pending.attack_name == "probe"
    assert not replaced_file.exists()
    assert {path: path.read_bytes() for path in all_manifests if path != pending_manifest} == {
        path: content for path, content in all_manifests.items() if path != pending_manifest
    }
    assert retained_file.read_text() == "preserve completed directory"

    # Execution settings form part of the checkpoint identity, not just the run's location.
    shorter = replace(prepared, configuration=replace(
        config, runs=replace(config.runs, decision_steps=1),
    ))
    shortened = Experiment.execute(shorter, resume=True)
    assert all(run.log.status == "completed" and len(run.log.steps) == 1 for run in shortened.runs)
    assert all(run.log.metadata["decision_steps"] == 1 for run in shortened.runs)
    assert all(run.log.metadata["selected_decision_steps"] == 2 for run in shortened.runs)
    assert not retained_file.exists()
    short_manifests = {path: path.read_bytes() for path in base.glob("*/manifest.json")}
    retained_file.write_text("retain unchanged shorter execution")
    Experiment.execute(shorter, resume=True)
    assert {path: path.read_bytes() for path in short_manifests} == short_manifests
    assert retained_file.read_text() == "retain unchanged shorter execution"

    changed = replace(shorter, configuration=replace(
        shorter.configuration, runs=replace(
            shorter.configuration.runs,
            treatments=tuple(replace(t, environment={"world": {"weather": "rain"}})
                             if t.name == "pending" else t for t in shorter.configuration.runs.treatments),
        ),
    ))
    replaced_file.write_text("discard changed treatment environment")
    changed_runs = Experiment.execute(changed, resume=True)
    pending = next(run for run in changed_runs.runs if run.treatment == "pending")
    assert pending.log.metadata["environment"]["world"]["weather"] == "rain"
    assert not replaced_file.exists()
    assert retained_file.read_text() == "retain unchanged shorter execution"
    assert {path: path.read_bytes() for path in short_manifests if path != pending_manifest} == {
        path: content for path, content in short_manifests.items() if path != pending_manifest
    }
