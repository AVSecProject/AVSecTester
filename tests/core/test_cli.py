"""The CLI runs and evaluates actual in-memory experiments through YAML configuration."""

import json

from typer.testing import CliRunner
import yaml

from avsectester import __version__, cli


runner = CliRunner()


def write_configuration(directory, *, output="results", world=None):
    directory.mkdir(parents=True, exist_ok=True)
    specification = {
        "dataset": {"type": "tests.fakes:MemoryDataset"},
        "simulator": {"type": "tests.fakes:MemorySimulator"},
        "avstack": {"type": "tests.fakes:ConstantAVStack"},
        "attacks": {"probe": {"type": "tests.fakes:MinimalAttack"}},
        "runs": {"output": output},
    }
    if world is not None:
        specification["runs"]["environment"] = {"world": world}
    path = directory / "experiment.yaml"
    path.write_text(yaml.safe_dump(specification))
    return path


def test_run(tmp_path):
    path = write_configuration(tmp_path)
    completed = runner.invoke(cli.app, ["run", str(path)])
    assert completed.exit_code == 0, completed.output
    output = tmp_path / "results"
    assert (output / "prepared.json").is_file()
    assert (output / "runs").is_dir()
    assert (output / "evaluation/report.json").is_file()
    assert (output / "evaluation/runs.csv").is_file()
    assert list((output / "evaluation/plots").glob("*.png"))
    assert str(output / "offline_report.html") in completed.output
    assert str(output / "evaluation_report.html") in completed.output
    assert (output / "offline_report.html").is_file()
    assert (output / "evaluation_report.html").is_file()
    assert str(output) in completed.output
    report = (output / "evaluation/report.json").read_bytes()
    repeated = runner.invoke(cli.app, ["run", str(path)])
    assert repeated.exit_code == 2 and "already exists" in repeated.output
    assert (output / "evaluation/report.json").read_bytes() == report

    second_path = write_configuration(tmp_path / "reused", world={"weather": "rain"})
    second_configuration = yaml.safe_load(second_path.read_text())
    second_configuration["runs"]["seeds"] = [1]
    second_path.write_text(yaml.safe_dump(second_configuration))
    mismatched_seed = runner.invoke(cli.app, ["run", str(second_path), "--prepared", str(output)])
    assert mismatched_seed.exit_code == 2, mismatched_seed.output
    assert "preparation configuration differs" in mismatched_seed.output
    second_configuration["runs"]["seeds"] = [0, 1]
    second_path.write_text(yaml.safe_dump(second_configuration))
    mismatched_environment = runner.invoke(cli.app, ["run", str(second_path), "--prepared", str(output)])
    assert mismatched_environment.exit_code == 2, mismatched_environment.output
    second_configuration["runs"].pop("environment")
    second_path.write_text(yaml.safe_dump(second_configuration))
    reused = runner.invoke(cli.app, ["run", str(second_path), "--prepared", str(output)])
    assert reused.exit_code == 0, reused.output
    assert (tmp_path / "reused/results/prepared.json").read_bytes() == (output / "prepared.json").read_bytes()

    failed_path = write_configuration(tmp_path / "failed", world={"fail": True})
    failed = runner.invoke(cli.app, ["run", str(failed_path)])
    assert failed.exit_code == 1 and "RuntimeError" in failed.output
    assert (failed_path.parent / "results/evaluation/report.json").is_file()
    interrupted_path = write_configuration(tmp_path / "interrupted", world={"interrupt": True})
    interrupted = runner.invoke(cli.app, ["run", str(interrupted_path)])
    assert interrupted.exit_code == 130 and "interrupted" in interrupted.output
    invalid_path = tmp_path / "invalid.yaml"
    invalid_path.write_text("unknown_setting: true\n")
    invalid = runner.invoke(cli.app, ["run", str(invalid_path)])
    assert invalid.exit_code == 2 and "Configuration error" in invalid.output


def test_run_resume(tmp_path):
    path = write_configuration(tmp_path)
    completed = runner.invoke(cli.app, ["run", str(path), "--resume"])
    assert completed.exit_code == 0, completed.output
    output = tmp_path / "results"
    manifests = {p: p.read_bytes() for p in (output / "runs").rglob("manifest.json")}
    assert manifests
    resumed = runner.invoke(cli.app, ["run", str(path), "--resume"])
    assert resumed.exit_code == 0, resumed.output
    assert {p: p.read_bytes() for p in manifests} == manifests
    conflicting = runner.invoke(cli.app, ["run", str(path), "--resume", "--prepared", str(output)])
    assert conflicting.exit_code == 2
    assert "--resume and --prepared" in conflicting.output
    configuration = yaml.safe_load(path.read_text())
    configuration["runs"]["seeds"] = [1]
    path.write_text(yaml.safe_dump(configuration))
    incompatible = runner.invoke(cli.app, ["run", str(path), "--resume"])
    assert incompatible.exit_code == 2 and "configuration differs" in incompatible.output


def test_evaluate(tmp_path):
    path = write_configuration(tmp_path)
    missing = runner.invoke(cli.app, ["evaluate", str(path)])
    assert missing.exit_code == 1 and "No saved experiment results" in missing.output
    assert runner.invoke(cli.app, ["run", str(path)]).exit_code == 0
    report_path = tmp_path / "results/evaluation/report.json"
    original = json.loads(report_path.read_text())
    evaluated = runner.invoke(cli.app, ["evaluate", str(path)])
    assert evaluated.exit_code == 0, evaluated.output
    assert json.loads(report_path.read_text()) == original

    configuration = yaml.safe_load(path.read_text())
    configuration["attacks"]["probe"]["options"] = {"decision_steps": 1}
    path.write_text(yaml.safe_dump(configuration))
    incompatible = runner.invoke(cli.app, ["evaluate", str(path)])
    assert incompatible.exit_code == 2 and "configuration differs" in incompatible.output
    configuration["attacks"]["probe"].pop("options")
    path.write_text(yaml.safe_dump(configuration))
    (tmp_path / "results/prepared.json").write_text("{broken JSON")
    corrupted = runner.invoke(cli.app, ["evaluate", str(path)])
    assert corrupted.exit_code == 1 and "Result load error" in corrupted.output


def test_version():
    completed = runner.invoke(cli.app, ["version"])
    assert completed.exit_code == 0
    assert completed.output.strip() == f"avsectester {__version__}"
