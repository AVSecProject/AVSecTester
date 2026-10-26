"""The CLI runs and evaluates actual in-memory experiments through YAML configuration."""

import json
import os
import shlex
import sys

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

    no_selector_path = write_configuration(tmp_path / "without-selector")
    no_selector = yaml.safe_load(no_selector_path.read_text())
    no_selector["attacks"] = {"supplied": {"type": "tests.fakes:MetadataOnlyAttack"}}
    no_selector_path.write_text(yaml.safe_dump(no_selector))
    invalid = runner.invoke(cli.app, ["run", str(no_selector_path)])
    assert invalid.exit_code == 2 and "Configuration error" in invalid.output
    assert "supplied" in invalid.output and "metadata.load" in invalid.output
    assert not (no_selector_path.parent / "results").exists()


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


def nre_configuration(directory):
    path = write_configuration(directory)
    specification = yaml.safe_load(path.read_text())
    data = directory / "data"
    data.mkdir()
    (data / "scene.usdz").touch()
    specification["dataset"] = {"type": "NuRecDataset", "options": {"dataroot": str(data)}}
    specification["simulator"] = {"type": "NuRecSimulator", "options": {
        "config": {"dt": .1}, "depth": "none", "physics_update_mode": "none", "route_generator": "none",
        "renderer": {"type": "avsectester.closed_loop.simulators.nurec:NuRecRenderer", "options": {
            "endpoint": "127.0.0.1:50152", "probe_directory": str(directory / "probes"),
            "server_probe_directory": "/mnt/probes",
        }},
    }}
    path.write_text(yaml.safe_dump(specification))
    return path


def fake_docker(directory, monkeypatch):
    bin_path = directory / "bin"
    bin_path.mkdir()
    state = directory / "docker-state"
    calls = directory / "docker-calls.jsonl"
    executable = bin_path / "docker"
    executable.write_text(f"#!{sys.executable}\n" + r'''
import json, os, sys
from pathlib import Path
state = Path(os.environ["FAKE_DOCKER_STATE"])
with open(os.environ["FAKE_DOCKER_CALLS"], "a") as stream:
    stream.write(json.dumps(sys.argv[1:]) + "\n")
arguments = sys.argv[1:]
mode = os.environ.get("FAKE_DOCKER_MODE", "ready")
if arguments[0] == "run":
    state.write_text("false" if mode == "exit" else "true")
    print("fake-container-id")
    raise SystemExit(1 if mode == "run-fail" else 0)
if arguments[0] == "inspect":
    if not state.exists():
        raise SystemExit(1)
    print(state.read_text())
elif arguments[0] == "logs":
    print("Serving on 0.0.0.0:50152" if mode == "ready" else "NRE startup diagnostic")
elif arguments[0] == "stop":
    if not state.exists():
        print("No such container", file=sys.stderr)
        raise SystemExit(1)
    state.write_text("false")
elif arguments[0] == "rm":
    state.unlink()
''')
    executable.chmod(0o755)
    monkeypatch.setenv("PATH", str(bin_path) + os.pathsep + os.environ.get("PATH", ""))
    monkeypatch.setenv("FAKE_DOCKER_STATE", str(state))
    monkeypatch.setenv("FAKE_DOCKER_CALLS", str(calls))
    return state, calls


def test_nre_serve(tmp_path, monkeypatch):
    path = nre_configuration(tmp_path)
    state, calls = fake_docker(tmp_path, monkeypatch)
    dry = runner.invoke(cli.app, ["nre", "serve", str(path), "--gpu", "2", "--dry-run"])
    assert dry.exit_code == 0, dry.output
    command = shlex.split(dry.output.strip())
    assert command[:4] == ["docker", "run", "-d", "--name"]
    assert command[4] == "avsectester-nre-50152" and "device=2" in command
    assert not calls.exists() and not (tmp_path / "probes").exists()
    served = runner.invoke(cli.app, ["nre", "serve", str(path), "--gpu", "2"])
    assert served.exit_code == 0, served.output
    assert "endpoint=127.0.0.1:50152" in served.output and "scenes=1" in served.output
    assert "Stop command:" in served.output and state.exists() and (tmp_path / "probes").is_dir()
    repeated = runner.invoke(cli.app, ["nre", "serve", str(path), "--gpu", "2"])
    assert repeated.exit_code == 1 and "already exists" in repeated.output
    runner.invoke(cli.app, ["nre", "stop", str(path)])
    monkeypatch.setenv("FAKE_DOCKER_MODE", "exit")
    failed = runner.invoke(cli.app, ["nre", "serve", str(path), "--gpu", "2"])
    assert failed.exit_code == 1 and "NRE startup diagnostic" in failed.output
    assert not state.exists()
    monkeypatch.setenv("FAKE_DOCKER_MODE", "timeout")
    timed_out = runner.invoke(cli.app, ["nre", "serve", str(path), "--gpu", "2", "--timeout", ".001"])
    assert timed_out.exit_code == 1 and "timed out" in timed_out.output and not state.exists()
    monkeypatch.setenv("PATH", "")
    unavailable = runner.invoke(cli.app, ["nre", "serve", str(path), "--gpu", "2"])
    assert unavailable.exit_code == 1 and "--dry-run" in unavailable.output
    assert runner.invoke(cli.app, ["nre", "serve", str(path), "--gpu", "2", "--dry-run"]).exit_code == 0
    invalid = write_configuration(tmp_path / "wrong-components")
    rejected = runner.invoke(cli.app, ["nre", "serve", str(invalid), "--gpu", "2", "--dry-run"])
    assert rejected.exit_code == 2, rejected.output


def test_nre_stop(tmp_path, monkeypatch):
    path = nre_configuration(tmp_path)
    state, calls = fake_docker(tmp_path, monkeypatch)
    served = runner.invoke(cli.app, ["nre", "serve", str(path), "--gpu", "1", "--name", "owned-nre"])
    assert served.exit_code == 0, served.output
    # Stop must work even after the dataset has moved; it needs only the renderer endpoint.
    (tmp_path / "data" / "scene.usdz").unlink()
    (tmp_path / "data").rmdir()
    stopped = runner.invoke(cli.app, ["nre", "stop", str(path), "--name", "owned-nre"])
    assert stopped.exit_code == 0 and "Stopped and removed" in stopped.output
    assert not state.exists()
    recorded = [json.loads(line) for line in calls.read_text().splitlines()]
    assert recorded[-2:] == [["stop", "owned-nre"], ["rm", "owned-nre"]]
    missing = runner.invoke(cli.app, ["nre", "stop", str(path), "--name", "owned-nre"])
    assert missing.exit_code == 1 and "No such container" in missing.output
    state.write_text("true")
    default = runner.invoke(cli.app, ["nre", "stop", str(path)])
    assert default.exit_code == 0, default.output
    recorded = [json.loads(line) for line in calls.read_text().splitlines()]
    assert recorded[-2:] == [["stop", "avsectester-nre-50152"], ["rm", "avsectester-nre-50152"]]
