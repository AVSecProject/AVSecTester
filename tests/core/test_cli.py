"""The CLI runs and evaluates actual in-memory experiments through YAML configuration."""

import json
import os
import shlex
import sys

import pytest
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
        "depth": "none", "physics_update_mode": "none", "route_generator": "none",
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
    owner = next((arguments[i + 1].split("=", 1)[1] for i, value in enumerate(arguments)
                  if value == "--label" and arguments[i + 1].startswith("avsectester.service-owner=")), "")
    state.with_suffix(".owner").write_text(owner)
    print("fake-container-id")
    raise SystemExit(1 if mode == "run-fail" else 0)
if arguments[0] == "inspect":
    if not state.exists():
        raise SystemExit(1)
    if "--format" in arguments and ".Config.Labels" in arguments[arguments.index("--format") + 1]:
        print("fake-container-id " + state.with_suffix(".owner").read_text())
    else:
        print(state.read_text())
elif arguments[0] == "logs":
    print("Serving on 0.0.0.0:50152\nStarting CATK traffic gRPC server on 0.0.0.0:6200" if mode == "ready" else "NRE startup diagnostic")
elif arguments[0] == "stop":
    if not state.exists():
        print("No such container", file=sys.stderr)
        raise SystemExit(1)
    state.write_text("false")
elif arguments[0] == "rm":
    state.unlink()
    state.with_suffix(".owner").unlink(missing_ok=True)
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


def traffic_configuration(directory, endpoint="127.0.0.1:6200"):
    path = nre_configuration(directory)
    specification = yaml.safe_load(path.read_text())
    specification["simulator"]["options"]["traffic"] = {
        "type": "avsectester.closed_loop.simulators.nurec:AlpaSimTraffic",
        "options": {"endpoint": endpoint},
    }
    path.write_text(yaml.safe_dump(specification))
    return path


def test_trafficsim_dry_run(tmp_path, monkeypatch):
    path = traffic_configuration(tmp_path)
    state, calls = fake_docker(tmp_path, monkeypatch)
    models = tmp_path / "not-created-models"
    monkeypatch.setenv("PATH", "")
    result = runner.invoke(cli.app, ["trafficsim", "serve", str(path), "--gpu", "2",
                                     "--models", str(models), "--dry-run"])
    assert result.exit_code == 0, result.output
    command = shlex.split(result.output.strip())
    assert command[:5] == ["docker", "run", "-d", "--name", "avsectester-trafficsim-6200"]
    assert "device=2" in command and "avsectester:trafficsim" in command
    assert f"{models}:/mnt/trafficsim-models:ro" in command
    assert "OMP_NUM_THREADS=1" in command and "MKL_NUM_THREADS=1" in command
    assert "catk.loader.prediction_steps=15" in command
    assert not calls.exists() and not state.exists() and not models.exists()
    assert not (tmp_path / "probes").exists()
    wrong = write_configuration(tmp_path / "missing-traffic")
    rejected = runner.invoke(cli.app, ["trafficsim", "serve", str(wrong), "--gpu", "2",
                                       "--models", str(models), "--dry-run"])
    assert rejected.exit_code == 2 and "AlpaSimTraffic" in rejected.output


def test_trafficsim_serve(tmp_path, monkeypatch):
    pytest.importorskip("alpasim_grpc.v0.traffic_pb2")
    from tests.fakes import TrafficServiceBoundary

    models = tmp_path / "models"
    models.mkdir()
    state, calls = fake_docker(tmp_path, monkeypatch)
    with TrafficServiceBoundary() as service:
        path = traffic_configuration(tmp_path / "configuration", service.endpoint)
        served = runner.invoke(cli.app, ["trafficsim", "serve", str(path), "--gpu", "1",
                                         "--models", str(models), "--name", "test-traffic"])
        assert served.exit_code == 0, served.output
        assert f"endpoint={service.endpoint}" in served.output and "scenes=1" in served.output
        assert "Stop command:" in served.output and state.exists()
        repeated = runner.invoke(cli.app, ["trafficsim", "serve", str(path), "--gpu", "1",
                                           "--models", str(models), "--name", "test-traffic"])
        assert repeated.exit_code == 1 and "already exists" in repeated.output
        stopped = runner.invoke(cli.app, ["trafficsim", "stop", str(path), "--name", "test-traffic"])
        assert stopped.exit_code == 0 and not state.exists()
    # A startup log alone is insufficient: metadata must be reachable too.
    failed = runner.invoke(cli.app, ["trafficsim", "serve", str(path), "--gpu", "1",
                                     "--models", str(models), "--timeout", ".2"])
    assert failed.exit_code == 1 and "timed out" in failed.output and not state.exists()
    assert "Starting CATK traffic gRPC server on" in failed.output
    recorded = [json.loads(line) for line in calls.read_text().splitlines()]
    assert recorded[-2][0] == "stop" and recorded[-1][0] == "rm"
    monkeypatch.setenv("PATH", "")
    unavailable = runner.invoke(cli.app, ["trafficsim", "serve", str(path), "--gpu", "1",
                                          "--models", str(models)])
    assert unavailable.exit_code == 1 and "--dry-run" in unavailable.output


def test_trafficsim_stop(tmp_path, monkeypatch):
    path = traffic_configuration(tmp_path)
    state, calls = fake_docker(tmp_path, monkeypatch)
    state.write_text("true")
    # Stopping needs only the traffic endpoint, not live archives or weights.
    (tmp_path / "data" / "scene.usdz").unlink()
    (tmp_path / "data").rmdir()
    stopped = runner.invoke(cli.app, ["trafficsim", "stop", str(path)])
    assert stopped.exit_code == 0 and "Stopped and removed" in stopped.output
    recorded = [json.loads(line) for line in calls.read_text().splitlines()]
    assert recorded[-2:] == [["stop", "avsectester-trafficsim-6200"],
                            ["rm", "avsectester-trafficsim-6200"]]
    assert not state.exists()
    missing = runner.invoke(cli.app, ["trafficsim", "stop", str(path)])
    assert missing.exit_code == 1 and "No such container" in missing.output
    state.write_text("true")
    named = runner.invoke(cli.app, ["trafficsim", "stop", str(path), "--name", "owned-traffic"])
    assert named.exit_code == 0, named.output
    recorded = [json.loads(line) for line in calls.read_text().splitlines()]
    assert recorded[-2:] == [["stop", "owned-traffic"], ["rm", "owned-traffic"]]


@pytest.fixture
def managed_nurec_cli(tmp_path, monkeypatch, nurec_usdz_path):
    from tests.fakes import NuRecRpcBoundary, NuRecRpcError

    state, calls = fake_docker(tmp_path, monkeypatch)
    boundary = NuRecRpcBoundary().install(monkeypatch)
    path = nre_configuration(tmp_path / "configuration")
    spec = yaml.safe_load(path.read_text())
    spec["dataset"] = {"type": "NuRecDataset", "options": {"usdz_paths": [str(nurec_usdz_path)]}}
    spec["simulator"]["options"]["object_occlusion"] = "geometry"
    spec["simulator"]["options"]["min_track_duration_s"] = 0
    spec["simulator"]["options"]["renderer"]["options"]["cameras"] = ["front"]
    spec["attacks"]["probe"]["options"] = {"decision_steps": 1, "max_segments": 1}
    path.write_text(yaml.safe_dump(spec))
    original = boundary.get_available_scenes

    def scenes(request, timeout=None):
        if not state.exists():
            raise NuRecRpcError("UNAVAILABLE", "not running")
        return original(request, timeout=timeout)

    monkeypatch.setattr(boundary, "get_available_scenes", scenes)
    return path, state, calls, boundary


def test_run_managed_nre(managed_nurec_cli, monkeypatch):
    path, state, calls, boundary = managed_nurec_cli
    monkeypatch.setenv("AVSECTESTER_HOST_GPU", "2")
    result = runner.invoke(cli.app, ["run", str(path), "--nre-name", "owned-nre"])
    assert result.exit_code == 0, result.output
    assert "Starting NRE" in result.output and "Stopped" in result.output
    assert not state.exists()
    recorded = [json.loads(line) for line in calls.read_text().splitlines()]
    command, = [row for row in recorded if row[0] == "run"]
    assert "device=2" in command
    assert recorded[-2:] == [["stop", "fake-container-id"], ["rm", "fake-container-id"]]
    assert (path.parent / "results/evaluation/report.json").is_file()
    before = calls.read_bytes()
    repeated = runner.invoke(cli.app, ["run", str(path)])
    assert repeated.exit_code == 2 and "already exists" in repeated.output
    assert calls.read_bytes() == before
    conflict = runner.invoke(cli.app, ["run", str(path), "--resume", "--prepared", str(path.parent / "results")])
    assert conflict.exit_code == 2 and calls.read_bytes() == before
    invalid = path.parent / "invalid.yaml"
    invalid.write_text("unknown: true")
    assert runner.invoke(cli.app, ["run", str(invalid)]).exit_code == 2
    assert calls.read_bytes() == before


def test_run_reused_nre(managed_nurec_cli):
    path, state, calls, boundary = managed_nurec_cli
    state.write_text("true")
    result = runner.invoke(cli.app, ["run", str(path)])
    assert result.exit_code == 0, result.output
    assert "Reusing NRE" in result.output and state.read_text() == "true"
    assert not calls.exists()


def test_run_nre_scene_mismatch(managed_nurec_cli):
    path, state, calls, boundary = managed_nurec_cli
    state.write_text("true")
    boundary.available_scenes = ("other-scene",)
    result = runner.invoke(cli.app, ["run", str(path)])
    assert result.exit_code == 2 and "scenes" in result.output
    assert state.read_text() == "true" and not calls.exists()


@pytest.mark.parametrize(("failure", "exit_code"), [(RuntimeError, 1), (KeyboardInterrupt, 130)])
def test_run_managed_nre_failure(managed_nurec_cli, monkeypatch, failure, exit_code):
    path, state, calls, boundary = managed_nurec_cli

    def failed_capture(request):
        raise failure("synthetic camera failure")

    monkeypatch.setattr(boundary, "render_rgb", failed_capture)
    result = runner.invoke(cli.app, ["run", str(path), "--nre-gpu", "0", "--nre-name", "owned-nre"])
    assert result.exit_code == exit_code, result.output
    assert not state.exists()
    recorded = [json.loads(line) for line in calls.read_text().splitlines()]
    assert recorded[-2:] == [["stop", "fake-container-id"], ["rm", "fake-container-id"]]


def test_run_service_gpu_configuration(managed_nurec_cli, monkeypatch):
    path, state, calls, boundary = managed_nurec_cli
    monkeypatch.setenv("AVSECTESTER_HOST_GPU", "all")
    result = runner.invoke(cli.app, ["run", str(path)])
    assert result.exit_code == 2, result.output
    assert "GPU" in result.output and not state.exists() and not calls.exists()


def test_run_resume_output_file(managed_nurec_cli):
    path, state, calls, boundary = managed_nurec_cli
    (path.parent / "results").write_text("not a directory")
    result = runner.invoke(cli.app, ["run", str(path), "--resume"])
    assert result.exit_code == 2 and "must be a directory" in result.output
    assert not calls.exists() and not boundary.scene_query_timeouts
