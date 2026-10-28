"""The CLI runs and evaluates actual in-memory experiments through YAML configuration."""

import json
import shlex

import pytest
from typer.testing import CliRunner
import yaml

from avsectester import __version__
from avsectester.attack import AttackSegment
from avsectester.experiment import save_metadata
import avsectester.cli as cli
from tests.fakes import (
    MemoryDataset, NuRecRpcBoundary, TrafficServiceBoundary, docker_process_boundary,
)


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


def test_run_prepared(tmp_path):
    path = write_configuration(tmp_path)
    dataset = MemoryDataset()
    frames = dataset.frames("scene")
    metadata = tmp_path / "segments.json"
    save_metadata((AttackSegment("scene", frames[0], frames[2]),), metadata,
                  dataset_type="tests.fakes:MemoryDataset")
    specification = yaml.safe_load(path.read_text())
    specification["metadata"] = {"load": str(metadata)}
    path.write_text(yaml.safe_dump(specification))
    assert runner.invoke(cli.app, ["run", str(path)]).exit_code == 0
    output = tmp_path / "results"
    second_path = write_configuration(tmp_path / "reused", world={"weather": "rain"})
    second_configuration = yaml.safe_load(second_path.read_text())
    second_configuration["metadata"] = {"load": str(metadata)}
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


def test_run_resume(tmp_path):
    path = write_configuration(tmp_path)
    dataset = MemoryDataset()
    frames = dataset.frames("scene")
    metadata = tmp_path / "segments.json"
    save_metadata((AttackSegment("scene", frames[0], frames[2]),), metadata,
                  dataset_type="tests.fakes:MemoryDataset")
    specification = yaml.safe_load(path.read_text())
    specification["metadata"] = {"load": str(metadata)}
    path.write_text(yaml.safe_dump(specification))
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

    file_path = write_configuration(tmp_path / "output-file")
    (file_path.parent / "results").write_text("not a directory")
    invalid = runner.invoke(cli.app, ["run", str(file_path), "--resume"])
    assert invalid.exit_code == 2 and "must be a directory" in invalid.output


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


def test_nre_serve(tmp_path, monkeypatch):
    invalid = write_configuration(tmp_path / "invalid-renderer")
    rejected = runner.invoke(cli.app, ["nre", "serve", str(invalid), "--gpu", "2"])
    assert rejected.exit_code == 2 and "NuRecRenderer" in rejected.output
    path = nre_configuration(tmp_path)
    state, calls = docker_process_boundary(tmp_path, monkeypatch)
    state = state / "avsectester-nre-50152"
    dry = runner.invoke(cli.app, ["nre", "serve", str(path), "--gpu", "2", "--dry-run"])
    assert dry.exit_code == 0, dry.output
    command = shlex.split(dry.output.strip())
    assert command[:4] == ["docker", "run", "-d", "--name"]
    assert command[4] == "avsectester-nre-50152" and "device=2" in command
    assert not calls.exists() and not (tmp_path / "probes").exists()
    served = runner.invoke(cli.app, ["nre", "serve", str(path), "--gpu", "2"])
    assert served.exit_code == 0, served.output
    assert "endpoint=127.0.0.1:50152" in served.output and "scenes=1" in served.output
    assert "Stop it with:" in served.output and state.exists() and (tmp_path / "probes").is_dir()
    runner.invoke(cli.app, ["nre", "stop", str(path)])
    monkeypatch.setenv("DOCKER_MODE", "exit")
    failed = runner.invoke(cli.app, ["nre", "serve", str(path), "--gpu", "2"])
    assert failed.exit_code == 1 and "startup diagnostic" in failed.output
    assert not state.exists()


def test_nre_stop(tmp_path, monkeypatch):
    path = nre_configuration(tmp_path)
    state, calls = docker_process_boundary(tmp_path, monkeypatch)
    state = state / "owned-nre"
    served = runner.invoke(cli.app, ["nre", "serve", str(path), "--gpu", "1", "--name", "owned-nre"])
    assert served.exit_code == 0, served.output
    # Stop must work even after the dataset has moved; it needs only the renderer endpoint.
    (tmp_path / "data" / "scene.usdz").unlink()
    (tmp_path / "data").rmdir()
    stopped = runner.invoke(cli.app, ["nre", "stop", str(path), "--name", "owned-nre"])
    assert stopped.exit_code == 0 and "Stopped and removed" in stopped.output
    assert not state.exists()
    missing = runner.invoke(cli.app, ["nre", "stop", str(path), "--name", "owned-nre"])
    assert missing.exit_code == 1 and "No such container" in missing.output
    state = state.with_name("avsectester-nre-50152")
    state.write_text("true")
    default = runner.invoke(cli.app, ["nre", "stop", str(path)])
    assert default.exit_code == 0, default.output
    assert not state.exists()


def traffic_configuration(directory, endpoint="127.0.0.1:6200"):
    path = nre_configuration(directory)
    specification = yaml.safe_load(path.read_text())
    specification["simulator"]["options"]["traffic"] = {
        "type": "avsectester.closed_loop.simulators.nurec:AlpaSimTraffic",
        "options": {"endpoint": endpoint},
    }
    path.write_text(yaml.safe_dump(specification))
    return path


def test_trafficsim_serve_dry_run(tmp_path, monkeypatch):
    invalid = write_configuration(tmp_path / "invalid-traffic")
    rejected = runner.invoke(cli.app, ["trafficsim", "serve", str(invalid), "--gpu", "2", "--models", str(tmp_path)])
    assert rejected.exit_code == 2 and "AlpaSimTraffic" in rejected.output
    path = traffic_configuration(tmp_path)
    state, calls = docker_process_boundary(tmp_path, monkeypatch)
    models = tmp_path / "uncreated-models"
    monkeypatch.setenv("PATH", "")
    result = runner.invoke(cli.app, [
        "trafficsim", "serve", str(path), "--gpu", "2", "--models", str(models), "--dry-run"])
    assert result.exit_code == 0 and "docker run" in result.output
    assert not calls.exists() and not list(state.iterdir()) and not models.exists()
    models.mkdir()
    unavailable = runner.invoke(cli.app, ["trafficsim", "serve", str(path), "--gpu", "2", "--models", str(models)])
    assert unavailable.exit_code == 1 and "--dry-run" in unavailable.output
    assert "Docker is not available on PATH" in unavailable.output


def test_trafficsim_serve(tmp_path, monkeypatch):
    pytest.importorskip("alpasim_grpc.v0.traffic_pb2")

    models = tmp_path / "models"
    models.mkdir()
    state, calls = docker_process_boundary(tmp_path, monkeypatch)
    state = state / "test-traffic"
    with TrafficServiceBoundary() as service:
        path = traffic_configuration(tmp_path / "configuration", service.endpoint)
        served = runner.invoke(cli.app, ["trafficsim", "serve", str(path), "--gpu", "1",
                                         "--models", str(models), "--name", "test-traffic"])
        assert served.exit_code == 0, served.output
        assert f"endpoint={service.endpoint}" in served.output and "scenes=1" in served.output
        assert "Stop it with:" in served.output and state.exists()
        stopped = runner.invoke(cli.app, ["trafficsim", "stop", str(path), "--name", "test-traffic"])
        assert stopped.exit_code == 0 and not state.exists()
    monkeypatch.setenv("DOCKER_MODE", "exit")
    failed = runner.invoke(cli.app, [
        "trafficsim", "serve", str(path), "--gpu", "1", "--models", str(models), "--name", "test-traffic"])
    assert failed.exit_code == 1 and "startup diagnostic" in failed.output
    assert not state.exists()


def test_trafficsim_stop(tmp_path, monkeypatch):
    path = traffic_configuration(tmp_path)
    state, calls = docker_process_boundary(tmp_path, monkeypatch)
    state = state / "avsectester-trafficsim-6200"
    state.write_text("true")
    # Stopping needs only the traffic endpoint, not live archives or weights.
    (tmp_path / "data" / "scene.usdz").unlink()
    (tmp_path / "data").rmdir()
    stopped = runner.invoke(cli.app, ["trafficsim", "stop", str(path)])
    assert stopped.exit_code == 0 and "Stopped and removed" in stopped.output
    assert not state.exists()
    missing = runner.invoke(cli.app, ["trafficsim", "stop", str(path)])
    assert missing.exit_code == 1 and "No such container" in missing.output
    state = state.with_name("owned-traffic")
    state.write_text("true")
    named = runner.invoke(cli.app, ["trafficsim", "stop", str(path), "--name", "owned-traffic"])
    assert named.exit_code == 0, named.output
    assert not state.exists()


@pytest.fixture
def managed_nurec_cli(tmp_path, monkeypatch, nurec_usdz_path):
    state, calls = docker_process_boundary(tmp_path, monkeypatch)
    state = state / "owned-nre"
    boundary = NuRecRpcBoundary(service_container=state).install(monkeypatch)
    path = nre_configuration(tmp_path / "configuration")
    spec = yaml.safe_load(path.read_text())
    spec["dataset"] = {"type": "NuRecDataset", "options": {"usdz_paths": [str(nurec_usdz_path)]}}
    spec["simulator"]["options"]["object_occlusion"] = "geometry"
    spec["simulator"]["options"]["min_track_duration_s"] = 0
    spec["simulator"]["options"]["renderer"]["options"]["cameras"] = ["front"]
    spec["attacks"]["probe"]["options"] = {"decision_steps": 1, "max_segments": 1}
    path.write_text(yaml.safe_dump(spec))
    return path, state, calls, boundary


def test_run_manages_local_services(managed_nurec_cli, monkeypatch):
    path, state, calls, boundary = managed_nurec_cli
    original = path.read_text()
    invalid_metadata = path.parent / "invalid-metadata.json"
    invalid_metadata.write_text(json.dumps({
        "format": "avsectester.metadata", "version": 1,
        "dataset_type": "user.data:OtherDataset", "segments": [],
    }))
    specification = yaml.safe_load(original)
    specification["metadata"] = {"load": str(invalid_metadata)}
    path.write_text(yaml.safe_dump(specification))
    rejected = runner.invoke(cli.app, ["run", str(path), "--nre-gpu", "2", "--nre-name", "owned-nre"])
    assert rejected.exit_code == 2 and "dataset_type" in rejected.output
    assert not calls.exists() and not state.exists()
    path.write_text(original)
    monkeypatch.setenv("AVSECTESTER_HOST_GPU", "all")
    invalid_service = runner.invoke(cli.app, ["run", str(path), "--nre-name", "owned-nre"])
    assert invalid_service.exit_code == 2 and "GPU" in invalid_service.output
    assert not calls.exists() and not state.exists()

    def failed_capture(request, timeout=None):
        raise RuntimeError("synthetic camera failure")

    monkeypatch.setattr(boundary, "render_rgb", failed_capture)
    failed = runner.invoke(cli.app, ["run", str(path), "--nre-gpu", "2", "--nre-name", "owned-nre"])
    assert failed.exit_code == 1, failed.output
    assert "synthetic camera failure" in failed.output and "Failed run" in failed.output
    assert "Starting NRE" in failed.output
    assert (path.parent / "results/evaluation/report.json").is_file()
    assert not state.exists()

    def interrupted_capture(request, timeout=None):
        raise KeyboardInterrupt("synthetic camera interruption")

    monkeypatch.setattr(boundary, "render_rgb", interrupted_capture)
    interrupted = runner.invoke(cli.app, [
        "run", str(path), "--resume", "--nre-gpu", "2", "--nre-name", "owned-nre"])
    assert interrupted.exit_code == 130, interrupted.output
    assert not state.exists()


def test_run_validation_errors(tmp_path):
    path = write_configuration(tmp_path)
    spec = yaml.safe_load(path.read_text())
    dataset = MemoryDataset()
    frames = dataset.frames("scene")
    source = tmp_path / "validation-segments.json"
    save_metadata((AttackSegment("scene", frames[0], frames[2]),), source,
                  dataset_type="tests.fakes:MemoryDataset")
    spec["metadata"] = {"load": str(source)}
    for failure, code, message in (("unexpected dataset failure", 1, "Preparation validation error:"),
                                    ("interrupt", 130, "Preparation validation interrupted.")):
        spec["dataset"]["options"] = {"scenarios_error": failure}
        path.write_text(yaml.safe_dump(spec))
        result = runner.invoke(cli.app, ["run", str(path)])
        assert result.exit_code == code, result.output
        assert message in result.output
        assert isinstance(result.exception, SystemExit)
