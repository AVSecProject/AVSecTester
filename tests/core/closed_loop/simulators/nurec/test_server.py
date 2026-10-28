"""Service commands and lifecycle use real configurations and a shared Docker boundary."""
import json
import re
import zipfile
from pathlib import Path

import pytest

from avsectester.closed_loop.simulators.nurec import AlpaSimTraffic, NuRecRenderer, NuRecSimulator
from avsectester.closed_loop.simulators.nurec.server import local_services
from avsectester.datasets.nurec import NuRecDataset
from tests.fakes import NuRecRpcBoundary, TrafficServiceBoundary, docker_process_boundary


def scene_copy(archive, destination, scene_id):
    """Copy a NuRec archive under a new path and scene ID."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as source, zipfile.ZipFile(destination, "w") as copy:
        for name in source.namelist():
            content = source.read(name)
            if name in {"rig_trajectories.json", "sequence_tracks.json"}:
                content = content.decode().replace('"scene"', f'"{scene_id}"')
                content = content.replace("@scene", f"@{scene_id}").encode()
            copy.writestr(name, content)
    return destination


@pytest.fixture
def docker_boundary(tmp_path, monkeypatch):
    return docker_process_boundary(tmp_path, monkeypatch)


@pytest.fixture
def service_configuration(nurec_usdz_path, docker_boundary, monkeypatch, tmp_path):
    state, _calls = docker_boundary
    boundary = NuRecRpcBoundary(service_container=state / "nre").install(monkeypatch)
    simulator = NuRecSimulator(NuRecRenderer(
        cameras=["front"], probe_directory=tmp_path / "probes",
        server_probe_directory="/server/assets"), time_step=.1, depth="none",
                               physics_update_mode="none", route_generator="none")
    monkeypatch.setenv("AVSECTESTER_HOST_GPU", "3")
    return NuRecDataset([nurec_usdz_path]), simulator, boundary


def test_local_services(service_configuration, docker_boundary, monkeypatch, tmp_path):
    dataset, simulator, boundary = service_configuration
    state, calls = docker_boundary
    archive = Path(dataset.usdz_paths[0])
    second = scene_copy(archive, tmp_path / "scenes" / "second-scene.usdz", "second-scene")
    dataset = NuRecDataset([archive, second])
    boundary.available_scenes = ("scene", "second-scene")
    messages = []
    with local_services(dataset, simulator, nre_name="nre", log=messages.append):
        assert (state / "nre").exists()
    assert not (state / "nre").exists()
    launch = next(json.loads(row) for row in calls.read_text().splitlines()
                  if json.loads(row)[0] == "run")
    label_index = launch.index("--label") + 1
    assert re.fullmatch(r"avsectester\.service-owner=[0-9a-f]{32}", launch[label_index])
    launch[label_index] = "avsectester.service-owner=<uuid>"
    assert launch == [
        "run", "--label", "avsectester.service-owner=<uuid>", "-d", "--name", "nre",
        "--network", "host", "--gpus", "device=3", "-e", "HOME=/tmp",
        "-v", f"{archive.resolve()}:/mnt/nre-data/{archive.name}:ro",
        "-v", f"{second.resolve()}:/mnt/nre-data/{second.name}:ro",
        "-v", f"{simulator.renderer.probe_directory.resolve()}:/server/assets:ro",
        "--entrypoint", "/app/internal/scripts/pycena/runtime/pycena_nrm_full",
        "nvcr.io/nvidia/nre/nre-ga:26.04", "serve-grpc", "--host=0.0.0.0",
        "--port=50051", "--artifact-glob=/mnt/nre-data/*.usdz", "--cache-size=2",
        "--enable-editing-actors",
    ]
    assert any("Starting NRE" in text for text in messages)
    assert any("Stopped and removed NRE" in text for text in messages)

    (state / "nre").write_text("true")
    messages.clear()
    with local_services(dataset, simulator, nre_name="nre", log=messages.append):
        pass
    assert (state / "nre").exists() and messages == ["Reusing NRE: endpoint=127.0.0.1:50051"]
    boundary.available_scenes = ("other-scene",)
    with pytest.raises(ValueError, match="missing dataset scenes: scene"):
        with local_services(dataset, simulator, nre_name="nre", log=messages.append):
            pass
    assert (state / "nre").exists()
    (state / "nre").unlink()
    boundary.available_scenes = ("scene", "second-scene")
    monkeypatch.setenv("AVSECTESTER_HOST_GPU", "all")
    with pytest.raises(ValueError, match="service GPU explicitly"):
        with local_services(dataset, simulator, nre_name="nre", log=messages.append):
            pass


def test_local_services_configuration_errors(service_configuration, docker_boundary, tmp_path):
    dataset, simulator, _boundary = service_configuration
    state, _calls = docker_boundary
    for endpoint, message in (
            ("remote.invalid:50051", "local endpoint"),
            ("localhost:0", "between 1 and 65535"),
            ("localhost:65536", "between 1 and 65535")):
        invalid = NuRecSimulator(
            NuRecRenderer(endpoint=endpoint, cameras=["front"]), time_step=.1, depth="none",
            physics_update_mode="none", route_generator="none")
        with pytest.raises(ValueError, match=message):
            with local_services(dataset, invalid, nre_name="nre"):
                pytest.fail("invalid service configuration started")

    archive = Path(dataset.usdz_paths[0])
    second = scene_copy(archive, tmp_path / "another" / archive.name, "second-scene")
    duplicate = NuRecDataset([archive, second])
    with pytest.raises(ValueError, match="filenames must be unique"):
        with local_services(duplicate, simulator, nre_name="nre"):
            pytest.fail("duplicate archive names accepted")
    (state / "nre").write_text("false")
    with pytest.raises(RuntimeError, match="already exists"):
        with local_services(dataset, simulator, nre_name="nre"):
            pytest.fail("existing container replaced")
    assert (state / "nre").exists()


def test_local_services_failure_and_cleanup(service_configuration, docker_boundary, monkeypatch):
    dataset, simulator, _boundary = service_configuration
    state, calls = docker_boundary
    for mode, message in (
            ("exit", "startup diagnostic"), ("timeout", "timed out"),
            ("run-fail", "injected docker run failure")):
        monkeypatch.setenv("DOCKER_MODE", mode)
        with pytest.raises(RuntimeError, match=message):
            with local_services(dataset, simulator, nre_name="nre", timeout_s=.001, log=lambda text: None):
                pass
        assert not (state / "nre").exists()
    monkeypatch.setenv("DOCKER_MODE", "interrupt")
    with pytest.raises(KeyboardInterrupt):
        with local_services(dataset, simulator, nre_name="nre", log=lambda text: None):
            pytest.fail("startup interruption ignored")
    assert not (state / "nre").exists()
    monkeypatch.setenv("DOCKER_MODE", "race")
    with pytest.raises(RuntimeError, match="No container owned by this invocation"):
        with local_services(dataset, simulator, nre_name="nre", log=lambda text: None):
            pytest.fail("ownership race ignored")
    assert json.loads((state / "nre").read_text())["owner"] == "someone-else"
    (state / "nre").unlink()
    monkeypatch.setenv("DOCKER_MODE", "ready")
    for error_type in (RuntimeError, KeyboardInterrupt):
        with pytest.raises(error_type, match="experiment failed"):
            with local_services(dataset, simulator, nre_name="nre", log=lambda text: None):
                raise error_type("experiment failed")
        assert not (state / "nre").exists()

    (state / "nre").write_text("true")
    with pytest.raises(RuntimeError, match="experiment failed"):
        with local_services(dataset, simulator, nre_name="nre", log=lambda text: None):
            raise RuntimeError("experiment failed")
    assert (state / "nre").exists()
    (state / "nre").unlink()

    messages = []
    with local_services(dataset, simulator, nre_name="nre", log=messages.append):
        monkeypatch.setenv("DOCKER_MODE", "stop-fail")
    assert any("Unable to stop NRE" in text for text in messages)
    monkeypatch.setenv("DOCKER_MODE", "ready")
    with pytest.raises(RuntimeError, match="original") as caught:
        with local_services(dataset, simulator, nre_name="nre", log=messages.append):
            monkeypatch.setenv("DOCKER_MODE", "stop-fail")
            raise RuntimeError("original")
    assert any("Unable to stop NRE" in note for note in caught.value.__notes__)
    monkeypatch.setenv("DOCKER_MODE", "ready")

    with local_services(dataset, simulator, nre_name="nre", log=messages.append):
        # Name replacement is an external Docker race; cleanup must retain the replacement.
        (state / "nre").write_text(json.dumps({"id": "foreign-id", "owner": "other", "running": True}))
    assert json.loads((state / "nre").read_text())["id"] == "foreign-id"


def test_local_services_with_traffic(nurec_usdz_path, docker_boundary, tmp_path, monkeypatch):
    pytest.importorskip("alpasim_grpc.v0.traffic_pb2")
    state, calls = docker_boundary
    monkeypatch.setenv("AVSECTESTER_HOST_GPU", "1")
    with TrafficServiceBoundary(service_container=state / "traffic") as service:
        NuRecRpcBoundary(service_container=state / "nre").install(
            monkeypatch, passthrough_endpoints=(service.endpoint,))
        dataset = NuRecDataset([nurec_usdz_path])
        simulator = NuRecSimulator(NuRecRenderer(cameras=["front"]), time_step=.1, depth="none",
                                   physics_update_mode="none", route_generator="none",
                                   traffic=AlpaSimTraffic(endpoint=service.endpoint))
        with pytest.raises(ValueError, match="--trafficsim-models"):
            with local_services(dataset, simulator, nre_name="nre", trafficsim_name="traffic",
                                log=lambda text: None):
                pass
        assert not (state / "nre").exists()

        models = tmp_path / "models"
        models.mkdir()
        messages = []

        def report(text):
            if "Stopped and removed trafficsim" in text:
                assert (state / "nre").exists()
            messages.append(text)

        with local_services(dataset, simulator, nre_name="nre", trafficsim_name="traffic",
                            trafficsim_models=models, log=report):
            assert (state / "nre").exists() and (state / "traffic").exists()
        assert not (state / "nre").exists() and not (state / "traffic").exists()
        assert any("Starting trafficsim" in text for text in messages)
        assert any("Stopped and removed trafficsim" in text for text in messages)
        launch = next(json.loads(row) for row in calls.read_text().splitlines()
                      if json.loads(row)[0] == "run" and "catk_trafficsim_server" in row)
        archive = Path(dataset.usdz_paths[0])
        label_index = launch.index("--label") + 1
        launch[label_index] = "avsectester.service-owner=<uuid>"
        assert launch == [
            "run", "--label", "avsectester.service-owner=<uuid>", "-d", "--name", "traffic",
            "--network", "host", "--gpus", "device=1", "-e", "HOME=/tmp",
            "-v", f"{archive.resolve()}:/mnt/nre-data/{archive.name}:ro",
            "-v", f"{models.resolve()}:/mnt/trafficsim-models:ro",
            "-e", "OMP_NUM_THREADS=1", "-e", "MKL_NUM_THREADS=1",
            "--entrypoint", "catk_trafficsim_server", "avsectester:trafficsim",
            f"server.port={service.endpoint.rsplit(':', 1)[1]}",
            "catk.loader.usdz_folder=/mnt/nre-data", "catk.loader.prediction_steps=15",
            "hydra.run.dir=/tmp/trafficsim-hydra",
        ]

        (state / "nre").write_text("true")
        with local_services(dataset, simulator, nre_name="nre", trafficsim_name="traffic",
                            trafficsim_models=models, log=report):
            assert (state / "traffic").exists()
        assert (state / "nre").exists() and not (state / "traffic").exists()

        (state / "traffic").write_text("true")
        messages.clear()
        with local_services(dataset, simulator, nre_name="nre", trafficsim_name="traffic",
                            log=messages.append):
            pass
        assert all(text.startswith("Reusing") for text in messages)
        assert (state / "nre").exists() and (state / "traffic").exists()
