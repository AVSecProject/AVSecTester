"""NRE launch argv is derived from public dataset and renderer configuration."""
from dataclasses import replace

import pytest

from avsectester.closed_loop.simulators.nurec import NuRecRenderer
from avsectester.closed_loop.simulators.nurec.server import default_renderer_container_name, renderer_command
from avsectester.datasets.nurec import NuRecDataset


def test_renderer_command(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    first, second = root / "first.usdz", root / "second.usdz"
    first.touch()
    second.touch()
    dataset = NuRecDataset(dataroot=root)
    probes = tmp_path / "probes"
    renderer = NuRecRenderer(endpoint="localhost:50152", probe_directory=probes,
                             server_probe_directory="/mnt/probes")
    command = renderer_command(dataset, renderer, gpu=2, name="test-nre", image="nre:test", cache_size=1)
    assert command == [
        "docker", "run", "-d", "--name", "test-nre", "--network", "host", "--gpus", "device=2",
        "-e", "HOME=/tmp", "-v", f"{first.resolve()}:/mnt/nre-data/first.usdz:ro",
        "-v", f"{second.resolve()}:/mnt/nre-data/second.usdz:ro",
        "-v", f"{probes.resolve()}:/mnt/probes:ro", "--entrypoint",
        "/app/internal/scripts/pycena/runtime/pycena_nrm_full", "nre:test", "serve-grpc",
        "--host=0.0.0.0", "--port=50152", "--artifact-glob=/mnt/nre-data/*.usdz",
        "--cache-size=1", "--enable-editing-actors",
    ]
    assert not probes.exists()
    same_path = renderer_command(dataset, replace(renderer, server_probe_directory=None),
                                 gpu="0", name="same", image="nre:test", cache_size=2)
    assert f"{probes.resolve()}:{probes.resolve()}:ro" in same_path
    no_probe = renderer_command(dataset, NuRecRenderer(endpoint="[::1]:50051"),
                                gpu=0, name="ipv6", image="nre:test", cache_size=2)
    assert no_probe.count("-v") == 2 and "--port=50051" in no_probe
    other = tmp_path / "other" / "first.usdz"
    other.parent.mkdir()
    other.touch()
    with pytest.raises(ValueError, match="unique"):
        renderer_command(NuRecDataset([first, other]), renderer,
                         gpu=0, name="duplicate", image="nre:test", cache_size=2)
    with pytest.raises(ValueError, match="local"):
        renderer_command(dataset, replace(renderer, endpoint="remote.example:50051"),
                         gpu=0, name="remote", image="nre:test", cache_size=2)
    with pytest.raises(ValueError, match="NuRecDataset"):
        renderer_command(object(), renderer, gpu=0, name="invalid", image="nre:test", cache_size=2)


def test_default_renderer_container_name():
    assert default_renderer_container_name(NuRecRenderer(endpoint="localhost:50152")) == "avsectester-nre-50152"
    assert default_renderer_container_name(NuRecRenderer(endpoint="[::1]:50051")) == "avsectester-nre-50051"
    with pytest.raises(ValueError, match="local"):
        default_renderer_container_name(NuRecRenderer(endpoint="remote.example:50051"))


def test_traffic_command(tmp_path):
    from avsectester.closed_loop.simulators.nurec import AlpaSimTraffic
    from avsectester.closed_loop.simulators.nurec.server import traffic_command

    root = tmp_path / "data"
    root.mkdir()
    first, second = root / "first.usdz", root / "second.usdz"
    first.touch()
    second.touch()
    models = tmp_path / "models"
    dataset = NuRecDataset(dataroot=root)
    traffic = AlpaSimTraffic(endpoint="localhost:6207")
    command = traffic_command(dataset, traffic, gpu=2, name="test-traffic",
                              image="traffic:test", models=models)
    assert command == [
        "docker", "run", "-d", "--name", "test-traffic", "--network", "host", "--gpus", "device=2",
        "-e", "HOME=/tmp", "-v", f"{first.resolve()}:/mnt/nre-data/first.usdz:ro",
        "-v", f"{second.resolve()}:/mnt/nre-data/second.usdz:ro",
        "-v", f"{models.resolve()}:/mnt/trafficsim-models:ro", "-e", "OMP_NUM_THREADS=1",
        "-e", "MKL_NUM_THREADS=1", "--entrypoint", "catk_trafficsim_server", "traffic:test",
        "server.port=6207", "catk.loader.usdz_folder=/mnt/nre-data",
        "catk.loader.prediction_steps=15", "hydra.run.dir=/tmp/trafficsim-hydra",
    ]
    assert not models.exists()
    duplicate = tmp_path / "other" / first.name
    duplicate.parent.mkdir()
    duplicate.touch()
    with pytest.raises(ValueError, match="unique"):
        traffic_command(NuRecDataset([first, duplicate]), traffic, gpu=0,
                        name="duplicate", image="traffic:test", models=models)
    with pytest.raises(ValueError, match="local"):
        traffic_command(dataset, AlpaSimTraffic(endpoint="remote.example:6200"), gpu=0,
                        name="remote", image="traffic:test", models=models)
    with pytest.raises(ValueError, match="NuRecDataset"):
        traffic_command(object(), traffic, gpu=0, name="invalid", image="traffic:test", models=models)


def test_default_traffic_container_name():
    from avsectester.closed_loop.simulators.nurec import AlpaSimTraffic
    from avsectester.closed_loop.simulators.nurec.server import default_traffic_container_name

    assert default_traffic_container_name(AlpaSimTraffic()) == "avsectester-trafficsim-6200"
    assert default_traffic_container_name(AlpaSimTraffic(endpoint="[::1]:6207")) == "avsectester-trafficsim-6207"
    with pytest.raises(ValueError, match="local"):
        default_traffic_container_name(AlpaSimTraffic(endpoint="remote.example:6200"))


@pytest.fixture
def docker_boundary(tmp_path, monkeypatch):
    """A deterministic docker executable replacing only the process boundary."""
    import os
    import sys

    directory = tmp_path / "bin"
    directory.mkdir()
    state = tmp_path / "containers"
    state.mkdir()
    calls = tmp_path / "docker.jsonl"
    docker = directory / "docker"
    docker.write_text(f"#!{sys.executable}\n" + r'''
import json, os, sys, uuid
from pathlib import Path
args = sys.argv[1:]
with open(os.environ["DOCKER_CALLS"], "a") as stream:
    stream.write(json.dumps(args) + "\n")
root = Path(os.environ["DOCKER_STATE"])
mode = os.environ.get("DOCKER_MODE", "ready")
def read(p):
    try:return json.loads(p.read_text())
    except (ValueError, FileNotFoundError):return None
def locate(key):
    direct=root/key
    if direct.exists():return direct
    for p in root.iterdir():
        data=read(p)
        if isinstance(data,dict) and data.get("id")==key:return p
    return direct
if args[0] == "run":
    name=args[args.index("--name")+1]
    label=args[args.index("--label")+1].split("=",1)[1] if "--label" in args else ""
    identity="id-"+uuid.uuid4().hex
    (root/name).write_text(json.dumps({"id":identity,"owner":"someone-else" if mode=="race" else label,"running":mode!="exit"}))
    print(identity)
    raise SystemExit(1 if mode in ("launch-fail","race") else 0)
p=locate(args[-1]);data=read(p)
if isinstance(data,bool):data={"id":"seed-"+p.name,"owner":"","running":data}
if args[0] == "inspect":
    if data is None:raise SystemExit(1)
    fmt=args[args.index("--format")+1] if "--format" in args else ""
    if ".Config.Labels" in fmt:print(data["id"]+" "+data["owner"])
    elif ".State.Running" in fmt:print(str(data["running"]).lower())
    else:print(json.dumps(data))
elif args[0] == "logs":
    print("Serving on\nStarting CATK traffic gRPC server on" if mode == "ready" else "startup diagnostic")
elif args[0] == "stop":
    if data is None:
        print("No such container", file=sys.stderr)
        raise SystemExit(1)
    data["running"]=False;p.write_text(json.dumps(data))
    if mode == "stop-fail":
        print("stop failed", file=sys.stderr)
        raise SystemExit(1)
elif args[0] == "rm":
    if data is None:raise SystemExit(1)
    p.unlink()
''')
    docker.chmod(0o755)
    monkeypatch.setenv("PATH", str(directory) + os.pathsep + os.environ.get("PATH", ""))
    monkeypatch.setenv("DOCKER_STATE", str(state))
    monkeypatch.setenv("DOCKER_CALLS", str(calls))
    return state, calls


def test_start_container(docker_boundary, monkeypatch):
    from avsectester.closed_loop.simulators.nurec.server import start_container, stop_container

    state, _calls = docker_boundary
    messages = []
    prepared = state / "prepared"
    command = ["docker", "run", "--name", "owned"]
    identity = start_container(command, name="owned", ready_log="Serving on", timeout_s=1.,
                               prepare=lambda: prepared.touch(), log=messages.append)
    assert identity.startswith("id-")
    assert prepared.exists() and (state / "owned").exists()
    assert messages == ["Starting container: owned"]
    with pytest.raises(RuntimeError, match="already exists"):
        start_container(command, name="owned", ready_log="Serving on", timeout_s=1., log=messages.append)
    stop_container("owned")
    monkeypatch.setenv("DOCKER_MODE", "race")
    with pytest.raises(RuntimeError, match="No container owned by this invocation"):
        start_container(command, name="owned", ready_log="Serving on", timeout_s=1., log=messages.append)
    assert (state / "owned").exists()
    import json
    race_calls = [json.loads(line) for line in _calls.read_text().splitlines()]
    assert race_calls[-2][0] == "run" and race_calls[-1][0] == "inspect"
    stop_container("owned")
    monkeypatch.setenv("DOCKER_MODE", "exit")
    with pytest.raises(RuntimeError, match="startup diagnostic"):
        start_container(command, name="owned", ready_log="Serving on", timeout_s=1., log=messages.append)
    assert not (state / "owned").exists()
    monkeypatch.setenv("DOCKER_MODE", "timeout")
    with pytest.raises(RuntimeError, match="timed out"):
        start_container(command, name="owned", ready_log="Serving on", timeout_s=.001, log=messages.append)
    assert not (state / "owned").exists()
    monkeypatch.setenv("DOCKER_MODE", "ready")

    def interrupt(remaining):
        raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        start_container(command, name="owned", ready_log="Serving on", timeout_s=1.,
                        readiness=interrupt, log=messages.append)
    assert not (state / "owned").exists()


def test_stop_container(docker_boundary):
    from avsectester.closed_loop.simulators.nurec.server import stop_container

    state, _calls = docker_boundary
    (state / "owned").write_text("true")
    stop_container("owned")
    assert not (state / "owned").exists()
    with pytest.raises(RuntimeError, match="No such container"):
        stop_container("owned")


@pytest.fixture
def service_configuration(nurec_usdz_path, docker_boundary, monkeypatch):
    from tests.fakes import NuRecRpcBoundary, NuRecRpcError, ConstantAVStack, MinimalAttack
    from avsectester.closed_loop.avstacks.driving_adapter import DrivingAdapter
    from avsectester.closed_loop.simulators.nurec import NuRecSimulator
    from avsectester.experiment import ExperimentConfiguration

    state, _calls = docker_boundary

    class RendererService(NuRecRpcBoundary):
        scenes = ("scene",)

        def get_available_scenes(self, request, timeout=None):
            if not (state / "nre").exists():
                raise NuRecRpcError("UNAVAILABLE", "not running")
            from types import SimpleNamespace
            return SimpleNamespace(scene_ids=self.scenes)

    boundary = RendererService().install(monkeypatch)
    renderer = NuRecRenderer(cameras=["front"])
    configuration = ExperimentConfiguration(
        NuRecDataset([nurec_usdz_path]),
        NuRecSimulator(renderer, time_step=.1, depth="none", physics_update_mode="none", route_generator="none"),
        DrivingAdapter(lambda seed: ConstantAVStack()), attacks={"test": MinimalAttack()},
    )
    monkeypatch.setenv("AVSECTESTER_HOST_GPU", "3")
    return configuration, boundary


def test_local_services(service_configuration, docker_boundary, monkeypatch):
    import json
    from avsectester.closed_loop.simulators.nurec.server import local_services

    configuration, boundary = service_configuration
    state, calls = docker_boundary
    messages = []
    with local_services(configuration.dataset, configuration.simulator, nre_name="nre", log=messages.append):
        assert (state / "nre").exists()
    assert not (state / "nre").exists()
    recorded = [json.loads(line) for line in calls.read_text().splitlines()]
    launch = next(call for call in recorded if call[0] == "run")
    assert "device=3" in launch
    assert any("Starting NRE" in text for text in messages)
    assert any("Stopped and removed NRE" in text for text in messages)
    (state / "nre").write_text("true")
    messages.clear()
    with local_services(configuration.dataset, configuration.simulator, nre_name="nre", log=messages.append):
        pass
    assert (state / "nre").exists() and messages == ["Reusing NRE: endpoint=127.0.0.1:50051"]
    boundary.scenes = ("other-scene",)
    with pytest.raises(ValueError, match="missing dataset scenes: scene"):
        with local_services(configuration.dataset, configuration.simulator, nre_name="nre", log=messages.append):
            pass
    assert (state / "nre").exists()
    (state / "nre").unlink()
    monkeypatch.setenv("AVSECTESTER_HOST_GPU", "all")
    with pytest.raises(ValueError, match="single nonnegative integer"):
        with local_services(configuration.dataset, configuration.simulator, nre_name="nre", log=messages.append):
            pass


@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt])
def test_local_services_failure(service_configuration, docker_boundary, error_type):
    from avsectester.closed_loop.simulators.nurec.server import local_services

    configuration, _boundary = service_configuration
    state, _calls = docker_boundary
    with pytest.raises(error_type):
        with local_services(configuration.dataset, configuration.simulator, nre_name="nre", log=lambda text: None):
            raise error_type("experiment stopped")
    assert not (state / "nre").exists()
    (state / "nre").write_text("true")
    with pytest.raises(error_type):
        with local_services(configuration.dataset, configuration.simulator, nre_name="nre", log=lambda text: None):
            raise error_type("experiment stopped")
    assert (state / "nre").exists()


def test_local_services_cleanup(service_configuration, docker_boundary, monkeypatch):
    from avsectester.closed_loop.simulators.nurec.server import local_services

    configuration, _boundary = service_configuration
    messages = []
    with local_services(configuration.dataset, configuration.simulator, nre_name="nre", log=messages.append):
        monkeypatch.setenv("DOCKER_MODE", "stop-fail")
    assert any("Unable to stop NRE" in text for text in messages)
    monkeypatch.setenv("DOCKER_MODE", "ready")
    with pytest.raises(ValueError, match="experiment error") as caught:
        with local_services(configuration.dataset, configuration.simulator, nre_name="nre", log=messages.append):
            monkeypatch.setenv("DOCKER_MODE", "stop-fail")
            raise ValueError("experiment error")
    assert any("Unable to stop NRE" in note for note in caught.value.__notes__)
    monkeypatch.setenv("DOCKER_MODE", "ready")
    import json
    state, _calls = docker_boundary
    with local_services(configuration.dataset, configuration.simulator, nre_name="nre", log=messages.append):
        # Another process removes our instance and creates a different container with its name.
        (state / "nre").write_text(json.dumps({"id": "foreign-id", "owner": "other", "running": True}))
    assert json.loads((state / "nre").read_text())["id"] == "foreign-id"


def test_local_traffic_services(nurec_usdz_path, docker_boundary, tmp_path, monkeypatch):
    grpc = pytest.importorskip("grpc")
    pytest.importorskip("alpasim_grpc.v0.traffic_pb2")
    from tests.fakes import CAMERA, ConstantAVStack, MinimalAttack, NuRecImageBoundary, TrafficServiceBoundary
    from avsectester.closed_loop.avstacks.driving_adapter import DrivingAdapter
    from avsectester.closed_loop.simulators.nurec import AlpaSimTraffic, NuRecSimulator
    from avsectester.closed_loop.simulators.nurec.server import local_services
    from avsectester.experiment import ExperimentConfiguration

    state, _calls = docker_boundary
    monkeypatch.setenv("AVSECTESTER_HOST_GPU", "2")
    with TrafficServiceBoundary() as service:
        configuration = ExperimentConfiguration(
            NuRecDataset([nurec_usdz_path]),
            NuRecSimulator(NuRecImageBoundary(CAMERA), time_step=.1, depth="none",
                           physics_update_mode="none", route_generator="none",
                           traffic=AlpaSimTraffic(endpoint=service.endpoint)),
            DrivingAdapter(lambda seed: ConstantAVStack()), attacks={"test": MinimalAttack()},
        )
        messages = []
        with local_services(configuration.dataset, configuration.simulator, trafficsim_name="traffic", log=messages.append):
            pass
        assert messages == [f"Reusing trafficsim: endpoint={service.endpoint}"]
        assert not (state / "traffic").exists()
        service.rpc_errors["get_available_scenes"] = (grpc.StatusCode.UNAVAILABLE, "offline")
        with pytest.raises(ValueError, match="--trafficsim-models"):
            with local_services(configuration.dataset, configuration.simulator, trafficsim_name="traffic", log=messages.append):
                pass
        models = tmp_path / "models"
        models.mkdir()
        with local_services(configuration.dataset, configuration.simulator, trafficsim_name="traffic", trafficsim_models=models,
                            log=messages.append):
            assert (state / "traffic").exists()
        assert not (state / "traffic").exists()
        assert any("Starting trafficsim" in line for line in messages)
        assert any("Stopped and removed trafficsim" in line for line in messages)


def test_local_services_order(nurec_usdz_path, docker_boundary, tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace

    grpc = pytest.importorskip("grpc")
    sensor = pytest.importorskip("alpasim_grpc.v0.sensorsim_pb2_grpc")
    from tests.fakes import ConstantAVStack, MinimalAttack, TrafficServiceBoundary
    from avsectester.closed_loop.avstacks.driving_adapter import DrivingAdapter
    from avsectester.closed_loop.simulators.nurec import AlpaSimTraffic, NuRecSimulator
    from avsectester.closed_loop.simulators.nurec.server import local_services
    from avsectester.experiment import ExperimentConfiguration

    state, calls = docker_boundary
    monkeypatch.setenv("AVSECTESTER_HOST_GPU", "1")

    class Unavailable(grpc.RpcError):
        def code(self):
            return grpc.StatusCode.UNAVAILABLE

        def details(self):
            return "renderer not running"

    def scenes(request, timeout=None):
        if not (state / "nre").exists():
            raise Unavailable()
        return SimpleNamespace(scene_ids=["scene"])

    monkeypatch.setattr(sensor, "SensorsimServiceStub", lambda channel: SimpleNamespace(get_available_scenes=scenes))
    with TrafficServiceBoundary() as service:
        service.rpc_errors["get_available_scenes"] = (grpc.StatusCode.UNAVAILABLE, "traffic not running")
        configuration = ExperimentConfiguration(
            NuRecDataset([nurec_usdz_path]),
            NuRecSimulator(NuRecRenderer(cameras=["front"]), time_step=.1, depth="none",
                           physics_update_mode="none", route_generator="none",
                           traffic=AlpaSimTraffic(endpoint=service.endpoint)),
            DrivingAdapter(lambda seed: ConstantAVStack()), attacks={"test": MinimalAttack()},
        )
        models = tmp_path / "models"
        models.mkdir()
        with local_services(configuration.dataset, configuration.simulator, nre_name="nre", trafficsim_name="traffic",
                            trafficsim_models=models, log=lambda text: None):
            assert (state / "nre").exists() and (state / "traffic").exists()
            owned_ids = [json.loads((state / key).read_text())["id"] for key in ("traffic", "nre")]
        recorded = [json.loads(line) for line in calls.read_text().splitlines()]
        assert [call[-1] for call in recorded if call[0] == "stop"] == owned_ids
        (state / "nre").write_text("true")
        calls.write_text("")
        with local_services(configuration.dataset, configuration.simulator, nre_name="nre", trafficsim_name="traffic",
                            trafficsim_models=models, log=lambda text: None):
            traffic_id = json.loads((state / "traffic").read_text())["id"]
        recorded = [json.loads(line) for line in calls.read_text().splitlines()]
        assert [call[-1] for call in recorded if call[0] == "stop"] == [traffic_id]
        assert (state / "nre").exists() and not (state / "traffic").exists()
