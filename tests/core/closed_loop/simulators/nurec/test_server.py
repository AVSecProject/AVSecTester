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
