"""Build local NuRec renderer and traffic-service launch commands."""
from __future__ import annotations

from pathlib import Path

from avsectester.datasets.nurec import NuRecDataset
from avsectester.validation import require_positive_integer

from .renderer import NuRecRenderer
from .traffic import AlpaSimTraffic


DEFAULT_NRE_IMAGE = "nvcr.io/nvidia/nre/nre-ga:26.04"
DEFAULT_TRAFFIC_IMAGE = "avsectester:trafficsim"


def _local_port(endpoint: str, service: str) -> int:
    try:
        host, port_text = endpoint.rsplit(":", 1)
        host = host.removeprefix("[").removesuffix("]")
        port = int(port_text)
        if host not in {"127.0.0.1", "localhost", "0.0.0.0", "::1"}:
            raise ValueError(f"{service} can only be started for a local endpoint")
        if not 1 <= port <= 65535:
            raise ValueError("endpoint port must be between 1 and 65535")
    except (TypeError, AttributeError, ValueError) as error:
        raise ValueError(f"Invalid local {service} endpoint {endpoint!r}: {error}") from error
    return port


def _archive_mounts(dataset: NuRecDataset) -> list[str]:
    archives = [Path(path).resolve() for path in dataset.usdz_paths]
    filenames = [path.name for path in archives]
    if len(set(filenames)) != len(filenames):
        raise ValueError("NuRec archive filenames must be unique; duplicate served scene IDs are ambiguous")
    mounts = []
    for path in archives:
        mounts.extend(["-v", f"{path}:/mnt/nre-data/{path.name}:ro"])
    return mounts


def _docker_run(gpu: str | int, name: str) -> list[str]:
    return ["docker", "run", "-d", "--name", name, "--network", "host", "--gpus",
            f"device={gpu}", "-e", "HOME=/tmp"]


def renderer_command(dataset: NuRecDataset, renderer: NuRecRenderer, *, gpu: str | int,
                     name: str, image: str, cache_size: int) -> list[str]:
    """Return Docker argv for one local renderer serving resolved, read-only archives.

    Archive basenames must be unique. This function creates no directories, starts
    no processes and connects to no service.
    """
    if not isinstance(dataset, NuRecDataset) or not isinstance(renderer, NuRecRenderer):
        raise ValueError("NRE serve requires NuRecDataset and NuRecRenderer")
    require_positive_integer(cache_size, "cache_size")
    port = _local_port(renderer.endpoint, "NRE")
    command = _docker_run(gpu, name) + _archive_mounts(dataset)
    if renderer.probe_directory is not None:
        client_path = Path(renderer.probe_directory).resolve()
        server_path = renderer.server_probe_directory or str(client_path)
        command.extend(["-v", f"{client_path}:{server_path}:ro"])
    command.extend([
        "--entrypoint", "/app/internal/scripts/pycena/runtime/pycena_nrm_full", image,
        "serve-grpc", "--host=0.0.0.0", f"--port={port}",
        "--artifact-glob=/mnt/nre-data/*.usdz", f"--cache-size={cache_size}",
        "--enable-editing-actors",
    ])
    return command


def default_renderer_container_name(renderer: NuRecRenderer) -> str:
    """Return the default renderer container name without reading a dataset."""
    if not isinstance(renderer, NuRecRenderer):
        raise ValueError("NRE requires NuRecRenderer")
    return f"avsectester-nre-{_local_port(renderer.endpoint, 'NRE')}"


def traffic_command(dataset: NuRecDataset, traffic: AlpaSimTraffic, *, gpu: str | int,
                    name: str, image: str, models: str | Path) -> list[str]:
    """Return Docker argv for local CATK traffic with read-only scenes and weights.

    The configured endpoint must be local. Every archive is mounted separately
    under /mnt/nre-data, so its basename must be unique. This function has no
    filesystem, process or network side effects.
    """
    if not isinstance(dataset, NuRecDataset) or not isinstance(traffic, AlpaSimTraffic):
        raise ValueError("Traffic serve requires NuRecDataset and AlpaSimTraffic")
    port = _local_port(traffic.endpoint, "trafficsim")
    command = _docker_run(gpu, name) + _archive_mounts(dataset)
    command.extend([
        "-v", f"{Path(models).resolve()}:/mnt/trafficsim-models:ro",
        "-e", "OMP_NUM_THREADS=1", "-e", "MKL_NUM_THREADS=1",
        "--entrypoint", "catk_trafficsim_server", image,
        f"server.port={port}", "catk.loader.usdz_folder=/mnt/nre-data",
        "catk.loader.prediction_steps=15", "hydra.run.dir=/tmp/trafficsim-hydra",
    ])
    return command


def default_traffic_container_name(traffic: AlpaSimTraffic) -> str:
    """Return the default traffic container name without reading a dataset."""
    if not isinstance(traffic, AlpaSimTraffic):
        raise ValueError("Traffic commands require AlpaSimTraffic")
    return f"avsectester-trafficsim-{_local_port(traffic.endpoint, 'trafficsim')}"
