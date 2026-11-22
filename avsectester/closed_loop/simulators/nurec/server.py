"""Build, start, reuse and stop local NuRec renderer and traffic-service containers."""
from __future__ import annotations

from pathlib import Path
from collections.abc import Callable, Iterator
from contextlib import contextmanager
import os
import re
from typing import Any

from avsectester.closed_loop.docker_containers import (
    remove_exited_container, start_container, stop_container,
)
from avsectester.closed_loop.simulators.base import Simulator
from avsectester.datasets.base import Dataset
from avsectester.datasets.nurec import NuRecDataset
from avsectester.validation import require_positive_integer, require_positive_number

from .renderer import NuRecRenderer
from .traffic import AlpaSimTraffic


DEFAULT_NRE_IMAGE = "nvcr.io/nvidia/nre/nre-ga:26.04"
DEFAULT_TRAFFIC_IMAGE = "avsectester:trafficsim"
# Host GPU index of the experiment container, set by the nurec-experiment Compose service.
# NVIDIA_VISIBLE_DEVICES cannot carry it: CDI runtimes replace it with "void" in the container.
HOST_GPU_VARIABLE = "AVSECTESTER_HOST_GPU"


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
    if renderer.asset_directory is not None:
        client_path = Path(renderer.asset_directory).resolve()
        server_path = renderer.server_asset_directory or str(client_path)
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


def _service_gpu(value: str | int | None) -> str:
    chosen = os.environ.get(HOST_GPU_VARIABLE) if value is None else str(value)
    if chosen is None or re.fullmatch(r"[0-9]+", chosen) is None:
        raise ValueError(f"Specify a service GPU explicitly, or set {HOST_GPU_VARIABLE} "
                         "to a single nonnegative integer host GPU index")
    return chosen


def _check_scenes(service: str, endpoint: str, served: tuple[str, ...], expected: set[str]) -> None:
    missing = expected - set(served)
    if missing:
        raise ValueError(f"The {service} at {endpoint} serves other scenes and is missing dataset scenes: "
                         + ", ".join(sorted(missing)))


@contextmanager
def local_services(
    dataset: Dataset, simulator: Simulator, *, nre_gpu: str | int | None = None,
    nre_name: str | None = None, trafficsim_gpu: str | int | None = None,
    trafficsim_models: str | Path | None = None, trafficsim_name: str | None = None,
    timeout_s: float = 600., log: Callable[[str], Any] = print,
) -> Iterator[None]:
    """Reuse compatible services or own local NRE/CATK containers for one experiment.

    Only services started here are stopped, in reverse order, on normal exit or failure.
    Cleanup errors become notes on the original exception; after success they are logged
    without replacing the experiment's result.
    """
    require_positive_number(timeout_s, "timeout_s")
    renderer = getattr(simulator, "renderer", None)
    traffic = getattr(simulator, "traffic", None)
    if not isinstance(renderer, NuRecRenderer) and traffic is None:
        yield
        return
    if not isinstance(dataset, NuRecDataset):
        raise ValueError("Automatic NuRec services require NuRecDataset")
    expected = {scenario.id for scenario in dataset.scenarios()}
    owned: list[tuple[str, str, str]] = []
    original: BaseException | None = None
    try:
        if isinstance(renderer, NuRecRenderer):
            served = renderer.served_scenes()
            if served is not None:
                _check_scenes("NRE", renderer.endpoint, served, expected)
                log(f"Reusing NRE: endpoint={renderer.endpoint}")
            else:
                name = nre_name or default_renderer_container_name(renderer)
                gpu = _service_gpu(nre_gpu)
                command = renderer_command(dataset, renderer, gpu=gpu, name=name,
                                           image=DEFAULT_NRE_IMAGE, cache_size=2)

                def prepare() -> None:
                    if renderer.asset_directory is not None:
                        Path(renderer.asset_directory).mkdir(parents=True, exist_ok=True)

                if remove_exited_container(name):
                    log(f"Removed exited NRE container: {name}")
                identity = start_container(command, name=name, ready_log="Serving on", timeout_s=timeout_s,
                                prepare=prepare,
                                log=lambda message: log(f"Starting NRE: endpoint={renderer.endpoint}, GPU={gpu}, container={name}"))
                owned.append(("NRE", name, identity))
        if traffic is not None:
            if not isinstance(traffic, AlpaSimTraffic):
                raise ValueError("Automatic traffic services require AlpaSimTraffic")
            served = traffic.served_scenes()
            if served is not None:
                _check_scenes("trafficsim", traffic.endpoint, served, expected)
                log(f"Reusing trafficsim: endpoint={traffic.endpoint}")
            else:
                if trafficsim_models is None:
                    raise ValueError("Starting trafficsim requires --trafficsim-models")
                models = Path(trafficsim_models)
                if not models.is_dir():
                    raise ValueError(f"Traffic model directory does not exist: {models}")
                name = trafficsim_name or default_traffic_container_name(traffic)
                gpu = _service_gpu(trafficsim_gpu)
                command = traffic_command(dataset, traffic, gpu=gpu, name=name,
                                          image=DEFAULT_TRAFFIC_IMAGE, models=models)
                if remove_exited_container(name):
                    log(f"Removed exited trafficsim container: {name}")
                identity = start_container(command, name=name, ready_log="Starting CATK traffic gRPC server on",
                                timeout_s=timeout_s,
                                readiness=lambda remaining: traffic.get_metadata(timeout_s=min(1., remaining)),
                                log=lambda message: log(f"Starting trafficsim: endpoint={traffic.endpoint}, GPU={gpu}, container={name}"))
                owned.append(("trafficsim", name, identity))
        yield
    except BaseException as error:
        original = error
        raise
    finally:
        for service, name, identity in reversed(owned):
            try:
                stop_container(identity)
            except BaseException as cleanup_error:
                message = f"Unable to stop {service} container {name}: {cleanup_error}"
                if original is not None:
                    original.add_note(message)
                log(message)
            else:
                log(f"Stopped and removed {service} container: {name}")


__all__ = ['local_services']
