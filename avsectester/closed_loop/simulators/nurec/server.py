"""Build a local NRE launch command from configured dataset and renderer paths."""
from __future__ import annotations

from pathlib import Path

from avsectester.datasets.nurec import NuRecDataset
from avsectester.validation import require_positive_integer

from .renderer import NuRecRenderer


DEFAULT_NRE_IMAGE = "nvcr.io/nvidia/nre/nre-ga:26.04"


def renderer_command(dataset: NuRecDataset, renderer: NuRecRenderer, *, gpu: str | int,
                     name: str, image: str, cache_size: int) -> list[str]:
    """Return Docker argv for one local renderer serving the dataset's resolved archives.

    Each archive has a separate read-only mount. Archive basenames must be unique because
    NRE's artifact glob would otherwise address different scenes through the same path.
    This function creates no directories, starts no processes and connects to no service.
    """
    if not isinstance(dataset, NuRecDataset) or not isinstance(renderer, NuRecRenderer):
        raise ValueError("NRE serve requires NuRecDataset and NuRecRenderer")
    require_positive_integer(cache_size, "cache_size")
    port = _endpoint_port(renderer)
    archives = [Path(path).resolve() for path in dataset.usdz_paths]
    filenames = [path.name for path in archives]
    if len(set(filenames)) != len(filenames):
        raise ValueError("NRE archive filenames must be unique; duplicate served scene IDs are ambiguous")
    command = ["docker", "run", "-d", "--name", name, "--network", "host", "--gpus",
               f"device={gpu}", "-e", "HOME=/tmp"]
    for path in archives:
        command.extend(["-v", f"{path}:/mnt/nre-data/{path.name}:ro"])
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


def default_container_name(renderer: NuRecRenderer) -> str:
    """Return the default local container name without reading a dataset."""
    if not isinstance(renderer, NuRecRenderer):
        raise ValueError("NRE requires NuRecRenderer")
    return f"avsectester-nre-{_endpoint_port(renderer)}"


def _endpoint_port(renderer: NuRecRenderer) -> int:
    try:
        host, port_text = renderer.endpoint.rsplit(":", 1)
        host = host.removeprefix("[").removesuffix("]")
        port = int(port_text)
        if host not in {"127.0.0.1", "localhost", "0.0.0.0", "::1"}:
            raise ValueError("NRE can only be started for a local endpoint")
        if not 1 <= port <= 65535:
            raise ValueError("NRE endpoint port must be between 1 and 65535")
    except (TypeError, AttributeError, ValueError) as error:
        raise ValueError(f"Invalid local NRE endpoint {renderer.endpoint!r}: {error}") from error
    return port
