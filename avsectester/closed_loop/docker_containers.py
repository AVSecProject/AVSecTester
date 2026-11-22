"""Manage local service containers through the Docker command-line interface."""

from __future__ import annotations

from collections.abc import Callable
import os
import re
import shutil
import subprocess
import time
from typing import Any, NoReturn
import uuid

from avsectester.validation import require_positive_number


def _docker(arguments: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["docker", *arguments], capture_output=True, text=True, check=False)


def _require_docker() -> None:
    if shutil.which("docker") is None:
        raise RuntimeError("Docker is not available on PATH")


def stop_container(name: str) -> None:
    """Stop and remove one explicitly named container, reporting either failure."""
    _require_docker()
    errors = []
    for operation in ("stop", "rm"):
        response = _docker([operation, name])
        if response.returncode:
            errors.append(response.stderr.strip() or response.stdout.strip()
                          or f"docker {operation} failed for container {name}")
    if errors:
        raise RuntimeError("; ".join(errors))


_OWNER_LABEL = "avsectester.service-owner"
_USER_LABEL = "avsectester.service-user"


def _owned_container(name: str, owner: str) -> str | None:
    # Read identity and label together so replacement of a name cannot mix two containers.
    response = _docker(["inspect", "--format",
                        '{{.Id}} {{ index .Config.Labels "avsectester.service-owner" }}', name])
    identity = response.stdout.strip().split(maxsplit=1)
    return identity[0] if response.returncode == 0 and len(identity) == 2 and identity[1] == owner else None


def remove_exited_container(name: str) -> bool:
    """Remove a stopped container that this user's earlier AVSecTester start left under ``name``.

    Running containers and containers without both AVSecTester labels for this user stay
    untouched. Return whether a container was removed.
    """
    _require_docker()
    state = _docker(["inspect", "--format", "{{.State.Running}}", name])
    if state.returncode or state.stdout.strip() != "false":
        return False
    response = _docker(["inspect", "--format",
                        '{{.Id}} {{ index .Config.Labels "avsectester.service-owner" }} '
                        '{{ index .Config.Labels "avsectester.service-user" }}', name])
    identity = response.stdout.strip().split()
    if (response.returncode or len(identity) != 3 or not re.fullmatch(r"[0-9a-f]{32}", identity[1])
            or identity[2] != str(os.getuid())):
        return False
    # Removal by identity without force refuses a container that started again meanwhile.
    removed = _docker(["rm", identity[0]])
    if removed.returncode:
        raise RuntimeError(removed.stderr.strip() or removed.stdout.strip()
                           or f"docker rm failed for container {name}")
    return True


def _failed_start(error: BaseException, name: str, identity: str | None) -> NoReturn:
    logs = "No container owned by this invocation; no container was stopped."
    if identity is not None:
        try:
            response = _docker(["logs", "--tail", "50", identity])
            logs = "\n".join((response.stdout + response.stderr).splitlines()[-50:])
        except BaseException as log_error:
            logs = f"Unable to read container logs: {log_error}"
        try:
            stop_container(identity)
        except BaseException as cleanup_error:
            error.add_note(f"Container cleanup failed for {name}: {cleanup_error}")
    if isinstance(error, Exception):
        failed = RuntimeError(f"{error}\nLast container logs:\n{logs}")
        for note in getattr(error, "__notes__", ()):
            failed.add_note(note)
        raise failed from error
    error.add_note(f"Last container logs:\n{logs}")
    raise error


def launch_container(command: list[str], name: str, *,
                     prepare: Callable[[], Any] | None = None) -> str:
    """Run detached Docker argv under unique ownership labels and return its container ID.

    Run the optional preparation callback after confirming the name is available. If Docker
    fails to launch, stop and remove only the container created by this invocation. Return
    on successful launch without waiting for service readiness.
    """
    _require_docker()
    if _docker(["inspect", name]).returncode == 0:
        raise RuntimeError(f"Container already exists: {name}; choose --name or stop it explicitly.")
    if prepare is not None:
        prepare()
    owner = uuid.uuid4().hex
    labeled_command = [*command[:2], "--label", f"{_OWNER_LABEL}={owner}",
                       "--label", f"{_USER_LABEL}={os.getuid()}", *command[2:]]
    try:
        launched = subprocess.run(labeled_command, capture_output=True, text=True, check=False)
        if launched.returncode:
            raise RuntimeError(launched.stderr.strip() or launched.stdout.strip() or "docker run failed")
        lines = launched.stdout.strip().splitlines()
        if not lines:
            raise RuntimeError(f"Docker did not return an identity for container {name}")
        return lines[-1]
    except BaseException as error:
        try:
            identity = _owned_container(name, owner)
        except BaseException as ownership_error:
            error.add_note(f"Unable to read container ownership for {name}: {ownership_error}")
            identity = None
        _failed_start(error, name, identity)


def start_container(
    command: list[str], *, name: str, ready_log: str, timeout_s: float,
    readiness: Callable[[float], Any] | None = None,
    prepare: Callable[[], Any] | None = None, log: Callable[[str], Any],
) -> str:
    """Start an absent local container and return its immutable Docker identity.

    Startup failures include the last fifty log lines and stop/remove only a container
    carrying this invocation's unique ownership label. Monitoring and cleanup use the
    immutable identity, so a replacement container with the same name is never stopped.
    Interruptions retain their original exception type.
    """
    require_positive_number(timeout_s, "timeout_s")

    def before_launch() -> None:
        if prepare is not None:
            prepare()
        log(f"Starting container: {name}")

    identity = launch_container(command, name, prepare=before_launch)
    try:
        deadline = time.monotonic() + timeout_s
        readiness_error = None
        while True:
            response = _docker(["logs", "--tail", "50", identity])
            logs = response.stdout + response.stderr
            state = _docker(["inspect", "--format", "{{.State.Running}}", identity])
            if state.returncode or state.stdout.strip() != "true":
                raise RuntimeError(f"Container {name} exited before becoming ready")
            remaining = deadline - time.monotonic()
            if ready_log in logs:
                if readiness is None:
                    return identity
                if remaining > 0:
                    try:
                        readiness(remaining)
                    except RuntimeError as error:
                        readiness_error = str(error)
                    else:
                        return identity
            if time.monotonic() >= deadline:
                detail = "" if readiness_error is None else f": {readiness_error}"
                raise RuntimeError(f"Container {name} startup timed out after {timeout_s:g} seconds{detail}")
            time.sleep(min(.5, max(0., deadline - time.monotonic())))
    except BaseException as error:
        _failed_start(error, name, identity)


__all__ = ["launch_container", "remove_exited_container", "start_container", "stop_container"]
