"""Container launch does not depend on a service-specific readiness message."""

import json
import os

import pytest

from avsectester.closed_loop.docker_containers import (
    launch_container, remove_exited_container, start_container, stop_container,
)
from tests.fakes import docker_process_boundary


def test_launch_container(tmp_path, monkeypatch):
    state, _ = docker_process_boundary(tmp_path, monkeypatch)
    monkeypatch.setenv("DOCKER_MODE", "timeout")
    name = "external-service"
    identity = launch_container(["docker", "run", "-d", "--name", name, "local:service"], name)
    try:
        assert json.loads((state / name).read_text())["id"] == identity
    finally:
        stop_container(identity)
    assert not (state / name).exists()


@pytest.mark.parametrize(("mode", "error_type", "message"), [
    ("exit", RuntimeError, "startup diagnostic"),
    ("timeout", RuntimeError, "timed out"),
    ("interrupt", KeyboardInterrupt, None),
])
def test_start_container_failure(tmp_path, monkeypatch, mode, error_type, message):
    state, _ = docker_process_boundary(tmp_path, monkeypatch)
    monkeypatch.setenv("DOCKER_MODE", mode)
    name = "external-service"
    with pytest.raises(error_type, match=message):
        start_container(["docker", "run", "-d", "--name", name, "local:service"],
                        name=name, ready_log="Serving on", timeout_s=.1, log=lambda text: None)
    assert not (state / name).exists()


def test_launch_container_ownership(tmp_path, monkeypatch):
    state, _ = docker_process_boundary(tmp_path, monkeypatch)
    monkeypatch.setenv("DOCKER_MODE", "race")
    name = "external-service"
    command = ["docker", "run", "-d", "--name", name, "local:service"]
    with pytest.raises(RuntimeError, match="This invocation owned no container to stop"):
        launch_container(command, name)
    assert json.loads((state / name).read_text())["owner"] == "someone-else"
    with pytest.raises(RuntimeError, match="already exists"):
        launch_container(command, name)
    assert json.loads((state / name).read_text())["owner"] == "someone-else"


def test_remove_exited_container(tmp_path, monkeypatch):
    state, _ = docker_process_boundary(tmp_path, monkeypatch)
    name = "external-service"
    identity = launch_container(["docker", "run", "-d", "--name", name, "local:service"], name)
    container = state / name
    launched = json.loads(container.read_text())
    assert not remove_exited_container(name)
    other_owner = {**launched["labels"], "avsectester.service-user": str(os.getuid() + 1)}
    container.write_text(json.dumps({**launched, "running": False, "labels": other_owner}))
    assert not remove_exited_container(name)
    assert container.exists()
    container.write_text(json.dumps({**launched, "running": False}))
    assert remove_exited_container(name)
    assert not container.exists()
    assert identity == launched["id"]


def test_stop_container_by_identity(tmp_path, monkeypatch):
    state, _ = docker_process_boundary(tmp_path, monkeypatch)
    name = "external-service"
    identity = launch_container(["docker", "run", "-d", "--name", name, "local:service"], name)
    container = state / name
    # Docker can allocate the name to another container before cleanup.
    container.write_text(json.dumps({"id": "replacement", "owner": "other", "running": True}))
    with pytest.raises(RuntimeError, match="No such container"):
        stop_container(identity)
    assert json.loads(container.read_text())["id"] == "replacement"
