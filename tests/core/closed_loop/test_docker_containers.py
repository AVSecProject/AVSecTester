"""Container launch does not depend on a service-specific readiness message."""

import json

from avsectester.closed_loop.docker_containers import launch_container, stop_container
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
