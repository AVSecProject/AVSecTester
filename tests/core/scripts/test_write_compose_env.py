"""Machine-local Compose settings are generated without changing the working checkout."""

import os
from pathlib import Path
import shutil
import subprocess


def test_write_compose_env(tmp_path):
    repository = tmp_path / "repository with spaces"
    script = repository / "scripts/preparation/write_compose_env.sh"
    script.parent.mkdir(parents=True)
    source = Path(__file__).resolve().parents[3] / "scripts/preparation/write_compose_env.sh"
    shutil.copy2(source, script)
    boundary = tmp_path / "bin"
    boundary.mkdir()
    docker = boundary / "docker"
    docker.write_text("#!/bin/sh\nexit 0\n")
    docker.chmod(0o755)
    stat = boundary / "stat"
    stat.write_text("#!/bin/sh\n[ \"$1\" = -c ] && [ \"$2\" = %g ] && [ \"$3\" = /var/run/docker.sock ] || exit 1\nprintf '996\\n'\n")
    stat.chmod(0o755)
    environment = {**os.environ, "PATH": f"{boundary}:{os.environ['PATH']}"}

    def run(*arguments):
        return subprocess.run(["bash", str(script), *arguments], cwd=tmp_path,
                              env=environment, text=True, capture_output=True)

    first = run()
    assert first.returncode == 0, first.stderr
    output = repository / ".env"
    entries = dict(line.split("=", 1) for line in output.read_text().splitlines())
    assert entries == {
        "AVSECTESTER_ROOT": f"'{repository}'", "AVSECTESTER_UID": str(os.getuid()),
        "AVSECTESTER_GID": str(os.getgid()), "DOCKER_GID": "996",
        "DOCKER_CLI": f"'{docker}'", "CARLA_SERVER_GPU": "0",
        "CARLA_EXPERIMENT_GPU": "0", "NUREC_GPU": "0",
    }
    assert first.stdout == output.read_text()
    output.write_text("local settings\n")
    rejected = run()
    assert rejected.returncode == 1 and "--force" in rejected.stderr
    assert output.read_text() == "local settings\n"
    forced = run("--force")
    assert forced.returncode == 0 and forced.stdout == first.stdout
