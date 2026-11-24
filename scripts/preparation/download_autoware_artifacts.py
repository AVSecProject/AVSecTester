#!/usr/bin/env python3
"""Download and verify the Autoware 1.9.0 model artifacts used by hosted Autoware.

Run on the Docker host with Python 3.11 or newer and curl. Files go to DATA_DIR/ml_models, and
DATA_DIR is the directory that `autoware_data_directory` names. The release files and archives
follow Autoware 1.9.0's ansible artifacts role and are checked against its SHA-256 values.
CenterPoint comes from a pinned Hugging Face commit and is checked against the sizes and hashes
that its file listing reports. Verified existing files are kept, and partial files continue
where they stopped, so running the command again resumes an interrupted download.
"""

from __future__ import annotations

from argparse import ArgumentParser
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import tarfile
import tempfile
from typing import Any
from urllib.parse import quote, urlsplit


MANIFEST_URL = ("https://raw.githubusercontent.com/autowarefoundation/autoware/1.9.0/"
                "ansible/roles/artifacts/tasks/main.yaml")
CENTERPOINT_REPOSITORY = "AutowareFoundation/lidar_centerpoint"
CENTERPOINT_REVISION = "4ccd240d715bfd06ca0bc12daf12ae77a39ad5b9"
CENTERPOINT_TREE_URL = (f"https://huggingface.co/api/models/{CENTERPOINT_REPOSITORY}/tree/"
                        f"{CENTERPOINT_REVISION}?recursive=true&expand=false")
RELEASE_HOSTS = frozenset({"awf.ml.dev.web.auto", "autoware-files.s3.us-west-2.amazonaws.com"})
_CURL_TIMEOUT = 28
_SHA256 = re.compile(r"[a-f0-9]{64}\Z")
_GIT_OID = re.compile(r"[a-f0-9]{40}\Z")


@dataclass(frozen=True)
class Artifact:
    """One file to download, its destination and its expected checksum."""

    url: str
    path: Path
    checksum: str
    algorithm: str = "sha256"
    size: int | None = None


@dataclass(frozen=True)
class Archive:
    """One downloaded archive and the directory it extracts into."""

    source: Path
    destination: Path
    strip_components: int


def fetch(url: str, destination: Path) -> None:
    """Download one HTTPS URL with curl, continuing a partial destination file."""
    subprocess.run([
        "curl", "--fail", "--silent", "--show-error", "--location", "--continue-at", "-",
        "--proto", "=https", "--proto-redir", "=https", "--retry", "2",
        "--connect-timeout", "20", "--max-time", "900", "--output", str(destination), url,
    ], check=True)


def digest(path: Path, algorithm: str) -> str:
    """Return a SHA-256 digest or a Git blob ID of one file."""
    with path.open("rb") as stream:
        if algorithm == "git-blob-sha1":
            result = hashlib.sha1(usedforsecurity=False)
            result.update(f"blob {path.stat().st_size}\0".encode("ascii"))
            while block := stream.read(1024 * 1024):
                result.update(block)
            return result.hexdigest()
        return hashlib.file_digest(stream, algorithm).hexdigest()


def confined(root: Path, name: str) -> Path:
    """Resolve a relative path below root without following symbolic links."""
    path = PurePosixPath(name)
    if not path.parts or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unconfined model path: {name}")
    target = root
    for part in path.parts:
        target /= part
        if target.is_symlink():
            raise ValueError(f"Refusing to follow a symbolic link in the model directory: {target}")
    return target


def _release_path(root: Path, value: str) -> Path:
    # The artifacts role's data_dir is the ml_models directory itself.
    prefix = "{{ data_dir }}/"
    if not value.startswith(prefix):
        raise ValueError(f"Unsupported artifact destination: {value}")
    return confined(root, value.removeprefix(prefix))


def _field(task: str, key: str) -> str:
    match = re.search(rf"^    {key}: (.+)$", task, re.MULTILINE)
    if match is None:
        raise ValueError(f"Missing {key} in task {task.splitlines()[0]!r}")
    return match[1].strip().strip('"')


def read_plan(root: Path, manifest: str,
              tree: Sequence[dict[str, Any]]) -> tuple[list[Artifact], list[Archive]]:
    """Read the ansible artifacts tasks and the CenterPoint file listing into one plan."""
    artifacts, archives = [], []
    for task in re.split(r"(?m)(?=^- name: )", manifest):
        if "\n  ansible.builtin.get_url:\n" in task:
            url = _field(task, "url")
            source = urlsplit(url)
            checksum = _field(task, "checksum").removeprefix("sha256:")
            if source.scheme != "https" or source.hostname not in RELEASE_HOSTS:
                raise ValueError(f"Unapproved artifact URL: {url}")
            if not _SHA256.fullmatch(checksum):
                raise ValueError(f"Invalid SHA-256 checksum for {url}")
            artifacts.append(Artifact(url, _release_path(root, _field(task, "dest")), checksum))
        elif "\n  ansible.builtin.unarchive:\n" in task:
            strip = "--strip-components=1" in task
            if "extra_opts:" in task and not strip:
                raise ValueError(f"Unsupported archive option in task {task.splitlines()[0]!r}")
            archives.append(Archive(_release_path(root, _field(task, "src")),
                                    _release_path(root, _field(task, "dest")), int(strip)))
    for entry in tree:
        if entry["type"] == "directory":
            continue
        if entry["type"] != "file":
            raise ValueError(f"Unexpected CenterPoint listing entry: {entry}")
        name, size = entry["path"], entry["size"]
        lfs = entry.get("lfs")
        if lfs:
            checksum, algorithm = lfs["oid"], "sha256"
            if size != lfs["size"] or not _SHA256.fullmatch(checksum):
                raise ValueError(f"Invalid large-file metadata for CenterPoint file {name}")
        else:
            checksum, algorithm = entry["oid"], "git-blob-sha1"
            if not _GIT_OID.fullmatch(checksum):
                raise ValueError(f"Invalid Git blob ID for CenterPoint file {name}")
        url = (f"https://huggingface.co/{CENTERPOINT_REPOSITORY}/resolve/"
               f"{CENTERPOINT_REVISION}/{quote(name)}")
        artifacts.append(Artifact(url, confined(root, f"lidar_centerpoint/{name}"),
                                  checksum, algorithm, size))
    paths = [artifact.path for artifact in artifacts]
    if len(set(paths)) != len(paths):
        raise ValueError("Two artifacts share a destination")
    if any(archive.source not in paths for archive in archives):
        raise ValueError("An archive has no verified download")
    return artifacts, archives


def _verify(artifact: Artifact, path: Path) -> None:
    size, actual = path.stat().st_size, digest(path, artifact.algorithm)
    if (artifact.size is not None and size != artifact.size) or actual != artifact.checksum:
        raise ValueError(f"{path} does not match its checksum. Expected {artifact.checksum}, "
                         f"obtained {actual}")


def download(root: Path, artifact: Artifact) -> bool:
    """Download one artifact unless a verified copy exists. Return whether it was downloaded."""
    if artifact.path.exists():
        _verify(artifact, artifact.path)
        return False
    artifact.path.parent.mkdir(parents=True, exist_ok=True)
    part = confined(root, str(artifact.path.relative_to(root)) + ".part")
    try:
        _verify(artifact, part)
    except (FileNotFoundError, ValueError):
        try:
            fetch(artifact.url, part)
        except subprocess.CalledProcessError as error:
            # A timeout keeps the partial file for the next run. Other errors restart the file.
            if error.returncode != _CURL_TIMEOUT:
                part.unlink(missing_ok=True)
            raise
        try:
            _verify(artifact, part)
        except ValueError:
            part.unlink()
            raise
    part.chmod(0o644)
    part.rename(artifact.path)
    return True


def extract(root: Path, archive: Archive) -> int:
    """Extract regular files and directories and keep existing identical files."""
    written = 0
    with tarfile.open(archive.source, "r:*") as stream:
        for member in stream.getmembers():
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts or not (member.isdir() or member.isreg()):
                raise ValueError(f"Unsafe archive member {member.name!r} in {archive.source}")
            components = path.parts[archive.strip_components:]
            if not components:
                continue
            relative = (archive.destination / Path(*components)).relative_to(root)
            output = confined(root, str(relative))
            if member.isdir():
                output.mkdir(parents=True, exist_ok=True)
                continue
            output.parent.mkdir(parents=True, exist_ok=True)
            incoming = stream.extractfile(member)
            if incoming is None:
                raise RuntimeError(f"Unable to read {member.name} from {archive.source}")
            part = confined(root, str(relative) + ".part")
            with incoming, part.open("wb") as target:
                shutil.copyfileobj(incoming, target)
            if output.exists():
                if digest(output, "sha256") != digest(part, "sha256"):
                    part.unlink()
                    raise ValueError(f"Existing file differs from its archive member: {output}")
                part.unlink()
            else:
                part.chmod(0o644 | (member.mode & 0o111))
                part.rename(output)
                written += 1
    return written


def main(argv: Sequence[str] | None = None) -> None:
    parser = ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Model data directory that will contain ml_models")
    parser.add_argument("--jobs", type=int, default=8, help="Parallel downloads")
    parser.add_argument("--list", action="store_true", help="Print the plan without downloading")
    arguments = parser.parse_args(argv)
    if arguments.data_dir.is_symlink():
        raise SystemExit("Pass the model data directory itself, not a symbolic link to it")
    root = arguments.data_dir / "ml_models"
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as directory:
        manifest, listing = Path(directory) / "main.yaml", Path(directory) / "tree.json"
        fetch(MANIFEST_URL, manifest)
        fetch(CENTERPOINT_TREE_URL, listing)
        artifacts, archives = read_plan(root, manifest.read_text(), json.loads(listing.read_text()))
    if arguments.list:
        for artifact in artifacts:
            print(artifact.path.relative_to(root))
        return
    failures = []
    with ThreadPoolExecutor(arguments.jobs) as pool:
        pending = {pool.submit(download, root, artifact): artifact for artifact in artifacts}
        for future in as_completed(pending):
            name = pending[future].path.relative_to(root)
            try:
                state = "downloaded" if future.result() else "verified"
            except Exception as error:
                failures.append(name)
                state = f"failed: {error}"
            print(f"{name}: {state}", flush=True)
    if failures:
        raise SystemExit(f"{len(failures)} artifacts failed. Run the command again to retry.")
    written = sum(extract(root, archive) for archive in archives)
    print(f"Verified {len(artifacts)} artifacts and extracted {len(archives)} archives "
          f"({written} new files) into {root}")


if __name__ == "__main__":
    main()
