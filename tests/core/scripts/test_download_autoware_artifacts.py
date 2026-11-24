"""Autoware artifact plans, verified downloads and confined archive extraction."""

import hashlib
import io
import subprocess
import tarfile

import pytest

from scripts.preparation import download_autoware_artifacts as artifacts


def _task(kind, **fields):
    lines = [f"- name: {kind} task", f"  ansible.builtin.{kind}:"]
    lines += [f"    {key}: {value}" for key, value in fields.items()]
    return "\n".join(lines) + "\n"


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def test_read_plan(tmp_path):
    root = tmp_path / "ml_models"
    manifest = (
        _task("file", path='"{{ data_dir }}/model"')
        + _task("get_url", url="https://awf.ml.dev.web.auto/model/a.onnx",
                dest='"{{ data_dir }}/model/a.onnx"', checksum=f"sha256:{'a' * 64}")
        + _task("get_url", url="https://awf.ml.dev.web.auto/model/b.tar.gz",
                dest='"{{ data_dir }}/model/b.tar.gz"', checksum=f"sha256:{'b' * 64}")
        + _task("unarchive", src='"{{ data_dir }}/model/b.tar.gz"', dest='"{{ data_dir }}/model/"')
    )
    tree = [{"type": "directory", "path": "tiny"},
            {"type": "file", "path": "tiny/encoder.onnx", "size": 3,
             "oid": "c" * 40, "lfs": {"oid": "d" * 64, "size": 3}},
            {"type": "file", "path": "README.md", "size": 2, "oid": "e" * 40}]
    plan, archives = artifacts.read_plan(root, manifest, tree)
    assert [(item.path.relative_to(root).as_posix(), item.algorithm, item.checksum) for item in plan] == [
        ("model/a.onnx", "sha256", "a" * 64), ("model/b.tar.gz", "sha256", "b" * 64),
        ("lidar_centerpoint/tiny/encoder.onnx", "sha256", "d" * 64),
        ("lidar_centerpoint/README.md", "git-blob-sha1", "e" * 40),
    ]
    assert plan[2].url.endswith(f"/resolve/{artifacts.CENTERPOINT_REVISION}/tiny/encoder.onnx")
    assert archives == [artifacts.Archive(root / "model/b.tar.gz", root / "model", 0)]
    for invalid, error in (
        (manifest.replace("awf.ml.dev.web.auto/model/a", "example.com/a"), "Unapproved artifact URL"),
        (manifest.replace('"{{ data_dir }}/model/a.onnx"', '"{{ data_dir }}/../a.onnx"'), "Unconfined"),
        (manifest.replace('src: "{{ data_dir }}/model/b', 'src: "{{ data_dir }}/model/c'),
         "archive has no verified download"),
    ):
        with pytest.raises(ValueError, match=error):
            artifacts.read_plan(root, invalid, tree)


def test_download_and_extract(tmp_path, monkeypatch):
    root = tmp_path / "ml_models"
    archive_bytes = io.BytesIO()
    with tarfile.open(fileobj=archive_bytes, mode="w:gz") as archive:
        for name, content in (("bundle/", None), ("bundle/weights.onnx", b"weights")):
            member = tarfile.TarInfo(name)
            if content is None:
                member.type = tarfile.DIRTYPE
                archive.addfile(member)
            else:
                member.size = len(content)
                archive.addfile(member, io.BytesIO(content))
    remote = {"https://awf.ml.dev.web.auto/b.tar.gz": archive_bytes.getvalue(),
              "https://awf.ml.dev.web.auto/a.onnx": b"model"}
    fetched, failures = [], []

    def fetch(url, destination):
        fetched.append(url)
        if failures:
            raise subprocess.CalledProcessError(failures.pop(), ["curl"])
        existing = destination.read_bytes() if destination.exists() else b""
        destination.write_bytes(existing + remote[url][len(existing):])

    monkeypatch.setattr(artifacts, "fetch", fetch)
    model = artifacts.Artifact("https://awf.ml.dev.web.auto/a.onnx", root / "model/a.onnx",
                               _sha256(b"model"))
    part = root / "model/a.onnx.part"
    part.parent.mkdir(parents=True)
    part.write_bytes(b"mo")
    failures.append(28)
    with pytest.raises(subprocess.CalledProcessError):
        artifacts.download(root, model)
    assert part.read_bytes() == b"mo"
    assert artifacts.download(root, model)
    assert not artifacts.download(root, model)
    assert fetched == [model.url] * 2 and model.path.read_bytes() == b"model"
    assert not part.exists()

    restarted = artifacts.Artifact(model.url, root / "model/d.onnx", _sha256(b"model"))
    (root / "model/d.onnx.part").write_bytes(b"bad!!")
    failures.append(22)
    with pytest.raises(subprocess.CalledProcessError):
        artifacts.download(root, restarted)
    assert not (root / "model/d.onnx.part").exists()
    assert artifacts.download(root, restarted) and restarted.path.read_bytes() == b"model"

    corrupted = artifacts.Artifact(model.url, root / "model/c.onnx", _sha256(b"other"))
    with pytest.raises(ValueError, match="does not match its checksum"):
        artifacts.download(root, corrupted)
    assert not corrupted.path.exists() and not list(root.rglob("*.part"))

    bundle = artifacts.Artifact("https://awf.ml.dev.web.auto/b.tar.gz", root / "bundle/b.tar.gz",
                                _sha256(remote["https://awf.ml.dev.web.auto/b.tar.gz"]))
    artifacts.download(root, bundle)
    unpacked = artifacts.Archive(bundle.path, root / "bundle", 1)
    assert artifacts.extract(root, unpacked) == 1
    assert (root / "bundle/weights.onnx").read_bytes() == b"weights"
    assert artifacts.extract(root, unpacked) == 0
    (root / "bundle/weights.onnx").write_bytes(b"changed")
    with pytest.raises(ValueError, match="differs from its archive member"):
        artifacts.extract(root, unpacked)

    unsafe = io.BytesIO()
    with tarfile.open(fileobj=unsafe, mode="w:gz") as archive:
        member = tarfile.TarInfo("../outside.onnx")
        member.size = 1
        archive.addfile(member, io.BytesIO(b"x"))
    (root / "unsafe.tar.gz").write_bytes(unsafe.getvalue())
    with pytest.raises(ValueError, match="Unsafe archive member"):
        artifacts.extract(root, artifacts.Archive(root / "unsafe.tar.gz", root, 0))
    assert not (tmp_path / "outside.onnx").exists()
