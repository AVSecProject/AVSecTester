"""Exercise checkpoint installation paths with a local download stand-in."""

import json
import os
import shutil
import subprocess
import sys

from scripts import REPO_ROOT


def test_fetch_models_exposes_the_nuscenes_detector_at_its_loader_path(tmp_path):
    root = tmp_path / "project"
    script = root / "scripts" / "preparation" / "fetch_models.sh"
    script.parent.mkdir(parents=True)
    shutil.copyfile(REPO_ROOT / "scripts/preparation/fetch_models.sh", script)
    detection_root = root / "third_party/avstack-core/third_party/mmdetection"
    detection_root.mkdir(parents=True)
    (root / "third_party/avstack-core/third_party/mmdetection3d").mkdir()
    binaries = tmp_path / "bin"
    binaries.mkdir()
    log = tmp_path / "downloads.jsonl"
    curl = binaries / "curl"
    curl.write_text(f"#!{sys.executable}\n" + '''
import json
import os
from pathlib import Path
import sys
arguments = sys.argv[1:]
destination = Path(arguments[arguments.index("-o") + 1])
destination.write_bytes(b"checkpoint fixture")
with open(os.environ["MODEL_DOWNLOAD_LOG"], "a") as stream:
    stream.write(json.dumps(arguments) + "\\n")
''')
    curl.chmod(0o755)
    environment = {**os.environ, "PATH": str(binaries) + os.pathsep + os.environ["PATH"],
                   "MODEL_DOWNLOAD_LOG": str(log)}
    completed = subprocess.run(["bash", str(script)], env=environment, capture_output=True,
                               text=True, timeout=30)
    assert completed.returncode == 0, completed.stderr
    name = "faster_rcnn_r50_fpn_1x_nuscenes.pth"
    loader_path = detection_root / "work_dirs/nuscenes" / name
    assert loader_path.read_bytes() == b"checkpoint fixture"
    assert loader_path.resolve() == root / "models/work_dirs_2d/nuscenes" / name
    downloads = [json.loads(line) for line in log.read_text().splitlines()]
    matching = [arguments for arguments in downloads if arguments[-1].endswith("/" + name)]
    assert len(matching) == 1
    assert matching[0][-1].endswith("/models/mmdet/work_dirs/nuscenes/" + name)
    # A second invocation must reuse the installed weight, including its linked loader path.
    before = log.read_text()
    repeated = subprocess.run(["bash", str(script)], env=environment, capture_output=True,
                              text=True, timeout=30)
    assert repeated.returncode == 0, repeated.stderr
    assert log.read_text() == before
