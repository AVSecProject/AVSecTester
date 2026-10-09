"""nuScenes -> ``SceneGT``: validated on REAL nuScenes GT (no synthetic boxes).

Reads real ``v1.0-trainval`` ground truth through the devkit and is **skipped** when the dataset/devkit
are absent — there is no fabricated stand-in. Asserts the adapter yields qualifying vehicles with real
3-D extent and that the requirement filter selects a rear-facing lead in the patch band. The
``RecordedFrameBackend`` plumbing is tested separately with a real on-disk image (no GT fabrication).
"""

import os

import pytest
from avsectester.scenarios.datasets.nuscenes import NuScenesDataset
from avsectester.scenarios.requirements import PHYSICAL_PATCH_HIDE_VEHICLE as REQ
from avsectester.scenarios.source import DatasetFilter

_ROOT = os.environ.get("AVSECTESTER_NUSCENES_ROOT", "/workspace/hdd/datasets/nuscenes")


def _has_nuscenes() -> bool:
    if not os.path.isdir(os.path.join(_ROOT, "v1.0-trainval")):
        return False
    try:
        import nuscenes  # noqa: F401
    except ImportError:
        return False
    return True


@pytest.mark.skipif(not _has_nuscenes(), reason="real nuScenes v1.0-trainval + devkit not present")
def test_reads_real_nuscenes_gt_and_filters():
    ds = NuScenesDataset(_ROOT, max_samples=80)
    hits = list(DatasetFilter(ds).scenarios(REQ))
    assert hits, "expected some qualifying frames among 80 real keyframes"
    target = hits[0].match.target
    assert target.category == "vehicle" and "front" in target.box2d
    assert 4.0 <= target.distance <= 25.0        # in the physical-patch distance band
    assert min(target.extent) > 0                # a real 3-D box has non-zero size (drives 3-D drawing)
