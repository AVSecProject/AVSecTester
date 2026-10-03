"""nuRec ``.usdz`` -> ``SceneGT``: validated against REAL scenes (no synthetic data).

These tests read real ``PhysicalAI-Autonomous-Vehicles-NuRec`` (26.01) ``.usdz`` artifacts and are
**skipped** when they are not present — there is no fabricated stand-in. They assert the reader recovers
sane ego-frame ground truth (rear-facing leads ahead, boxes inside the fisheye frame) and that the
requirement filter yields genuine matches on real data. The projection was also confirmed visually by
overlaying the boxes on the scene's rendered ``.mp4`` (see ``tmp/compare/nurec_gt_*.png``).
"""

import math
import os

import pytest
from avsectester.scenarios.datasets.nurec import NuRecDataset
from avsectester.scenarios.requirements import PHYSICAL_PATCH_HIDE_VEHICLE as REQ

_ROOT = os.environ.get(
    "AVSECTESTER_NUREC_ROOT",
    "/workspace/hdd/datasets/PhysicalAI-Autonomous-Vehicles-NuRec/sample_set/26.01_release",
)
_UUIDS = ("023b7fcc-671c-40e3-9bd2-c66b0b073fbc", "0245ff75-aa3f-46b7-ba87-16a7afb841af")
_SCENES = [os.path.join(_ROOT, u, f"{u}.usdz") for u in _UUIDS]

pytestmark = pytest.mark.skipif(
    not all(os.path.exists(p) for p in _SCENES),
    reason="real nuRec .usdz artifacts not present (license-gated download)")


def test_reads_real_usdz_ego_frame_gt():
    scene = next(NuRecDataset([_SCENES[0]], keyframe=0.5).scenes())
    assert scene.source["dataset"] == "nurec" and scene.source["scene_id"].startswith("clipgt-")
    ahead = [o for o in scene.objects if o.category == "vehicle" and o.ahead and "front" in o.box2d]
    assert len(ahead) >= 3                              # a busy road: several leads project ahead
    cam = scene.cameras["front"]
    for o in ahead:
        x1, y1, x2, y2 = o.box2d["front"]              # every 2-D box lies inside the fisheye frame
        assert 0 <= x1 < x2 <= cam.width and 0 <= y1 < y2 <= cam.height
        assert abs(o.yaw) <= math.pi + 1e-6            # yaw is a real heading in [-pi, pi]


def test_real_filtering_yields_matches():
    # scanning frames of a real clip, some qualify for physical_patch_hide_vehicle (rear-facing lead in
    # the distance/area band) and some do not — real, non-trivial filtering.
    hits = sum(REQ.match(next(NuRecDataset([_SCENES[0]], keyframe=k / 20).scenes())) is not None
               for k in range(21))
    assert 1 <= hits < 21
