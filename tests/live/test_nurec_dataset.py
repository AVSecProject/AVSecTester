"""Read and filter ground truth from a NuRec 26.01 scene when the dataset is available.

These tests check the USDZ reader and projections. They do not exercise the rendering service or
closed-loop interactions.
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
_UUID = "023b7fcc-671c-40e3-9bd2-c66b0b073fbc"
_SCENE = os.path.join(_ROOT, _UUID, f"{_UUID}.usdz")

pytestmark = pytest.mark.skipif(
    not os.path.isfile(_SCENE),
    reason="real nuRec .usdz artifacts not present (license-gated download)")


def test_reads_real_usdz_ego_frame_gt():
    scene = next(NuRecDataset([_SCENE], keyframe=0.5).scenes())
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
    hits = sum(REQ.match(next(NuRecDataset([_SCENE], keyframe=k / 20).scenes())) is not None
               for k in range(21))
    assert 1 <= hits < 21
