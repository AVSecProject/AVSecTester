"""Detection-manipulation vector: injection + removal methods (no GPU, no CARLA).

Needs avstack geometry/detections but not a neural detector, so it is fast and runs
wherever the stack is installed. Covers both methods sharing the vector: phantom detection
injection (false positive) and detection removal (false negative).
"""
from __future__ import annotations

import pytest

pytest.importorskip("avstack")

import numpy as np
from avsectester.attacks.detection_manipulation import (
    DetectionRemovalAttack,
    PhantomDetectionAttack,
)
from avstack.datastructs import DataContainer
from avstack.geometry import Attitude, Box3D, GlobalOrigin3D, Position, ReferenceFrame
from avstack.modules.perception.detections import BoxDetection


def _ref() -> ReferenceFrame:
    return ReferenceFrame(x=np.zeros(3), q=np.quaternion(1), reference=GlobalOrigin3D)


def _detection(ref: ReferenceFrame, x: float = 5.0) -> BoxDetection:
    box = Box3D(Position(np.array([x, 0.0, -1.5]), ref), Attitude(np.quaternion(1), ref),
                [1.6, 1.8, 4.0], where_is_t="bottom")
    return BoxDetection(
        data=box, noise=np.array([1, 1, 1, 0.1, 0.1, 0.1]) ** 2,
        source_identifier="lidar", reference=ref, obj_type="Car", score=0.8,
    )


def test_appends_phantom_in_detection_frame():
    ref = _ref()
    data = DataContainer(0, 0.0, [_detection(ref)], "lidar")
    out = PhantomDetectionAttack(target_xyz=[12.0, 0.0, -1.5], obj_type="Car").apply(data)
    assert len(out) == 2
    phantom = next(d for d in out if getattr(d, "ID", None) == 90002)
    assert phantom.obj_type == "Car"
    assert list(phantom.position.x) == [12.0, 0.0, -1.5]
    assert phantom.reference is ref  # placed in the detections' own frame


def test_score_is_carried_for_downstream_gating():
    ref = _ref()
    data = DataContainer(0, 0.0, [_detection(ref)], "lidar")
    out = PhantomDetectionAttack(score=0.42).apply(data)
    assert next(d for d in out if d.ID == 90002).score == 0.42


def test_validate_rejects_bad_score():
    with pytest.raises(ValueError, match="score"):
        PhantomDetectionAttack(score=1.5).validate(spec=None)


def test_injection_and_removal_share_the_vector_seams():
    assert PhantomDetectionAttack.seams is DetectionRemovalAttack.seams
    assert PhantomDetectionAttack.seams == ("perception_out",)


def test_removal_drops_nearest_forward_detection():
    ref = _ref()
    data = DataContainer(
        0, 0.0,
        [_detection(ref, x=20.0), _detection(ref, x=8.0), _detection(ref, x=-6.0)],
        "lidar",
    )
    for i, d in enumerate(data):
        d.ID = 10 + i  # 10@20m, 11@8m, 12@-6m(behind)
    out = DetectionRemovalAttack(corridor=3.0, max_range=40.0).apply(data)
    assert {d.ID for d in out} == {10, 12}  # nearest forward (11 @ 8 m) suppressed


def test_removal_never_targets_an_injected_phantom():
    ref = _ref()
    data = DataContainer(0, 0.0, [_detection(ref, x=8.0)], "lidar")
    data[0].ID = 5
    # inject a closer phantom, then run removal: it must drop the real one, not the phantom
    PhantomDetectionAttack(target_xyz=[4.0, 0.0, -1.5]).apply(data)
    attack = DetectionRemovalAttack()
    out = attack.apply(data)
    ids = {d.ID for d in out}
    assert 90002 in ids and 5 not in ids  # phantom kept, real detection removed
    assert attack._removed_id == 5
