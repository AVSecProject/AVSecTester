"""Detection-level phantom attack: geometry + hook-seam behavior (no GPU, no CARLA).

Needs avstack geometry/detections but not a neural detector, so it is fast and runs
wherever the stack is installed.
"""
from __future__ import annotations

import pytest

pytest.importorskip("avstack")

import numpy as np
from avsectester.attacks.perception import PhantomDetectionAttack
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
