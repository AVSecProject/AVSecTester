"""Attack apply() geometry at seams (avstack; no CARLA, no GPU)."""

from __future__ import annotations

import pytest

pytest.importorskip("avstack")

import numpy as np
from avsectester.attacks import (
    DetectionRemovalAttack,
    ObjectRemovalAttack,
    ObjectSpoofingAttack,
    PhantomDetectionAttack,
)
from avsectester.core import Context, Frame, Seam
from avsectester.envs.common import make_object
from avstack.datastructs import DataContainer
from avstack.geometry import Attitude, Box3D, GlobalOrigin3D, Position, ReferenceFrame
from avstack.modules.perception.detections import BoxDetection


def _det(ref, x):
    box = Box3D(Position(np.array([x, 0.0, -1.5]), ref), Attitude(np.quaternion(1), ref),
                [1.6, 1.8, 4.0], where_is_t="bottom")
    return BoxDetection(data=box, noise=np.array([1, 1, 1, 0.1, 0.1, 0.1]) ** 2,
                        source_identifier="l", reference=ref, obj_type="Car", score=0.8)


def _out_ctx():
    return Context(Frame(index=0, timestamp=0.0, ego=None), Seam.PERCEPTION_OUT)


def _in_ctx(ego):
    return Context(Frame(index=0, timestamp=0.0, ego=ego), Seam.PERCEPTION_INPUT)


def test_phantom_appends_a_detection():
    ref = ReferenceFrame(x=np.zeros(3), q=np.quaternion(1), reference=GlobalOrigin3D)
    data = DataContainer(0, 0.0, [_det(ref, 5.0)], "l")
    out = PhantomDetectionAttack(target_xyz=[12.0, 0.0, -1.5]).apply(data, _out_ctx())
    assert len(out) == 2 and any(getattr(d, "ID", None) == 90002 for d in out)


def test_detection_removal_drops_nearest_forward():
    ref = ReferenceFrame(x=np.zeros(3), q=np.quaternion(1), reference=GlobalOrigin3D)
    dets = [_det(ref, 20.0), _det(ref, 8.0)]
    for i, d in enumerate(dets):
        d.ID = 10 + i
    out = DetectionRemovalAttack().apply(DataContainer(0, 0.0, dets, "l"), _out_ctx())
    assert {d.ID for d in out} == {10}  # 8 m (nearest forward) removed


def test_object_spoof_appends_ahead():
    ego = make_object("car", 0, [0, 0, 0], 0.0)
    out = ObjectSpoofingAttack(target_xyz=[12.0, 0.0, 0.0]).apply(
        DataContainer(0, 0.0, [], "gt"), _in_ctx(ego))
    assert len(out) == 1 and out[0].position.x[0] == pytest.approx(12.0, abs=1e-6)


def test_object_removal_drops_forward_obstacle():
    ego = make_object("car", 0, [0, 0, 0], 0.0)
    objs = [make_object("car", 1, [15, 0, 0], 0.0), make_object("car", 2, [10, 8, 0], 0.0)]
    out = ObjectRemovalAttack(corridor=3.0, max_range=40.0).apply(
        DataContainer(0, 0.0, objs, "gt"), _in_ctx(ego))
    assert {o.ID for o in out} == {2}  # lead vehicle (ID 1) hidden
