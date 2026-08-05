"""Offline regression tests for the LiDAR-spoofing vector (object-level seam).

Needs the avstack stack (geometry / perception / tracking); skipped in core-only CI.
Does NOT need CARLA — it exercises the attack hooks + perception + tracker directly.
Covers both methods that share the vector: object spoofing (false positive) and object
removal (false negative).
"""
from __future__ import annotations

import pytest

pytest.importorskip("avstack")

import numpy as np
from avsectester.attacks.lidar_spoofing import (
    LidarSpoofAttack,
    ObjectRemovalAttack,
    ObjectSpoofingAttack,
)
from avstack.datastructs import DataContainer
from avstack.environment.objects import ObjectState
from avstack.geometry import (
    Acceleration,
    AngularVelocity,
    Attitude,
    GlobalOrigin3D,
    Position,
    Velocity,
)


def _ego_state(t: float = 0.0) -> ObjectState:
    pos = Position(np.zeros(3), GlobalOrigin3D)
    att = Attitude(np.quaternion(1), GlobalOrigin3D)  # heading +x
    ego = ObjectState("car", ID=0)
    ego.set(
        t,
        pos,
        None,
        Velocity(np.zeros(3), GlobalOrigin3D),
        Acceleration(np.zeros(3), GlobalOrigin3D),
        att,
        AngularVelocity(np.quaternion(1), GlobalOrigin3D),
    )
    return ego


def test_inject_appends_phantom_ahead():
    attack = LidarSpoofAttack(target_xyz=[12.0, 0.0, 0.0])
    ego = _ego_state()
    data = DataContainer(0, 0.0, [], "gt")
    out = attack.apply(data, ego_state=ego)
    assert len(out) == 1
    phantom = out[0]
    assert phantom.obj_type == "car"
    assert hasattr(phantom, "box")
    # placed ~12 m ahead of the ego (+x), on the ego's axis
    assert phantom.position.x[0] == pytest.approx(12.0, abs=1e-6)
    assert abs(phantom.position.x[1]) < 1e-6


def test_phantom_position_is_world_fixed_across_frames():
    attack = LidarSpoofAttack(target_xyz=[12.0, 0.0, 0.0])
    p0 = attack.apply(DataContainer(0, 0.0, [], "gt"), ego_state=_ego_state(0.0))[0]
    # ego moves forward 5 m; phantom must stay put in the world (fixed obstacle)
    ego2 = _ego_state(1.0)
    ego2.position.x[0] = 5.0
    p1 = attack.apply(DataContainer(1, 1.0, [], "gt"), ego_state=ego2)[0]
    assert np.allclose(p0.position.x, p1.position.x)


def test_phantom_propagates_to_confirmed_track():
    from avstack.modules.perception.object3d import Passthrough3DObjectDetector
    from avstack.modules.tracking.tracker3d import BasicBoxTracker3D

    attack = LidarSpoofAttack(target_xyz=[12.0, 0.0, 0.0])
    perception = Passthrough3DObjectDetector()
    tracker = BasicBoxTracker3D()
    for f in range(6):
        data = attack.apply(DataContainer(f, f * 0.05, [], "gt"), ego_state=_ego_state(f * 0.05))
        dets = perception(data, frame=f)
        assert len(dets) == 1
        tracker(dets, platform=GlobalOrigin3D)
    assert len(tracker.tracks_confirmed) == 1


def test_validate_rejects_over_budget():
    with pytest.raises(ValueError, match="budget"):
        LidarSpoofAttack(n_points=5000).validate(spec=None)


def test_lidar_spoof_alias_is_object_spoofing():
    assert LidarSpoofAttack is ObjectSpoofingAttack


def test_spoofing_and_removal_share_the_vector_bindings():
    assert ObjectSpoofingAttack.bindings is ObjectRemovalAttack.bindings


def _obj(oid: int, xyz, t: float = 0.0):
    pos = Position(np.asarray(xyz, dtype=float), GlobalOrigin3D)
    att = Attitude(np.quaternion(1), GlobalOrigin3D)
    o = ObjectState("car", ID=oid)
    o.set(
        t, pos, None,
        Velocity(np.zeros(3), GlobalOrigin3D),
        Acceleration(np.zeros(3), GlobalOrigin3D),
        att,
        AngularVelocity(np.quaternion(1), GlobalOrigin3D),
    )
    return o


def test_removal_deletes_nearest_forward_obstacle():
    # ego at origin heading +x; a lead obstacle 15 m ahead, one 8 m to the side, one behind.
    data = DataContainer(0, 0.0, [_obj(1, [15, 0, 0]), _obj(2, [10, 8, 0]), _obj(3, [-5, 0, 0])], "gt")
    attack = ObjectRemovalAttack(corridor=3.0, max_range=40.0)
    out = attack.apply(data, ego_state=_ego_state())
    ids = {o.ID for o in out}
    assert ids == {2, 3}  # the lead obstacle (ID 1) was hidden
    assert attack._removed_id == 1


def test_removal_targets_explicit_id():
    data = DataContainer(0, 0.0, [_obj(1, [15, 0, 0]), _obj(7, [20, 0, 0])], "gt")
    out = ObjectRemovalAttack(target_id=7).apply(data, ego_state=_ego_state())
    assert {o.ID for o in out} == {1}


def test_removal_noop_when_nothing_ahead():
    data = DataContainer(0, 0.0, [_obj(3, [-5, 0, 0])], "gt")  # only an object behind
    out = ObjectRemovalAttack().apply(data, ego_state=_ego_state())
    assert {o.ID for o in out} == {3}
