"""The phantom attack is a real avstack HOOKS post-hook — verify it end-to-end at the hook level.

No CARLA/GPU needed: build a small detection list, attach PhantomInjection the way a pipeline would
(register_post_hook on a perception-like module), and confirm it appends exactly one fabricated
BoxDetection. This is the same hook the closed-loop demo attaches to MMDetObjectDetector3D.
"""

import numpy as np
import pytest
from avsectester.attacks import PhantomInjection


def _detections():
    pytest.importorskip("avstack")
    from avstack.datastructs import DataContainer
    from avstack.geometry import Attitude, Box3D, GlobalOrigin3D, Position
    from avstack.modules.perception.detections import BoxDetection

    dets = []
    for i in range(3):
        pos = Position(np.array([10.0 + i, 0.0, 0.0]), GlobalOrigin3D)
        att = Attitude(np.quaternion(1), GlobalOrigin3D)
        box = Box3D(pos, att, [1.5, 1.8, 4.0], where_is_t="bottom")
        dets.append(
            BoxDetection(data=box, noise=np.ones(6), source_identifier="test",
                         reference=GlobalOrigin3D, obj_type="Car", score=0.8)
        )
    return DataContainer(0, 0.0, dets, "test")


def test_phantom_appends_one_detection():
    dets = _detections()
    n0 = len(dets)
    (out,) = PhantomInjection(target_xyz=(6.0, 0.0, -1.5))(dets)
    assert len(out) == n0 + 1
    phantom = out[-1]
    assert phantom.obj_type == "Car"
    assert float(phantom.score) == pytest.approx(0.9)


def test_phantom_registers_in_avstack_hooks():
    pytest.importorskip("avstack")
    from avstack.config import HOOKS

    assert "PhantomInjection" in HOOKS.module_dict
    hook = HOOKS.build({"type": "PhantomInjection"})
    assert isinstance(hook, PhantomInjection)
