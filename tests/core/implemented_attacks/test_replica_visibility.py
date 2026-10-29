"""ReplicaVisibility counts a decision when any selected camera sees the replica."""
import pytest

from avsectester.evaluation import ComparisonGroup, UnavailableMetricValue
from avsectester.implemented_attacks.stop_sign_patch_attack.metrics import ReplicaVisibility
from avsectester.rendering import ObjectVisibility
from avsectester.runlog import RunLog, StepRecord


def visibility(**pixels):
    return {"stop_sign_replica": {camera: ObjectVisibility("scene_visibility", (), count)
                                  for camera, count in pixels.items()}}


def test_camera_selection(segment):
    patched = RunLog(status="completed", steps=[
        StepRecord(0, object_visibility=visibility(front=100, tele=0)),
        StepRecord(1, object_visibility=visibility(front=20, tele=150)),
        StepRecord(2, object_visibility=visibility(front=20, tele=20)),
        StepRecord(3, object_visibility={"stop_sign_replica": {}}),
    ])
    clean = RunLog(status="completed", steps=[StepRecord(0)])
    group = ComparisonGroup(segment, 0, {"patched": patched, "clean": clean})
    # By default every recorded camera counts; a decision is visible when any camera meets the threshold.
    values = ReplicaVisibility().evaluate(group)
    assert values["patched"]["visible_steps"] == 2
    assert isinstance(values["clean"]["visible_steps"], UnavailableMetricValue)
    assert ReplicaVisibility(camera_ids=["front"]).evaluate(group)["patched"]["visible_steps"] == 1
    assert ReplicaVisibility(camera_ids=["side"]).evaluate(group)["patched"]["visible_steps"] == 0
    for camera_ids in ([], [""], "front"):
        with pytest.raises(ValueError, match="camera_ids"):
            ReplicaVisibility(camera_ids=camera_ids)
