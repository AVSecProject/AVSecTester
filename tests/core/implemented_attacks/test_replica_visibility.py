"""ReplicaVisibility chooses one recorded camera or requires an explicit selection."""
import pytest

from avsectester.evaluation import ComparisonGroup, UnavailableMetricValue
from avsectester.implemented_attacks.stop_sign_patch_attack.metrics import ReplicaVisibility
from avsectester.rendering import ObjectVisibility
from avsectester.runlog import RunLog, StepRecord


def test_camera_selection(segment):
    one = RunLog(status="completed", steps=[
        StepRecord(0, object_visibility={"stop_sign_replica": {
            "front": ObjectVisibility("scene_visibility", (), 100)}}),
        StepRecord(1, object_visibility={"stop_sign_replica": {}}),
    ])
    clean = RunLog(status="completed", steps=[StepRecord(0)])
    group = ComparisonGroup(segment, 0, {"patched": one, "clean": clean})
    values = ReplicaVisibility().evaluate(group)
    assert values["patched"]["visible_steps"] == 1
    assert isinstance(values["clean"]["visible_steps"], UnavailableMetricValue)
    multiple = RunLog(status="completed", steps=[
        StepRecord(0, object_visibility={"stop_sign_replica": {
            "front": ObjectVisibility("scene_visibility", (), 100),
            "side": ObjectVisibility("scene_visibility", (), 150)}}),
    ])
    with pytest.raises(ValueError, match="camera_id explicitly"):
        ReplicaVisibility().evaluate(ComparisonGroup(segment, 0, {"patched": multiple}))
