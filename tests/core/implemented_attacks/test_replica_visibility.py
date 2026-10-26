"""Replica visibility counts use saved capture evidence, including unavailable treatments."""
from avsectester.evaluation import ComparisonGroup, UnavailableMetricValue
from avsectester.implemented_attacks.stop_sign_patch_attack.metrics import ReplicaVisibility
from avsectester.implemented_attacks.stop_sign_patch_attack.online import REPLICA_OBJECT_ID
from avsectester.rendering import ObjectVisibility
from avsectester.runlog import RunLog, StepRecord


def test_replica_visibility(segment):
    log = RunLog(status="completed", steps=[
        StepRecord(0, object_visibility={REPLICA_OBJECT_ID: {
            "front": ObjectVisibility("scene_visibility", (), 150)}}),
        StepRecord(1, object_visibility={REPLICA_OBJECT_ID: {
            "front": ObjectVisibility("scene_visibility", (), 99)}}),
        StepRecord(2, object_visibility={REPLICA_OBJECT_ID: {
            "side": ObjectVisibility("scene_visibility", (), 500)}}),
        StepRecord(3, object_visibility={REPLICA_OBJECT_ID: {
            "front": ObjectVisibility("scene_visibility", (), 100)}}),
        StepRecord(4, object_visibility={}),
    ])
    clean = RunLog(status="completed", steps=[StepRecord(0)])
    group = ComparisonGroup(segment, 0, {"attacked": log, "clean": clean, "empty": RunLog()})
    metric = ReplicaVisibility(camera_id="front")
    values = metric.evaluate(group)
    assert set(metric.describe().fields) == set(values["attacked"])
    assert metric.describe().fields["visible_steps"].unit == "decision steps"
    assert values["attacked"]["visible_steps"] == 2
    assert isinstance(values["clean"]["visible_steps"], UnavailableMetricValue)
    assert values["clean"]["visible_steps"].reason
    assert isinstance(values["empty"]["visible_steps"], UnavailableMetricValue)
    assert values["empty"]["visible_steps"].reason
