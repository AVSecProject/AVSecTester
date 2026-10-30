"""SignDetection separates undetected runs from runs whose replica is never visible."""
import pytest

from avsectester.evaluation import ComparisonGroup, UnavailableMetricValue
from avsectester.implemented_attacks.stop_sign_patch_attack.metrics import SignDetection
from avsectester.rendering import ObjectVisibility
from avsectester.runlog import RunLog, StepRecord


def run(*pixels):
    return RunLog(status="completed", steps=[
        StepRecord(index, components={"detections": []}, object_visibility={"stop_sign_replica": {
            "front": ObjectVisibility("scene_visibility", (), count)}})
        for index, count in enumerate(pixels)])


def test_undetected_and_invisible_runs(segment):
    group = ComparisonGroup(segment, 0, {"undetected": run(150, 150), "invisible": run(0, 0)})
    values = SignDetection(camera_id="front").evaluate(group)
    assert values["undetected"]["detected_steps"] == 0
    assert values["undetected"]["detection_rate"] == 0.
    assert isinstance(values["undetected"]["first_detection_line_distance_m"], UnavailableMetricValue)
    assert isinstance(values["invisible"]["detection_rate"], UnavailableMetricValue)
    with pytest.raises(ValueError, match="camera_id"):
        SignDetection(camera_id="")
    with pytest.raises(ValueError, match="min_visible_pixels"):
        SignDetection(camera_id="front", min_visible_pixels=-1)
