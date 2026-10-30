"""Every metric uses the same numerical and missing-evidence contract template."""

from operator import itemgetter
from dataclasses import replace
from types import SimpleNamespace

import numpy as np

import pytest

from avsectester.data_types import Detection
from avsectester.evaluation import (
    ComparisonGroup, DetectionMetrics, DrivingMetrics, ReasoningPhraseRate, StopImpact,
    StopLineMetrics, TreatmentPair,
    TrajectoryDeviation, UnavailableMetricValue, signed_stop_line_distances,
)
from avsectester.runlog import EventRecord, RunLog
from avsectester.maps import ScenarioMap, StopLine
from avsectester.rendering import ObjectVisibility
from avsectester.implemented_attacks.stop_sign_patch_attack.metrics import (
    AttackSuccess, ReplicaVisibility, SignDetection,
)


@pytest.fixture
def make_group(segment):
    return lambda runs: ComparisonGroup(segment, 0, runs)


@pytest.fixture
def make_pair():
    return lambda first, second: TreatmentPair("clean", "attacked", first, second)


@pytest.fixture
def stop_line_group(segment):
    """A selected map stop line, independent of the trajectory's recorded route."""
    def make(runs, *, line_x=1.25, line_end_y=1.):
        line = StopLine("line", np.array([[line_x, -line_end_y, 0.],
                                          [line_x, line_end_y, 0.]]), "stop", "stop_sign")
        selected = replace(segment, user_data={"stop_line_id": line.id})
        return ComparisonGroup(selected, 0, runs,
                               ScenarioMap(stop_lines={line.id: line}))
    return make


def unchanged(value):
    return value


def without_capture(log):
    for step in log.steps:
        step.physical_pre = None
    return log


@pytest.fixture
def sign_group(segment):
    """A stop-sign attack segment that stores its stop line points in user_data."""
    points = np.array([[1.25, -1., 0.], [1.25, 1., 0.]])
    return lambda runs: ComparisonGroup(replace(segment, user_data={"stop_line_points": points}), 0, runs)


@pytest.fixture
def metric_inputs(make_actor, make_log, make_group, make_pair, stop_line_group, sign_group):
    return {"actor": make_actor, "log": make_log, "group": make_group, "pair": make_pair,
            "stop_line_group": stop_line_group, "sign_group": sign_group}


def with_reasoning(log):
    for step, text in zip(log.steps, ("STOP SIGN ahead", "road clear")):
        step.components["policy"] = {"reasoning_text": text}
    return log


def with_replica_visibility(log):
    log.steps[0].object_visibility = {"stop_sign_replica": {
        "front": ObjectVisibility("scene_visibility", (), 12)}}
    log.steps[1].object_visibility = {"stop_sign_replica": {
        "front": ObjectVisibility("scene_visibility", (), 2)}}
    return log


def with_sign_detections(log):
    # Visible, hidden and visible decisions. The last two log a detection.
    for step, pixels, detections in zip(log.steps, (12, 2, 12), ([], [{"score": .9}], [{"score": .8}])):
        step.object_visibility = {"stop_sign_replica": {
            "front": ObjectVisibility("scene_visibility", (), pixels)}}
        step.components["detections"] = detections
    return log


CASES = [
    pytest.param(
        DrivingMetrics, {},
        lambda data: data["group"]({"attacked": data["log"](
            [2., 0.], [0., 1.], actors={"lead": data["actor"](10.)})}),
        lambda data: data["group"]({"attacked": RunLog()}), itemgetter("attacked"), {
            "final_speed": 0., "peak_speed": 2., "mean_speed": 1., "stopped": True,
            "brake_steps": 1, "min_distance": 8., "collision": False,
        }, id="driving"),
    pytest.param(
        TrajectoryDeviation, {},
        lambda data: data["pair"](data["log"]([2., 2.]), data["log"]([2., 2.], shift=3.)),
        lambda data: data["pair"](data["log"]([2., 2.]), RunLog()), unchanged,
        {"mean_deviation": 3., "max_deviation": 3., "final_deviation": 3.}, id="trajectory"),
    pytest.param(
        DetectionMetrics, {"category_map": {"Car": "vehicle"}},
        lambda data: data["group"]({"attacked": data["log"](
            [2.], actors={"lead": data["actor"](10., category="vehicle")}, detections=[
                Detection((10., 0., 80.), (4., 2., 2.), 0., "Car"),
                Detection((20., 0., 0.), (1., 1., 1.), 0.),
                Detection((100., 0., 0.), (1., 1., 1.), 0., "Car")])}),
        lambda data: data["group"]({"attacked": without_capture(data["log"]([2.]))}),
        itemgetter("attacked"), {
            "tp": 1, "fp": 1, "fn": 0, "precision": .5, "recall": 1., "f1": 2 / 3,
        }, id="detection"),
    pytest.param(
        StopImpact, {},
        lambda data: data["pair"](data["log"]([2.] * 3), data["log"]([0.] * 3, [1.] * 3)),
        lambda data: data["pair"](data["log"]([2., 2.]), RunLog()), unchanged, {
            "a_to_b.reference_drove": True, "a_to_b.induced_braking": True,
            "a_to_b.induced_stop": True, "a_to_b.suppressed_stop": False,
            "a_to_b.overdrive_steps": 0, "b_to_a.reference_drove": False,
            "b_to_a.induced_braking": False, "b_to_a.induced_stop": False,
            "b_to_a.suppressed_stop": True, "b_to_a.overdrive_steps": 3,
        }, id="stop-impact"),
    pytest.param(
        StopLineMetrics, {},
        lambda data: data["stop_line_group"]({"attacked": data["log"](
            [4., 0.], actors={"lead": data["actor"](10., category="vehicle")})}),
        lambda data: data["group"]({"attacked": data["log"]([4., 0.])}), itemgetter("attacked"), {
            "stopped_before_line": False, "crossed_without_stop": True, "line_crossed": True,
            "speed_at_line": 3., "min_speed_before_line": 4., "max_deceleration": 40.,
            "time_to_stop": .2, "min_vehicle_distance_after_line": 8.,
            "initial_distance_to_line": 1.25, "final_distance_to_line": -.75,
            "distance_to_line_at_stop": -.75, "distance_to_line_at_min_speed": 1.25,
            "distance_to_line_at_brake_onset": .25,
        }, id="stop-line"),
    pytest.param(
        ReasoningPhraseRate, {"phrase": "stop sign"},
        lambda data: data["group"]({"attacked": with_reasoning(data["log"]([2., 2.]))}),
        lambda data: data["group"]({"attacked": data["log"]([2., 2.])}), itemgetter("attacked"),
        {"phrase_rate": .5}, id="reasoning-phrase"),
    pytest.param(
        ReplicaVisibility, {"camera_ids": ["front"], "min_visible_pixels": 5},
        lambda data: data["group"]({"attacked": with_replica_visibility(data["log"]([2., 2.]))}),
        lambda data: data["group"]({"attacked": data["log"]([2., 2.])}), itemgetter("attacked"),
        {"visible_steps": 1}, id="replica-visibility"),
    pytest.param(
        SignDetection, {"camera_id": "front", "min_visible_pixels": 5},
        lambda data: data["sign_group"]({"attacked": with_sign_detections(data["log"]([2., 2., 2.]))}),
        lambda data: data["group"]({"attacked": data["log"]([2., 2.])}), itemgetter("attacked"),
        {"detected_steps": 2, "detection_rate": .5, "first_detection_line_distance_m": .25},
        id="sign-detection"),
    pytest.param(
        AttackSuccess, {},
        lambda data: data["stop_line_group"](
            {"clean": data["log"]([4., 2., 0.]), "attacked": data["log"]([4.] * 6)}, line_x=5.),
        lambda data: data["group"]({"clean": data["log"]([4., 2., 0.]), "attacked": data["log"]([4.] * 6)}),
        itemgetter("attacked"), {"success": True}, id="attack-success"),

]


@pytest.mark.parametrize("metric_type,options,build_input,build_missing,select_result,expected", CASES)
def test_metric_contract(metric_type, options, build_input, build_missing, select_result, expected, metric_inputs):
    metric = metric_type(**options)
    measurements = metric.evaluate(build_input(metric_inputs))
    values = select_result(measurements)
    assert set(metric.describe().fields) == set(values)
    for name, value in expected.items():
        assert values[name] == pytest.approx(value), name
    missing = select_result(metric.evaluate(build_missing(metric_inputs)))
    assert all(isinstance(missing[name], UnavailableMetricValue) and missing[name].reason
               for name in expected)


def test_stop_line_metrics(make_log, make_actor, stop_line_group):
    with pytest.raises(ValueError, match="nonnegative"):
        StopLineMetrics(brake_deceleration_mps2=-1.)
    with pytest.raises(ValueError, match="nonnegative"):
        StopLineMetrics(endpoint_tolerance_m=-1.)

    log = make_log([8., 4.])
    values = StopLineMetrics().evaluate(stop_line_group({"attacked": log}))["attacked"]
    assert values["speed_at_line"] == 7.  # x=1.25 is one quarter of the x=1 -> 2 step.
    stopping = make_log([4., 2., 0.])
    stopped = StopLineMetrics().evaluate(stop_line_group({"attacked": stopping}, line_x=5.))["attacked"]
    assert stopped["distance_to_line_at_stop"] == 2.
    assert stopped["distance_to_line_at_min_speed"] == 2.
    assert stopped["distance_to_line_at_brake_onset"] == 4.
    assert stopped["time_to_stop"] == pytest.approx(.3)
    uniform = StopLineMetrics().evaluate(stop_line_group({"attacked": make_log([3., 3.])}))["attacked"]
    assert uniform["line_crossed"] is True
    assert uniform["distance_to_line_at_min_speed"] == 1.25  # First tied minimum before crossing.
    assert isinstance(uniform["distance_to_line_at_stop"], UnavailableMetricValue)
    assert isinstance(uniform["distance_to_line_at_brake_onset"], UnavailableMetricValue)
    distant = make_log([8., 4.])
    for step in distant.steps:
        step.physical_pre = replace(step.physical_pre,
                                    ego=make_actor(step.physical_pre.ego.pose[0, 3], 4.01))
        step.physical_post = replace(step.physical_post,
                                     ego=make_actor(step.physical_post.ego.pose[0, 3], 4.01))
    beyond = StopLineMetrics().evaluate(stop_line_group({"attacked": distant}))["attacked"]
    assert beyond["line_crossed"] is False
    assert beyond["crossed_without_stop"] is False
    assert isinstance(beyond["speed_at_line"], UnavailableMetricValue)
    initialization = RunLog(events=[EventRecord("initialization", data={
        "physical_state": log.steps[0].physical_pre,
    })])
    initial = StopLineMetrics().evaluate(stop_line_group({"attacked": initialization}))["attacked"]
    assert initial["initial_distance_to_line"] == 1.25
    assert initial["min_speed_before_line"] == 8.
    assert initial["line_crossed"] is False
    assert isinstance(initial["max_deceleration"], UnavailableMetricValue)
    assert isinstance(initial["distance_to_line_at_brake_onset"], UnavailableMetricValue)


def test_reasoning_phrase_rate(make_log, make_group):
    log = make_log([1., 1., 1.])
    log.steps[0].components["policy"] = SimpleNamespace(reasoning_text="STOP SIGN ahead")
    log.steps[1].components["policy"] = {"reasoning_text": ""}
    metric = ReasoningPhraseRate("stop sign")
    values = metric.evaluate(make_group({"attacked": log}))["attacked"]
    assert values["phrase_steps"] == 1
    assert values["reasoning_steps"] == 2
    assert isinstance(values["phrase_rate"], UnavailableMetricValue)


def test_signed_stop_line_distances():
    distances, crossing = signed_stop_line_distances(
        [[0., 4., 50.], [2., 4., -50.]], [[1.25, -1., 0.], [1.25, 1., 0.]],
    )
    assert crossing == (0, .625)
    assert distances == pytest.approx([1.25, -.75])

    # Both bends and repeated vertices retain the first crossing along the trace.
    line = np.array([[1., -1., 0.], [1., -1., 0.], [1., 1., 0.], [3., 1., 0.]])
    distances, crossing = signed_stop_line_distances(
        [[0., 0., 9.], [2., 0., -5.], [2., 2., 0.]], line,
    )
    assert crossing == (0, .5)
    assert distances == pytest.approx([1., -1., -1.])
    np.testing.assert_array_equal(line[:, 0], [1., 1., 1., 3.])
    with pytest.raises(ValueError, match="M >= 2"):
        signed_stop_line_distances([[0., 0., 0.]], [[1., 0., 0.]])
    with pytest.raises(ValueError, match="nonzero horizontal"):
        signed_stop_line_distances([[0., 0., 0.]], [[1., 0., 0.], [1., 0., 1.]])
