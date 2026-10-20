"""RunLog metrics cover real measurements, unavailable evidence and pairwise comparisons."""

import numpy as np
import pytest

from avsectester.closed_loop.datatypes import Control, Detection, WorldSnapshot
from avsectester.evaluation import (
    DetectionMetrics, DrivingMetrics, StopImpact, TrajectoryDeviation, UnavailableMetricValue,
)


def test_driving_uses_physical_speed_commands_and_role_centres(actor_factory, log_factory, group_factory):
    values = DrivingMetrics().evaluate(group_factory({
        "clean": log_factory([0, 3, .5], [0, 1, 1], actors={"host": actor_factory(10)})},
        roles={"host": ("host",)}))["clean"]
    assert values == dict(final_speed=.5, peak_speed=3, mean_speed=pytest.approx(3.5 / 3),
                          brake_steps=2, stopped=True, min_distance=7, min_role_distance=7, collision=False)


def test_collision_is_oriented_and_uses_local_box_centre(actor_factory, log_factory, group_factory):
    def collision(actor):
        return DrivingMetrics().evaluate(group_factory({
            "clean": log_factory([1], actors={"other": actor})}))["clean"]["collision"]
    assert collision(actor_factory(3.5, yaw=np.pi / 4)) is True
    assert collision(actor_factory(20, box_center=(-18, 0, 0))) is True
    assert collision(actor_factory(0, 4, yaw=np.pi / 2)) is False
    assert collision(actor_factory(10, extent=None)).value_type == "boolean"
    assert collision(actor_factory(10, extent=(0, 0, 0))) is False
    assert collision(actor_factory(1, extent=(0, 0, 0))) is True


def test_missing_values_carry_reasons_without_mutating_metric(log_factory, group_factory):
    metric = DrivingMetrics()
    empty = metric.evaluate(group_factory({"clean": log_factory([])}))["clean"]
    assert all(isinstance(value, UnavailableMetricValue) for value in empty.values())
    assert empty["final_speed"].reason
    assert empty["stopped"].value_type == "boolean"
    assert vars(metric) == {"stop_speed": .5}
    run = log_factory([1])
    run.steps[0].executed_command = Control(trajectory=[[1, 0, 0]])
    unavailable = metric.evaluate(group_factory({"clean": run}))["clean"]["brake_steps"]
    assert isinstance(unavailable, UnavailableMetricValue) and "actuator" in unavailable.reason


def test_trajectory_is_symmetric_and_matches_decision_indices(log_factory, pair_factory):
    metric = TrajectoryDeviation()
    first, second = log_factory([1, 1]), log_factory([1, 1], shift=3)
    values = metric.evaluate(pair_factory(first, second))
    assert values == dict(mean_deviation=3, max_deviation=3, final_deviation=3)
    assert metric.evaluate(pair_factory(second, first)) == values
    second.steps[1].step = 9
    assert isinstance(metric.evaluate(pair_factory(first, second))["mean_deviation"], UnavailableMetricValue)


def test_detection_greedy_matching_and_role_rates(actor_factory, log_factory, group_factory):
    predictions = (Detection((10.1, 0, 0), (4, 2, 2), 0, "Car", .9),
                   Detection((10.2, 0, 0), (4, 2, 2), 0, "Car", .8),
                   Detection((30, 0, 0), (4, 2, 2), 0, "Car", .8))
    run = log_factory([1], actors={"host": actor_factory(10), "miss": actor_factory(20)}, detections=predictions)
    values = DetectionMetrics().evaluate(group_factory({"clean": run}, {
        "host": ("host",), "missing": ("absent",)}))["clean"]
    assert (values["tp"], values["fp"], values["fn"]) == (1, 2, 1)
    assert values["precision"] == pytest.approx(1 / 3)
    assert values["recall"] == .5 and values["f1"] == .4
    assert values["role_detection_rate.host"] == 1
    assert isinstance(values["role_detection_rate.missing"], UnavailableMetricValue)
    assert "tn" not in values and "ap" not in values


def test_detection_uses_true_ego_box_centres_and_category_mapping(actor_factory, log_factory, group_factory):
    run = log_factory([1], shift=5,
                      actors={"host": actor_factory(15, box_center=(1, 0, 0)), "far": actor_factory(100)},
                      detections=(Detection((11, 0, 0), (4, 2, 2), 0, "Car", .9),))
    metric = DetectionMetrics(category_map={"Car": "vehicle", "car": "vehicle"})
    assert metric.evaluate(group_factory({"clean": run}))["clean"]["tp"] == 1
    assert DetectionMetrics(category_map={"Car": "person", "car": "vehicle"}).evaluate(
        group_factory({"clean": run}))["clean"]["fp"] == 1
    rotated = log_factory([1], actors={"host": actor_factory(0, 10)},
                          detections=(Detection((10, 0, 0), (4, 2, 2), 0),))
    rotated.steps[0].physical_pre = WorldSnapshot(0, 0, ego=actor_factory(yaw=np.pi / 2),
                                                 actors=rotated.steps[0].physical_pre.actors)
    assert DetectionMetrics().evaluate(group_factory({"clean": rotated}))["clean"]["tp"] == 1


def test_detection_range_filters_predictions_and_truths_in_bev(actor_factory, log_factory, group_factory):
    run = log_factory([1], actors={"near": actor_factory(5, z=100), "far": actor_factory(100)},
                      detections=(Detection((5, 0, 100), (0, 0, 0), 0),
                                  Detection((100, 0, 0), (0, 0, 0), 0)))
    assert DetectionMetrics(max_range=10).evaluate(group_factory({"run": run}))["run"]["tp"] == 1
    assert DetectionMetrics(max_range=10).evaluate(group_factory({"run": run}))["run"]["fp"] == 0
    assert DetectionMetrics(max_range=None).evaluate(group_factory({"run": run}))["run"]["tp"] == 2
    metric = DetectionMetrics(category_map={"car": "car"})
    run.steps[0].physical_pre.actors["near"] = actor_factory(5, category=None)
    assert metric.evaluate(group_factory({"run": run}))["run"]["tp"] == 0


def test_detection_unavailable_and_empty_ratios_are_explicit(log_factory, group_factory):
    with pytest.raises(ValueError, match="TestDriver"):
        DetectionMetrics().evaluate(group_factory({"clean": log_factory([1], detections=None)}))
    values = DetectionMetrics().evaluate(group_factory({"clean": log_factory([1])}))["clean"]
    assert values["tp"] == values["fp"] == values["fn"] == 0
    assert all(isinstance(values[name], UnavailableMetricValue) for name in ("precision", "recall", "f1"))
    run = log_factory([1])
    run.steps[0].physical_pre = WorldSnapshot(0, 0)
    assert isinstance(DetectionMetrics().evaluate(group_factory({"clean": run}))["clean"]["tp"], UnavailableMetricValue)


def test_stop_impact_reports_both_directions_and_duration_boundaries(log_factory, pair_factory):
    first, second = log_factory([0, 3, 0, 0, 0]), log_factory([0, 3, 2, 2, 2])
    values = StopImpact().evaluate(pair_factory(first, second))
    assert values["a_to_b.reference_drove"] is True
    assert values["a_to_b.suppressed_stop"] is True
    assert values["a_to_b.overdrive_steps"] == 3
    assert values["b_to_a.induced_stop"] is True
    assert StopImpact(min_overdrive_steps=4).evaluate(pair_factory(first, second))["a_to_b.suppressed_stop"] is False
    result = StopImpact().evaluate(pair_factory(log_factory([0, 3, 3]), log_factory([0, 3, .5], [0, 1, 1])))
    assert result["a_to_b.induced_braking"] and result["a_to_b.induced_stop"]
    assert "attack_succeeded" not in result


@pytest.mark.parametrize("constructor,kwargs", [
    (DrivingMetrics, {"stop_speed": -1}), (DetectionMetrics, {"max_range": 0}),
    (DetectionMetrics, {"match_distance": float("inf")}), (StopImpact, {"min_overdrive_steps": 0}),
])
def test_metric_options_are_validated(constructor, kwargs):
    with pytest.raises(ValueError):
        constructor(**kwargs)


def test_missing_extent_does_not_discard_recorded_box_centre(actor_factory, log_factory, group_factory):
    run = log_factory([1], actors={"host": actor_factory(10, extent=None, box_center=(1, 0, 0))},
                      detections=(Detection((11, 0, 0), (4, 2, 2), 0),))
    values = DrivingMetrics().evaluate(group_factory({"run": run}))["run"]
    assert values["min_distance"] == 10
    assert isinstance(values["collision"], UnavailableMetricValue)
    assert DetectionMetrics().evaluate(group_factory({"run": run}))["run"]["tp"] == 1


def test_empty_pair_and_missing_pose_return_explained_fields(log_factory, pair_factory):
    values = StopImpact().evaluate(pair_factory(log_factory([]), log_factory([1])))
    assert all(isinstance(value, UnavailableMetricValue) for value in values.values())
    assert values["a_to_b.induced_stop"].value_type == "boolean"
    first, second = log_factory([1]), log_factory([1])
    second.steps[0].physical_post = None
    assert isinstance(TrajectoryDeviation().evaluate(pair_factory(first, second))["mean_deviation"], UnavailableMetricValue)
