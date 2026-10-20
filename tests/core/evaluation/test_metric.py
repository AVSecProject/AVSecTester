"""Every metric uses the same numerical and missing-evidence contract template."""

from operator import itemgetter

import pytest

from avsectester.closed_loop.datatypes import Detection
from avsectester.evaluation import (
    DetectionMetrics, DrivingMetrics, StopImpact, TrajectoryDeviation, UnavailableMetricValue,
)
from avsectester.runlog import RunLog


def unchanged(value):
    return value


def without_capture(log):
    for step in log.steps:
        step.physical_pre = None
    return log


@pytest.fixture
def metric_inputs(make_actor, make_log, make_group, make_pair):
    return {"actor": make_actor, "log": make_log, "group": make_group, "pair": make_pair}


CASES = [
    pytest.param(
        DrivingMetrics, {},
        lambda data: data["group"]({"attacked": data["log"](
            [2., 0.], [0., 1.], actors={"lead": data["actor"](10.)})}),
        lambda data: data["group"]({"attacked": RunLog()}), itemgetter("attacked"), {
            "final_speed": 0., "peak_speed": 2., "mean_speed": 1., "stopped": True,
            "brake_steps": 1, "min_distance": 8., "min_role_distance": 8., "collision": False,
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
            "role_detection_rate.host": 1.,
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
]


@pytest.mark.parametrize("metric_type,options,build_input,build_missing,select_result,expected", CASES)
def test_evaluate(metric_type, options, build_input, build_missing, select_result, expected, metric_inputs):
    metric = metric_type(**options)
    values = select_result(metric.evaluate(build_input(metric_inputs)))
    for name, value in expected.items():
        assert values[name] == pytest.approx(value), name
    missing = select_result(metric.evaluate(build_missing(metric_inputs)))
    assert missing and all(isinstance(value, UnavailableMetricValue) and value.reason
                           for value in missing.values())
