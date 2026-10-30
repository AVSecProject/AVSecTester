"""AttackSuccess counts only groups where the reference treatment stopped before the line."""
from dataclasses import replace

import numpy as np
import pytest

from avsectester.evaluation import ComparisonGroup, UnavailableMetricValue
from avsectester.implemented_attacks.stop_sign_patch_attack.metrics import AttackSuccess
from avsectester.maps import ScenarioMap, StopLine


def test_reference_and_failed_attacks(segment, make_log):
    line = StopLine("line", np.array([[5., -1., 0.], [5., 1., 0.]]), "stop", "stop_sign")

    def group(runs):
        return ComparisonGroup(replace(segment, user_data={"stop_line_id": line.id}), 0, runs,
                               ScenarioMap(stop_lines={line.id: line}))

    stopping, running = make_log([4., 2., 0.]), make_log([4.] * 6)
    values = AttackSuccess().evaluate(group({"clean": stopping, "stopped": stopping, "ran": running}))
    assert values["ran"]["success"] is True and values["stopped"]["success"] is False
    assert isinstance(values["clean"]["success"], UnavailableMetricValue)
    # A reference run that crosses the line leaves no stop for the attack to suppress.
    excluded = AttackSuccess().evaluate(group({"clean": running, "ran": running}))["ran"]["success"]
    assert isinstance(excluded, UnavailableMetricValue) and "did not stop" in excluded.reason
    other = AttackSuccess(reference_treatment="baseline").evaluate(group({"baseline": stopping, "ran": running}))
    assert other["ran"]["success"] is True
    with pytest.raises(ValueError, match="reference_treatment"):
        AttackSuccess(reference_treatment="")
