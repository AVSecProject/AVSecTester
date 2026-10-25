"""Small recorded worlds used by metric, report and visualization tests."""

import pytest

from avsectester.evaluation import ComparisonGroup, TreatmentPair


@pytest.fixture
def make_group(segment):
    return lambda runs: ComparisonGroup("attack", segment, 0, runs)


@pytest.fixture
def make_pair():
    return lambda first, second: TreatmentPair("clean", "attacked", first, second)


@pytest.fixture
def stop_line_group(segment):
    """A selected map stop line, independent of the trajectory's recorded route."""
    from dataclasses import replace

    import numpy as np

    from avsectester.maps import ScenarioMap, StopLine

    def make(runs, *, line_x=1.25, line_end_y=1.):
        line = StopLine("line", np.array([[line_x, -line_end_y, 0.],
                                          [line_x, line_end_y, 0.]]), "stop", "stop_sign")
        selected = replace(segment, user_data={"stop_line_id": line.id})
        return ComparisonGroup("attack", selected, 0, runs,
                               ScenarioMap(stop_lines={line.id: line}))
    return make
