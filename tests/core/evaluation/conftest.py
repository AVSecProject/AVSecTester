"""Small recorded worlds used by metric, report and visualization tests."""

import pytest

from avsectester.evaluation import ComparisonGroup, TreatmentPair


@pytest.fixture
def make_group(segment):
    return lambda runs: ComparisonGroup("attack", segment, 0, runs)


@pytest.fixture
def make_pair():
    return lambda first, second: TreatmentPair("clean", "attacked", first, second)
