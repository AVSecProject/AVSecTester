"""Shared numeric constraints through their public functions."""

import numpy as np
import pytest

from avsectester.validation import (
    require_nonnegative_integer, require_nonnegative_number,
    require_positive_integer, require_positive_number,
)


def test_require_nonnegative_integer():
    require_nonnegative_integer(0, "position")
    require_nonnegative_integer(7, "position")
    for value in (-1, True, np.inf, np.nan):
        with pytest.raises(ValueError, match="^position must be a nonnegative integer$"):
            require_nonnegative_integer(value, "position")


def test_require_positive_integer():
    require_positive_integer(1, "steps")
    require_positive_integer(7, "steps")
    for value in (0, True, np.inf, np.nan):
        with pytest.raises(ValueError, match="^steps must be a positive integer$"):
            require_positive_integer(value, "steps")


def test_require_nonnegative_number():
    require_nonnegative_number(0, "distance")
    require_nonnegative_number(np.float32(.25), "distance")
    for value in (-.1, True, np.inf, np.nan, "0"):
        with pytest.raises(ValueError, match="^distance must be a nonnegative number$"):
            require_nonnegative_number(value, "distance")


def test_require_positive_number():
    require_positive_number(1, "time_step")
    require_positive_number(np.float64(.25), "time_step")
    for value in (0, True, np.inf, np.nan, 1j):
        with pytest.raises(ValueError, match="^time_step must be a positive number$"):
            require_positive_number(value, "time_step")
