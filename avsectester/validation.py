"""Reusable value checks for framework components and user implementations.

Each function raises ``ValueError`` with the supplied parameter name when the value does not
meet its constraint. Accepted values are left unchanged; these checks perform no conversion.
"""

import math
from numbers import Real


def require_nonnegative_integer(value: int, name: str) -> None:
    """Require an integer greater than or equal to zero, excluding booleans."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


def require_positive_integer(value: int, name: str) -> None:
    """Require an integer greater than zero, excluding booleans, without converting it."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def require_nonnegative_number(value: float, name: str) -> None:
    """Require a finite real number greater than or equal to zero, excluding booleans."""
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a nonnegative number")


def require_positive_number(value: float, name: str) -> None:
    """Require a finite real number greater than zero, excluding booleans."""
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive number")
