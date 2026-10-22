"""Native CARLA depth bytes decode to metric optical-axis depth."""

from types import SimpleNamespace

import numpy as np
import pytest

from avsectester.closed_loop.simulators.carla.sensors import decode_carla_depth


def test_decode_carla_depth():
    for metres in (0., 17.25):
        encoded = round(metres / 1000 * (2**24 - 1))
        bgra = np.array([[[encoded >> 16, (encoded >> 8) & 255, encoded & 255, 255]]], np.uint8)
        image = SimpleNamespace(height=1, width=1, raw_data=bgra.tobytes())
        assert decode_carla_depth(image)[0, 0] == pytest.approx(metres, abs=.00003)
