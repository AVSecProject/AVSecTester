"""Configuration summaries describe data without traversing arbitrary object state."""

from pathlib import Path

import numpy as np
import pytest

from avsectester.experiment.configuration_summary import summary_value, type_name
from avsectester.data_types import Control


def test_summary_value():
    summarized = summary_value({
        "image": Path("texture.png"), "shape": (1, 3), "level": np.float32(.5),
        "pixels": np.array([1, 2, 3], dtype=np.uint8),
        "command": Control(throttle=.5),
    })
    assert summarized == {
        "image": "texture.png", "shape": [1, 3], "level": .5,
        "pixels": {"array_dtype": "|u1", "array_shape": [3],
                   "sha256": "039058c6f2c0cb492c533b0a4d14ef77cc0f78abccced5287d84a1a2011cfb81"},
        "command": {"type": "avsectester.data_types:Control", "parameters": {
            "throttle": .5, "steer": 0., "brake": 0., "trajectory": None,
        }},
    }
    with pytest.raises(TypeError, match="Cannot summarize configuration parameter object"):
        summary_value(object())
    with pytest.raises(ValueError, match="finite numbers"):
        summary_value(float("nan"))


def test_type_name():
    assert type_name(Control()) == "avsectester.data_types:Control"
