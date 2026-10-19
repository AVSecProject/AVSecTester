"""Stage telemetry records outputs when emitted, before downstream mutation."""

import numpy as np

from avsectester.closed_loop.avstacks.modular import _StageCapture


def test_stage_capture_detaches_output_before_the_next_component_runs():
    capture = _StageCapture()
    output = [{"score": 0.8, "box": np.array([1.0, 2.0, 3.0])}]
    assert capture(output)[0] is output
    output[0]["score"] = 0.0
    output[0]["box"][:] = 99
    assert capture.last[0]["score"] == 0.8
    np.testing.assert_array_equal(capture.last[0]["box"], [1.0, 2.0, 3.0])
    previous = capture.last
    capture([])
    assert capture.last == [] and previous[0]["score"] == 0.8
    assert capture._recorder.log.path is None
    assert capture._recorder.log.steps == []
