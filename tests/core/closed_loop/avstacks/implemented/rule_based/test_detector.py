"""Stop-sign detector filtering at the learned-model boundary."""

import weakref

import numpy as np
import pytest

from avsectester.closed_loop.avstacks.implemented.rule_based.detector import CocoStopSignDetector
from tests.fakes import CocoDetectionModel


def test_detect(tmp_path):
    boxes = ((1., 2., 10., 12.), (3., 4., 9., 11.), (2., 3., 8., 10.), (4., 5., 9., 12.))
    model = CocoDetectionModel(boxes=boxes, scores=(.9, .49, .8, .5), labels=(13, 13, 1, 13))
    detector = CocoStopSignDetector("unused.pth", device="cpu", score_threshold=.5, model=model)
    rgb = np.full((20, 20, 3), 30, dtype=np.uint8)
    detections = detector.detect(rgb)
    np.testing.assert_array_equal([box for box, score in detections], [boxes[0], boxes[3]])
    assert [score for box, score in detections] == pytest.approx([.9, .5])
    assert not model.training and model.roi_heads.score_thresh == 0.
    empty = CocoStopSignDetector(
        "unused.pth", device="cpu", model=CocoDetectionModel(boxes=(), scores=(), labels=()),
    )
    assert empty.detect(rgb) == []
    with pytest.raises(ValueError, match="uint8 RGB"):
        detector.detect(np.ones((20, 20, 3)))
    with pytest.raises(FileNotFoundError, match="Existing detector weights"):
        CocoStopSignDetector(tmp_path / "missing.pth", device="cpu").detect(rgb)


def test_close():
    model = CocoDetectionModel()
    reference = weakref.ref(model)
    detector = CocoStopSignDetector("unused.pth", device="cpu", model=model)
    detector.detect(np.full((20, 20, 3), 30, dtype=np.uint8))
    del model
    detector.close()
    assert detector.model is None
    assert reference() is None
    with pytest.raises(RuntimeError, match="closed"):
        detector.detect(np.full((20, 20, 3), 30, dtype=np.uint8))
