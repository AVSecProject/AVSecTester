"""Stop-sign detector filtering at the learned-model boundary."""

import weakref

import numpy as np
import pytest
import torch

from avsectester.closed_loop.avstacks.implemented.rule_based.detector import CocoStopSignDetector
from avsectester.data_types import TargetSpec
from tests.fakes import CocoDetectionModel, ImageScoreDetector


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


def test_target_score():
    model = ImageScoreDetector(
        [[0., 0., 4., 4.], [0., 0., 4., 4.], [0., 0., 1., 1.], [0., 0., 4., 4.]],
        [13, 1, 13, 13], [.2, .94, .93, .7],
    )
    detector = CocoStopSignDetector("unused.pth", device="cpu", model=model)
    image = torch.full((3, 8, 8), .5, requires_grad=True)
    score = detector.target_score(image, TargetSpec((0., 0., 4., 4.), "stop sign"))
    assert score.item() == pytest.approx(.75)
    gradient, = torch.autograd.grad(score, image)
    torch.testing.assert_close(gradient, torch.full_like(image, .1 / image.numel()))
    assert model.roi_heads.score_thresh == 0.
    assert not model.training and not model.weight.requires_grad
    with pytest.raises(ValueError, match="finite float RGB"):
        detector.target_score(torch.ones(8, 8), TargetSpec((0., 0., 4., 4.)))
    with pytest.raises(ValueError, match="TargetSpec for a stop sign"):
        detector.target_score(image, TargetSpec((0., 0., 4., 4.), "car"))
    with pytest.raises(ValueError, match="finite float RGB"):
        detector.target_score(torch.full_like(image, float("nan")), TargetSpec((0., 0., 4., 4.)))
    with pytest.raises(ValueError, match="positive width and height"):
        detector.target_score(image, TargetSpec((4., 0., 0., 4.)))
    detector.close()
    with pytest.raises(RuntimeError, match="closed"):
        detector.target_score(image, TargetSpec((0., 0., 4., 4.)))


def test_target_score_without_matching_detections():
    image = torch.ones(3, 4, 4, requires_grad=True)
    empty = CocoStopSignDetector("unused.pth", device="cpu", model=ImageScoreDetector([], [], []))
    zero = empty.target_score(image, TargetSpec((0., 0., 2., 2.)))
    assert zero.item() == 0.
    gradient, = torch.autograd.grad(zero, image)
    torch.testing.assert_close(gradient, torch.zeros_like(image))
    unrelated = CocoStopSignDetector("unused.pth", device="cpu", model=ImageScoreDetector(
        [[0., 0., 2., 2.], [2., 2., 4., 4.]], [1, 13], [.9, .8],
    ))
    unmatched = unrelated.target_score(image, TargetSpec((0., 0., 2., 2.)))
    assert unmatched.item() == 0.
    gradient, = torch.autograd.grad(unmatched, image)
    torch.testing.assert_close(gradient, torch.zeros_like(image))
