"""YOLOX preprocessing, classwise deployment filtering and differentiable target confidence."""

import weakref

import numpy as np
import pytest
import torch

from avsectester.closed_loop.avstacks.implemented.modules.detector import YOLOXDetector
from avsectester.data_types import TargetSpec
from tests.fakes import yolox_detection_model


def prediction(box, objectness, probabilities):
    """Construct one decoded learned-model output row from a network-pixel xyxy box."""
    x1, y1, x2, y2 = box
    classes = [0.] * 80
    for index, probability in probabilities.items():
        classes[index] = probability
    return [(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1, objectness, *classes]


def test_detect(tmp_path):
    box = (2., 3., 10., 15.)
    model = yolox_detection_model([
        prediction(box, .9, {2: .9, 11: .8}),
        prediction(box, .9, {2: .8}),
        prediction(box, .9, {0: .85}),
        prediction((12., 2., 15., 7.), 1., {0: .49}),
    ])
    detector = YOLOXDetector("unused.pth", input_size=32, device="cpu", model=model)
    rgb = np.empty((40, 64, 3), dtype=np.uint8)
    rgb[:] = (10, 20, 30)
    detections = detector.detect(rgb)
    assert [label for _, _, label in detections] == ["car", "person"]
    assert [score for _, score, _ in detections] == pytest.approx([.81, .765])
    np.testing.assert_allclose([box for box, _, _ in detections], [[4, 6, 20, 30]] * 2)
    assert model.last_input.shape == (1, 3, 32, 32)
    torch.testing.assert_close(model.last_input[0, :, :20],
                               torch.tensor([30., 20., 10.])[:, None, None].expand(3, 20, 32))
    assert torch.all(model.last_input[:, :, 20:] == 114)
    assert not model.training and not model.weight.requires_grad
    detector.detect(rgb[:10, :20])
    torch.testing.assert_close(model.last_input[0, :, :10, :20],
                               torch.tensor([30., 20., 10.])[:, None, None].expand(3, 10, 20))
    assert torch.all(model.last_input[:, :, 10:] == 114)
    assert torch.all(model.last_input[:, :, :, 20:] == 114)
    empty = YOLOXDetector("unused.pth", device="cpu", model=yolox_detection_model([]))
    assert empty.detect(rgb) == []
    with pytest.raises(ValueError, match="uint8 RGB"):
        detector.detect(np.ones((20, 20, 3)))
    with pytest.raises(FileNotFoundError, match="Existing detector weights"):
        YOLOXDetector(tmp_path / "missing.pth", device="cpu").detect(rgb)
    malformed = YOLOXDetector("unused.pth", device="cpu", model=torch.nn.Identity())
    with pytest.raises(ValueError, match=r"shape \(1, N, 85\)"):
        malformed.detect(rgb)


def test_target_score():
    model = yolox_detection_model([
        prediction((0., 0., 4., 4.), .6, {2: .9, 0: .2}),
        prediction((0., 0., 4., 4.), .95, {2: .4, 0: .95}),
        prediction((2., 0., 6., 4.), .85, {2: .95}),
        prediction((2.5, 0., 6.5, 4.), .99, {2: .99}),
    ], image_weight=.1)
    detector = YOLOXDetector("unused.pth", device="cpu", model=model)
    image = torch.full((3, 8, 8), .5, requires_grad=True)
    target = TargetSpec((0., 0., 4., 4.), "car")
    score = detector.target_score(image, target)
    dependency = (.5 * 64 + 114 / 255 * 960) / 1024 * .1
    assert score.item() == pytest.approx((.85 + dependency) * .95)
    gradient, = torch.autograd.grad(score, image)
    torch.testing.assert_close(gradient, torch.full_like(image, .095 / (3 * 32 * 32)))
    person = detector.target_score(image, TargetSpec(target.box, "person"))
    assert person.item() == pytest.approx((.95 + dependency) * .95)
    zero = detector.target_score(image, TargetSpec((6., 6., 8., 8.), "car"))
    assert zero.item() == 0.
    gradient, = torch.autograd.grad(zero, image)
    torch.testing.assert_close(gradient, torch.zeros_like(image))
    rgb = np.full((8, 8, 3), 128, dtype=np.uint8)
    deployed = detector.detect(rgb)
    same_class = []
    for box, confidence, label in deployed:
        intersection = np.maximum(0, np.minimum(box[2:], target.box[2:])
                                  - np.maximum(box[:2], target.box[:2])).prod()
        union = np.maximum(0, box[2:] - box[:2]).prod() + 16 - intersection
        if label == "car" and intersection / union >= .3:
            same_class.append(confidence)
    scored = detector.target_score(torch.from_numpy(rgb).permute(2, 0, 1).float() / 255, target)
    assert max(same_class) <= scored.item()
    unambiguous = YOLOXDetector("unused.pth", device="cpu", model=yolox_detection_model([
        prediction((0., 0., 4., 4.), .9, {2: .8}),
    ], image_weight=.1))
    same_class = unambiguous.detect(rgb)
    scored = unambiguous.target_score(
        torch.from_numpy(rgb).permute(2, 0, 1).float() / 255, target,
    )
    assert same_class[0][2] == "car"
    assert same_class[0][1] == pytest.approx(scored.item())
    with pytest.raises(ValueError, match="finite float RGB"):
        detector.target_score(torch.ones(8, 8), target)
    with pytest.raises(ValueError, match="COCO class name"):
        detector.target_score(image, TargetSpec(target.box, "unknown class"))
    with pytest.raises(ValueError, match="COCO class name"):
        detector.target_score(image, TargetSpec(target.box))
    with pytest.raises(ValueError, match="positive width and height"):
        detector.target_score(image, TargetSpec((4., 0., 0., 4.), "car"))
    detector.close()
    with pytest.raises(RuntimeError, match="closed"):
        detector.target_score(image, target)


def test_close():
    model = yolox_detection_model()
    reference = weakref.ref(model)
    detector = YOLOXDetector("unused.pth", device="cpu", model=model)
    detector.detect(np.full((20, 20, 3), 30, dtype=np.uint8))
    del model
    detector.close()
    assert detector.model is None
    assert reference() is None
    with pytest.raises(RuntimeError, match="closed"):
        detector.detect(np.full((20, 20, 3), 30, dtype=np.uint8))
