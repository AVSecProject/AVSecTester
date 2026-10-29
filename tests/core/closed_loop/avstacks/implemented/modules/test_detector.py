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
    with pytest.raises(ValueError, match=r"shape \(B, N, 85\)"):
        malformed.detect(rgb)


def test_target_scores():
    model = yolox_detection_model([
        prediction((0., 0., 4., 4.), .6, {2: .9, 0: .2}),
        prediction((0., 0., 4., 4.), .95, {2: .4, 0: .95}),
        prediction((2., 0., 6., 4.), .85, {2: .95}),
        prediction((2.5, 0., 6.5, 4.), .99, {2: .99}),
    ], image_weight=.1)
    detector = YOLOXDetector("unused.pth", device="cpu", model=model)
    images = torch.stack([torch.full((3, 8, 8), value) for value in (.5, .25, .75)])
    images.requires_grad_(True)
    targets = [
        TargetSpec((0., 0., 4., 4.), "car"),
        TargetSpec((0., 0., 4., 4.), "person"),
        TargetSpec((6., 6., 8., 8.), "car"),
    ]
    scores = detector.target_scores(images, targets)
    assert scores.shape == (3,)
    dependencies = [(value * 64 + 114 / 255 * 960) / 1024 * .1 for value in (.5, .25)]
    torch.testing.assert_close(
        scores, torch.tensor([(.85 + dependencies[0]) * .95,
                              (.95 + dependencies[1]) * .95, 0.]),
    )
    singles = torch.stack([
        detector.target_scores(images[index:index + 1], [target])[0]
        for index, target in enumerate(targets)
    ])
    torch.testing.assert_close(scores, singles)
    first_grad, = torch.autograd.grad(scores[0], images, retain_graph=True)
    torch.testing.assert_close(first_grad[0], torch.full_like(images[0], .095 / (3 * 32 * 32)))
    assert torch.count_nonzero(first_grad[1:]) == 0
    second_grad, = torch.autograd.grad(scores[1], images, retain_graph=True)
    torch.testing.assert_close(second_grad[1], torch.full_like(images[1], .095 / (3 * 32 * 32)))
    assert torch.count_nonzero(second_grad[[0, 2]]) == 0
    zero_grad, = torch.autograd.grad(scores[2], images)
    torch.testing.assert_close(zero_grad, torch.zeros_like(images))

    target = targets[0]
    rgb = np.full((8, 8, 3), 128, dtype=np.uint8)
    deployed = detector.detect(rgb)
    same_class = []
    for box, confidence, label in deployed:
        intersection = np.maximum(0, np.minimum(box[2:], target.box[2:])
                                  - np.maximum(box[:2], target.box[:2])).prod()
        union = np.maximum(0, box[2:] - box[:2]).prod() + 16 - intersection
        if label == "car" and intersection / union >= .3:
            same_class.append(confidence)
    inputs = torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0) / 255
    scored = detector.target_scores(inputs, [target])[0]
    assert max(same_class) <= scored.item()
    unambiguous = YOLOXDetector("unused.pth", device="cpu", model=yolox_detection_model([
        prediction((0., 0., 4., 4.), .9, {2: .8}),
    ], image_weight=.1))
    same_class = unambiguous.detect(rgb)
    scored = unambiguous.target_scores(inputs, [target])[0]
    assert same_class[0][2] == "car"
    assert same_class[0][1] == pytest.approx(scored.item())
    with pytest.raises(ValueError, match="finite float RGB"):
        detector.target_scores(torch.ones(8, 8), [target])
    with pytest.raises(ValueError, match="length"):
        detector.target_scores(images, [target])
    with pytest.raises(ValueError, match="COCO class name"):
        detector.target_scores(inputs, [TargetSpec(target.box, "unknown class")])
    with pytest.raises(ValueError, match="COCO class name"):
        detector.target_scores(inputs, [TargetSpec(target.box)])
    with pytest.raises(ValueError, match="positive width and height"):
        detector.target_scores(inputs, [TargetSpec((4., 0., 0., 4.), "car")])
    detector.close()
    with pytest.raises(RuntimeError, match="closed"):
        detector.target_scores(inputs, [target])


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
