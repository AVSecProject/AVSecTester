"""SessionScorer converts complete image batches and preserves requested input gradients."""

import numpy as np
import pytest
import torch

from avsectester.attack.optimization import SessionScorer
from avsectester.closed_loop.avstacks import DrivingAdapter, ScorerDescriptor
from avsectester.data_types import TargetSpec
from avsectester.execution.services import ClosedLoopServices
from tests.fakes import ConstantAVStack, MemoryDataset, MemorySimulator


def test_target_scores_float_tensor():
    descriptor = ScorerDescriptor(
        "confidence", {"images": "BCHW float tensor [0,1]"},
        {"scores": "scalar confidence"}, {"scores": "tensor"}, ("images",),
    )

    def confidence(avstack, images, targets):
        assert images.shape == (2, 3, 4, 5)
        assert len(targets) == 2
        return images.mean(dim=(1, 2, 3))

    adapter = DrivingAdapter(lambda seed: ConstantAVStack(), scorers=[(descriptor, confidence)])
    targets = [TargetSpec((0, 0, 5, 4))] * 2
    with ClosedLoopServices(MemoryDataset(), MemorySimulator(), adapter) as services:
        with services.session(0) as session:
            scorer = SessionScorer(
                session, "confidence", require_grad=True,
                input_format="BCHW float tensor [0,1]",
            )
            images = torch.stack([torch.full((3, 4, 5), .25),
                                  torch.full((3, 4, 5), .75)]).requires_grad_()
            scores = scorer.target_scores(images, targets)
            torch.testing.assert_close(scores, torch.tensor([.25, .75]))
            gradient, = torch.autograd.grad(scores[0], images)
            torch.testing.assert_close(gradient[0], torch.full((3, 4, 5), 1 / 60))
            torch.testing.assert_close(gradient[1], torch.zeros(3, 4, 5))
            with torch.no_grad():
                converted = scorer.target_scores(
                    [np.full((4, 5, 3), 51, np.uint8),
                     np.full((4, 5, 3), 204, np.uint8)], targets)
            torch.testing.assert_close(converted, torch.tensor([.2, .8]))
            with pytest.raises(ValueError, match="target|length"):
                scorer.target_scores(images, targets[:1])


def test_target_scores_uint8_images():
    descriptor = ScorerDescriptor(
        "confidence", {"images": "list of HWC RGB uint8"},
        {"scores": "scalar confidence"}, {"scores": "array"},
    )

    def confidence(avstack, images, targets):
        assert len(images) == len(targets) == 2
        assert all(image.dtype == np.uint8 and image.shape == (4, 5, 3) for image in images)
        return np.asarray([image.mean() for image in images])

    adapter = DrivingAdapter(lambda seed: ConstantAVStack(), scorers=[(descriptor, confidence)])
    targets = [TargetSpec((0, 0, 5, 4))] * 2
    with ClosedLoopServices(MemoryDataset(), MemorySimulator(), adapter) as services:
        with services.session(0) as session:
            scorer = SessionScorer(
                session, "confidence", require_grad=False,
                input_format="list of HWC RGB uint8",
            )
            images = torch.stack([torch.full((3, 4, 5), .25),
                                  torch.full((3, 4, 5), .75)])
            np.testing.assert_array_equal(scorer.target_scores(images, targets), [63., 191.])
            np.testing.assert_array_equal(scorer.target_scores(
                [np.full((4, 5, 3), 80, np.uint8),
                 np.full((4, 5, 3), 120, np.uint8)], targets), [80., 120.])
            with pytest.raises(ValueError, match="cannot preserve.*grad"):
                scorer.target_scores(images.requires_grad_(), targets)
