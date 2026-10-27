"""SessionScorer adapts declared image formats without silently dropping gradients."""
import numpy as np
import pytest
import torch

from avsectester.attack.optimization import SessionScorer
from avsectester.closed_loop.avstacks import DrivingAdapter, ScorerDescriptor
from avsectester.data_types import TargetSpec
from avsectester.execution.services import ClosedLoopServices
from tests.fakes import ConstantAVStack, MemoryDataset, MemorySimulator


def test_target_score():
    descriptor = ScorerDescriptor("confidence", {"image": "HWC RGB uint8"},
                                  {"score": "scalar"}, {"score": "float"})

    def confidence(avstack, image, target):
        assert image.dtype == np.uint8 and image.shape == (4, 5, 3)
        return float(image.mean())

    adapter = DrivingAdapter(lambda seed: ConstantAVStack(), scorers=[(descriptor, confidence)])
    with ClosedLoopServices(MemoryDataset(), MemorySimulator(), adapter) as services:
        with services.session(0) as session:
            scorer = SessionScorer(session, "confidence", require_grad=False,
                                   input_format="HWC RGB uint8")
            target = TargetSpec((0, 0, 5, 4))
            image = torch.full((3, 4, 5), .25)
            assert scorer.target_score(image, target) == 63.
            assert scorer.target_score(np.full((4, 5, 3), 80, np.uint8), target) == 80.
            with pytest.raises(ValueError, match="cannot preserve image gradients"):
                scorer.target_score(image.requires_grad_(), target)
