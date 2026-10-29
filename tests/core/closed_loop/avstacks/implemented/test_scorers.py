"""Native batch scorer numerics and bindings around a tiny learned-model boundary."""

import numpy as np
import pytest
import torch

from avsectester.data_types import TargetSpec
from avsectester.closed_loop.avstacks.implemented.camera_forward_collision import CameraForwardCollisionAVStack
from avsectester.closed_loop.avstacks.implemented.modular import modular_driving_adapter
from avsectester.execution.services import ClosedLoopServices
from tests.fakes import MemoryDataset, MemorySimulator


@pytest.fixture
def rpn_case():
    from avsectester.closed_loop.avstacks.implemented.scorers import MMDetRPNObjectnessScorer

    images = torch.full((3, 3, 16, 16), .5, requires_grad=True)
    return MMDetRPNObjectnessScorer, images, float(torch.sigmoid(torch.tensor(.5))), True


@pytest.fixture
def inference_case():
    from avsectester.closed_loop.avstacks.implemented.scorers import MMDetInferenceScorer

    images = [np.full((16, 16, 3), (30, 60, 90), np.uint8) for _ in range(3)]
    return MMDetInferenceScorer, images, .9, False


@pytest.mark.parametrize("case_fixture,binding", [
    pytest.param("rpn_case", "mmdet.rpn_objectness", id="rpn-objectness"),
    pytest.param("inference_case", "mmdet.inference", id="inference"),
])
def test_target_scores(case_fixture, binding, request, camera_detector_boundary):
    scorer_type, images, expected, differentiable = request.getfixturevalue(case_fixture)
    scorer = scorer_type(camera_detector_boundary)
    targets = [
        TargetSpec((8., 8., 16., 16.)),
        TargetSpec((100., 100., 104., 104.)),
        TargetSpec((0., 0., 16., 16.)),
    ]
    values = scorer.target_scores(images, targets)
    np.testing.assert_allclose(torch.as_tensor(values).detach().cpu().numpy(), [expected, 0., expected])
    with pytest.raises(ValueError, match="length"):
        scorer.target_scores(images, targets[:1])

    adapter = modular_driving_adapter(lambda seed: CameraForwardCollisionAVStack(), scorers=(binding,))
    with ClosedLoopServices(MemoryDataset(), MemorySimulator(), adapter) as services:
        with services.session(0) as session:
            descriptor, = session.scorers()
            assert descriptor.id == binding
            assert set(descriptor.inputs) == {"images"}
            assert set(descriptor.outputs) == {"scores"}
            result = session.score(binding, images, targets, require_grad=differentiable)
            np.testing.assert_allclose(torch.as_tensor(result.value).detach().cpu().numpy(),
                                       [expected, 0., expected])


def test_rpn_score_retains_image_gradient(camera_detector_boundary):
    from avsectester.closed_loop.avstacks.implemented.scorers import MMDetRPNObjectnessScorer

    images = torch.full((2, 3, 16, 16), .5, requires_grad=True)
    scores = MMDetRPNObjectnessScorer(camera_detector_boundary).target_scores(
        images, [TargetSpec((0., 0., 4., 4.)), TargetSpec((100., 100., 104., 104.))],
    )
    scores.sum().backward()
    assert torch.isfinite(images.grad).all()
    assert images.grad[0, :, :4, :4].abs().sum() > 0
    assert torch.count_nonzero(images.grad[0, :, 4:, :]) == 0
    assert torch.count_nonzero(images.grad[0, :, :, 4:]) == 0
    assert torch.count_nonzero(images.grad[1]) == 0


def test_inference_score_uses_bgr_input(camera_detector_boundary):
    from avsectester.closed_loop.avstacks.implemented.scorers import MMDetInferenceScorer

    images = [np.full((16, 16, 3), color, np.uint8) for color in ((30, 60, 90), (10, 20, 40))]
    MMDetInferenceScorer(camera_detector_boundary).target_scores(
        images, [TargetSpec((0., 0., 16., 16.))] * 2,
    )
    assert len(camera_detector_boundary.last_images) == 2
    for actual, image in zip(camera_detector_boundary.last_images, images):
        np.testing.assert_array_equal(actual, image[:, :, ::-1])
