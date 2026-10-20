"""Native scorer numerics and adapter bindings around a tiny learned-model boundary."""

import numpy as np
import pytest

from avsectester.attack.optimization import TargetSpec
from avsectester.closed_loop.avstacks.camera_forward_collision import CameraForwardCollisionAVStack
from avsectester.closed_loop.driving_adapters import modular_driving_adapter
from avsectester.closed_loop.services import ClosedLoopServices
from tests.fakes import MemoryDataset, MemorySimulator


@pytest.fixture
def rpn_case():
    torch = pytest.importorskip("torch")
    from avsectester.closed_loop.avstacks.scorers import MMDetRPNObjectnessScorer

    image = torch.full((3, 16, 16), .5, requires_grad=True)
    return MMDetRPNObjectnessScorer, image, float(torch.sigmoid(torch.tensor(.5))), True


@pytest.fixture
def inference_case():
    from avsectester.closed_loop.avstacks.scorers import MMDetInferenceScorer

    image = np.full((16, 16, 3), (30, 60, 90), np.uint8)
    return MMDetInferenceScorer, image, .9, False


@pytest.mark.parametrize("case_fixture,binding", [
    pytest.param("rpn_case", "mmdet.rpn_objectness", id="rpn-objectness"),
    pytest.param("inference_case", "mmdet.inference", id="inference"),
])
def test_target_score(case_fixture, binding, request, camera_detector_boundary):
    scorer_type, image, expected, differentiable = request.getfixturevalue(case_fixture)
    scorer = scorer_type(camera_detector_boundary)
    target = TargetSpec((8., 8., 16., 16.))
    value = scorer.target_score(image, target)
    assert float(value) == pytest.approx(expected)
    assert float(scorer.target_score(image, TargetSpec((100., 100., 104., 104.)))) == 0.

    adapter = modular_driving_adapter(lambda seed: CameraForwardCollisionAVStack(), scorers=(binding,))
    with ClosedLoopServices(MemoryDataset(), MemorySimulator(), adapter) as services:
        with services.session(0) as session:
            descriptor, = session.scorers()
            assert descriptor.id == binding
            result = session.score(binding, image, target, require_grad=differentiable)
            assert float(result.value) == pytest.approx(expected)


def test_rpn_score_retains_image_gradient(camera_detector_boundary):
    torch = pytest.importorskip("torch")
    from avsectester.closed_loop.avstacks.scorers import MMDetRPNObjectnessScorer

    image = torch.full((3, 16, 16), .5, requires_grad=True)
    score = MMDetRPNObjectnessScorer(camera_detector_boundary).target_score(
        image, TargetSpec((0., 0., 4., 4.)))
    score.backward()
    assert torch.isfinite(image.grad).all()
    assert image.grad[:, :4, :4].abs().sum() > 0
    assert torch.count_nonzero(image.grad[:, 4:, :]) == 0
    assert torch.count_nonzero(image.grad[:, :, 4:]) == 0


def test_inference_score_uses_bgr_input(camera_detector_boundary):
    from avsectester.closed_loop.avstacks.scorers import MMDetInferenceScorer

    image = np.full((16, 16, 3), (30, 60, 90), np.uint8)
    MMDetInferenceScorer(camera_detector_boundary).target_score(image, TargetSpec((0., 0., 16., 16.)))
    np.testing.assert_array_equal(camera_detector_boundary.last_image, image[:, :, ::-1])
