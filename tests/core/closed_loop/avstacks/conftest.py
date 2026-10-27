"""Camera observations and learned-model boundaries shared by driving-system tests."""
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def alpamayo_model_boundary(monkeypatch):
    base = pytest.importorskip("alpasim_driver.models.base")
    pytest.importorskip("alpasim_grpc.v0.egodriver_pb2")
    pytest.importorskip("alpasim_driver.navigation")

    class PolicyModel:
        def __init__(self):
            self.closed = False
            self.last_input = None
            self.prediction = base.ModelPrediction(
                candidate_positions=np.array([[[1., 0., 0.], [2., 0., 0.]]]),
                candidate_rotations=np.tile(np.eye(3), (1, 2, 1, 1)),
                selected_index=0, selected_plan="forward", reasoning_text="clear",
            )

        def predict(self, inputs):
            self.last_input = inputs
            return self.prediction

        def close(self):
            self.closed = True

    class ModelLoader:
        @classmethod
        def from_config(cls, **options):
            model = PolicyModel()
            model.options = options
            return model

    # Keep real AlpaSim protobufs, navigation and model input/output records. Only weights
    # and checkpoint-backed construction are replaced.
    name = "alpasim_driver.models.alpamayo1_5_model"
    module = ModuleType(name)
    module.Alpamayo15Model = ModelLoader
    monkeypatch.setitem(sys.modules, name, module)
    return SimpleNamespace(create=PolicyModel)


@pytest.fixture
def camera_detector_boundary(monkeypatch):
    """Replace checkpoint-backed image inference, leaving the real camera policy intact."""
    pytest.importorskip("avstack.config")
    api = pytest.importorskip("mmdet.apis")
    import torch
    from mmdet.models.data_preprocessors import DetDataPreprocessor
    from avstack.config import MODELS
    import avstack.modules.perception.object2dfv  # noqa: F401

    class ImageModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gain = torch.nn.Parameter(torch.tensor(1.))
            self.data_preprocessor = DetDataPreprocessor()
            self.boxes = [[8., 8., 16., 16.]]
            self.labels = [0]
            self.last_image = None

        def extract_feat(self, images):
            values = torch.nn.functional.avg_pool2d(images.mean(dim=1, keepdim=True), 4)
            return (values * self.gain / 255.,)

        def rpn_head(self, features):
            return (list(features), [])

        def predict(self, rgb):
            self.last_image = rgb.copy()
            return SimpleNamespace(pred_instances=SimpleNamespace(
                bboxes=torch.tensor(self.boxes).reshape(-1, 4),
                scores=torch.full((len(self.boxes),), .9), labels=torch.tensor(self.labels),
            ))

    model = ImageModel()
    # Only the learned detector is replaced; preprocessing, scorer and driving adapter stay real.
    monkeypatch.setitem(MODELS.module_dict, "MMDetObjectDetector2D",
                        lambda **options: SimpleNamespace(model=model))
    monkeypatch.setattr(api, "inference_detector", lambda detector, image: detector.predict(image))
    return model


@pytest.fixture
def camera_observation():
    from avsectester.data_types import Observation
    return lambda: Observation(0., 0, {"front": np.zeros((32, 32, 3), np.uint8)})


@pytest.fixture
def alpamayo_observations():
    from tests.fakes import recorded_policy_observations

    return recorded_policy_observations


@pytest.fixture
def policy_past(alpamayo_observations):
    return alpamayo_observations()[:-1]


@pytest.fixture
def policy_observation(alpamayo_observations):
    return lambda: alpamayo_observations()[-1]


@pytest.fixture
def rule_based_detector_model():
    from tests.fakes import coco_detection_model

    return coco_detection_model(boxes=((48., 38., 52., 42.),))


@pytest.fixture
def rule_based_observation():
    from avsectester.closed_loop.simulators.ego_dynamics import PlanarEgoState
    from avsectester.data_types import Observation
    from avsectester.geometry import CameraCalibration, PinholeCamera
    from avsectester.geometry.cameras import BODY_FROM_OPTICAL

    return lambda: Observation(
        0., 0, {"front": np.full((80, 100, 3), 30, np.uint8)},
        {"front": CameraCalibration(PinholeCamera(
            np.array([[100., 0., 50.], [0., 100., 40.], [0., 0., 1.]]), 100, 80),
            BODY_FROM_OPTICAL)},
        PlanarEgoState(speed=10.), 10.,
    )


@pytest.fixture
def rule_based_detector(rule_based_detector_model):
    from avsectester.closed_loop.avstacks.implemented.rule_based import CocoStopSignDetector

    return CocoStopSignDetector("synthetic-local-weights", device="cpu", model=rule_based_detector_model)
