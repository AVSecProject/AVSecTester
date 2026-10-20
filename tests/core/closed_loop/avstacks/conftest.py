"""Small stand-ins at the AlpaSim model and schema boundary; no checkpoints are loaded."""
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def alpamayo_model_boundary(monkeypatch):
    names = (
        "alpasim_driver", "alpasim_driver.models", "alpasim_driver.models.base",
        "alpasim_driver.models.alpamayo1_5_model", "alpasim_driver.schema",
        "alpasim_grpc", "alpasim_grpc.v0", "alpasim_grpc.v0.common_pb2",
    )
    modules = {name: ModuleType(name) for name in names}
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
        if "." in name:
            parent, attribute = name.rsplit(".", 1)
            setattr(modules[parent], attribute, module)
    base = modules["alpasim_driver.models.base"]
    base.CameraFrame = base.PredictionInput = SimpleNamespace
    base.DriveCommand = SimpleNamespace(STRAIGHT="straight")
    for name in ("PoseAtTime", "Pose", "Vec3", "Quat"):
        setattr(modules["alpasim_grpc.v0.common_pb2"], name, SimpleNamespace)
    modules["alpasim_driver.schema"].ModelConfig = SimpleNamespace

    class PolicyModel:
        def __init__(self):
            self.closed = False
            self.last_input = None

        def predict(self, inputs):
            self.last_input = inputs
            return SimpleNamespace(candidate_positions=np.array([[[1., 0., 0.], [2., 0., 0.]]]),
                                   selected_index=0, selected_plan="forward", reasoning_text="clear")

        def close(self):
            self.closed = True

    loaded = []

    class ModelLoader:
        @classmethod
        def from_config(cls, **options):
            model = PolicyModel()
            loaded.append(model)
            return model

    modules["alpasim_driver.models.alpamayo1_5_model"].Alpamayo15Model = ModelLoader
    return SimpleNamespace(create=PolicyModel, loaded=loaded)


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
def native_dependencies():
    pytest.importorskip("avstack.config")


@pytest.fixture
def camera_observation():
    from avsectester.closed_loop.datatypes import Observation
    return lambda: Observation(0., 0, {"front": np.zeros((32, 32, 3), np.uint8)})
