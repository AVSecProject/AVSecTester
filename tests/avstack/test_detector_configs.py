"""Build each shipped detector configuration and forward a small synthetic input.

Checkpoint weights are intentionally omitted; this verifies model/configuration compatibility,
not detector accuracy. Optional MMDetection installations may skip these native component cases.
"""
from pathlib import Path
from inspect import signature

import numpy as np
import pytest
import yaml

pytest.importorskip("avstack.config")

from avstack.modules.perception.object2dfv import MMDetObjectDetector2D
from avstack.modules.perception.object3d import MMDetObjectDetector3D
from avsectester.closed_loop.avstacks.implemented.camera_forward_collision import CameraForwardCollisionAVStack


def configured_options(avstack_type):
    for path in sorted((Path(__file__).resolve().parents[2] / "configs").glob("*.yaml")):
        specification = yaml.safe_load(path.read_text())["avstack"]
        if specification["type"] == avstack_type:
            yield path.stem, specification.get("options", {})


def image_detectors():
    defaults = signature(CameraForwardCollisionAVStack).parameters
    return [pytest.param("fasterrcnn", options.get("detector_dataset",
                         defaults["detector_dataset"].default), id=name)
            for name, options in configured_options("CameraForwardCollisionAVStack")]


def point_cloud_detectors():
    return [pytest.param(options["pipeline_cfg"]["perception"].get("model", "pointpillars"),
                         options["pipeline_cfg"]["perception"].get("dataset", "kitti"),
                         carla_metadata if options["pipeline_cfg"]["perception"].get(
                             "dataset", "kitti").startswith("carla-") else standard_metadata, id=name)
            for name, options in configured_options("ModularAVStack")]


def carla_metadata():
    from avsectester.closed_loop.avstacks.implemented.modular import register_carla_dataset
    register_carla_dataset()


def standard_metadata():
    pass


def detector_configuration(detector_type, model_name, detector_dataset):
    Config = pytest.importorskip("mmengine.config").Config
    from avstack.modules.perception import base

    config_file = detector_type.parse_mm_model_from_checkpoint(
        model_name, detector_dataset, "latest")[1]
    roots = (Path(base.mm2d_root), Path(base.mm3d_root),
             Path(__file__).resolve().parents[2] / "models")
    config_path = next((root / config_file for root in roots if (root / config_file).is_file()), None)
    if config_path is None:
        pytest.skip(f"Initialize the optional detector configuration checkout: {config_file}")
    config = Config.fromfile(config_path)
    if "backbone" in config.model:
        config.model.backbone.init_cfg = None
    return config


@pytest.mark.parametrize("model_name,detector_dataset", image_detectors())
def test_image_detector(model_name, detector_dataset):
    import torch
    api = pytest.importorskip("mmdet.apis")
    from mmdet.structures import DetDataSample

    config = detector_configuration(MMDetObjectDetector2D, model_name, detector_dataset)
    model = api.init_detector(config, checkpoint=None, device="cpu")
    sample = DetDataSample(metainfo={"img_shape": (64, 64), "ori_shape": (64, 64),
                                    "scale_factor": (1., 1.), "batch_input_shape": (64, 64)})
    with torch.no_grad():
        result = model(torch.zeros((1, 3, 64, 64)), [sample], mode="tensor")
    assert result is not None


@pytest.mark.parametrize("model_name,detector_dataset,prepare_metadata", point_cloud_detectors())
def test_point_cloud_detector(model_name, detector_dataset, prepare_metadata):
    import torch
    api = pytest.importorskip("mmdet3d.apis")
    from mmdet3d.structures import Det3DDataSample, LiDARInstance3DBoxes

    config = detector_configuration(MMDetObjectDetector3D, model_name, detector_dataset)
    prepare_metadata()
    model = api.init_model(config, checkpoint=None, device="cpu")
    sample = Det3DDataSample(metainfo={"box_type_3d": LiDARInstance3DBoxes})
    points = np.array([[5., .1, -.5, .8], [5.1, .2, -.4, .7],
                       [10., 1., -.2, .5], [10.2, 1.1, -.3, .4]], dtype=np.float32)
    with torch.no_grad():
        prepared = model.data_preprocessor({
            "inputs": {"points": [torch.from_numpy(points)]}, "data_samples": [sample],
        }, training=False)
        # This SDK exposes tensor inference through its feature extractor and prediction head.
        _, point_features = model.extract_feat(prepared["inputs"], [sample.metainfo])
        result = model.pts_bbox_head(point_features)
    assert all(tensor.ndim == 4 and torch.isfinite(tensor).all()
               for level in result for tensor in level)
