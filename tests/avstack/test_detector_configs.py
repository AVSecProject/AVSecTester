"""Bundled native detector pairs are parsed without loading checkpoints or models."""
from pathlib import Path
from inspect import signature

import pytest
import yaml

pytest.importorskip('avstack.config')
from avstack.modules.perception.object2dfv import MMDetObjectDetector2D
from avstack.modules.perception.object3d import MMDetObjectDetector3D
from avsectester.closed_loop.avstacks.camera_forward_collision import CameraForwardCollisionAVStack


def configured_detectors():
    for path in sorted((Path(__file__).resolve().parents[2] / 'configs').glob('*.yaml')):
        specification = yaml.safe_load(path.read_text())['avstack']
        options = specification.get('options', {})
        if specification['type'] == 'CameraForwardCollisionAVStack':
            defaults = signature(CameraForwardCollisionAVStack).parameters
            yield pytest.param(MMDetObjectDetector2D, 'fasterrcnn',
                               options.get('detector_dataset', defaults['detector_dataset'].default),
                               id=path.stem)
        elif specification['type'] == 'ModularAVStack':
            perception = options['pipeline_cfg']['perception']
            assert perception['type'] == 'MMDetObjectDetector3D'
            yield pytest.param(MMDetObjectDetector3D, perception.get('model', 'pointpillars'),
                               perception.get('dataset', 'kitti'), id=path.stem)


@pytest.mark.parametrize('detector,model,dataset', list(configured_detectors()))
def test_bundled_detector_pair_is_supported(detector, model, dataset):
    parsed = detector.parse_mm_model_from_checkpoint(model, dataset, 'latest')
    assert parsed is not None
