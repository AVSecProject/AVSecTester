"""A real dataset adapter, selected image insertion and replay need no ego cuboid size."""
from types import SimpleNamespace

import numpy as np
from PIL import Image

from avsectester.attack.attack import OfflineContext
from avsectester.attack.implemented_attacks.object_insertion_attack import ObjectInsertionAttack
from avsectester.attack.implemented_attacks.object_insertion_attack.assets import SignAsset
from avsectester.attack.insertion.compositor import PatchCompositor
from avsectester.attack.placement import WorldPlacement
from avsectester.closed_loop.avstack import AVStack
from avsectester.closed_loop.datatypes import Control
from avsectester.closed_loop.runner import ClosedLoopRunner, RecordingPolicy, RunRequest
from avsectester.closed_loop.services import ClosedLoopServices, RenderRequest
from avsectester.closed_loop.simulators.recorded import RecordedReplaySimulator
from avsectester.datasets.nurec import _matrix_quaternion
from avsectester.datasets.nuscenes import NuScenesDataset


def recorded_dataset(tmp_path):
    scene = {'token': 'scene', 'name': 'recorded-road', 'first_sample_token': 'sample-0'}
    optical_to_ego = np.array([[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]])
    qx, qy, qz, qw = _matrix_quaternion(optical_to_ego)
    records = {
        'scene': {'scene': scene}, 'sample': {}, 'sample_data': {}, 'ego_pose': {},
        'calibrated_sensor': {'front-calibration': {
            'rotation': [qw, qx, qy, qz], 'translation': [0., 0., 1.5],
            'camera_intrinsic': [[80., 0., 64.], [0., 80., 48.], [0., 0., 1.]],
        }},
    }
    for position in range(2):
        filename = f'front-{position}.png'
        Image.fromarray(np.full((96, 128, 3), 100 + position, np.uint8)).save(tmp_path / filename)
        records['sample'][f'sample-{position}'] = {
            'token': f'sample-{position}', 'scene_token': 'scene', 'anns': [],
            'next': 'sample-1' if position == 0 else '',
            'prev': 'sample-0' if position == 1 else '',
            'data': {'CAM_FRONT': f'image-{position}'},
        }
        records['sample_data'][f'image-{position}'] = {
            'timestamp': position * 500_000, 'ego_pose_token': f'ego-{position}',
            'calibrated_sensor_token': 'front-calibration', 'width': 128, 'height': 96,
            'filename': filename,
        }
        # nuScenes ego_pose contains a pose but no bounding-box dimensions.
        records['ego_pose'][f'ego-{position}'] = {
            'rotation': [1., 0., 0., 0.], 'translation': [position * .5, 0., 0.],
        }
    dataset = NuScenesDataset(str(tmp_path), version='v1.0-mini')
    dataset._nusc = SimpleNamespace(scene=[scene], get=lambda table, token: records[table][token])
    return dataset


def test_nuscenes_provider_render_selection_and_online_insertion_preserve_unknown_ego_extent(tmp_path):
    dataset = recorded_dataset(tmp_path)
    simulator = RecordedReplaySimulator()
    frames = dataset.frames('scene')
    record = dataset.frame_data('scene', frames[0])
    assert record.ego.extent is None and record.actors == {}
    texture = np.full((32, 32, 4), [230, 20, 30, 255], np.uint8)
    attack = ObjectInsertionAttack(
        decision_steps=1, sensor='front', placement=WorldPlacement((8., 0., 0.)),
        assets=SignAsset(face=texture, width_m=2., mount_height_m=.5, post=False),
        compositor=PatchCompositor(harmonize=False),
    )
    with ClosedLoopServices(dataset, simulator) as services:
        with services.session(0) as session:
            rendered = session.render(RenderRequest('scene', frames[0], 'front'))
            assert rendered.view.ego.extent is None
        metadata = attack.build_metadata(services.scenario('scene', seed=0))
        assert len(metadata) == 1
        assert metadata[0].start == frames[0] and metadata[0].stop == frames[1]
        artifacts = attack.run_offline(OfflineContext(services, metadata))
    class ImageReader(AVStack):
        def __call__(self, observation):
            self.image = observation.sensor_data['front'].copy()
            return Control()
    avstack = ImageReader()
    simulator_runtime = simulator.create_runtime()
    segment = metadata[0]
    log = ClosedLoopRunner().run(simulator_runtime, avstack, RunRequest(
        1, start=dataset.scenario_start('scene', segment.start),
        attacks=(attack.create_runtime(segment, artifacts),),
        recording=RecordingPolicy(('front',), ('raw', 'model_input')),
    ))
    assert log.status == 'completed'
    assert log.steps[0].physical_pre.ego.extent is None
    assert not np.array_equal(avstack.image, log.steps[0].raw_observation.sensor_data['front'])
    np.testing.assert_array_equal(avstack.image, log.steps[0].model_observation.sensor_data['front'])
    assert any(effect.get('render', {}).get('rendered') is True for effect in log.steps[0].effects)
