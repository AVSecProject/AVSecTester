"""Sensor corruption invariants, deterministic environments and native image write-back."""
import numpy as np
import pytest

from avsectester.closed_loop.datatypes import Observation, read_camera_rgb, write_camera_rgb
from avsectester.closed_loop.environment import SensorEnvironment
from avsectester.closed_loop.sensor_transforms import (
    CORRUPTIONS, AlbumentationsCorruption, ChromaticAberration,
)


@pytest.mark.parametrize('name', [name for name in CORRUPTIONS if name != 'AlbumentationsCorruption'])
def test_corruption_preserves_shape_dtype_and_is_seeded(name):
    image = np.random.default_rng(1).integers(0, 256, (32, 48, 3), dtype=np.uint8)
    transform = CORRUPTIONS[name](severity=.8)
    actual = transform.apply(image, np.random.default_rng(2))
    assert actual.shape == image.shape and actual.dtype == np.uint8
    assert not np.array_equal(actual, image)
    np.testing.assert_array_equal(actual, transform.apply(image, np.random.default_rng(2)))


def test_environment_composes_corruptions_in_configured_order():
    image = np.full((8, 12, 3), 70, np.uint8)
    observation = Observation(0, 0, sensor_data={'front': image})
    specs = [{'type': 'Brightness', 'options': {'severity': .5}},
             {'type': 'Fog', 'options': {'severity': .4}}]
    actual = SensorEnvironment({'sensors': specs}, seed=2).apply(observation, 7)
    brightness = SensorEnvironment({'sensors': specs[:1]}, seed=2).apply(observation, 7)
    expected = SensorEnvironment({'sensors': specs[1:]}, seed=2).apply(brightness, 7)
    np.testing.assert_array_equal(actual.sensor_data['front'], expected.sensor_data['front'])
    np.testing.assert_array_equal(observation.sensor_data['front'], image)


def test_native_camera_writeback_keeps_source_independent():
    class ImageData:
        def __init__(self, image):
            self.rgb_image = self.data = image

    image = np.random.default_rng(1).integers(0, 256, (32, 48, 3), dtype=np.uint8)
    source = ImageData(image)
    observation = Observation(0, 0, sensor_data={'camera': source})
    assert read_camera_rgb(observation)[0] == 'camera'
    actual = write_camera_rgb(observation, 'camera', ChromaticAberration(.9).apply(image, None))
    assert actual.sensor_data['camera'] is not source
    assert not np.array_equal(actual.sensor_data['camera'].data, image)
    np.testing.assert_array_equal(source.data, image)


@pytest.mark.parametrize('name', ['Fog', 'Rain', 'Snow', 'GaussNoise', 'ISONoise',
                                  'MotionBlur', 'DefocusBlur', 'JPEGCompression'])
def test_named_albumentations_corruption_is_deterministic(name):
    image = np.random.default_rng(1).integers(0, 256, (32, 48, 3), dtype=np.uint8)
    transform = AlbumentationsCorruption(name, severity=.6)
    actual = transform.apply(image, np.random.default_rng(2))
    assert actual.shape == image.shape and actual.dtype == np.uint8
    np.testing.assert_array_equal(actual, transform.apply(image, np.random.default_rng(2)))
