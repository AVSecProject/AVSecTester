"""Scenario augmentation: corruption operators, the seeded pipeline, and the perturb lift.

Pure image/logic tests (no dataset/simulator) — a controlled random image is a fair input for checking
shape/dtype invariants, severity-0 identity, determinism, and the write-back, not a claim about real data.
"""

import numpy as np
import pytest
from avsectester.plane import Observation
from avsectester.simulators.augment import (
    CORRUPTIONS,
    AugmentationPipeline,
    ChromaticAberration,
    Fog,
    GaussianNoise,
    Rain,
    common_corruptions,
    compose,
    read_camera_rgb,
    sensor_augmentation,
    write_camera_rgb,
)

# operators whose effect vanishes exactly at severity 0 (no base noise / minimum kernel)
_IDENTITY_AT_ZERO = ["Fog", "Rain", "Snow", "Brightness", "Contrast", "Gamma", "ColorTemp",
                     "GaussianNoise", "MotionBlur", "ChromaticAberration"]


def _img(seed=1):
    return np.random.default_rng(seed).integers(0, 256, (32, 48, 3), dtype=np.uint8)


@pytest.mark.parametrize("name", list(CORRUPTIONS))
def test_every_corruption_preserves_shape_dtype_and_changes_image(name):
    img, rng = _img(), np.random.default_rng(0)
    out = CORRUPTIONS[name](severity=0.8).apply(img, rng)
    assert out.shape == img.shape and out.dtype == np.uint8
    assert not np.array_equal(out, img)                     # severity 0.8 actually corrupts


@pytest.mark.parametrize("name", _IDENTITY_AT_ZERO)
def test_severity_zero_is_identity(name):
    img = _img()
    assert np.array_equal(CORRUPTIONS[name](severity=0.0).apply(img, np.random.default_rng(0)), img)


def test_random_corruption_is_deterministic_given_the_rng():
    img = _img()
    a = Rain(0.7).apply(img, np.random.default_rng(42))
    b = Rain(0.7).apply(img, np.random.default_rng(42))
    assert np.array_equal(a, b)                             # same seed -> identical realisation


def test_pipeline_seeds_by_frame_so_paired_runs_match():
    img = _img()
    p_clean = AugmentationPipeline([GaussianNoise(0.5), Fog(0.4)], seed=7)
    p_attacked = AugmentationPipeline([GaussianNoise(0.5), Fog(0.4)], seed=7)
    # same seed + frame -> pixel-identical corruption (so impact() measures only the attack)
    assert np.array_equal(p_clean.apply(img, frame=3), p_attacked.apply(img, frame=3))
    # different frames -> different corruption
    assert not np.array_equal(p_clean.apply(img, frame=3), p_clean.apply(img, frame=4))


def test_common_corruptions_covers_every_operator():
    suite = common_corruptions(0.5)
    assert len(suite) == len(CORRUPTIONS)
    assert all(isinstance(p, AugmentationPipeline) and len(p.corruptions) == 1 for p in suite)


def test_sensor_augmentation_reads_writes_ndarray_and_imagedata():
    rgb = _img()
    # ndarray payload (NuRec / dataset replay)
    obs = Observation(t=0.0, frame=2, sensor_data={"front": rgb})
    out = sensor_augmentation([Fog(0.6)], camera="front")(obs)
    assert isinstance(out.sensor_data["front"], np.ndarray)
    assert not np.array_equal(out.sensor_data["front"], rgb)
    assert np.array_equal(obs.sensor_data["front"], rgb)   # original untouched (functional)

    # ImageData-like payload (avstack / CARLA): .rgb_image read, .data buffer swapped
    class _ImageData:
        def __init__(self, arr):
            self.rgb_image = arr
            self.data = arr

    im = _ImageData(rgb.copy())
    obs2 = Observation(t=0.0, frame=0, sensor_data={"camera-0": im})
    key, got = read_camera_rgb(obs2)
    assert key == "camera-0" and np.array_equal(got, rgb)
    out2 = write_camera_rgb(obs2, "camera-0", ChromaticAberration(0.9).apply(rgb, np.random.default_rng(0)))
    payload = out2.sensor_data["camera-0"]
    assert payload is not im and hasattr(payload, "data") and not np.array_equal(payload.data, rgb)


def test_sensor_augmentation_passes_through_when_no_camera():
    obs = Observation(t=0.0, frame=0, sensor_data={})
    assert sensor_augmentation([Fog(0.5)])(obs) is obs


def test_compose_applies_perturbs_in_order_and_skips_none():
    rgb = _img()
    obs = Observation(t=0.0, frame=0, sensor_data={"front": rgb})
    brighten = sensor_augmentation([CORRUPTIONS["Brightness"](0.5)], camera="front")
    fog = sensor_augmentation([Fog(0.5)], camera="front")
    both = compose(brighten, None, fog)(obs)
    expect = fog(brighten(obs))
    assert np.array_equal(both.sensor_data["front"], expect.sensor_data["front"])
    assert np.array_equal(compose(brighten, None)(obs).sensor_data["front"],
                          brighten(obs).sensor_data["front"])  # None skipped
