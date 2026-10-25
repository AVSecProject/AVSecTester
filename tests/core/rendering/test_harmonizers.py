"""Exposure harmonization uses one differentiable linear-light calculation."""
from dataclasses import replace

import numpy as np
import pytest
import torch

from avsectester.rendering import ClassicHarmonizer, ExposureHarmonizer


@pytest.mark.parametrize(("reference", "reference_linear"), [("ring", .2), ("under", .1)])
def test_harmonize_tensor(reference, reference_linear):
    mask = torch.zeros(9, 9, dtype=torch.bool)
    mask[3:6, 3:6] = True
    background = torch.full((3, 9, 9), .4845292045)  # linear .2
    background[:, mask] = .3491902126  # linear .1
    composite = background.clone()
    composite[:, mask] = .6651850846  # linear .4
    composite[:, 0, 0] = torch.tensor((.9, .8, .7))
    harmonizer = ExposureHarmonizer(reference, ring_px=1, gain_range=(.1, 2.), feather_sigma=0.)
    gain = reference_linear / (.4 + 1e-6)
    assert harmonizer.exposure_gain(composite, mask, background).item() == pytest.approx(gain)
    result = harmonizer.harmonize_tensor(composite, mask, background, composite)
    expected = 1.055 * (.4 * gain) ** (1 / 2.4) - .055
    torch.testing.assert_close(result[:, mask], torch.full((3, 9), expected))
    torch.testing.assert_close(result[:, ~mask], composite[:, ~mask], atol=0, rtol=0)
    # The nominal composite alone sets the gain; the composite only receives it.
    nominal = composite.clone()
    nominal[:, mask] = .4845292045  # linear .2
    nominal_gain = reference_linear / (.2 + 1e-6)
    result = harmonizer.harmonize_tensor(composite, mask, background, nominal)
    expected = 1.055 * (.4 * nominal_gain) ** (1 / 2.4) - .055
    torch.testing.assert_close(result[:, mask], torch.full((3, 9), expected))
    with pytest.raises(ValueError, match="nominal composite"):
        harmonizer.harmonize_tensor(composite, mask, background, nominal[:, :8])
    assert replace(harmonizer, gain_range=(.6, .7)).exposure_gain(
        composite, mask, background).item() == pytest.approx(.6)
    assert replace(harmonizer, gain_range=(.05, .1)).exposure_gain(
        composite, mask, background).item() == pytest.approx(.1)

    colors = composite.clone()
    colors[2, mask] = 0.
    colors.requires_grad_()
    harmonizer.harmonize_tensor(colors, mask, background, colors)[0, mask].sum().backward()
    assert torch.isfinite(colors.grad).all() and colors.grad[:, mask].abs().sum() > 0
    empty = torch.zeros_like(mask)
    assert harmonizer.exposure_gain(composite, empty, background) is None
    torch.testing.assert_close(harmonizer.harmonize_tensor(composite, empty, background, composite), composite)
    ring = ExposureHarmonizer(reference="ring")
    full = torch.ones_like(mask)
    assert ring.exposure_gain(composite, full, background) is None
    torch.testing.assert_close(ring.harmonize_tensor(composite, full, background, composite), composite)
    with pytest.raises(NotImplementedError, match="ClassicHarmonizer.*not differentiable"):
        ClassicHarmonizer().harmonize_tensor(composite, mask, background, composite)


def test_feathering():
    mask = torch.zeros(9, 9, dtype=torch.bool)
    mask[2:7, 2:7] = True
    background = torch.full((3, 9, 9), .2)
    composite = torch.full((3, 9, 9), .8)
    harmonizer = ExposureHarmonizer(reference="under", gain_range=(.5, .5), feather_sigma=1.)
    sharp = replace(harmonizer, feather_sigma=0.).harmonize_tensor(composite, mask, background, composite)
    feathered = harmonizer.harmonize_tensor(composite, mask, background, composite)
    torch.testing.assert_close(feathered[:, ~mask], composite[:, ~mask], atol=0, rtol=0)
    assert torch.all(feathered[:, mask] < sharp[:, mask])
    assert torch.all(feathered[:, mask] > background[:, mask])
    assert torch.all(feathered[:, 4, 4] > feathered[:, 2, 2])
    tiny = torch.ones((1, 1), dtype=torch.bool)
    assert torch.isfinite(harmonizer.harmonize_tensor(
        composite[:, :1, :1], tiny, background[:, :1, :1], composite[:, :1, :1])).all()


def test_numpy_call():
    composite = np.full((7, 8, 3), (224, 127, 0), dtype=np.uint8)
    background = np.full_like(composite, (31, 73, 104))
    mask = np.zeros((7, 8), dtype=np.uint8)
    mask[1:6, 2:6] = 255
    harmonizer = ExposureHarmonizer(reference="under")
    composite_tensor = torch.from_numpy(composite).permute(2, 0, 1).float() / 255
    tensor_result = harmonizer.harmonize_tensor(
        composite_tensor, torch.from_numpy(mask > 0),
        torch.from_numpy(background).permute(2, 0, 1).float() / 255, composite_tensor,
    )
    expected = (tensor_result * 255).round().to(torch.uint8).permute(1, 2, 0).numpy()
    result = harmonizer(composite, mask, background)
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(result[mask == 0], composite[mask == 0])
    for options in ({"reference": "other"}, {"ring_px": -1},
                    {"feather_sigma": float("nan")}, {"gain_range": (1., .5)}):
        with pytest.raises(ValueError):
            ExposureHarmonizer(**options)
