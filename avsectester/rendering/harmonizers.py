"""Image harmonization with optional in-process PCTNet inference."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
import math
from typing import ClassVar, TYPE_CHECKING
from pathlib import Path
import numpy as np

from avsectester.validation import require_nonnegative_integer, require_nonnegative_number

if TYPE_CHECKING:
    import torch

log = logging.getLogger(__name__)


class Harmonizer(ABC):
    """Adjust the pasted foreground to fit the background. Must preserve the patch's texture."""

    differentiable: ClassVar[bool] = False

    def harmonize_tensor(
        self, composite: torch.Tensor, mask: torch.Tensor, background: torch.Tensor,
        nominal_composite: torch.Tensor,
    ) -> torch.Tensor:
        """Adjust CHW floating sRGB only within the HW boolean visible mask.

        ``nominal_composite`` shows each surface's nominal texture where one is set and equals
        ``composite`` elsewhere. Estimate scene statistics from it, so texture edits such as an
        adversarial patch do not change the adjustment applied to the rest of the object.
        """
        raise NotImplementedError(f"{type(self).__name__} is not differentiable")

    @abstractmethod
    def __call__(
        self, composite_rgb: np.ndarray, mask: np.ndarray, background_rgb: np.ndarray
    ) -> np.ndarray: ...


def _linear_rgb(srgb: torch.Tensor) -> torch.Tensor:
    import torch

    return torch.where(srgb <= .04045, srgb / 12.92, ((srgb + .055) / 1.055).pow(2.4))


def _srgb_rgb(linear: torch.Tensor) -> torch.Tensor:
    import torch

    # The inactive power branch must also have a finite derivative at zero.
    return torch.where(linear <= .0031308, 12.92 * linear,
                       1.055 * linear.clamp_min(.0031308).pow(1 / 2.4) - .055)


def _reflection_indices(length: int, radius: int, device: torch.device) -> torch.Tensor:
    import torch

    indices = torch.arange(-radius, length + radius, device=device)
    if length == 1:
        return torch.zeros_like(indices)
    period = 2 * (length - 1)
    indices = indices.remainder(period)
    return torch.minimum(indices, period - indices)


def _feather_mask(mask: torch.Tensor, sigma: float, dtype: torch.dtype) -> torch.Tensor:
    import torch
    import torch.nn.functional as F

    radius = int(round(4 * sigma))
    if radius == 0:
        return mask.to(dtype)
    coordinates = torch.arange(-radius, radius + 1, device=mask.device, dtype=dtype)
    kernel = torch.exp(-.5 * (coordinates / sigma).square())
    kernel = kernel / kernel.sum()
    weights = mask.to(dtype)[None, None]
    # Reflect without repeating the border, even for images smaller than the kernel.
    weights = weights.index_select(-1, _reflection_indices(mask.shape[1], radius, mask.device))
    weights = F.conv2d(weights, kernel[None, None, None, :])
    weights = weights.index_select(-2, _reflection_indices(mask.shape[0], radius, mask.device))
    return F.conv2d(weights, kernel[None, None, :, None])[0, 0]


def _check_tensor_inputs(composite: torch.Tensor, mask: torch.Tensor, background: torch.Tensor) -> None:
    import torch

    if (composite.ndim != 3 or composite.shape[0] != 3 or background.shape != composite.shape
            or mask.shape != composite.shape[1:]):
        raise ValueError("Harmonizer inputs require RGB CHW images and an HW mask")
    if (not composite.is_floating_point() or not background.is_floating_point()
            or mask.dtype != torch.bool):
        raise TypeError("Harmonizer inputs require floating RGB tensors and a boolean mask")
    if (background.device != composite.device or mask.device != composite.device
            or background.dtype != composite.dtype):
        raise ValueError("Harmonizer inputs must share one device and image dtype")


@dataclass(frozen=True)
class ExposureHarmonizer(Harmonizer):
    """Match visible foreground exposure to a background reference in linear RGB.

    The ring reference samples square outward dilation minus the visible mask.
    The under reference samples the original background below the visible mask.
    One clipped luminance gain, estimated from the nominal composite, scales all
    channels of the composite, with optional Gaussian feathering against the
    background. Only visible-mask pixels change. NumPy and differentiable tensor
    calls share this calculation; NumPy has no nominal composite, estimates the
    gain from the composite itself and rounds to uint8 only at the end.
    """

    reference: str = "ring"
    ring_px: int = 12
    gain_range: tuple[float, float] = (.05, 1.2)
    feather_sigma: float = 1.2
    differentiable: ClassVar[bool] = True

    def __post_init__(self) -> None:
        if self.reference not in {"ring", "under"}:
            raise ValueError("reference must be ring or under")
        require_nonnegative_integer(self.ring_px, "ring_px")
        require_nonnegative_number(self.feather_sigma, "feather_sigma")
        if len(self.gain_range) != 2 or any(
            isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number))
            or not math.isfinite(float(value)) or float(value) <= 0
            for value in self.gain_range
        ) or self.gain_range[0] > self.gain_range[1]:
            raise ValueError("gain_range must be two positive finite values in increasing order")
        object.__setattr__(self, "gain_range", tuple(float(value) for value in self.gain_range))

    def exposure_gain(
        self, composite: torch.Tensor, mask: torch.Tensor, background: torch.Tensor,
    ) -> torch.Tensor | None:
        """Return the clipped gain from ``composite``'s visible luminance to the reference.

        Inputs use CHW floating sRGB in [0, 1] and an HW boolean mask. Harmonization passes
        the nominal composite here, so this measures exactly the gain used for rendering.
        Returns None for an empty visible or reference region.
        """
        import torch.nn.functional as F

        _check_tensor_inputs(composite, mask, background)
        if not mask.any():
            return None
        reference = mask
        if self.reference == "ring":
            dilated = F.max_pool2d(
                mask.to(composite.dtype)[None, None], 2 * self.ring_px + 1,
                stride=1, padding=self.ring_px,
            )[0, 0] > 0
            reference = dilated & ~mask
        if not reference.any():
            return None
        luminance_weights = composite.new_tensor((.2126, .7152, .0722))[:, None]
        foreground_y = (_linear_rgb(composite[:, mask]) * luminance_weights).sum(dim=0)
        reference_y = (_linear_rgb(background[:, reference]) * luminance_weights).sum(dim=0)
        return (reference_y.mean() / (foreground_y.mean() + 1e-6)).clamp(
            self.gain_range[0], self.gain_range[1])

    def harmonize_tensor(
        self, composite: torch.Tensor, mask: torch.Tensor, background: torch.Tensor,
        nominal_composite: torch.Tensor,
    ) -> torch.Tensor:
        import torch

        _check_tensor_inputs(composite, mask, background)
        if (nominal_composite.shape != composite.shape or nominal_composite.dtype != composite.dtype
                or nominal_composite.device != composite.device):
            raise ValueError("The nominal composite must match the composite's shape, dtype and device")
        gain = self.exposure_gain(nominal_composite, mask, background)
        if gain is None:
            return composite
        adjusted = _srgb_rgb(_linear_rgb(composite) * gain).clamp(0, 1)
        if self.feather_sigma > 0:
            alpha = _feather_mask(mask, self.feather_sigma, composite.dtype)[None]
            adjusted = alpha * adjusted + (1 - alpha) * background
        return torch.where(mask[None], adjusted, composite)

    def __call__(
        self, composite_rgb: np.ndarray, mask: np.ndarray, background_rgb: np.ndarray,
    ) -> np.ndarray:
        import torch

        composite = torch.from_numpy(np.ascontiguousarray(composite_rgb)).permute(2, 0, 1).float() / 255
        background = torch.from_numpy(np.ascontiguousarray(background_rgb)).permute(2, 0, 1).float() / 255
        visible = torch.from_numpy(np.ascontiguousarray(mask > 0))
        with torch.no_grad():
            output = self.harmonize_tensor(composite, visible, background, composite)
        return (output.clamp(0, 1) * 255).round().to(torch.uint8).permute(1, 2, 0).numpy()


class ClassicHarmonizer(Harmonizer):
    """Reinhard color transfer (match the patch's Lab mean/std to the scene) + Poisson seamless blend.

    Reliable, in-process, no model weights; texture-preserving (Poisson keeps the patch's gradients
    while shifting color/brightness to the background).

    Both steps pull the patch's *colour* toward its surroundings, which suits a texture whose exact hue
    does not matter but washes out objects whose colour carries meaning (a red STOP sign turns grey-brown
    on a grey road). For those, ``preserve_chroma=True`` instead scales only the Lab lightness by an
    exposure gain (surrounding mean / object mean, clipped to ``gain_range``), keeping a/b and the
    object's internal contrast; ``blend="feather"`` replaces the Poisson solve with a feathered alpha
    edge, which keeps the object's own colours. Defaults reproduce the original behaviour.
    """

    def __init__(
        self,
        color_transfer: bool = True,
        poisson: bool = True,
        preserve_chroma: bool = False,
        blend: str = "poisson",
        feather: float = 1.2,
        gain_range: tuple = (0.35, 1.2),
    ) -> None:
        self.color_transfer = color_transfer
        self.poisson = poisson and blend == "poisson"
        self.preserve_chroma = preserve_chroma
        self.gain_range = gain_range
        self.blend = blend
        self.feather = feather

    def __call__(self, composite_rgb, mask, background_rgb):
        import cv2

        out = composite_rgb.copy()
        m = mask > 0
        if self.color_transfer and m.any():
            lab = cv2.cvtColor(out, cv2.COLOR_RGB2LAB).astype(np.float32)
            bg_lab = cv2.cvtColor(background_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
            # dilate the mask to sample the surrounding scene for the target statistics
            ring = cv2.dilate(mask, np.ones((25, 25), np.uint8)) > 0
            ring &= ~m
            if ring.any() and self.preserve_chroma:
                # exposure-style gain on lightness only: the scene's light level scales the object
                # while its own contrast (white legend vs red field) and hue survive
                gain = np.clip(bg_lab[ring, 0].mean() / (lab[m, 0].mean() + 1e-5), *self.gain_range)
                lab[m, 0] *= gain
                out = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
            elif ring.any():
                for c in range(3):
                    fmu, fsd = lab[m, c].mean(), lab[m, c].std() + 1e-5
                    bmu, bsd = bg_lab[ring, c].mean(), bg_lab[ring, c].std() + 1e-5
                    lab[m, c] = (lab[m, c] - fmu) * (bsd / fsd) + bmu
                out = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
        if self.blend == "feather" and self.feather > 0 and m.any():
            a = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (0, 0), self.feather)[:, :, None]
            out = (out.astype(np.float32) * a + background_rgb.astype(np.float32) * (1 - a)).astype(
                np.uint8
            )
        if self.poisson and m.any():
            ys, xs = np.where(m)
            center = (int((xs.min() + xs.max()) / 2), int((ys.min() + ys.max()) / 2))
            out = cv2.seamlessClone(out, background_rgb, mask, center, cv2.NORMAL_CLONE)
        return out


class PCTNetHarmonizer(Harmonizer):
    """Learned harmonization via libcom's **PCTNet**, run **in-process** (no subprocess, no separate env).

    PCTNet is a self-contained color-transform CNN needing only torch / torchvision / numpy / einops —
    all compatible with the avstack stack (torch 2.1) — so we load just the ``pct_net`` module from
    ``libcom_directory``, bypassing libcom's package ``__init__`` (which eagerly imports
    diffusers-based models). PCTNet gives a milder, texture-preserving harmonization than the classic
    Poisson blend — better for keeping an adversarial pattern intact. Falls back to
    :class:`ClassicHarmonizer` on any error (or set ``strict`` to raise instead).

    ``libcom_directory`` is the repository root containing ``libcom/image_harmonization``.
    ``device`` is a PyTorch device string; inference falls back to CPU when CUDA is unavailable.
    """

    def __init__(
        self, libcom_directory: str | Path, device: str = "cuda:0",
        weights: str | Path | None = None, strict: bool = False,
    ) -> None:
        self.libcom_directory = Path(libcom_directory)
        self.device = device
        self.weights = weights
        self.strict = strict
        self._net = None
        self._dev = None
        self._fallback = ClassicHarmonizer()

    def _load(self):
        if self._net is not None:
            return self._net
        import importlib.util
        import sys
        import types

        import torch

        root = self.libcom_directory / "libcom" / "image_harmonization"
        src = root / "source"
        # dummy parent packages so pct_net's absolute imports resolve WITHOUT running libcom/__init__
        # (which imports diffusers/pytorch-lightning models that conflict with the avstack stack).
        for name in ("libcom", "libcom.image_harmonization", "libcom.image_harmonization.source"):
            if name not in sys.modules:
                mod = types.ModuleType(name)
                mod.__path__ = []
                sys.modules[name] = mod

        def _load_file(name, path):
            spec = importlib.util.spec_from_file_location(name, str(path))
            mod = importlib.util.module_from_spec(spec)
            sys.modules[name] = mod
            spec.loader.exec_module(mod)
            return mod

        _load_file("libcom.image_harmonization.source.functions", src / "functions.py")
        pct = _load_file("libcom.image_harmonization.source.pct_net", src / "pct_net.py")
        weights = Path(self.weights) if self.weights else self._resolve_weights(root)
        self._dev = self.device if torch.cuda.is_available() else "cpu"
        net = pct.PCTNet()
        net.load_state_dict(torch.load(str(weights), map_location="cpu", weights_only=True))
        self._net = net.to(self._dev).eval()
        log.info("PCTNet harmonizer loaded in-process (%s, %s)", weights.name, self._dev)
        return self._net

    def _resolve_weights(self, root):
        cand = root / "pretrained_models" / "PCTNet.pth"
        if cand.exists():
            return cand
        from huggingface_hub import hf_hub_download

        cand.parent.mkdir(parents=True, exist_ok=True)
        return Path(hf_hub_download("BCMIZB/Libcom_pretrained_models", "PCTNet.pth",
                                    local_dir=str(cand.parent)))

    def __call__(self, composite_rgb, mask, background_rgb):
        try:
            import cv2
            import torch

            net = self._load()
            img = np.ascontiguousarray(composite_rgb, dtype=np.uint8)  # RGB HxWx3
            m = (np.asarray(mask) > 0).astype(np.uint8) * 255
            img_lr, mask_lr = cv2.resize(img, (256, 256)), cv2.resize(m, (256, 256))

            def _t(a):
                return torch.from_numpy(a).float().div(255).permute(2, 0, 1).to(self._dev)

            def _tm(a):
                return torch.from_numpy(a).float().div(255)[None].to(self._dev)

            with torch.no_grad():  # PCTNet.forward adds the batch dim itself; pass (C,H,W)
                out = net(_t(img_lr), _t(img), _tm(mask_lr), _tm(m))
            res = torch.clamp(255.0 * out.squeeze(0).permute(1, 2, 0), 0, 255).cpu().numpy()
            log.info("harmonized frame with in-process PCTNet")
            return res.astype(np.uint8)
        except Exception as exc:
            if self.strict:
                raise RuntimeError(f"PCTNet harmonization failed: {exc}") from exc
            log.warning("PCTNet failed (%s) -> classic fallback", str(exc)[-200:])
            return self._fallback(composite_rgb, mask, background_rgb)
