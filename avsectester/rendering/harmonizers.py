"""Image harmonization with optional in-process PCTNet inference."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
import numpy as np

log = logging.getLogger(__name__)


class Harmonizer(ABC):
    """Adjust the pasted foreground to fit the background. Must preserve the patch's texture."""

    @abstractmethod
    def __call__(
        self, composite_rgb: np.ndarray, mask: np.ndarray, background_rgb: np.ndarray
    ) -> np.ndarray: ...


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
    all compatible with the avstack stack (torch 2.1) — so we load just the ``pct_net`` module from the
    ``third_party/libcom`` submodule, bypassing libcom's package ``__init__`` (which eagerly imports
    diffusers-based models). PCTNet gives a milder, texture-preserving harmonization than the classic
    Poisson blend — better for keeping an adversarial pattern intact. Falls back to
    :class:`ClassicHarmonizer` on any error (or set ``strict`` to raise instead).
    """

    _SUBMODULE = "third_party/libcom/libcom/image_harmonization"

    def __init__(self, device: int = 0, weights: str | None = None, strict: bool = False) -> None:
        self.device_id = device
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
        from pathlib import Path

        import torch

        root = Path(__file__).resolve().parents[2] / self._SUBMODULE
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
        self._dev = f"cuda:{self.device_id}" if torch.cuda.is_available() else "cpu"
        net = pct.PCTNet()
        net.load_state_dict(torch.load(str(weights), map_location="cpu", weights_only=True))
        self._net = net.to(self._dev).eval()
        log.info("PCTNet harmonizer loaded in-process (%s, %s)", weights.name, self._dev)
        return self._net

    def _resolve_weights(self, root):
        from pathlib import Path

        cand = root / "pretrained_models" / "PCTNet.pth"
        if cand.exists():
            return cand
        from huggingface_hub import hf_hub_download  # downloaded once, then cached in the submodule

        cand.parent.mkdir(parents=True, exist_ok=True)
        return Path(
            hf_hub_download(
                "BCMIZB/Libcom_pretrained_models", "PCTNet.pth", local_dir=str(cand.parent)
            )
        )

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
