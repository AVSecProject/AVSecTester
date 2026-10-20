"""Concrete threat models (``Perturbation``): a physical patch and a whole-image L-inf perturbation.

Both plug into the same ``GradientOptimizer`` (PGD) — the algorithm only calls ``apply``/``project``.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .interface import AdversarialSample, Perturbation


class PatchPerturbation(Perturbation):
    """A rectangular texture warped with a user-supplied sampling grid and visibility mask.

    δ is the patch pixels ``(3, Hp, Wp)`` in [0,1]; ``apply`` warps+composites it into the sample's
    background using calibrated ``sampling_grid``/``mask``, with photometric jitter; ``project``
    clamps to [0,1] (the whole surface is adversarial); ``export`` yields the deploy texture.
    """

    def __init__(self, size=(128, 128), device="cuda:0", init="random", seed=0, alpha=None):
        if (len(size) != 2 or any(isinstance(value, bool) or not isinstance(value, int)
                                  or value <= 0 for value in size)):
            raise ValueError("size must contain two positive integers")
        if init not in {"random", "gray"}:
            raise ValueError("init must be random or gray")
        if alpha is not None:
            alpha = np.asarray(alpha)
            if alpha.shape != tuple(size) or alpha.dtype != np.uint8:
                raise ValueError("alpha must be a uint8 mask matching the texture size")
            alpha = alpha.copy()
        self.hp, self.wp = size
        self.device = device
        self.init_mode = init
        self.seed = seed
        self.alpha = alpha

    def init(self) -> torch.Tensor:
        g = torch.Generator().manual_seed(self.seed)
        d = (torch.rand(3, self.hp, self.wp, generator=g) if self.init_mode == "random"
             else torch.full((3, self.hp, self.wp), 0.5))
        return d.to(self.device).requires_grad_(True)

    def apply(self, sample: AdversarialSample, delta: torch.Tensor) -> torch.Tensor:
        x = sample.x.to(delta.device)
        out_h, out_w = x.shape[1:]
        grid = torch.as_tensor(sample.context["sampling_grid"], device=delta.device,
                               dtype=delta.dtype)
        mask = torch.as_tensor(sample.context["mask"], device=delta.device, dtype=delta.dtype)
        if grid.shape != (out_h, out_w, 2) or mask.shape != (1, out_h, out_w):
            raise ValueError("Calibrated patch grid/mask must match the background dimensions")
        warped = F.grid_sample(delta.unsqueeze(0), grid.unsqueeze(0), mode="bilinear",
                               padding_mode="border", align_corners=False)[0]
        coverage = F.grid_sample(torch.ones_like(delta[:1]).unsqueeze(0), grid.unsqueeze(0),
                                 mode="bilinear", padding_mode="zeros", align_corners=False)[0]
        mask = mask * coverage
        b = float(sample.context.get("brightness", 1.0))
        c = float(sample.context.get("contrast", 1.0))
        warped = (((warped - 0.5) * c + 0.5) * b).clamp(0, 1)
        return x * (1 - mask) + warped * mask

    def project(self, delta: torch.Tensor) -> torch.Tensor:
        return delta.clamp(0, 1)

    def export(self, delta: torch.Tensor) -> np.ndarray:
        rgb = (delta.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        alpha = (np.full((*rgb.shape[:2], 1), 255, np.uint8) if self.alpha is None
                 else np.asarray(self.alpha, dtype=np.uint8)[..., None])
        return np.concatenate([rgb, alpha], axis=2)


class LinfImage(Perturbation):
    """A whole-image additive L-inf perturbation — same PGD, different threat model (no export)."""

    def __init__(self, epsilon=8 / 255, image_hw=(600, 800), device="cuda:0"):
        self.epsilon = epsilon
        self.h, self.w = image_hw
        self.device = device

    def init(self) -> torch.Tensor:
        return torch.zeros(3, self.h, self.w, device=self.device, requires_grad=True)

    def apply(self, sample: AdversarialSample, delta: torch.Tensor) -> torch.Tensor:
        return (sample.x.to(delta.device) + delta).clamp(0, 1)

    def project(self, delta: torch.Tensor) -> torch.Tensor:
        return delta.clamp(-self.epsilon, self.epsilon)
