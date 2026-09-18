"""Concrete threat models (``Perturbation``): a physical patch and a whole-image L-inf perturbation.

Both plug into the same ``GradientAttack`` (PGD) — the algorithm only calls ``apply``/``project``.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .geometry import rear_panel_homography  # noqa: F401 - re-exported for callers
from .interface import AdvSample, Perturbation


def _perspective_paste(patch: torch.Tensor, h_mat: torch.Tensor, out_h: int, out_w: int):
    """Warp ``patch`` (3,Hp,Wp) into an (out_h,out_w) canvas by ``h_mat`` (patch-px -> image-px).

    Returns ``(warped (3,out_h,out_w), mask (1,out_h,out_w))``, differentiable in ``patch``.
    """
    dev = patch.device
    hp, wp = patch.shape[1:]
    ys, xs = torch.meshgrid(torch.arange(out_h, device=dev), torch.arange(out_w, device=dev),
                            indexing="ij")
    img = torch.stack([xs.flatten(), ys.flatten(), torch.ones(out_h * out_w, device=dev)], 0).float()
    src = torch.inverse(h_mat.to(dev).float()) @ img  # image-px -> patch-px
    src = src[:2] / src[2:3].clamp(min=1e-8)
    gx = src[0] / max(wp - 1, 1) * 2 - 1
    gy = src[1] / max(hp - 1, 1) * 2 - 1
    grid = torch.stack([gx, gy], -1).reshape(1, out_h, out_w, 2)
    warped = F.grid_sample(patch.unsqueeze(0), grid, mode="bilinear", padding_mode="zeros",
                           align_corners=True)[0]
    mask = (((gx >= -1) & (gx <= 1) & (gy >= -1) & (gy <= 1)).reshape(1, out_h, out_w)).float()
    return warped, mask


class PatchPerturbation(Perturbation):
    """A rectangular texture warped onto a target's rear panel (the physical-patch surrogate).

    δ is the patch pixels ``(3, Hp, Wp)`` in [0,1]; ``apply`` warps+composites it into the sample's
    background at ``sample.context['placement']`` (a homography) with photometric jitter; ``project``
    clamps to [0,1] (the whole surface is adversarial); ``export`` yields the deploy texture.
    """

    def __init__(self, size=(128, 128), device="cuda:0", init="random", seed=0):
        self.hp, self.wp = size
        self.device = device
        self.init_mode = init
        self.seed = seed

    def init(self) -> torch.Tensor:
        g = torch.Generator().manual_seed(self.seed)
        d = (torch.rand(3, self.hp, self.wp, generator=g) if self.init_mode == "random"
             else torch.full((3, self.hp, self.wp), 0.5))
        return d.to(self.device).requires_grad_(True)

    def apply(self, sample: AdvSample, delta: torch.Tensor) -> torch.Tensor:
        x = sample.x.to(delta.device)
        out_h, out_w = x.shape[1:]
        h_mat = sample.context["placement"]
        h_mat = h_mat if torch.is_tensor(h_mat) else torch.as_tensor(h_mat)
        warped, mask = _perspective_paste(delta, h_mat, out_h, out_w)
        b = float(sample.context.get("brightness", 1.0))
        c = float(sample.context.get("contrast", 1.0))
        warped = (((warped - 0.5) * c + 0.5) * b).clamp(0, 1)
        return x * (1 - mask) + warped * mask

    def project(self, delta: torch.Tensor) -> torch.Tensor:
        return delta.clamp(0, 1)

    def export(self, delta: torch.Tensor) -> np.ndarray:
        rgb = (delta.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        alpha = np.full((*rgb.shape[:2], 1), 255, np.uint8)
        return np.concatenate([rgb, alpha], axis=2)


class LinfImage(Perturbation):
    """A whole-image additive L-inf perturbation — same PGD, different threat model (no export)."""

    def __init__(self, epsilon=8 / 255, image_hw=(600, 800), device="cuda:0"):
        self.epsilon = epsilon
        self.h, self.w = image_hw
        self.device = device

    def init(self) -> torch.Tensor:
        return torch.zeros(3, self.h, self.w, device=self.device, requires_grad=True)

    def apply(self, sample: AdvSample, delta: torch.Tensor) -> torch.Tensor:
        return (sample.x.to(delta.device) + delta).clamp(0, 1)

    def project(self, delta: torch.Tensor) -> torch.Tensor:
        return delta.clamp(-self.epsilon, self.epsilon)
