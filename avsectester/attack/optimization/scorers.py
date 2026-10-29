"""Adapt batches to a closed-loop session's declared scorer input format."""
from __future__ import annotations

import numpy as np

from .interface import Scorer


class SessionScorer(Scorer):
    """Call the bound batch scorer, preserving requested gradients when gradients are enabled."""

    def __init__(self, session, scorer_id, *, require_grad, input_format=None, device="cpu"):
        self.session, self.scorer_id = session, scorer_id
        self.differentiable = require_grad
        self.input_format, self.device = input_format, device

    def _inputs(self, images):
        import torch
        if self.input_format == "BCHW float tensor [0,1]":
            if isinstance(images, (list, tuple)):
                if not images or any(not isinstance(im, np.ndarray) or im.ndim != 3
                                     or im.shape[2] != 3 or im.dtype != np.uint8 for im in images):
                    raise ValueError("BCHW conversion requires a list of HWC RGB uint8 images")
                if any(im.shape != images[0].shape for im in images):
                    raise ValueError("Images in the same batch must have equal dimensions")
                images = torch.stack([torch.as_tensor(im.copy()).permute(2, 0, 1).float() / 255
                                      for im in images])
            if (not torch.is_tensor(images) or images.ndim != 4 or images.shape[1] != 3
                    or 0 in images.shape or not images.is_floating_point()
                    or not bool(torch.isfinite(images).all())
                    or bool((images < 0).any()) or bool((images > 1).any())):
                raise ValueError("Scorer requires finite BCHW float RGB images in [0, 1]")
            return images.to(self.device)
        if self.input_format == "list of HWC RGB uint8":
            if torch.is_tensor(images):
                if images.requires_grad:
                    raise ValueError("A HWC uint8 inference scorer cannot preserve image gradients")
                if images.ndim != 4 or images.shape[1] != 3:
                    raise ValueError("Scorer requires a BCHW image tensor")
                images = [(im.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                          for im in images]
            if (not isinstance(images, (list, tuple)) or not images
                    or any(not isinstance(im, np.ndarray) or im.ndim != 3 or im.shape[2] != 3
                           or im.dtype != np.uint8 or 0 in im.shape for im in images)):
                raise ValueError("Scorer requires a list of HWC RGB uint8 images")
            return images
        return images

    def target_scores(self, images, targets):
        """Return one bound score per image; target and image batch lengths must agree."""
        import torch
        inputs = self._inputs(images)
        if len(inputs) != len(targets):
            raise ValueError("images and targets must have the same batch length")
        return self.session.score(self.scorer_id, inputs, targets,
                                  require_grad=self.differentiable and torch.is_grad_enabled()).value


__all__ = [
    'SessionScorer',
]
