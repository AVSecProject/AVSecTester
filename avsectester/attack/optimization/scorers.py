"""Score arbitrary images using a closed-loop session's driving-system binding."""
from __future__ import annotations

import numpy as np

from .interface import Scorer


class SessionScorer(Scorer):
    """Use a session scorer for user-generated images, preserving requested gradients."""

    def __init__(self, session, scorer_id, *, require_grad, input_format=None, device="cpu"):
        self.session, self.scorer_id = session, scorer_id
        self.differentiable = require_grad
        self.input_format, self.device = input_format, device

    def _input(self, image):
        """Adapt only the formats explicitly declared by the driving scorer.

        Unknown/custom format descriptions are passed through: the user's binding owns its
        representation. Built-in descriptors make their CHW-float / HWC-uint8 seam explicit.
        """
        import torch
        if self.input_format == "CHW float tensor [0,1]":
            if isinstance(image, np.ndarray):
                if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
                    raise ValueError("CHW scorer conversion requires HWC RGB uint8 rendered images")
                return torch.as_tensor(image.copy(), device=self.device).permute(2, 0, 1).float()/255
            if not torch.is_tensor(image) or image.ndim != 3 or image.shape[0] != 3:
                raise ValueError("Scorer requires a CHW float image tensor")
            return image.to(self.device)
        if self.input_format == "HWC RGB uint8":
            if torch.is_tensor(image):
                if image.requires_grad:
                    raise ValueError("A HWC uint8 inference scorer cannot preserve image gradients")
                image = (image.clamp(0, 1).permute(1, 2, 0).cpu().numpy()*255).astype(np.uint8)
            if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
                raise ValueError("Scorer requires a HWC RGB uint8 image")
        return image

    def target_score(self, image, target):
        # PGD's final report runs under no_grad. Requiring a graph there would reject a
        # valid differentiable scorer merely because no gradient is requested for reporting.
        import torch
        return self.session.score(self.scorer_id, self._input(image), target,
                                  require_grad=self.differentiable and torch.is_grad_enabled()).value
