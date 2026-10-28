"""Local COCO stop-sign detection shared by rule driving and detector scorers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, TYPE_CHECKING

import numpy as np

from avsectester.data_types import TargetSpec
from avsectester.validation import require_nonnegative_number

if TYPE_CHECKING:
    import torch


STOP_SIGN_LABEL = 13


def load_stop_sign_model(weights_path: str | Path, device: str = "cuda:0") -> Any:
    """Load a frozen COCO_V1 Faster R-CNN from an existing local checkpoint.

    Neither the detector nor its backbone downloads weights. The model is in evaluation
    mode with a zero internal score threshold so each caller can apply its own threshold.
    Frozen model parameters still allow differentiation with respect to an input image.
    """
    import torch
    from torchvision.models.detection import (
        FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn,
    )

    categories = FasterRCNN_ResNet50_FPN_Weights.COCO_V1.meta["categories"]
    if categories.index("stop sign") != STOP_SIGN_LABEL:
        raise ValueError("COCO_V1 metadata does not identify stop sign as category 13")
    path = Path(weights_path)
    if not path.is_file():
        raise FileNotFoundError(f"Existing detector weights are required: {path}")
    model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.roi_heads.score_thresh = 0.
    model.to(device)
    model.eval()
    model.requires_grad_(False)
    return model


class CocoStopSignDetector:
    """Detect only COCO category 13 in uint8 RGB images using a local checkpoint.

    Loading is lazy. An injected model replaces only the learned-model boundary and follows
    torchvision's detection-model interface; it is prepared identically to the real model.
    """

    def __init__(self, weights_path: str | Path, device: str = "cuda:0",
                 score_threshold: float = .5, model: Any = None) -> None:
        require_nonnegative_number(score_threshold, "score_threshold")
        if score_threshold > 1:
            raise ValueError("score_threshold must be at most 1")
        self.weights_path = Path(weights_path)
        self.device = str(device)
        self.score_threshold = score_threshold
        self.model = model
        self._loaded_model: Any = None
        self._closed = False

    def _model_for_detection(self) -> Any:
        import torch

        if self._loaded_model is not None:
            return self._loaded_model
        model = self.model
        if model is None:
            model = load_stop_sign_model(self.weights_path, self.device)
        else:
            if not isinstance(model, torch.nn.Module):
                raise TypeError("model must implement torchvision's torch.nn.Module detection interface")
            if not hasattr(model, "roi_heads") or not hasattr(model.roi_heads, "score_thresh"):
                raise TypeError("model must expose roi_heads.score_thresh")
            model.roi_heads.score_thresh = 0.
            model.to(self.device)
            model.eval()
            model.requires_grad_(False)
        self._loaded_model = model
        return model

    def detect(self, rgb: np.ndarray) -> list[tuple[np.ndarray, float]]:
        """Return original-image xyxy boxes and scores at or above score_threshold."""
        import torch

        if self._closed:
            raise RuntimeError("CocoStopSignDetector has been closed")
        if (not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.ndim != 3
                or rgb.shape[2] != 3 or 0 in rgb.shape):
            raise ValueError("rgb must be a nonempty uint8 RGB (H, W, 3) image")
        image = torch.from_numpy(np.array(rgb, copy=True)).permute(2, 0, 1)
        image = image.to(device=self.device, dtype=torch.float32) / 255
        with torch.inference_mode():
            prediction = self._model_for_detection()([image])[0]
        boxes = prediction["boxes"].detach().cpu().numpy()
        scores = prediction["scores"].detach().cpu().numpy()
        labels = prediction["labels"].detach().cpu().numpy()
        return [(np.array(box, copy=True), float(score))
                for box, score, label in zip(boxes, scores, labels)
                if int(label) == STOP_SIGN_LABEL and float(score) >= self.score_threshold]

    def target_score(self, image: torch.Tensor, target: TargetSpec) -> torch.Tensor:
        """Return a scalar confidence for a float RGB ``(3, H, W)`` image in [0, 1].

        The target uses original-image xyxy pixel coordinates. No matching box yields
        ``image.sum() * 0``, keeping a valid zero-gradient connection for PGD.
        ``target.label`` may be None or ``"stop sign"``.
        """
        import torch

        if self._closed:
            raise RuntimeError("CocoStopSignDetector has been closed")
        if (not isinstance(image, torch.Tensor) or image.ndim != 3 or image.shape[0] != 3
                or 0 in image.shape or not image.is_floating_point()
                or not bool(torch.isfinite(image).all())
                or bool((image < 0).any()) or bool((image > 1).any())):
            raise ValueError("image must be finite float RGB (3, H, W) values in [0, 1]")
        if not isinstance(target, TargetSpec) or target.label not in (None, "stop sign"):
            raise ValueError("target must be a TargetSpec for a stop sign")
        image = image.to(self.device)
        box = torch.as_tensor(target.box, dtype=image.dtype, device=image.device)
        if (box.shape != (4,) or not bool(torch.isfinite(box).all())
                or bool((box[2:] <= box[:2]).any())):
            raise ValueError("target.box must be finite xyxy coordinates with positive width and height")
        prediction = self._model_for_detection()([image])[0]
        boxes, scores, labels = prediction["boxes"], prediction["scores"], prediction["labels"]
        if not len(boxes):
            return image.sum() * 0
        intersection_size = (torch.minimum(boxes[:, 2:], box[2:])
                             - torch.maximum(boxes[:, :2], box[:2])).clamp(min=0)
        intersection = intersection_size.prod(dim=1)
        areas = (boxes[:, 2:] - boxes[:, :2]).clamp(min=0).prod(dim=1)
        union = areas + (box[2:] - box[:2]).prod() - intersection
        overlaps = intersection / union.clamp(min=torch.finfo(image.dtype).eps)
        matching = (labels == STOP_SIGN_LABEL) & (overlaps >= .3)
        if not bool(matching.any()):
            return image.sum() * 0
        return scores[matching].max()

    def close(self) -> None:
        """Release the model; a closed detector cannot detect or score again."""
        self._closed = True
        self._loaded_model = None
        self.model = None


__all__ = [
    'CocoStopSignDetector',
]
