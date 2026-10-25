"""A local COCO Faster R-CNN proxy for stop-sign patch optimization."""
from __future__ import annotations

from pathlib import Path
from typing import Any, TYPE_CHECKING

from avsectester.attack.optimization import Scorer
from avsectester.data_types import TargetSpec

if TYPE_CHECKING:
    import torch


class StopSignDetector(Scorer):
    """Score stop-sign detections overlapping a specified image-space target box.

    The real model is loaded lazily from ``weights_path``; neither detector nor
    backbone weights are downloaded. COCO_V1 metadata is checked for the stop-sign
    category, whose ID is 13. An injected ``model`` replaces only the learned-model
    boundary and must follow torchvision's detection-model interface.

    In evaluation mode, ``roi_heads.score_thresh=0`` preserves low-score detections.
    The score is the maximum stop-sign confidence among surviving boxes with IoU
    at least 0.3 against ``TargetSpec.box``. NMS and the discrete choice of matching
    boxes are not differentiable; gradients flow through the selected confidence.
    Frozen model parameters do not prevent gradients reaching the input image.
    """

    stop_sign_label = 13

    def __init__(
        self, weights_path: str | Path = "fasterrcnn_resnet50_fpn_coco-258fb6c6.pth",
        device: str = "cpu", model: Any = None,
    ) -> None:
        self.weights_path = Path(weights_path)
        self.device = str(device)
        self.model = model
        self._loaded_model: Any = None

    def _model_for_scoring(self) -> Any:
        import torch

        if self._loaded_model is not None:
            return self._loaded_model
        model = self.model
        if model is None:
            from torchvision.models.detection import (
                FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn,
            )

            categories = FasterRCNN_ResNet50_FPN_Weights.COCO_V1.meta["categories"]
            if categories.index("stop sign") != self.stop_sign_label:
                raise ValueError("COCO_V1 metadata does not identify stop sign as category 13")
            if not self.weights_path.is_file():
                raise FileNotFoundError(f"Existing detector weights are required: {self.weights_path}")
            model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)
            state = torch.load(self.weights_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state)
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

    def target_score(self, image: torch.Tensor, target: TargetSpec) -> torch.Tensor:
        """Return a scalar confidence for a float RGB ``(3, H, W)`` image in [0, 1].

        The target uses original-image xyxy pixel coordinates. No matching box yields
        ``image.sum() * 0``, keeping a valid zero-gradient connection for PGD.
        ``target.label`` may be None or ``"stop sign"``.
        """
        import torch

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
        prediction = self._model_for_scoring()([image])[0]
        boxes, scores, labels = prediction["boxes"], prediction["scores"], prediction["labels"]
        if not len(boxes):
            return image.sum() * 0
        intersection_size = (torch.minimum(boxes[:, 2:], box[2:])
                             - torch.maximum(boxes[:, :2], box[:2])).clamp(min=0)
        intersection = intersection_size.prod(dim=1)
        areas = (boxes[:, 2:] - boxes[:, :2]).clamp(min=0).prod(dim=1)
        union = areas + (box[2:] - box[:2]).prod() - intersection
        overlaps = intersection / union.clamp(min=torch.finfo(image.dtype).eps)
        matching = (labels == self.stop_sign_label) & (overlaps >= .3)
        if not bool(matching.any()):
            return image.sum() * 0
        return scores[matching].max()
