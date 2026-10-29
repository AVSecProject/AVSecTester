"""General COCO object detection and differentiable confidence scoring with YOLOX."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, TYPE_CHECKING

import numpy as np

from avsectester.data_types import TargetSpec
from avsectester.validation import (
    require_nonnegative_number, require_positive_integer, require_positive_number,
)

if TYPE_CHECKING:
    import torch


COCO_CLASSES = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
)


def load_yolox_model(model_name: str, weights_path: str | Path, device: str) -> Any:
    """Load a decoded, frozen YOLOX model from an existing local checkpoint."""
    import torch

    path = Path(weights_path)
    if not path.is_file():
        raise FileNotFoundError(f"Existing detector weights are required: {path}")
    from yolox.exp import get_exp

    model = get_exp(None, model_name).get_model()
    state = torch.load(path, map_location="cpu", weights_only=True)["model"]
    model.load_state_dict(state)
    model.head.decode_in_inference = True
    model.eval()
    model.requires_grad_(False)
    return model.to(device)


class YOLOXDetector:
    """Detect COCO objects and score labeled target boxes using a local YOLOX checkpoint.

    Model loading is lazy. Images are downscaled without enlargement, converted to BGR in
    [0, 255], and padded on the right and bottom with 114 to multiples of 32. Injected models
    replace the learned-model boundary and return decoded YOLOX predictions.
    """

    def __init__(
        self, weights_path: str | Path, model_name: str = "yolox-m", input_size: int = 1280,
        score_threshold: float = .5, nms_iou: float = .45, device: str = "cuda:0",
        model: Any = None,
    ) -> None:
        require_positive_integer(input_size, "input_size")
        require_nonnegative_number(score_threshold, "score_threshold")
        require_positive_number(nms_iou, "nms_iou")
        if score_threshold > 1:
            raise ValueError("score_threshold must be at most 1")
        if nms_iou > 1:
            raise ValueError("nms_iou must be at most 1")
        if not isinstance(model_name, str) or not model_name:
            raise ValueError("model_name must be a nonempty string")
        if model is not None:
            import torch

            if not isinstance(model, torch.nn.Module):
                raise TypeError("model must be a torch.nn.Module")
        self.weights_path = Path(weights_path)
        self.model_name = model_name
        self.input_size = input_size
        self.score_threshold = score_threshold
        self.nms_iou = nms_iou
        self.device = str(device)
        self.model = model
        self._loaded_model: Any = None
        self._closed = False

    def _model_for_detection(self) -> Any:
        if self._closed:
            raise RuntimeError("YOLOXDetector has been closed")
        if self._loaded_model is None:
            model = self.model
            if model is None:
                model = load_yolox_model(self.model_name, self.weights_path, self.device)
            else:
                model.eval()
                model.requires_grad_(False)
                model.to(self.device)
            self._loaded_model = model
        return self._loaded_model

    def _network_input(self, images: torch.Tensor) -> tuple[torch.Tensor, float]:
        import torch.nn.functional as functional

        height, width = images.shape[-2:]
        scale = min(self.input_size / max(height, width), 1.)
        resized = functional.interpolate(
            images, size=(max(1, round(height * scale)), max(1, round(width * scale))),
            mode="bilinear", align_corners=False,
        )
        bgr = resized[:, [2, 1, 0]] * 255
        pad_h, pad_w = (-bgr.shape[-2]) % 32, (-bgr.shape[-1]) % 32
        return functional.pad(bgr, (0, pad_w, 0, pad_h), value=114), scale

    def _prediction(self, images: torch.Tensor) -> torch.Tensor:
        import torch

        network_input, scale = self._network_input(images)
        prediction = self._model_for_detection()(network_input)
        if (not isinstance(prediction, torch.Tensor) or prediction.ndim != 3
                or prediction.shape[0] != images.shape[0]
                or prediction.shape[2] != 5 + len(COCO_CLASSES)):
            raise ValueError("YOLOX model must return decoded predictions with shape (B, N, 85)")
        center, size = prediction[..., :2], prediction[..., 2:4]
        boxes = torch.cat((center - size / 2, center + size / 2), dim=-1) / scale
        return torch.cat((boxes, prediction[..., 4:]), dim=-1)

    def detect(self, rgb: np.ndarray) -> list[tuple[np.ndarray, float, str]]:
        """Return original-image xyxy boxes, confidences and COCO labels after classwise NMS."""
        import torch
        from torchvision.ops import batched_nms

        if self._closed:
            raise RuntimeError("YOLOXDetector has been closed")
        if (not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.ndim != 3
                or rgb.shape[2] != 3 or 0 in rgb.shape):
            raise ValueError("rgb must be a nonempty uint8 RGB (H, W, 3) image")
        image = torch.from_numpy(np.array(rgb, copy=True)).permute(2, 0, 1)
        image = image.to(device=self.device, dtype=torch.float32) / 255
        self._model_for_detection()
        with torch.inference_mode():
            prediction = self._prediction(image.unsqueeze(0))[0]
            probabilities, labels = prediction[:, 5:].max(dim=1)
            scores = prediction[:, 4] * probabilities
            selected = scores >= self.score_threshold
            boxes, scores, labels = prediction[selected, :4], scores[selected], labels[selected]
            kept = batched_nms(boxes, scores, labels, self.nms_iou)
        return [
            (box.detach().cpu().numpy().copy(), float(score), COCO_CLASSES[int(label)])
            for box, score, label in zip(boxes[kept], scores[kept], labels[kept])
        ]

    def target_scores(
        self, images: torch.Tensor, targets: Sequence[TargetSpec],
    ) -> torch.Tensor:
        """Return one differentiable pre-NMS confidence per BCHW image and target.

        Images are finite float RGB (B, 3, H, W) in [0, 1]. Each target names a COCO class and
        an original-image box. Among anchors with target IoU at least 0.3, each score is the
        largest objectness times that class probability. It is no smaller than the deployed
        same-class detection score at that target. No match returns an input-connected zero.
        """
        import torch

        if self._closed:
            raise RuntimeError("YOLOXDetector has been closed")
        if (not isinstance(images, torch.Tensor) or images.ndim != 4 or images.shape[1] != 3
                or 0 in images.shape or not images.is_floating_point()
                or not bool(torch.isfinite(images).all())
                or bool((images < 0).any()) or bool((images > 1).any())):
            raise ValueError("images must be finite float RGB (B, 3, H, W) values in [0, 1]")
        if not isinstance(targets, Sequence) or len(targets) != images.shape[0]:
            raise ValueError("targets length must equal the image batch size")
        images = images.to(device=self.device, dtype=torch.float32)
        target_boxes = []
        for target in targets:
            if not isinstance(target, TargetSpec) or target.label not in COCO_CLASSES:
                raise ValueError("target.label must be a COCO class name")
            box = torch.as_tensor(target.box, dtype=images.dtype, device=images.device)
            if (box.shape != (4,) or not bool(torch.isfinite(box).all())
                    or bool((box[2:] <= box[:2]).any())):
                raise ValueError("target.box must be finite xyxy coordinates with positive width and height")
            target_boxes.append(box)
        predictions = self._prediction(images)
        result = []
        for index, (prediction, target, box) in enumerate(zip(predictions, targets, target_boxes)):
            boxes = prediction[:, :4]
            intersection_size = (torch.minimum(boxes[:, 2:], box[2:])
                                 - torch.maximum(boxes[:, :2], box[:2])).clamp(min=0)
            intersection = intersection_size.prod(dim=1)
            areas = (boxes[:, 2:] - boxes[:, :2]).clamp(min=0).prod(dim=1)
            union = areas + (box[2:] - box[:2]).prod() - intersection
            overlaps = intersection / union.clamp(min=torch.finfo(images.dtype).eps)
            matching = overlaps >= .3
            if not bool(matching.any()):
                result.append(images[index].sum() * 0)
            else:
                scores = prediction[:, 4] * prediction[:, 5 + COCO_CLASSES.index(target.label)]
                result.append(scores[matching].max())
        return torch.stack(result)

    def close(self) -> None:
        """Release the model. A closed detector cannot detect or score again."""
        self._closed = True
        self._loaded_model = None
        self.model = None


__all__ = [
    'YOLOXDetector',
]
