"""Differentiable RPN objectness and inference-confidence scorers for mmdet detectors.

Each input has its own target box and score. RPN scoring runs the batch through preprocessing,
feature extraction and the RPN once. RPN objectness is the mean objectness of candidate cells
inside the target box: a differentiable proxy for an object being detected there. Minimizing it
suppresses proposals in that region. Inference scoring queries a list of RGB images together.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

import avsectester.data_types as data_types


# ResNet-FPN P2..P6 strides for the RPN cls_score levels (matches faster_rcnn_r50_fpn).
FPN_STRIDES = (4, 8, 16, 32, 64)


class MMDetRPNObjectnessScorer:
    """RPN-objectness confidence in the target box for an mmdet two-stage detector (white-box)."""

    differentiable = True
    'Whether input-image gradients are supported by this component.'

    def __init__(self, model, strides=FPN_STRIDES):
        self.model = model
        self.strides = strides
        self.device = next(model.parameters()).device

    def target_scores(
        self, images: torch.Tensor, targets: Sequence[data_types.TargetSpec],
    ) -> torch.Tensor:
        """Return one differentiable mean RPN objectness per BCHW image and target box."""
        from mmdet.structures import DetDataSample

        if (not isinstance(images, torch.Tensor) or images.ndim != 4 or images.shape[1] != 3
                or 0 in images.shape or not images.is_floating_point()
                or not bool(torch.isfinite(images).all())
                or bool((images < 0).any()) or bool((images > 1).any())):
            raise ValueError("images must be finite float RGB (B, 3, H, W) values in [0, 1]")
        if len(targets) != images.shape[0]:
            raise ValueError("targets length must equal the image batch size")
        x = images.to(self.device) * 255.0
        h, w = int(x.shape[2]), int(x.shape[3])
        records = []
        for _ in targets:
            ds = DetDataSample()
            ds.set_metainfo({"img_shape": (h, w), "ori_shape": (h, w), "scale_factor": (1.0, 1.0),
                             "pad_shape": (h, w), "batch_input_shape": (h, w)})
            records.append(ds)
        data = self.model.data_preprocessor({"inputs": list(x), "data_samples": records}, False)
        feats = self.model.extract_feat(data["inputs"])
        cls_scores = self.model.rpn_head(feats)[0]
        result = []
        for index, target in enumerate(targets):
            x0, y0, x1, y1 = target.box
            vals = []
            for lvl, cs in enumerate(cls_scores):
                stride = self.strides[lvl] if lvl < len(self.strides) else 2 ** (lvl + 2)
                _, hl, wl = cs.shape[1:]
                jc = (torch.arange(wl, device=cs.device) + 0.5) * stride
                ic = (torch.arange(hl, device=cs.device) + 0.5) * stride
                mx = (jc >= x0) & (jc <= x1)
                my = (ic >= y0) & (ic <= y1)
                if mx.any() and my.any():
                    vals.append(cs[index][:, my][:, :, mx].sigmoid().reshape(-1))
            result.append(torch.cat(vals).mean() if vals else x[index].sum() * 0)
        return torch.stack(result)


class MMDetInferenceScorer:
    """Black-box (non-differentiable) scorer: the detector's actual max confidence in the target box.

    Runs ``inference_detector`` on a list of uint8 RGB frames and returns each target region's
    maximum detected confidence. Input gradients are unavailable.
    """

    differentiable = False
    'Whether input-image gradients are supported by this component.'

    def __init__(self, model):
        self.model = model

    def target_scores(self, images, targets: Sequence[data_types.TargetSpec]) -> list[float]:
        """Return one matched detection confidence per RGB image and target box."""
        import numpy as np
        from mmdet.apis import inference_detector

        if len(images) != len(targets):
            raise ValueError("targets length must equal the image batch size")
        if not images:
            return []
        converted = []
        for image in images:
            img = np.asarray(image)
            if img.ndim != 3 or img.shape[2] != 3 or img.dtype != np.uint8 or 0 in img.shape:
                raise ValueError("images must be a list of nonempty HWC RGB uint8 arrays")
            converted.append(img[:, :, ::-1])
        results = inference_detector(self.model, converted)
        values = []
        for result, target in zip(results, targets):
            inst = result.pred_instances
            scores = inst.scores.detach().cpu().numpy()
            boxes = inst.bboxes.detach().cpu().numpy()
            x0, y0, x1, y1 = target.box
            best = 0.0
            for box, score in zip(boxes, scores):
                cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                if x0 <= cx <= x1 and y0 <= cy <= y1:
                    best = max(best, float(score))
            values.append(best)
        return values


__all__ = [
    'MMDetRPNObjectnessScorer',
    'MMDetInferenceScorer',
]
