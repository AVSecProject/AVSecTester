"""White-box ``Scorer`` over an mmdet two-stage detector: RPN objectness in the target region.

The deployed AV perception is an avstack ``MMDetObjectDetector2D`` whose underlying ``self.model`` is
a raw mmdet ``nn.Module``. ``target_score`` runs a differentiable forward
(``data_preprocessor -> extract_feat -> rpn_head``) and returns the mean RPN objectness over anchor
cells inside the target box — a differentiable proxy for "an object is detected here". Minimizing it
suppresses proposals in the region, i.e. object removal.
"""

from __future__ import annotations

import torch

from .interface import Scorer, TargetSpec

# ResNet-FPN P2..P6 strides for the RPN cls_score levels (matches faster_rcnn_r50_fpn).
FPN_STRIDES = (4, 8, 16, 32, 64)


class MMDetRPNObjectnessScorer(Scorer):
    """RPN-objectness confidence in the target box for an mmdet two-stage detector (white-box)."""

    differentiable = True

    def __init__(self, model, strides=FPN_STRIDES):
        self.model = model
        self.strides = strides
        self.device = next(model.parameters()).device

    def target_score(self, image: torch.Tensor, target: TargetSpec) -> torch.Tensor:
        from mmdet.structures import DetDataSample

        x = image.to(self.device) * 255.0  # the preprocessor expects a 0..255 image
        h, w = int(x.shape[1]), int(x.shape[2])
        ds = DetDataSample()
        ds.set_metainfo({"img_shape": (h, w), "ori_shape": (h, w), "scale_factor": (1.0, 1.0),
                         "pad_shape": (h, w), "batch_input_shape": (h, w)})
        data = self.model.data_preprocessor({"inputs": [x], "data_samples": [ds]}, False)
        feats = self.model.extract_feat(data["inputs"])
        cls_scores = self.model.rpn_head(feats)[0]  # list of (1, A, Hl, Wl) objectness logits

        x0, y0, x1, y1 = target.box
        vals = []
        for lvl, cs in enumerate(cls_scores):
            s = self.strides[lvl] if lvl < len(self.strides) else 2 ** (lvl + 2)
            _, hl, wl = cs.shape[1:]
            jc = (torch.arange(wl, device=cs.device) + 0.5) * s  # cell centers in image px
            ic = (torch.arange(hl, device=cs.device) + 0.5) * s
            mx = (jc >= x0) & (jc <= x1)
            my = (ic >= y0) & (ic <= y1)
            if mx.any() and my.any():
                sub = cs[0][:, my][:, :, mx]  # (A, ny, nx) logits inside the box
                vals.append(sub.sigmoid().reshape(-1))
        if not vals:
            return torch.zeros((), device=self.device)
        return torch.cat(vals).mean()


class MMDetInferenceScorer(Scorer):
    """Black-box (non-differentiable) scorer: the detector's actual max confidence in the target box.

    Runs the full ``inference_detector`` on a rendered ``(H,W,3)`` uint8 RGB frame — the true metric an
    object-removal attack must drive below the detection threshold. For gradient-free attacks (NES).
    """

    differentiable = False

    def __init__(self, model):
        self.model = model

    def target_score(self, image, target: TargetSpec) -> float:
        import numpy as np
        from mmdet.apis import inference_detector

        img = np.asarray(image)
        inst = inference_detector(self.model, img[:, :, ::-1]).pred_instances
        scores = inst.scores.detach().cpu().numpy()
        boxes = inst.bboxes.detach().cpu().numpy()
        x0, y0, x1, y1 = target.box
        best = 0.0
        for b, sc in zip(boxes, scores):
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            if x0 <= cx <= x1 and y0 <= cy <= y1:
                best = max(best, float(sc))
        return best
