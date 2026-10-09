#!/usr/bin/env python
"""Cut real pedestrians out of nuScenes camera images -> RGBA assets for the person-poster attack.

Picks large, fully visible, unoccluded pedestrian boxes from an mmdet3d ``nuscenes_infos_*_mono3d.coco.json``
(2-D boxes per camera image), segments each with SAM prompted by its box, and keeps clean full-body
masks. Writes ``<out>/person_XXX.png`` (RGBA, tight crop, transparent background) and ``index.json``
(source image, box, 3-D height, SAM score).

nuScenes is CC BY-NC-SA 4.0: the cutouts are derived data — keep them out of the repo, next to your
datasets (e.g. ``/workspace/hdd/users/<you>/assets/pedestrians``), and credit nuScenes where shown.

    python -m scripts.preparation.extract_person_cutouts --nuscenes /workspace/hdd/datasets/nuscenes \
        --out /workspace/hdd/users/<you>/assets/pedestrians --n 12
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def iou(a, b) -> float:
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    ix = max(0.0, min(ax2, bx2) - max(a[0], b[0]))
    iy = max(0.0, min(ay2, by2) - max(a[1], b[1]))
    inter = ix * iy
    return inter / (a[2] * a[3] + b[2] * b[3] - inter + 1e-9)


def candidates(coco: dict, min_h: float, margin: float = 8.0) -> list[dict]:
    """Full-body pedestrians: tall, inside the frame, person-shaped, overlapping no other object."""
    by_img: dict = {}
    for a in coco["annotations"]:
        by_img.setdefault(a["image_id"], []).append(a)
    images = {im["id"]: im for im in coco["images"]}
    out = []
    for img_id, anns in by_img.items():
        im = images[img_id]
        for a in anns:
            x, y, w, h = a["bbox"]
            if a["category_name"] != "pedestrian" or h < min_h:
                continue
            if a.get("attribute_name") not in ("pedestrian.standing", "pedestrian.moving"):
                continue
            if x < margin or y < margin or x + w > im["width"] - margin or y + h > im["height"] - margin:
                continue
            if not 2.0 <= h / w <= 4.5:
                continue
            if any(o is not a and iou(a["bbox"], o["bbox"]) > 0.02 for o in anns):
                continue
            out.append({"file": im["file_name"], "bbox": [x, y, w, h], "height_m": a["bbox_cam3d"][4]})
    return sorted(out, key=lambda c: -c["bbox"][3])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nuscenes", required=True, help="nuScenes root (with samples/ and the mono3d json)")
    ap.add_argument("--ann", default="nuscenes_infos_val_mono3d.coco.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=12, help="cutouts to keep")
    ap.add_argument("--pool", type=int, default=60, help="candidates to segment before ranking")
    ap.add_argument("--min-h", type=float, default=350.0, help="min box height (px)")
    ap.add_argument("--sam", default="facebook/sam-vit-huge")
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()

    import cv2
    import torch
    from transformers import SamModel, SamProcessor

    root, out = Path(args.nuscenes), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cands = candidates(json.loads((root / args.ann).read_text()), args.min_h)
    # at most one person per source image, for variety
    seen, pool = set(), []
    for c in cands:
        if c["file"] not in seen:
            seen.add(c["file"])
            pool.append(c)
        if len(pool) >= args.pool:
            break
    print(f"[cutouts] {len(cands)} candidates, segmenting {len(pool)}")

    dev = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    proc, sam = SamProcessor.from_pretrained(args.sam), SamModel.from_pretrained(args.sam).to(dev).eval()
    kept = []
    for c in pool:
        rgb = cv2.cvtColor(cv2.imread(str(root / c["file"])), cv2.COLOR_BGR2RGB)
        x, y, w, h = c["bbox"]
        inputs = proc(rgb, input_boxes=[[[x, y, x + w, y + h]]], return_tensors="pt").to(dev)
        with torch.no_grad():
            res = sam(**inputs, multimask_output=True)
        masks = proc.image_processor.post_process_masks(
            res.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu())[0][0]
        scores = res.iou_scores.cpu()[0, 0]
        k = int(scores.argmax())
        m = masks[k].numpy().astype(np.uint8)
        # keep the largest connected component (drop stray blobs)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
        if n < 2:
            continue
        m = (lab == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
        x0, y0, x1, y1 = int(x), int(y), int(x + w), int(y + h)
        box_m = m[y0:y1, x0:x1]
        fill = float(box_m.mean())
        rows = np.where(box_m.any(axis=1))[0]
        spans = rows.size and (rows[0] < 0.06 * h) and (rows[-1] > 0.94 * h)  # head to feet
        if not (0.30 <= fill <= 0.75 and spans and m[:y0].sum() + m[y1:].sum() < 0.02 * m.sum()):
            continue
        ys, xs = np.where(m)
        crop = np.dstack([rgb, m * 255])[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        kept.append((float(scores[k]), c, crop))
    kept.sort(key=lambda t: -t[0])
    index = []
    for i, (score, c, crop) in enumerate(kept[:args.n]):
        name = f"person_{i:03d}.png"
        cv2.imwrite(str(out / name), cv2.cvtColor(crop, cv2.COLOR_RGBA2BGRA))
        index.append({"file": name, "source": c["file"], "bbox": c["bbox"], "height_m": c["height_m"],
                      "sam_score": round(score, 3)})
    (out / "index.json").write_text(json.dumps({"source": "nuScenes (CC BY-NC-SA 4.0)", "cutouts": index},
                                               indent=1))
    print(f"[cutouts] kept {len(index)} -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
