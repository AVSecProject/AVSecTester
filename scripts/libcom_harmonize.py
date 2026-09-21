#!/usr/bin/env python
"""Run libcom image harmonization on a composite+mask, out-of-process (isolated `libcom` conda env).

libcom's deps conflict with the avstack stack, so ``LibcomHarmonizer`` shells out to this script in a
dedicated env. Reads a composite PNG + a binary mask PNG, writes the harmonized PNG.

    conda run -n libcom python scripts/libcom_harmonize.py --comp comp.png --mask mask.png \
        --out out.png --model PCTNet
"""

import argparse
import sys

import numpy as np
from PIL import Image


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--comp", required=True)
    ap.add_argument("--mask", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="PCTNet", help="libcom harmonization model (PCTNet / CDTNet)")
    ap.add_argument("--device", type=int, default=0)
    args = ap.parse_args()

    from libcom import ImageHarmonizationModel

    net = ImageHarmonizationModel(device=args.device, model_type=args.model)
    result = net(args.comp, args.mask)  # libcom returns a BGR ndarray (or a list); normalize below
    if isinstance(result, (list, tuple)):
        result = result[0]
    arr = np.asarray(result)
    if arr.ndim == 3 and arr.shape[2] == 3:  # libcom is BGR -> save as RGB
        arr = arr[:, :, ::-1]
    Image.fromarray(arr.astype(np.uint8)).save(args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
