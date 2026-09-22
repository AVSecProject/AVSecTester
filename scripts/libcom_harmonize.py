#!/usr/bin/env python
"""libcom image harmonization, out-of-process (isolated `libcom` conda env).

libcom's deps conflict with the avstack stack, so ``LibcomHarmonizer`` runs harmonization here in a
dedicated env. Two modes:

* one-shot:  ``... --comp comp.png --mask mask.png --out out.png --model PCTNet``
* server:    ``... --serve --model PCTNet`` — load the model ONCE, then read one request per line on
  stdin (``<comp>\\t<mask>\\t<out>``) and reply ``OK`` / ``ERR <msg>`` per line on stdout. This keeps
  the model resident so a whole sequence harmonizes at ~1 img/s instead of reloading weights each call.
"""

import argparse
import sys

import numpy as np
from PIL import Image


def _harmonize(net, comp_path: str, mask_path: str, out_path: str) -> None:
    result = net(comp_path, mask_path)  # libcom returns a BGR ndarray (or a list); normalize
    if isinstance(result, (list, tuple)):
        result = result[0]
    arr = np.asarray(result)
    if arr.ndim == 3 and arr.shape[2] == 3:  # libcom is BGR -> save as RGB
        arr = arr[:, :, ::-1]
    Image.fromarray(arr.astype(np.uint8)).save(out_path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--comp")
    ap.add_argument("--mask")
    ap.add_argument("--out")
    ap.add_argument("--serve", action="store_true", help="persistent server: one request per stdin line")
    ap.add_argument("--model", default="PCTNet", help="libcom harmonization model (PCTNet / CDTNet)")
    ap.add_argument("--device", type=int, default=0)
    args = ap.parse_args()

    from libcom import ImageHarmonizationModel

    net = ImageHarmonizationModel(device=args.device, model_type=args.model)

    if not args.serve:
        _harmonize(net, args.comp, args.mask, args.out)
        return 0

    print("READY", flush=True)  # signal the parent the model is loaded and we can stream
    for line in sys.stdin:
        line = line.rstrip("\n")
        if not line:
            continue
        if line == "QUIT":
            break
        try:
            comp, mask, out = line.split("\t")
            _harmonize(net, comp, mask, out)
            print("OK", flush=True)
        except Exception as exc:  # noqa: BLE001 - report per-request; keep the server alive
            print("ERR " + str(exc).replace("\n", " ")[:400], flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
