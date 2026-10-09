#!/usr/bin/env python
"""Run the NuRec insertion demo with three independently oriented host-bound patches.

Use ``--usdz`` for the scene archive and ``--host`` for a stable recorded vehicle track ID.
The saved RGB frames are the perturbed observation supplied at the driving-model input seam.
"""

from scripts.demos.nurec.nurec_insertion_demo import main


if __name__ == "__main__":
    raise SystemExit(main(patch_only=True))
