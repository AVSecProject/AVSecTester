#!/usr/bin/env python
"""End-to-end: the real Alpamayo policy driving on real NuRec imagery, via the interface's run().

    NuRecBackend (in-process loop; TrajectoryFollower dynamics; NuRecRenderer -> nre-ga render_rgb)
        --Observation(rendered camera, ego)-->  AlpamayoAVStack (real Alpamayo-1.5-10B)
        <--Control(trajectory)---------------  (rig-frame waypoints) --> follower advances the ego

Prerequisites (see docs / prior setup):
  1. nre-ga renderer serving a NuRec scene on :50051 --
       docker run -d --name nre --net=host --gpus '"device=2"' -e HOME=/tmp \
         -v <scenes-dir>:/mnt/nre-data \
         --entrypoint /app/internal/scripts/pycena/runtime/pycena_nrm_full \
         nvcr.io/nvidia/nre/nre-ga:26.04 serve-grpc --host=0.0.0.0 --port=50051 \
         '--artifact-glob=/mnt/nre-data/**/*.usdz' --cache-size=2 --enable-editing-actors
  2. Run in the AlpaSim driver env (Python 3.12 + alpasim_driver + the local Alpamayo checkpoint):
       cd /workspace/nvme/qzzhang/alpasim
       HF_HOME=/workspace/hdd/models/huggingface PYTHONPATH=<AVSecTester> \
         uv run python scripts/alpamayo_nurec_demo.py            # or --stub for black frames

Pass --stub to swap NuRecRenderer for StubRenderer (black frames; no renderer needed).
"""

import argparse
import sys
import time

from avsectester.alpamayo import AlpamayoAVStack
from avsectester.backend import run
from avsectester.nurec import NuRecBackend, NuRecRenderer, StubRenderer, TrajectoryFollower

CAM = "camera_front_wide_120fov"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("frames", nargs="?", type=int, default=8)
    ap.add_argument("--stub", action="store_true", help="use black stub frames (no renderer)")
    ap.add_argument("--scene", default="01d503d4", help="NuRec scene id substring")
    ap.add_argument("--endpoint", default="127.0.0.1:50051", help="nre-ga renderer endpoint")
    ap.add_argument("--gpu", type=int, default=1, help="CUDA device for Alpamayo")
    args = ap.parse_args()

    renderer = (
        StubRenderer(cameras=[CAM])
        if args.stub
        else NuRecRenderer(endpoint=args.endpoint, scene_id=args.scene, cameras=[CAM])
    )
    backend = NuRecBackend(
        {"dt": 0.1, "ego0": {"speed": 5.0}}, renderer=renderer, dynamics=TrajectoryFollower()
    )
    stack = AlpamayoAVStack(device=f"cuda:{args.gpu}", camera_ids=[CAM])

    kind = "stub (black)" if args.stub else "NuRec"
    print(f"[demo] Alpamayo + {kind} renderer, {args.frames} frames ...")
    t0 = time.time()
    trace = run(backend, stack, args.frames)
    for r in trace.records:
        print(f"  f{r.frame:02d} t={r.t:.1f}s  speed={r.speed:5.2f} m/s")
    print(
        f"END-TO-END OK: Alpamayo drove on {kind} imagery {len(trace.records)} frames in "
        f"{time.time() - t0:.0f}s; peak_speed={trace.peak_speed:.2f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
