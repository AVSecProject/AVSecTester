#!/usr/bin/env python
"""End-to-end: the real Alpamayo policy driving on real NuRec imagery, via the interface's run().

    NuRecSimulator (in-process loop; TrajectoryFollower dynamics; NuRecRenderer -> nre-ga render_rgb)
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
       cd /path/to/AVSecTester
       /path/to/alpasim/.venv/bin/python -m scripts.demos.nurec.alpamayo_nurec_demo
       # Add --stub for black frames instead of NuRec rendering.

Pass --stub to swap NuRecRenderer for StubRenderer (black frames; no renderer needed).
"""

import argparse
import sys
import time

from avsectester.backend import run
from avsectester.runner import run_prepared
from avsectester.scenarios.setup import scenario_from_config
from avsectester.closed_loop.simulators.nurec import (
    NuRecSimulator,
    NuRecRenderer,
    StubRenderer,
    TrajectoryFollower,
)
from avsectester.closed_loop.simulators.viz import REPO_TMP, camera_view, record_run
from avsectester.closed_loop.avstacks.alpamayo import AlpamayoAVStack

CAM = "camera_front_wide_120fov"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("frames", nargs="?", type=int, default=8)
    ap.add_argument("--stub", action="store_true", help="use black stub frames (no renderer)")
    ap.add_argument("--scene", default="01d503d4", help="NuRec scene id substring")
    ap.add_argument("--endpoint", default="127.0.0.1:50051", help="nre-ga renderer endpoint")
    ap.add_argument("--gpu", type=int, default=1, help="CUDA device for Alpamayo")
    ap.add_argument("--checkpoint", help="Alpamayo weights directory (or set AVSECTESTER_ALPAMAYO_CHECKPOINT)")
    ap.add_argument("--save-frames", nargs="?", const=str(REPO_TMP / "alpamayo_nurec"),
                    default=None, help="save each rendered frame (default <repo>/tmp/alpamayo_nurec)")
    args = ap.parse_args()

    renderer = (
        StubRenderer(cameras=[CAM])
        if args.stub
        else NuRecRenderer(endpoint=args.endpoint, scene_id=args.scene, cameras=[CAM])
    )
    scenario = scenario_from_config(
        {"dt": 0.1, "ego0": {"speed": 5.0}},
        backend_factory=lambda config: NuRecSimulator(config, renderer=renderer, dynamics=TrajectoryFollower()),
        provenance={"scene_id": args.scene, "renderer": "stub" if args.stub else "nurec"},
    )

    kind = "stub (black)" if args.stub else "NuRec"
    print(f"[demo] Alpamayo + {kind} renderer, {args.frames} frames ...")
    t0 = time.time()
    def record(backend, stack, frames, *, runtime):
        if args.save_frames:
            return record_run(backend, stack, frames, hooks=runtime,
                              out_dir=args.save_frames, visualize=camera_view)
        return run(backend, stack, frames, runtime=runtime)

    trace = run_prepared(
        scenario,
        lambda: AlpamayoAVStack(checkpoint_path=args.checkpoint, device=f"cuda:{args.gpu}", camera_ids=[CAM]),
        args.frames, run_fn=record,
    )["clean"]
    for r in trace.records:
        print(f"  f{r.frame:02d} t={r.t:.1f}s  speed={r.speed:5.2f} m/s")
    print(
        f"END-TO-END OK: Alpamayo drove on {kind} imagery {len(trace.records)} frames in "
        f"{time.time() - t0:.0f}s; peak_speed={trace.peak_speed:.2f}"
    )
    if args.save_frames:
        print(f"[output]   saved {len(trace.records)} frames to {args.save_frames}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
