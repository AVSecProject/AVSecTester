#!/usr/bin/env python
"""End-to-end: the real Alpamayo policy driving the in-process NuRec world backend via run().

    NuRecBackend (owns the loop; TrajectoryFollower dynamics; renderer -> camera Observation)
        --Observation(camera, ego)-->  AlpamayoAVStack (real Alpamayo-1.5-10B)
        <--Control(trajectory)--------  (64 rig-frame waypoints)  --> follower advances the ego

Runs in the AlpaSim driver env (Python 3.12 + alpasim_driver + the local Alpamayo checkpoint). The
renderer is StubRenderer (black frames) by default so the whole framework loop runs with the real
model here; swap in NuRecRenderer (nre-ga + a NuRec scene) for photorealistic input.

    cd /workspace/nvme/qzzhang/alpasim
    HF_HOME=/workspace/hdd/models/huggingface \
    PYTHONPATH=/workspace/nvme/qzzhang/AVSecProject/AVSecTester \
    uv run python scripts/alpamayo_nurec_demo.py
"""

import sys

from avsectester.alpamayo import AlpamayoAVStack
from avsectester.backend import run
from avsectester.nurec import NuRecBackend, StubRenderer, TrajectoryFollower

CAM = "camera_front_wide_120fov"


def main() -> int:
    backend = NuRecBackend(
        {"dt": 0.1, "ego0": {"speed": 5.0}},
        renderer=StubRenderer(cameras=[CAM], height=480, width=640),
        dynamics=TrajectoryFollower(),
    )
    stack = AlpamayoAVStack(device="cuda:1", camera_ids=[CAM], context_length=4)
    trace = run(backend, stack, frames=10)
    for r in trace.records:
        print(f"  f{r.frame:02d} t={r.t:.1f}s  speed={r.speed:5.2f} m/s")
    print(
        f"END-TO-END OK: Alpamayo drove NuRecBackend {len(trace.records)} frames; "
        f"peak_speed={trace.peak_speed:.2f} final_speed={trace.final_speed:.2f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
