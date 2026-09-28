#!/usr/bin/env python
"""Closed loop: does a fake object in the camera change how Alpamayo drives?

The real Alpamayo-1.5-10B policy drives the NuRec scene twice from the same start — clean, then with
an object inserted into its camera stream by ``perturb(Observation)`` (the same insertion as
``nurec_object_demo.py``: a roadside STOP sign by default). Its planned trajectories move the ego
(``TrajectoryFollower``), so the attacked run sees the sign from wherever the policy actually drives.
Per frame we record the ego speed and Alpamayo's reasoning text; the driving-impact verdict is
:func:`avsectester.metric.impact` on the two traces.

Run in an AlpaSim driver env (Python 3.12 + alpasim_driver) with an nre-ga server (docs/SETUP.md §4):

    python scripts/alpamayo_attack_demo.py --endpoint 127.0.0.1:50051 --object stop --frames 60 \
        --gpu 1 --harm-gpu 0

Outputs under ``tmp/alpamayo_<object>/``: ``speed.png`` (clean vs attacked), ``side_by_side.gif``,
``trace.json`` (speed + reasoning per frame, impact verdict).
"""

import argparse
import json
import sys
import time
from pathlib import Path

from avsectester.attacks.sign_spoof import roadside_sign_insert
from avsectester.metric import impact
from avsectester.simulators.nurec import NuRecBackend, NuRecRenderer, TrajectoryFollower
from avsectester.simulators.patch_insertion import PatchCompositor, frame_perturbation
from avsectester.simulators.viz import record_run, save_gif
from avsectester.stacks.alpamayo import AlpamayoAVStack
from nurec_object_demo import OBJECTS, build_object, make_harmonizer, side_by_side

REPO = Path(__file__).resolve().parents[1]
CAM = "camera_front_wide_120fov"


class Recording(AlpamayoAVStack):
    """AlpamayoAVStack that also logs each step's speed input and reasoning text."""

    def reset(self, observation):
        super().reset(observation)
        self.log = []

    def __call__(self, observation):
        control = super().__call__(observation)
        self.log.append({"t": round(observation.t, 2), "speed": round(float(observation.ego_speed), 3),
                         "reasoning": self.last_reasoning})
        return control


def drive(args, stack, perturb_of=None):
    renderer = NuRecRenderer(endpoint=args.endpoint, scene_id=args.scene, cameras=[CAM])
    backend = NuRecBackend({"dt": 0.1, "ego0": {"speed": args.speed}}, renderer=renderer,
                           dynamics=TrajectoryFollower())
    try:
        renderer.load_scene(args.scene)
        perturb = perturb_of(renderer) if perturb_of else None
        trace = record_run(backend, stack, args.frames, out_dir=args.out / ("attacked" if perturb else "clean"),
                           perturb=perturb, collect=True)
        return trace, list(stack.log)
    finally:
        backend.close()


def plot_speed(clean, attacked, path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 3.2))
    for trace, label, col in ((clean, "clean", "tab:green"), (attacked, "attacked", "tab:red")):
        ax.plot([r.t for r in trace.records], [r.speed for r in trace.records], label=label, color=col)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("ego speed (m/s)")
    ax.set_title(title, fontsize=10)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--speed", type=float, default=5.0, help="initial ego speed (m/s)")
    ap.add_argument("--scene", default="01d503d4")
    ap.add_argument("--endpoint", default="127.0.0.1:50051")
    ap.add_argument("--object", choices=sorted(OBJECTS), default="stop")
    ap.add_argument("--asset", default=None, help="person cut-out PNG (standee / billboard)")
    ap.add_argument("--harmonizer", choices=["none", "classic", "chroma", "libcom"], default="libcom")
    ap.add_argument("--x", type=float, default=None)
    ap.add_argument("--y", type=float, default=None)
    ap.add_argument("--yaw", type=float, default=0.15)
    ap.add_argument("--size", type=float, default=None)
    ap.add_argument("--mount", type=float, default=None)
    ap.add_argument("--ground-z", type=float, default=0.0)
    ap.add_argument("--soften", type=float, default=0.6)
    ap.add_argument("--gpu", type=int, default=1, help="CUDA device for Alpamayo")
    ap.add_argument("--harm-gpu", type=int, default=0, help="CUDA device for the PCTNet harmonizer")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    args.out = Path(args.out or REPO / "tmp" / f"alpamayo_{args.object}")
    args.out.mkdir(parents=True, exist_ok=True)

    sign, _face, _aspect = build_object(args)
    compositor = PatchCompositor(make_harmonizer(args.harmonizer, args.harm_gpu))

    def perturb_of(r):
        insert = roadside_sign_insert(sign, compositor, r.camera_model(),
                                      lambda o: r.cam_from_world(o.vehicle_state), soften=args.soften)
        return frame_perturbation(insert, camera=CAM)

    stack = Recording(device=f"cuda:{args.gpu}", camera_ids=[CAM])
    t0 = time.time()
    print(f"[alpamayo] clean drive, {args.frames} frames")
    clean, clean_log = drive(args, stack)
    print(f"[alpamayo] attacked drive ({args.object} at x={sign.x}, y={sign.y}; {args.harmonizer})")
    attacked, att_log = drive(args, stack, perturb_of)
    verdict = impact(clean, attacked)
    print(f"[alpamayo] {time.time() - t0:.0f}s; clean final {clean.records[-1].speed:.2f} m/s, "
          f"attacked final {attacked.records[-1].speed:.2f} m/s; verdict: {verdict}")

    plot_speed(clean, attacked, args.out / "speed.png",
               f"Alpamayo-1.5 on NuRec: clean vs {args.object} ({args.harmonizer})")
    save_gif([side_by_side(c, a) for c, a in zip(clean.frames, attacked.frames)],
             args.out / "side_by_side.gif", fps=10)
    (args.out / "trace.json").write_text(json.dumps({
        "object": args.object, "at": [sign.x, sign.y], "harmonizer": args.harmonizer,
        "verdict": str(verdict),
        "clean": clean_log, "attacked": att_log,
    }, indent=1))
    print(f"[output] {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
