"""Record a full security test (clean vs attacked) with images + data, and save analysis.

Runs the neural CarlaBackend twice (clean, then phantom-attacked), recording per-frame BEV
+ RGB images and per-frame data, then writes the escalation analysis (comparison plot,
metrics, report). Manual/hardware: needs a live CARLA 0.9.15 server + carla-vehicle weights.

    python scripts/record_run.py [frames] [--no-camera]

Output: results/carla_neural_<stamp>/{clean,attacked}/ + timeline_compare.png + summary.md
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def _run(outdir: Path, attack, frames: int, camera: bool):
    from avsectester.backends.carla_backend import CarlaBackend
    from avsectester.viz import RunRecorder

    be = CarlaBackend(
        perception="neural", nn_dataset="carla-vehicle", nn_threshold=0.3,
        n_npcs=8, frames=frames, target_speed=6.0, brake_distance=8.0, record_camera=camera,
    )
    be.build(None)
    rec = RunRecorder(outdir, brake_distance=be.brake_distance, brake_corridor=be.brake_corridor)
    be.set_recorder(rec)
    if attack is not None:
        attack.reset()
        be.attach(attack, attack.resolve_binding(be.profile()).seam)
    try:
        list(be.run())
    finally:
        be.close()
        rec.finalize(title=outdir.name)
    return rec.records


def main() -> int:
    from avsectester.attacks.detection_manipulation import PhantomDetectionAttack
    from avsectester.core.engine import ExperimentResult
    from avsectester.metrics.escalation import EscalationMetric
    from avsectester.monitors.trace import build_trace
    from avsectester.reports import render_report
    from avsectester.viz import compare_runs

    frames = 40
    camera = "--no-camera" not in sys.argv
    for a in sys.argv[1:]:
        if a.isdigit():
            frames = int(a)

    stamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    root = Path("results") / f"carla_neural_{stamp}"
    root.mkdir(parents=True, exist_ok=True)
    print(f"recording to {root}  (frames={frames}, camera={camera})")

    print("== CLEAN ==")
    clean = _run(root / "clean", None, frames, camera)
    print("== ATTACKED ==")
    attacked = _run(root / "attacked",
                    PhantomDetectionAttack(target_xyz=[6.0, 0.0, -1.5], score=0.9), frames, camera)

    # analysis: comparison plot + escalation metric + report
    compare_runs(clean, attacked, root / "timeline_compare.png")
    out = EscalationMetric().compute(build_trace(clean, "clean"), build_trace(attacked, "attacked"))
    result = ExperimentResult(
        name=root.name, metrics=out["metrics"], dag=out["dag"],
        clean_trace=build_trace(clean, "clean"), attacked_trace=build_trace(attacked, "attacked"),
    )
    (root / "metrics.json").write_text(json.dumps(out["metrics"], indent=2, default=str))
    (root / "summary.md").write_text(render_report(result))

    print(f"\nDONE. Results in {root}")
    print(f"  escalated={out['metrics'].get('escalated')}  "
          f"clean_final={clean[-1]['ego_speed']:.1f}  attacked_final={attacked[-1]['ego_speed']:.1f}")
    print(f"  images: {root}/clean/frames, {root}/attacked/frames")
    print("  data:   records.jsonl per run; metrics.json; summary.md; timeline_compare.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
