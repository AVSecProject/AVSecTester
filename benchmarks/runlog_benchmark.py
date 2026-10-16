"""Measure synthetic driving-log recording without running a simulator.

Run from the repository root:
    python -m benchmarks.runlog_benchmark --steps 128 --image-every 5

The fixtures use 800x600 RGB input, 20 boxes and 12 trajectory waypoints per
step. They measure this storage implementation, not simulator throughput or
compressed image size. Each case runs in a fresh process so RSS peaks do not
carry over. Temporary output is removed after measurement.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import resource
import subprocess
import sys
import tempfile
import time
import tracemalloc

import numpy as np

from avsectester.plane import Control, Observation, WorldSnapshot
from avsectester.runlog import Recorder, StepRecord, read_table


def _rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def measure(steps: int, image_every: int) -> dict:
    with tempfile.TemporaryDirectory(prefix="avsectester-runlog-benchmark-") as directory:
        root = Path(directory)
        recorder = Recorder(root, retain_steps=False, chunk_steps=32, payload_shard_bytes=8 * 1024 * 1024)
        image = np.arange(800 * 600 * 3, dtype=np.uint8).reshape(600, 800, 3)
        baseline_rss = _rss_bytes()
        tracemalloc.start()
        start = time.perf_counter()
        images = 0
        for i in range(steps):
            sensors = {}
            if image_every and i % image_every == 0:
                sensors["camera_front"] = image
                images += 1
            recorder.record_step(StepRecord(
                step=i,
                raw_observation=Observation(i * 0.05, i),
                model_observation=Observation(i * 0.05, i, sensor_data=sensors),
                physical_pre=WorldSnapshot(i * 0.05, i, {"position": [i * .2, 0., 0.]}, 4.),
                physical_post=WorldSnapshot((i + 1) * 0.05, i + 1, {"position": [(i + 1) * .2, 0., 0.]}, 4.),
                produced_command=Control(trajectory=[((j * .2, 0., 0.), (1., 0., 0., 0.), i * 50000 + (j + 1) * 50000) for j in range(12)]),
                executed_command=Control(throttle=0.2),
                components={"perception": [{"label": "car", "box": [j * 2., 3., 15., 8.], "score": 0.9} for j in range(20)]},
                actions=[{"stage": "perception.post", "name": "example", "applied_t": i * 0.05}],
                telemetry={"n_detections": 20},
            ))
        recorder.finish()
        write_seconds = time.perf_counter() - start
        _, peak_traced = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_rss = _rss_bytes()
        read_start = time.perf_counter()
        table = read_table(root, columns=["step", "physical_post", "executed_command"])
        selected_seconds = time.perf_counter() - read_start
        assert table.num_rows == steps
        files = [path for path in root.rglob("*") if path.is_file()]
        return {
            "fixture": "scalars_nested" if not image_every else "scalars_nested_sampled_rgb",
            "steps": steps, "image_every": image_every, "images": images,
            "recording_seconds": write_seconds,
            "stored_bytes": sum(path.stat().st_size for path in files),
            "file_count": len(files),
            "payload_file_count": len(list((root / "payloads").glob("*.bin"))),
            "peak_traced_allocations_bytes": peak_traced,
            "peak_process_rss_bytes": peak_rss,
            "rss_peak_increase_bytes": max(0, peak_rss - baseline_rss),
            "selected_column_read_seconds": selected_seconds,
            "selected_columns": table.column_names,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--image-every", type=int, default=5)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.steps <= 0 or args.image_every < 0:
        parser.error("steps must be positive; image-every must be nonnegative")
    if args.child:
        print(json.dumps(measure(args.steps, args.image_every)))
        return
    cases = []
    for every in (0, args.image_every):
        result = subprocess.run([
            sys.executable, "-m", "benchmarks.runlog_benchmark", "--child",
            "--steps", str(args.steps), "--image-every", str(every),
        ], check=True, capture_output=True, text=True)
        cases.append(json.loads(result.stdout))
    print(json.dumps({"note": "Synthetic shapes; raw RGB shards, not simulator performance.", "cases": cases}, indent=2))


if __name__ == "__main__":
    main()
