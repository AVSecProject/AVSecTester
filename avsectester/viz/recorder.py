"""Per-tick recorder for the evaluation side: CARLA screenshots + a bird's-eye detection view.

``Recorder`` is an ``observer(frame, outcome)`` for :func:`avsectester.core.run`: each tick it
saves the forward RGB camera image (a CARLA screenshot) and a bird's-eye-view (BEV) render of
the LiDAR cloud with the detector's boxes — real detections green, an injected phantom
(ID in ``PHANTOM_IDS``) red — plus the ego and the brake corridor. Injected phantoms show up as
a red box sitting in the corridor with no supporting points, which is exactly the attack.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

PHANTOM_IDS = (90001, 90002)


def _points_from_lidar(cloud: Any) -> np.ndarray | None:
    """[N,4] xyz+intensity from an avstack CarlaLidar payload, or None."""
    if cloud is None:
        return None
    try:
        return np.frombuffer(bytes(cloud.data.raw_data), dtype=np.float32).reshape(-1, 4)
    except Exception:  # noqa: BLE001 - best-effort; some payloads store points differently
        try:
            return np.asarray(cloud.data.x)[:, :4]
        except Exception:  # noqa: BLE001
            return None


def _rgb_from_camera(cam: Any) -> np.ndarray | None:
    if cam is None:
        return None
    arr = np.asarray(cam.data)
    return arr[:, :, ::-1] if arr.ndim == 3 and arr.shape[2] >= 3 else None  # BGR->RGB


class Recorder:
    def __init__(self, outdir: str | Path, brake_distance: float = 8.0,
                 brake_corridor: float = 2.5, bev_range: float = 40.0) -> None:
        self.outdir = Path(outdir)
        self.frames_dir = self.outdir / "frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.brake_distance = brake_distance
        self.brake_corridor = brake_corridor
        self.bev_range = bev_range
        self.records: list[dict] = []
        self._fh = (self.outdir / "records.jsonl").open("w", encoding="utf-8")

    def __call__(self, frame: Any, outcome: Any) -> None:
        i = frame.index
        detections = outcome.extras.get("detections", [])
        rec = {**outcome.record,
               "detections": [self._det_summary(d) for d in detections]}
        self.records.append(rec)
        self._fh.write(json.dumps(rec) + "\n")

        rgb = _rgb_from_camera(frame.sensors.get("camera"))
        if rgb is not None:
            from PIL import Image
            Image.fromarray(rgb[:, :, :3].astype("uint8")).save(self.frames_dir / f"rgb_{i:04d}.png")

        points = _points_from_lidar(frame.sensors.get("lidar"))
        self._render_bev(i, rec, points, detections)

    @staticmethod
    def _det_summary(d: Any) -> dict:
        p = d.position.x
        return {"type": getattr(d, "obj_type", "?"),
                "score": round(float(getattr(d, "score", 0.0) or 0.0), 3),
                "forward": round(float(p[0]), 2), "left": round(float(p[1]), 2),
                "phantom": getattr(d, "ID", None) in PHANTOM_IDS}

    def _render_bev(self, i: int, rec: dict, points: np.ndarray | None, detections: list) -> None:
        r = self.bev_range
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_facecolor("#0e1116")
        ax.add_patch(Rectangle((-self.brake_corridor, 0), 2 * self.brake_corridor,
                               self.brake_distance, color="#e2b33d", alpha=0.12))  # brake corridor
        if points is not None and len(points):
            p = points[:: max(1, len(points) // 9000)]
            ax.scatter(p[:, 1], p[:, 0], s=0.4, c="#6b7280", alpha=0.5, linewidths=0)
        for d in detections:
            pos = d.position.x
            w, ln = self._box_wl(d)
            phantom = getattr(d, "ID", None) in PHANTOM_IDS
            color = "#ef4444" if phantom else "#22c55e"
            ax.add_patch(Rectangle((pos[1] - w / 2, pos[0] - ln / 2), w, ln,
                                   fill=False, edgecolor=color, linewidth=1.8))
            ax.text(pos[1], pos[0] + ln / 2 + 0.6,
                    f"{getattr(d, 'obj_type', '?')} {float(getattr(d, 'score', 0) or 0):.2f}",
                    color=color, fontsize=7, ha="center")
        ax.plot(0, 0, marker="^", color="#38bdf8", markersize=12)  # ego
        ax.set_xlim(-r, r); ax.set_ylim(-5, 2 * r); ax.invert_xaxis(); ax.set_aspect("equal")
        ax.set_xlabel("lateral (m, left +)"); ax.set_ylabel("forward (m)")
        brake = "BRAKING" if rec.get("braking") else "drive"
        ax.set_title(f"frame {i:03d}  speed={rec.get('ego_speed', 0):.1f} m/s  [{brake}]",
                     color="#e5e7eb", fontsize=10)
        ax.tick_params(colors="#9ca3af")
        for s in ax.spines.values():
            s.set_color("#374151")
        fig.tight_layout()
        fig.savefig(self.frames_dir / f"bev_{i:04d}.png", dpi=90, facecolor="#0e1116")
        plt.close(fig)

    @staticmethod
    def _box_wl(d: Any) -> tuple[float, float]:
        b = getattr(d, "box", None)
        w, ln = getattr(b, "w", None), getattr(b, "l", None)
        return (float(w) if w else 1.8, float(ln) if ln else 4.0)

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()
