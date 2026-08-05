"""Record a security test run as per-frame images + data, and plot the analysis.

``RunRecorder`` is duck-typed onto a backend via ``backend.set_recorder(rec)``: the backend
calls ``rec.capture(record, points=..., detections=..., rgb=...)`` each tick. The recorder
writes, under ``outdir``:

- ``frames/bev_XXXX.png`` -- bird's-eye view of the LiDAR cloud with detection boxes
  (real detections green, injected phantoms red), the ego, and the brake corridor.
- ``frames/rgb_XXXX.png`` -- the RGB camera image, if provided.
- ``records.jsonl``       -- one JSON object per frame (the record dict + detections list).
- ``timeline.png``        -- ego speed / detection count / braking over the run.

``compare_runs`` overlays two runs (e.g. clean vs attacked) into one figure.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

_PHANTOM_IDS = (90001, 90002)


def _det_summary(det: Any) -> dict[str, Any]:
    """Extract a small, serializable summary of one avstack detection."""
    p = det.position.x
    return {
        "type": getattr(det, "obj_type", "?"),
        "score": round(float(getattr(det, "score", 0.0) or 0.0), 3),
        "forward": round(float(p[0]), 2),
        "left": round(float(p[1]), 2),
        "up": round(float(p[2]), 2),
        "phantom": getattr(det, "ID", None) in _PHANTOM_IDS,
    }


def _box_wl(det: Any) -> tuple[float, float]:
    b = getattr(det, "box", None)
    w, l = getattr(b, "w", None), getattr(b, "l", None)
    return (float(w) if w else 1.8, float(l) if l else 4.0)


class RunRecorder:
    def __init__(
        self,
        outdir: str | Path,
        bev_range: float = 40.0,
        brake_distance: float = 8.0,
        brake_corridor: float = 2.5,
        save_frames: bool = True,
    ) -> None:
        self.outdir = Path(outdir)
        self.frames_dir = self.outdir / "frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.bev_range = bev_range
        self.brake_distance = brake_distance
        self.brake_corridor = brake_corridor
        self.save_frames = save_frames
        self.records: list[dict[str, Any]] = []
        self._fh = open(self.outdir / "records.jsonl", "w", encoding="utf-8")  # noqa: SIM115

    # -- per-frame ------------------------------------------------------------
    def capture(
        self,
        record: dict[str, Any],
        *,
        points: Any = None,
        detections: Any = None,
        rgb: Any = None,
    ) -> None:
        dets = [_det_summary(d) for d in (detections or [])]
        row = {**record, "detections": dets}
        self.records.append(row)
        self._fh.write(json.dumps(row) + "\n")
        if not self.save_frames:
            return
        i = record["frame"]
        if points is not None or detections is not None:
            self._render_bev(i, record, points, detections or [])
        if rgb is not None:
            self._save_rgb(i, rgb)

    def _render_bev(self, i: int, record: dict[str, Any], points: Any, detections: list) -> None:
        r = self.bev_range
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_facecolor("#0e1116")
        # brake corridor (ego frame: forward = up, lateral = left)
        ax.add_patch(Rectangle((-self.brake_corridor, 0), 2 * self.brake_corridor,
                               self.brake_distance, color="#e2b33d", alpha=0.12))
        if points is not None and len(points):
            p = points[:: max(1, len(points) // 9000)]  # subsample for speed
            ax.scatter(p[:, 1], p[:, 0], s=0.4, c="#6b7280", alpha=0.5, linewidths=0)
        for d in detections:
            pos = d.position.x
            w, l = _box_wl(d)
            phantom = getattr(d, "ID", None) in _PHANTOM_IDS
            color = "#ef4444" if phantom else "#22c55e"
            ax.add_patch(Rectangle((pos[1] - w / 2, pos[0] - l / 2), w, l,
                                   fill=False, edgecolor=color, linewidth=1.8))
            label = f"{getattr(d, 'obj_type', '?')} {float(getattr(d, 'score', 0) or 0):.2f}"
            ax.text(pos[1], pos[0] + l / 2 + 0.6, label, color=color, fontsize=7, ha="center")
        ax.plot(0, 0, marker="^", color="#38bdf8", markersize=12)  # ego
        ax.set_xlim(-r, r)
        ax.set_ylim(-5, 2 * r)
        ax.invert_xaxis()  # left of the ego on the left of the image
        ax.set_aspect("equal")
        ax.set_xlabel("lateral (m, left +)")
        ax.set_ylabel("forward (m)")
        brake = "BRAKING" if record.get("braking") else "drive"
        ax.set_title(f"frame {i:03d}  speed={record.get('ego_speed', 0):.1f} m/s  [{brake}]",
                     color="#e5e7eb", fontsize=10)
        ax.tick_params(colors="#9ca3af")
        for s in ax.spines.values():
            s.set_color("#374151")
        fig.tight_layout()
        fig.savefig(self.frames_dir / f"bev_{i:04d}.png", dpi=90, facecolor="#0e1116")
        plt.close(fig)

    def _save_rgb(self, i: int, rgb: Any) -> None:
        from PIL import Image

        Image.fromarray(rgb[:, :, :3].astype("uint8")).save(self.frames_dir / f"rgb_{i:04d}.png")

    # -- finalize -------------------------------------------------------------
    def finalize(self, title: str = "run") -> None:
        self._fh.close()
        _save_timeline(self.records, self.outdir / "timeline.png", title,
                       self.brake_distance)

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()


def _series(records: list[dict[str, Any]], key: str) -> list[float]:
    return [r.get(key, 0.0) for r in records]


def _save_timeline(records: list[dict[str, Any]], path: Path, title: str, brake_distance: float) -> None:
    t = _series(records, "t")
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    a1.plot(t, _series(records, "ego_speed"), color="#2563eb", label="ego speed (m/s)")
    a1.fill_between(t, 0, max(_series(records, "ego_speed") or [1]),
                    where=[bool(r.get("braking")) for r in records],
                    color="#ef4444", alpha=0.12, label="braking")
    a1.set_ylabel("speed (m/s)")
    a1.legend(loc="upper right", fontsize=8)
    a1.set_title(title)
    a2.plot(t, _series(records, "n_detections"), color="#16a34a", label="detections")
    a2.plot(t, _series(records, "n_tracks"), color="#9333ea", label="tracks")
    a2.set_ylabel("count")
    a2.set_xlabel("time (s)")
    a2.legend(loc="upper right", fontsize=8)
    for a in (a1, a2):
        a.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def compare_runs(clean: list[dict[str, Any]], attacked: list[dict[str, Any]], path: str | Path) -> None:
    """Overlay two runs' speed and detection timelines (clean vs attacked)."""
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    a1.plot(_series(clean, "t"), _series(clean, "ego_speed"), color="#2563eb", label="clean")
    a1.plot(_series(attacked, "t"), _series(attacked, "ego_speed"), color="#ef4444", label="attacked")
    a1.set_ylabel("ego speed (m/s)")
    a1.legend(loc="upper right")
    a1.set_title("clean vs attacked")
    a2.plot(_series(clean, "t"), _series(clean, "n_detections"), color="#2563eb", label="clean detections")
    a2.plot(_series(attacked, "t"), _series(attacked, "n_detections"), color="#ef4444", label="attacked detections")
    a2.set_ylabel("detections")
    a2.set_xlabel("time (s)")
    a2.legend(loc="upper right")
    for a in (a1, a2):
        a.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
