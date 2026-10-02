"""Offline regressions for experiment method selection and saved driving results."""

import importlib
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from avsectester.metric import impact
from avsectester.simulators.nurec import StubRenderer


@pytest.fixture
def demo(monkeypatch):
    # Scripts import shared helpers as siblings, just as when invoked from the CLI.
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    return importlib.import_module("alpamayo_attack_demo")


@pytest.mark.parametrize("failure_stage", ["load", "inference"])
def test_experiment_pctnet_failure_raises_without_classic_fallback(demo, monkeypatch, failure_stage):
    harmonizer = demo.make_harmonizer("libcom", gpu=0)

    def fail(*args):
        raise RuntimeError(f"controlled {failure_stage} failure")

    def unexpected_fallback(*args):
        pytest.fail("An experiment labelled libcom must not run the classic harmonizer")

    monkeypatch.setattr(harmonizer, "_fallback", unexpected_fallback)
    if failure_stage == "load":
        monkeypatch.setattr(harmonizer, "_load", fail)
    else:
        monkeypatch.setattr(harmonizer, "_load", lambda: fail)
        harmonizer._dev = "cpu"

    frame = np.full((16, 16, 3), 90, np.uint8)
    mask = np.full((16, 16), 255, np.uint8)
    with pytest.raises(RuntimeError, match=f"PCTNet harmonization failed: controlled {failure_stage}"):
        harmonizer(frame, mask, frame)


def test_saved_steps_pair_reasoning_with_post_step_trace(demo, monkeypatch, tmp_path):
    class Renderer(StubRenderer):
        def __init__(self, **kwargs):
            super().__init__(cameras=kwargs["cameras"], height=8, width=8)

        def camera_model(self):
            return None  # The synthetic insertion below needs no geometry.

    class Model:
        def predict(self, observation):
            attacked = bool(observation.sensor_data[demo.CAM].any())
            speed = ([1.0, 0.0] if attacked else [3.0, 2.0])[observation.frame]
            return SimpleNamespace(
                candidate_positions=np.array([[[speed * 0.1, 0.0, 0.0]]]),
                reasoning_text=f"{'attacked' if attacked else 'clean'} step {observation.frame}",
            )

    def load_model(stack):
        stack._model = Model()

    # Keep the real stack adapter, run loop and trajectory follower; replace only external inputs.
    monkeypatch.setattr(demo, "NuRecRenderer", Renderer)
    monkeypatch.setattr(demo.AlpamayoAVStack, "_load", load_model)
    monkeypatch.setattr(demo.AlpamayoAVStack, "_prediction_input", lambda self, obs: obs)
    monkeypatch.setattr(demo, "roadside_sign_insert", lambda *a, **kw: lambda obs, rgb: rgb + 10)
    plotted = {}

    def capture_plot(clean, attacked, path, title):
        plotted.update(clean=clean, attacked=attacked)

    monkeypatch.setattr(demo, "plot_speed", capture_plot)
    monkeypatch.setattr(
        demo.sys, "argv",
        ["alpamayo_attack_demo.py", "--frames", "2", "--speed", "5", "--harmonizer", "none",
         "--out", str(tmp_path)],
    )
    assert demo.main() == 0

    saved = json.loads((tmp_path / "trace.json").read_text())
    for label, speeds in (("clean", [3.0, 2.0]), ("attacked", [1.0, 0.0])):
        rows = saved[label]
        assert len(rows) == 2
        assert [row["input_t"] for row in rows] == pytest.approx([0.0, 0.1])
        assert [row["t"] for row in rows] == pytest.approx([0.1, 0.2])
        assert [row["speed"] for row in rows] == pytest.approx(speeds)
        assert [row["reasoning"] for row in rows] == [f"{label} step 0", f"{label} step 1"]
        for row, record in zip(rows, plotted[label].records, strict=True):
            assert {key: row[key] for key in asdict(record)} == asdict(record)
    assert saved["verdict"] == str(impact(plotted["clean"], plotted["attacked"]))
