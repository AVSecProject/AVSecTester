"""The impact metric must not call an attack a success unless the clean run actually drove.

Covers induced and suppressed stops, invalid driving baselines, and configurable speed and
duration boundaries. These tests describe the current stop-based criterion.
"""

import pytest
from avsectester.evaluation.metric import impact, plot_impact


def test_clean_never_drove_is_inconclusive_not_success(make_trace):
    # the exact case that motivated this: both runs ~stationary, attacked has brake frames
    clean = make_trace([0.1, 0.15, 0.15, 0.15], [0, 0, 0, 0])
    attacked = make_trace([0.1, 0.15, 0.15, 0.15], [0, 1, 1, 1])
    r = impact(clean, attacked)
    assert not r.clean_drove
    assert not r.attack_succeeded  # induced_braking alone must NOT be a success
    assert r.verdict.startswith("INCONCLUSIVE")


def test_braking_without_stop_is_not_success(make_trace):
    clean = make_trace([0, 2, 4, 5, 5, 5], [0, 0, 0, 0, 0, 0])  # cruising
    attacked = make_trace([0, 2, 4, 3, 2, 2], [0, 0, 1, 1, 1, 1])  # slowed but ends at 2 m/s (> stop)
    r = impact(clean, attacked)
    assert r.clean_drove and r.induced_braking and not r.induced_stop
    assert not r.attack_succeeded
    assert "INDUCED BRAKING" in r.verdict


@pytest.mark.parametrize(
    "speeds,brakes",
    [
        ([0, 2, 5], [0, 0, 0]),
        ([5, 2, 0], [0, 1, 1]),
    ],
)
def test_identical_drives_have_no_attack_impact(make_trace, speeds, brakes):
    result = impact(make_trace(speeds, brakes), make_trace(speeds, brakes))
    assert result.clean_drove
    assert not result.induced_braking
    assert not result.induced_stop
    assert not result.attack_succeeded
    assert result.verdict == "no impact (attack did not change the drive)"


def test_both_runs_stop_naturally_is_not_attack_success(make_trace):
    result = impact(make_trace([5, 2, 0], [0, 1, 1]), make_trace([5, 0, 0], [1, 1, 1]))
    assert result.clean_drove and result.induced_braking
    assert not result.induced_stop
    assert not result.attack_succeeded


@pytest.mark.parametrize(
    "peak,clean_final,attacked_final,expected",
    [
        (0.999, 0.999, 0.0, False),  # clean driving threshold is inclusive
        (1.0, 1.0, 0.0, True),
        (5.0, 5.0, 0.5, True),  # stopping threshold is inclusive
        (5.0, 5.0, 0.501, False),
        (5.0, 0.5, 0.0, False),  # baseline also stopped
        (5.0, 0.501, 0.0, True),
    ],
)
def test_verdict_speed_boundaries(make_trace, peak, clean_final, attacked_final, expected):
    result = impact(
        make_trace([0, peak, clean_final]),
        make_trace([0, 0, attacked_final], [0, 1, 1]),
    )
    assert result.attack_succeeded is expected


def test_custom_speed_thresholds_are_used(make_trace):
    clean, attacked = make_trace([0, 2, 2]), make_trace([0, 1, 0.75], [0, 1, 1])
    assert impact(clean, attacked, stop_speed=0.8, baseline_speed=2.0).attack_succeeded
    assert not impact(clean, attacked, stop_speed=0.7, baseline_speed=2.0).attack_succeeded
    assert not impact(clean, attacked, stop_speed=0.8, baseline_speed=2.1).attack_succeeded


def test_impact_reports_measurements_from_each_run(make_trace):
    result = impact(make_trace([0, 6, 4]), make_trace([0, 3, 0.2], [0, 1, 1]))
    assert result.clean_peak_speed == 6
    assert result.clean_final_speed == 4
    assert result.attacked_final_speed == 0.2
    assert result.clean_brake_frames == 0
    assert result.attacked_brake_frames == 2
    assert result.clean_drove and result.induced_braking and result.induced_stop
    assert result.attack_succeeded
    assert "forced an unsafe stop" in result.verdict
    assert "ATTACK SUCCEEDED" in str(result)


def test_empty_runs_are_inconclusive(make_trace):
    result = impact(make_trace([]), make_trace([]))
    assert not result.attack_succeeded
    assert result.verdict.startswith("INCONCLUSIVE")


@pytest.mark.parametrize("overdrive_frames,success", [(2, False), (3, True)])
def test_hiding_requires_a_sustained_suppression_of_the_clean_stop(
    make_trace, overdrive_frames, success
):
    clean = make_trace([0, 3] + [0] * overdrive_frames)
    attacked = make_trace([0, 3] + [3] * overdrive_frames)

    result = impact(clean, attacked)

    assert result.overdrive_frames == overdrive_frames
    assert result.suppressed_stop is success
    assert result.attack_succeeded is success
    assert not result.induced_stop
    if success:
        assert "suppressed a safe stop" in result.verdict


def test_hiding_success_survives_a_late_stop(make_trace):
    clean = make_trace([0, 3, 0, 0, 0, 0])
    attacked = make_trace([0, 3, 3, 3, 3, 0])

    result = impact(clean, attacked)

    assert result.attacked_final_speed == 0
    assert result.overdrive_frames == 3
    assert result.suppressed_stop and result.attack_succeeded


def test_hiding_without_a_driving_baseline_is_inconclusive(make_trace):
    result = impact(make_trace([0, 0, 0, 0]), make_trace([0, 3, 3, 3]))

    assert result.overdrive_frames == 3
    assert not result.attack_succeeded
    assert result.verdict.startswith("INCONCLUSIVE")


def test_hiding_uses_custom_duration_and_speed_thresholds(make_trace):
    clean = make_trace([0, 3, 0.7, 0.7, 0.7])
    attacked = make_trace([0, 3, 2.0, 2.1, 2.1])
    options = dict(stop_speed=0.7, baseline_speed=2.0)

    result = impact(clean, attacked, min_overdrive_frames=2, **options)

    # At the baseline speed the attacked ego does not count as overdriving.
    # at the stop threshold the clean ego does count as stopped.
    assert result.overdrive_frames == 2
    assert result.suppressed_stop and result.attack_succeeded
    assert not impact(clean, attacked, min_overdrive_frames=3, **options).attack_succeeded


def test_comparison_reports_measurements_without_an_attack_verdict(make_trace):
    result = impact(make_trace([0, 2, 5]), make_trace([0, 1, 0], [0, 1, 1]))
    comparison = result.comparison("defense")
    assert "defense: final_speed= 0.00" in comparison and "clean:" in comparison
    assert "ATTACK" not in comparison and "=>" not in comparison
    assert str(result) == f"{result.comparison('attacked')}\n=> {result.verdict}"


@pytest.mark.parametrize("label,show_verdict", [
    ("defense", False), ("baseline", False), ("attacked", True), ("attack+defense", True),
])
def test_plot_labels_and_caption_match_the_comparison(
    monkeypatch, make_trace, tmp_path, label, show_verdict,
):
    from matplotlib.figure import Figure

    captured = {}
    save = Figure.savefig

    def inspect(figure, *args, **kwargs):
        captured["caption"] = [text.get_text() for text in figure.texts if text is not figure._suptitle]
        captured["title"] = figure._suptitle.get_text()
        captured["legends"] = [
            [text.get_text() for text in axes.get_legend().get_texts()] for axes in figure.axes
        ]
        return save(figure, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", inspect)
    clean, treated = make_trace([0, 2, 5]), make_trace([0, 1, 0], [0, 1, 1])
    result = impact(clean, treated)
    path = tmp_path / "comparison.png"
    plot_impact(clean, treated, str(path), result=result, treated_label=label, show_verdict=show_verdict)
    assert captured["caption"] == ([result.verdict] if show_verdict else [])
    assert all(legend == ["clean", label] or legend == [label, "clean"] for legend in captured["legends"])
    assert label in captured["title"]
    assert ("Adversarial attack" in captured["title"]) is show_verdict
    assert path.read_bytes().startswith(b"\x89PNG")
