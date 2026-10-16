"""The nuScenes insertion demo skips keyframes it cannot replay for the requested decisions."""

from types import SimpleNamespace

from avsectester.datasets.base import FrameRef


def test_recorded_window_skips_keyframes_near_the_end_of_their_scene():
    from scripts.demos.nuscenes import nuscenes_object_demo as demo

    frames = (FrameRef("s", 0, "a", 1.0), FrameRef("s", 1, "b", 1.5))
    nusc = SimpleNamespace(
        sample_data=[{"filename": f"samples/{token}.jpg", "is_key_frame": True, "sample_token": token}
                     for token in ("a", "b")],
        get=lambda table, token: {"token": token, "scene_token": "s"},
    )

    def no_window(*args):
        raise AssertionError("a skipped keyframe must not build a window or source data")

    dataset = SimpleNamespace(_load=lambda: nusc, frames=lambda scenario: frames,
                              decision_window=no_window, source_data=no_window)
    assert demo.recorded_window(dataset, "samples/b.jpg", 1) is None  # last keyframe
    assert demo.recorded_window(dataset, "samples/a.jpg", 2) is None  # needs 3 frames
