"""Traffic light textures and roadside placement."""

import numpy as np


def test_traffic_lights_payload():
    from avsectester.attacks.object_insertion.traffic_light import (
        aspect_of,
        roadside_rig,
        signal_head_rgba,
        traffic_lights_rgba,
    )

    head = signal_head_rgba("red", height=300)
    assert head.shape[2] == 4 and (head[..., 3] == 255).any()  # RGBA, has opaque pixels
    # the lit (top, red) third is redder than the dark (bottom, green-off) third
    top = head[: head.shape[0] // 3][head[: head.shape[0] // 3, :, 3] > 0]
    bot = head[2 * head.shape[0] // 3:][head[2 * head.shape[0] // 3:, :, 3] > 0]
    assert top[:, 0].mean() > bot[:, 0].mean() + 20

    board = traffic_lights_rgba(n=3, lit="red", height=240)
    assert (board[..., 3] == 255).all()  # opaque board behind the heads
    assert 0.7 < aspect_of(board) < 1.0  # a wide-ish board (3 heads in a row)
    assert board.shape[1] > board.shape[0]  # wider than tall

    ((face, _), *_rest) = roadside_rig(board, x=26.0, y=-6.0, mount_height=2.2).planes()[-1:]
    assert np.isclose(face[3, 2], 2.2)  # bottom edge at the mount height
