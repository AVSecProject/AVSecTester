import math
import os
import socket
from threading import Event, Thread, get_ident
from time import monotonic, sleep

import numpy as np
import pytest

from avsectester.closed_loop.ros_interception.channel import RelayChannel, RelayConnection
from avsectester.closed_loop.ros_interception.wire import receive_frame, send_frame


def _wait_until(condition):
    deadline = monotonic() + 5.
    while not condition() and monotonic() < deadline:
        sleep(.001)
    assert condition()


def test_relay_channel_routes_frames(tmp_path):
    # Use a long framework path, as its container can see the directory at a shorter mount.
    directory = tmp_path / ("run-" * 30)
    directory.mkdir()
    path = directory / "relay.sock"
    channel = RelayChannel(path, relay_count=2)
    received = []
    reader_threads = []
    entered, release = Event(), Event()

    def handle(frame):
        reader_threads.append(get_ident())
        received.append((frame["relay_index"], frame["sequence"]))
        if frame["sequence"] == "first":
            entered.set()
            assert release.wait(5.)
        return {"kind": "reply", "sequence": frame["sequence"]}

    def connect(peer):
        peer.settimeout(5.)
        parent = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            peer.connect(f"/proc/self/fd/{parent}/{path.name}")
        finally:
            os.close(parent)

    try:
        assert channel.relay_count == 0
        with (
            socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as left,
            socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as right,
        ):
            connect(left)
            send_frame(left, {"kind": "ready"})
            _wait_until(lambda: channel.relay_count == 1)
            assert not channel.ready
            with pytest.raises(RuntimeError, match="has not connected all relays"):
                channel.send({"kind": "origin"})
            connect(right)
            send_frame(right, {"kind": "ready"})
            _wait_until(lambda: channel.ready)
            assert channel.relay_count == 2
            channel.start(handle)

            send_frame(left, {"kind": "message", "sequence": "first"})
            assert entered.wait(5.)
            send_frame(left, {"kind": "message", "sequence": "second"})
            # The writer may emit a control frame while a reader is preparing a reply.
            channel.send({"kind": "origin"})
            assert receive_frame(left) == receive_frame(right) == {"kind": "origin"}
            assert received == [(0, "first")]
            release.set()
            assert receive_frame(left) == {"kind": "reply", "sequence": "first"}
            assert receive_frame(left) == {"kind": "reply", "sequence": "second"}

            send_frame(left, {"kind": "message", "id": 7, "sequence": "left"})
            send_frame(right, {"kind": "message", "id": 7, "sequence": "right"})
            assert receive_frame(left) == {"kind": "reply", "sequence": "left"}
            assert receive_frame(right) == {"kind": "reply", "sequence": "right"}
            channel.send({"kind": "stop"})
            assert receive_frame(left) == receive_frame(right) == {"kind": "stop"}

            assert reader_threads and get_ident() not in reader_threads
            assert received[:2] == [(0, "first"), (0, "second")]
            assert set(received[2:]) == {(0, "left"), (1, "right")}
            send_frame(right, {"kind": "error", "reason": "relay graph failed"})
            _wait_until(lambda: not channel.ready)
            with pytest.raises(RuntimeError, match="relay graph failed"):
                channel.check()
            send_frame(left, {"kind": "message", "sequence": "after failure"})
            channel.close()
            assert len(received) == 4
    finally:
        release.set()
        channel.close()
    assert not path.exists()


def test_relay_channel_rejects_messages_before_start(tmp_path):
    path = tmp_path / "early.sock"
    channel = RelayChannel(path)
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as peer:
            peer.connect(os.fspath(path))
            send_frame(peer, {"kind": "ready"})
            _wait_until(lambda: channel.ready)
            send_frame(peer, {"kind": "message", "id": 7})
            _wait_until(lambda: not channel.ready)
            with pytest.raises(RuntimeError, match="sent 'message' before message processing started"):
                channel.check()
    finally:
        channel.close()


def test_relay_notifies_frames_and_disconnect(tmp_path):
    channel = RelayChannel(tmp_path / "peer.sock")
    notified = Event()
    relay = RelayConnection(channel.path, notify=notified.set)
    try:
        relay.send({"kind": "ready"})
        _wait_until(lambda: channel.ready)
        channel.send({"kind": "origin"})
        assert notified.wait(5.)
        (frame,) = relay.drain()
        assert frame["kind"] == "origin"
        notified.clear()

        channel.close()
        assert notified.wait(5.)
        with pytest.raises(RuntimeError, match="socket disconnected") as error:
            relay.check()
        assert isinstance(error.value.__cause__, EOFError)
    finally:
        relay.close()
        channel.close()


def test_wire_large_frame_and_partial_disconnect():
    dtype = np.dtype({"names": ["x", "flags", "sample"], "formats": ["<f4", "u1", (">i2", (2,))],
                      "offsets": [0, 8, 12], "itemsize": 20})
    raw = bytearray(range(80))
    points = np.frombuffer(raw, dtype=dtype)[::2]
    large = np.arange(2 ** 21, dtype=np.int32)
    frame = {"nested": [{"points": points, "opaque": bytes(raw[:4]), "gain": float("nan")}],
             "large": large, "small": [b"a"] * 1100}
    received = []
    sender, reader = socket.socketpair()
    with sender, reader:
        reader.settimeout(5.)
        worker = Thread(target=lambda: received.append(receive_frame(reader)))
        worker.start()
        send_frame(sender, frame)
        worker.join(5.)
        assert not worker.is_alive()
    (actual,) = received
    nested = actual["nested"][0]
    assert nested["points"].dtype == dtype
    assert nested["points"].tobytes() == bytes(raw[:20] + raw[40:60])
    assert nested["points"].flags.writeable
    assert nested["opaque"] == bytes(raw[:4]) and math.isnan(nested["gain"])
    np.testing.assert_array_equal(actual["large"], large)
    assert actual["small"] == frame["small"]

    sender, reader = socket.socketpair()
    with sender, reader:
        reader.settimeout(5.)
        sender.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4096)

        def interrupted_write():
            try:
                send_frame(sender, {"large": large})
            except (OSError, EOFError):
                pass  # The other thread has stopped this write before it could complete.

        worker = Thread(target=interrupted_write)
        worker.start()
        try:
            assert reader.recv(1, socket.MSG_PEEK)
            sender.shutdown(socket.SHUT_WR)
            with pytest.raises(EOFError, match="socket disconnected"):
                receive_frame(reader)
        finally:
            sender.close()
            worker.join(5.)
        assert not worker.is_alive()
