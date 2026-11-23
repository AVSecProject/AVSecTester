"""Local relay socket transport for framed messages."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import suppress
from functools import partial
import os
from pathlib import Path
from queue import Empty, SimpleQueue
import socket
from threading import Event, Lock, Thread
from time import time_ns
from typing import Any

from avsectester.closed_loop.simulators.messages import MessageChannel
from avsectester.validation import require_positive_integer

from .wire import receive_frame, send_frame


_SOCKET_BUFFER_SIZE = 8 * 1024 * 1024


def _check_failure(error: Exception | None) -> None:
    if error is not None:
        raise RuntimeError(f"ROS interception channel failed: {error}") from error


class _Connection:
    def __init__(self, stream: socket.socket, *,
                 on_frame: Callable[[dict[str, Any]], Mapping[str, Any] | None],
                 on_failure: Callable[[Exception], None]) -> None:
        self._stream = stream
        self._on_frame = on_frame
        self._on_failure = on_failure
        self._outgoing: SimpleQueue[Mapping[str, Any] | Event | None] = SimpleQueue()
        self._closed = Event()
        self._write_lock = Lock()
        # Large frames generally fit in one kernel write. The kernel may clamp these sizes
        # and the wire reader and writer handle partial transfers.
        for option in (socket.SO_SNDBUF, socket.SO_RCVBUF):
            stream.setsockopt(socket.SOL_SOCKET, option, _SOCKET_BUFFER_SIZE)
        self._reader = Thread(target=self._read, daemon=True, name="ros-interception-read")
        self._writer = Thread(target=self._write, daemon=True, name="ros-interception-write")
        self._reader.start()
        self._writer.start()

    def _read(self) -> None:
        try:
            while not self._closed.is_set():
                frame = receive_frame(self._stream)
                frame["read_ns"] = time_ns()
                reply = self._on_frame(frame)
                if reply is not None:
                    self._send_frame(reply)
        except Exception as error:
            if not self._closed.is_set():
                self._on_failure(error)

    def _write(self) -> None:
        try:
            while not self._closed.is_set():
                frame = self._outgoing.get()
                if frame is None:
                    return
                if isinstance(frame, Event):
                    frame.set()
                else:
                    self._send_frame(frame)
        except Exception as error:
            if not self._closed.is_set():
                self._on_failure(error)

    def _send_frame(self, frame: Mapping[str, Any]) -> None:
        with self._write_lock:
            send_frame(self._stream, frame)

    def send(self, frame: Mapping[str, Any]) -> None:
        if self._closed.is_set():
            raise RuntimeError("ROS interception channel is closed")
        self._outgoing.put(frame)

    def flush(self, timeout: float) -> bool:
        done = Event()
        self._outgoing.put(done)
        return done.wait(timeout)

    def close(self) -> None:
        self._closed.set()
        self._outgoing.put(None)
        with suppress(OSError):
            self._stream.shutdown(socket.SHUT_RDWR)
        self._reader.join()
        self._writer.join()
        self._stream.close()


class RelayChannel(MessageChannel):
    """Listen for a fixed set of relays on a run-owned Unix socket path."""

    def __init__(self, path: Path, *, relay_count: int = 1) -> None:
        require_positive_integer(relay_count, "relay count")
        self.path = path
        self._expected = relay_count
        self._connections: list[_Connection] = []
        self._ready: set[int] = set()
        self._handler: Callable[[dict[str, Any]], Mapping[str, Any] | None] | None = None
        self._lock = Lock()
        self._closed = False
        self._error: Exception | None = None
        self._listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            self._bind()
        except BaseException:
            self._listener.close()
            raise
        try:
            self._listener.listen(relay_count)
            self._listener.settimeout(0.1)
        except BaseException:
            self._listener.close()
            self.path.unlink()
            raise
        self._acceptor = Thread(target=self._accept, daemon=True, name="ros-interception-accept")
        self._acceptor.start()

    def _bind(self) -> None:
        path = os.fsencode(self.path)
        if len(path) < 108:
            self._listener.bind(path)
            return
        # The Linux sockaddr limit is checked before following the procfd symlink.
        parent = os.open(self.path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            self._listener.bind(f"/proc/self/fd/{parent}/{self.path.name}")
        finally:
            os.close(parent)

    def _accept(self) -> None:
        try:
            while not self._closed and len(self._connections) < self._expected:
                try:
                    stream, _ = self._listener.accept()
                except TimeoutError:
                    continue
                with self._lock:
                    if self._closed:
                        stream.close()
                        return
                    try:
                        relay_index = len(self._connections)
                        connection = _Connection(
                            stream, on_frame=partial(self._received, relay_index),
                            on_failure=self._fail,
                        )
                    except BaseException:
                        stream.close()
                        raise
                    self._connections.append(connection)
        except Exception as error:
            self._fail(error)

    def _fail(self, error: Exception) -> None:
        with self._lock:
            if not self._closed and self._error is None:
                self._error = error

    def _received(self, relay_index: int, frame: dict[str, Any]) -> Mapping[str, Any] | None:
        if self._closed or self._error is not None:
            return None
        kind = frame.get("kind")
        if kind == "ready":
            with self._lock:
                self._ready.add(relay_index)
            return None
        if kind == "error":
            raise RuntimeError(str(frame["reason"]))
        handler = self._handler
        if handler is None:
            raise RuntimeError(f"ROS relay {relay_index} sent {kind!r} before message processing started")
        frame["relay_index"] = relay_index
        reply = handler(frame)
        return reply if not self._closed and self._error is None else None

    def start(self, handler: Callable[[dict[str, Any]], Mapping[str, Any] | None]) -> None:
        """Install the handler for frames requiring processing."""
        with self._lock:
            self.check()
            if self._closed:
                raise RuntimeError("ROS interception channel is closed")
            if self._handler is not None:
                raise RuntimeError("ROS interception channel already started")
            self._handler = handler

    def send(self, frame: Mapping[str, Any]) -> None:
        """Queue a control frame once every expected relay is connected."""
        with self._lock:
            self.check()
            if self._closed:
                raise RuntimeError("ROS interception channel is closed")
            if len(self._connections) != self._expected:
                raise RuntimeError("ROS interception channel has not connected all relays")
            for connection in self._connections:
                connection.send(frame)

    def check(self) -> None:
        """Raise the first saved relay or transport failure."""
        _check_failure(self._error)

    @property
    def ready(self) -> bool:
        """Whether all expected relays have connected and reported readiness."""
        with self._lock:
            return (self._error is None and not self._closed
                    and len(self._ready) == self._expected)

    @property
    def relay_count(self) -> int:
        """Number of relays that have connected to this channel."""
        return len(self._connections)

    def close(self) -> None:
        """Release all relay connections and unlink the owned listener path."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._listener.close()
        self._acceptor.join()
        for connection in self._connections:
            connection.close()
        self.path.unlink(missing_ok=True)


class RelayConnection:
    """Connect to the framework and optionally notify the relay of frames and failures."""

    def __init__(self, path: Path, *, notify: Callable[[], None] | None = None) -> None:
        self._incoming: SimpleQueue[dict[str, Any]] = SimpleQueue()
        self._wake = Event()
        self._notify = notify
        self._error: Exception | None = None
        self._failure_lock = Lock()
        stream = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            stream.connect(str(path))
            self._connection = _Connection(
                stream, on_frame=self._received, on_failure=self._fail)
        except BaseException:
            stream.close()
            raise

    def _received(self, frame: dict[str, Any]) -> None:
        self._incoming.put(frame)
        self._signal()

    def _fail(self, error: Exception) -> None:
        with self._failure_lock:
            if self._error is not None:
                return
            self._error = error
        self._signal()

    def _signal(self) -> None:
        self._wake.set()
        if self._notify is not None:
            self._notify()

    def check(self) -> None:
        """Raise a stored stream or framing failure on the caller thread."""
        _check_failure(self._error)

    def wait(self, timeout: float | None = None) -> bool:
        """Wait for a frame, failure or closure, with an optional timeout in seconds."""
        return self._wake.wait(timeout)

    def drain(self) -> list[dict[str, Any]]:
        """Return currently queued frames in stream order without waiting for more."""
        self.check()
        self._wake.clear()
        received = []
        while True:
            try:
                received.append(self._incoming.get_nowait())
            except Empty:
                return received

    def send(self, frame: Mapping[str, Any]) -> None:
        """Queue a frame without blocking on socket I/O."""
        self.check()
        self._connection.send(frame)

    def flush(self, timeout: float) -> None:
        """Wait for previous outgoing frames to reach the socket."""
        self.check()
        if not self._connection.flush(timeout):
            self.check()
            raise TimeoutError("Timed out sending ROS interception frames")

    def close(self) -> None:
        """Stop both workers and release the connected socket."""
        self._connection.close()
        self._wake.set()


__all__ = ["RelayChannel", "RelayConnection"]
