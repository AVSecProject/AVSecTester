"""Frame JSON data and NumPy buffers on a byte stream shared by the framework and its relay."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from itertools import islice
import json
import math
import os
import socket
import struct
from typing import Any

import numpy as np

from avsectester.structured_arrays import describe_dtype, dtype_from_description, item_bytes


_PREFIX = struct.Struct("!QI")
_MAX_FRAME_SIZE = 512 * 1024 * 1024
_BUFFER = "$buffer"
_FLOAT = "$float"
_MAX_IOV = os.sysconf("SC_IOV_MAX")


def _encode(value: Any, buffers: list[memoryview], offset: list[int]) -> Any:
    if isinstance(value, (np.ndarray, bytes)):
        if isinstance(value, bytes):
            description = None
            buffer = memoryview(value)
        else:
            description = {"dtype": describe_dtype(value.dtype), "shape": value.shape}
            if value.flags.c_contiguous:
                buffer = memoryview(value.reshape(-1).view(np.uint8)).cast("B")
            else:
                buffer = memoryview(item_bytes(value))
        start = offset[0]
        offset[0] += buffer.nbytes
        buffers.append(buffer)
        return {_BUFFER: [start, buffer.nbytes, description]}
    if isinstance(value, np.generic):
        return _encode(value.item(), buffers, offset)
    if isinstance(value, float) and not math.isfinite(value):
        return {_FLOAT: struct.pack("!d", value).hex()}
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) or key in (_BUFFER, _FLOAT) for key in value):
            raise TypeError("ROS wire mappings require string keys outside the format's tags")
        return {key: _encode(item, buffers, offset) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_encode(item, buffers, offset) for item in value]
    return value


def _decode(value: Any, buffers: memoryview) -> Any:
    if isinstance(value, dict):
        if _BUFFER in value:
            start, size, description = value[_BUFFER]
            if start < 0 or size < 0 or start + size > len(buffers):
                raise ValueError("ROS wire buffer lies outside its frame")
            buffer = buffers[start:start + size]
            if description is None:
                return bytes(buffer)
            dtype = dtype_from_description(description["dtype"])
            array = np.ndarray(tuple(description["shape"]), dtype=dtype, buffer=buffer)
            if array.nbytes != size:
                raise ValueError("ROS wire buffer size does not match the array dtype and shape")
            return array
        if _FLOAT in value:
            return struct.unpack("!d", bytes.fromhex(value[_FLOAT]))[0]
        return {key: _decode(item, buffers) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode(item, buffers) for item in value]
    return value


def send_frame(stream: socket.socket, message: Mapping[str, Any]) -> None:
    """Send one framed mapping. Array bytes remain separate from the JSON header."""
    buffers: list[memoryview] = []
    offset = [0]
    header = json.dumps(_encode(message, buffers, offset), separators=(",", ":"),
                        allow_nan=False).encode()
    length = len(header) + offset[0]
    if length > _MAX_FRAME_SIZE:
        raise ValueError("ROS wire frame exceeds the maximum size")
    remaining = deque([memoryview(_PREFIX.pack(length, len(header)) + header),
                       *(buffer for buffer in buffers if buffer)])
    while remaining:
        sent = stream.sendmsg(islice(remaining, _MAX_IOV))
        if not sent:
            raise EOFError("ROS relay socket disconnected")
        while remaining and sent >= len(remaining[0]):
            sent -= len(remaining.popleft())
        if sent:
            remaining[0] = remaining[0][sent:]


def _read_exact(stream: socket.socket, size: int) -> bytearray:
    result = bytearray(size)
    view = memoryview(result)
    while view:
        count = stream.recv_into(view, len(view), socket.MSG_WAITALL)
        if not count:
            raise EOFError("ROS relay socket disconnected")
        view = view[count:]
    return result


def receive_frame(stream: socket.socket) -> dict[str, Any]:
    """Read one complete frame. The decoded arrays own writable receive storage."""
    length, header_size = _PREFIX.unpack(_read_exact(stream, _PREFIX.size))
    if not 0 < header_size <= length <= _MAX_FRAME_SIZE:
        raise ValueError("Invalid ROS wire frame length")
    frame = memoryview(_read_exact(stream, length))
    message = _decode(json.loads(frame[:header_size].tobytes()), frame[header_size:])
    if not isinstance(message, dict):
        raise ValueError("ROS wire frame must contain a mapping")
    return message


__all__ = ["receive_frame", "send_frame"]
